import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from elder_portrait.tag_schema import (
    find_unknown_tags,
    join_bio_tokens,
    normalize_event_bio,
    split_bio_tokens,
)


STRONG_PUNCT = set("。！？!?；;")
SECONDARY_PUNCT = set("，、,:：")
TRIM_CHARS = set([" ", "\t", "\n", "\r", "\u3000"])


def split_base_spans(text: str) -> List[Tuple[int, int]]:
    n = len(text)
    spans: List[Tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in STRONG_PUNCT:
            spans.append((start, i + 1))
            start = i + 1
    if start < n:
        spans.append((start, n))
    return spans


def split_long_span(
    text: str,
    start: int,
    end: int,
    target_max_len: int,
    target_min_len: int,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cur = start
    while end - cur > target_max_len:
        window_end = min(end, cur + target_max_len)
        cut = None
        for j in range(window_end - 1, cur + target_min_len - 1, -1):
            ch = text[j]
            if ch in STRONG_PUNCT or ch in SECONDARY_PUNCT:
                cut = j + 1
                break
        if cut is None:
            cut = window_end
        spans.append((cur, cut))
        cur = cut
    spans.append((cur, end))
    return spans


def trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
    s, e = start, end
    while s < e and text[s] in TRIM_CHARS:
        s += 1
    while e > s and text[e - 1] in TRIM_CHARS:
        e -= 1
    return s, e


def sentence_spans(
    text: str,
    target_max_len: int = 180,
    target_min_len: int = 50,
    min_segment_len: int = 20,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for s, e in split_base_spans(text):
        spans.extend(
            split_long_span(
                text=text,
                start=s,
                end=e,
                target_max_len=target_max_len,
                target_min_len=target_min_len,
            )
        )

    cleaned: List[Tuple[int, int]] = []
    for s, e in spans:
        s, e = trim_span(text, s, e)
        if s < e:
            cleaned.append((s, e))

    merged: List[Tuple[int, int]] = []
    for s, e in cleaned:
        if merged and (e - s) < min_segment_len:
            ps, _ = merged[-1]
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split newly appended long rows in storywell_2 into sentence-level rows."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="dataset/storywell_2.csv",
        help="Path to storywell_2.csv",
    )
    parser.add_argument(
        "--split_id_min",
        type=int,
        default=38,
        help="Only split rows with id >= this value.",
    )
    parser.add_argument(
        "--target_max_len",
        type=int,
        default=180,
        help="Target max sentence length when forced split.",
    )
    parser.add_argument(
        "--target_min_len",
        type=int,
        default=50,
        help="Min length before searching backward punctuation for forced split.",
    )
    parser.add_argument(
        "--min_segment_len",
        type=int,
        default=20,
        help="If a segment is shorter than this and has previous segment, merge into previous.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before overwrite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_cols = {"id", "text", "event_bio", "sentiment"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    keep_df = df[df["id"] < args.split_id_min].copy().sort_values("id")
    split_df = df[df["id"] >= args.split_id_min].copy().sort_values("id")

    if split_df.empty:
        print(f"No rows with id >= {args.split_id_min}; nothing changed.")
        return

    rows = []
    next_id = args.split_id_min
    split_count_before = len(split_df)
    split_count_after = 0

    for _, row in split_df.iterrows():
        text = str(row["text"])
        tags = split_bio_tokens(str(row["event_bio"]))
        if len(tags) != len(text):
            raise ValueError(
                f"id={int(row['id'])} text/tag length mismatch: "
                f"text={len(text)} tags={len(tags)}"
            )

        spans = sentence_spans(
            text=text,
            target_max_len=args.target_max_len,
            target_min_len=args.target_min_len,
            min_segment_len=args.min_segment_len,
        )
        for s, e in spans:
            sub_text = text[s:e]
            sub_bio = normalize_event_bio(join_bio_tokens(tags[s:e], prefer_tab=True))
            if len(split_bio_tokens(sub_bio)) != len(sub_text):
                raise ValueError(
                    f"Split mismatch at original id={int(row['id'])}: "
                    f"text={len(sub_text)} tags={len(split_bio_tokens(sub_bio))}"
                )
            rows.append(
                {
                    "id": int(next_id),
                    "text": sub_text,
                    "event_bio": sub_bio,
                    "sentiment": int(row["sentiment"]),
                }
            )
            next_id += 1
            split_count_after += 1

    result_df = pd.concat(
        [keep_df[["id", "text", "event_bio", "sentiment"]], pd.DataFrame(rows)],
        axis=0,
        ignore_index=True,
    )
    result_df = result_df.sort_values("id").reset_index(drop=True)
    result_df["id"] = result_df["id"].astype(int)
    result_df["sentiment"] = result_df["sentiment"].astype(int)
    result_df["text"] = result_df["text"].astype(str)
    result_df["event_bio"] = result_df["event_bio"].astype(str).map(normalize_event_bio)

    unknown_tags = find_unknown_tags(result_df["event_bio"])
    if unknown_tags:
        raise ValueError(
            "Found unknown tags after split: " + ", ".join(unknown_tags[:20])
        )

    if args.backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak_split")
        csv_path.replace(backup_path)
        print(f"Backup created: {backup_path}")

    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Split rows before: {split_count_before}")
    print(f"Split rows after : {split_count_after}")
    print(f"Total rows now   : {len(result_df)}")
    print(
        "Sentiment dist   :",
        result_df["sentiment"].value_counts().sort_index().to_dict(),
    )


if __name__ == "__main__":
    main()
