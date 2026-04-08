import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from elder_portrait.tag_schema import STORYWELL_ALLOWED_TAG_SET


LINE_PATTERN = re.compile(r"^(.*?)[ \t]+(.+?)\s*$")
STRONG_PUNCT = set("。！？!?；;")
WEAK_PUNCT = set("，,、：:")
TRIM_CHARS = set([" ", "\t", "\n", "\r", "\u3000"])


def parse_line(line: str) -> Tuple[str, str]:
    m = LINE_PATTERN.match(line)
    if not m:
        raise ValueError(f"Invalid BIO line: {line!r}")
    ch = m.group(1)
    tag = m.group(2)
    if ch == "":
        ch = " "
    return ch, tag


def parse_bio_file(path: Path) -> Tuple[str, List[str]]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    chars: List[str] = []
    tags: List[str] = []

    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        ch, tag = parse_line(line)
        chars.append(ch)
        tags.append(tag)

    if not chars:
        raise ValueError(f"No valid BIO lines in {path}")
    if len(chars) != len(tags):
        raise ValueError(
            f"Length mismatch in {path.name}: text={len(chars)}, tags={len(tags)}"
        )
    return "".join(chars), tags


def trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
    s, e = start, end
    while s < e and text[s] in TRIM_CHARS:
        s += 1
    while e > s and text[e - 1] in TRIM_CHARS:
        e -= 1
    return s, e


def split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    n = len(text)
    spans: List[Tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in STRONG_PUNCT:
            spans.append((start, i + 1))
            start = i + 1
    if start < n:
        spans.append((start, n))

    cleaned: List[Tuple[int, int]] = []
    for s, e in spans:
        s, e = trim_span(text, s, e)
        if s < e:
            cleaned.append((s, e))

    # Split extra-long spans by weak punctuation.
    final_spans: List[Tuple[int, int]] = []
    for s, e in cleaned:
        if e - s <= 220:
            final_spans.append((s, e))
            continue
        cur = s
        while e - cur > 220:
            window_end = min(e, cur + 220)
            cut = None
            for j in range(window_end - 1, cur + 40, -1):
                if text[j] in WEAK_PUNCT or text[j] in STRONG_PUNCT:
                    cut = j + 1
                    break
            if cut is None:
                cut = window_end
            final_spans.append((cur, cut))
            cur = cut
        if cur < e:
            final_spans.append((cur, e))

    merged: List[Tuple[int, int]] = []
    for s, e in final_spans:
        s, e = trim_span(text, s, e)
        if s >= e:
            continue
        if merged and (e - s) < 12:
            ps, _ = merged[-1]
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def build_rows(
    file_paths: List[Path],
    default_sentiment: int,
) -> Tuple[List[Dict], List[str]]:
    rows: List[Dict] = []
    unknown_tags: set[str] = set()
    next_id = 1

    for file_path in file_paths:
        text, tags = parse_bio_file(file_path)
        spans = split_sentence_spans(text)
        for sent_idx, (s, e) in enumerate(spans, start=1):
            sub_text = text[s:e]
            sub_tags = tags[s:e]
            if len(sub_text) != len(sub_tags):
                raise ValueError(
                    f"Split mismatch in {file_path.name}: text={len(sub_text)}, tags={len(sub_tags)}"
                )
            for t in sub_tags:
                if t not in STORYWELL_ALLOWED_TAG_SET:
                    unknown_tags.add(t)

            # Preserve raw tags exactly; use TAB to separate per-char tags so tags
            # containing spaces are still lossless.
            event_bio = "\t".join(sub_tags)
            rows.append(
                {
                    "id": next_id,
                    "text": sub_text,
                    "event_bio": event_bio,
                    "sentiment": int(default_sentiment),
                    "source_file": file_path.name,
                    "source_sentence_index": sent_idx,
                }
            )
            next_id += 1

    return rows, sorted(unknown_tags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a new sentence-split dataset from StoryWell .txt.bio files "
            "while preserving raw (possibly incompatible) BIO tags."
        )
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="dataset/storywell_raw_split_new.csv",
    )
    parser.add_argument(
        "--default_sentiment",
        type=int,
        default=1,
        help="Sentiment label for all generated rows.",
    )
    parser.add_argument(
        "--bio_files",
        nargs="+",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.bio_files]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"BIO file not found: {p}")

    rows, unknown = build_rows(paths, default_sentiment=args.default_sentiment)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "text",
            "event_bio",
            "sentiment",
            "source_file",
            "source_sentence_index",
        ],
    )
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_path}")
    print(f"Rows : {len(df)}")
    print(f"Files: {len(paths)}")
    print("Sentiment distribution:", df["sentiment"].value_counts().sort_index().to_dict())
    print("Unknown (incompatible) tag count:", len(unknown))
    if unknown:
        print("Unknown tags sample:", unknown[:30])
    print("event_bio separator: TAB (\\t)")


if __name__ == "__main__":
    main()

