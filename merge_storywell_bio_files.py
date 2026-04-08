import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from elder_portrait.tag_schema import (
    find_unknown_tags,
    join_bio_tokens,
    normalize_event_bio,
    split_bio_tokens,
)


TAG_MAP: Dict[str, str] = {
    "Activity_pro": "B-Social Activity_pro",
    "background_pro": "B-Education background_pro",
    "B-TRAVEL": "B-Social Activity_pro",
    "I-TRAVEL": "I-Social Activity_pro",
    "B-LEISURE": "B-Interest_pro",
    "I-LEISURE": "I-Interest_pro",
}


def parse_bio_file(path: Path) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    chars: List[str] = []
    tags: List[str] = []
    bad_lines = 0

    for line in text.splitlines():
        if not line.strip():
            continue
        m = re.match(r"^(.)\s+(.+?)\s*$", line)
        if not m:
            bad_lines += 1
            continue
        ch = m.group(1)
        tag_raw = m.group(2)
        tag = TAG_MAP.get(tag_raw, tag_raw)
        chars.append(ch)
        tags.append(tag)

    if not chars:
        raise ValueError(f"No valid token-tag lines parsed from: {path}")
    if bad_lines > 0:
        print(f"Warning: skipped {bad_lines} malformed lines in {path.name}")

    merged_text = "".join(chars)
    merged_bio = normalize_event_bio(join_bio_tokens(tags, prefer_tab=True))
    if len(split_bio_tokens(merged_bio)) != len(merged_text):
        raise ValueError(
            f"Length mismatch after parse in {path.name}: "
            f"text={len(merged_text)}, tags={len(split_bio_tokens(merged_bio))}"
        )
    return merged_text, merged_bio


def infer_sentiment_from_name(name: str) -> int:
    stem = Path(name).name
    if stem.startswith("12."):
        return 2
    if stem.startswith("13."):
        return 1
    if stem.startswith("15."):
        return 0
    if "白崇禧" in stem:
        return 1
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge selected StoryWell BIO files into storywell_2.csv"
    )
    parser.add_argument(
        "--target_csv",
        type=str,
        default="dataset/storywell_2.csv",
        help="Target CSV path to append merged rows.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup file before overwrite.",
    )
    parser.add_argument(
        "--bio_files",
        nargs="+",
        required=True,
        help="BIO file paths to merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_csv = Path(args.target_csv)
    if not target_csv.exists():
        raise FileNotFoundError(f"Target CSV not found: {target_csv}")

    df = pd.read_csv(target_csv, encoding="utf-8-sig")
    required_cols = {"id", "text", "event_bio", "sentiment"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Target CSV missing columns: {sorted(missing)}")

    bio_paths = [Path(p) for p in args.bio_files]
    for p in bio_paths:
        if not p.exists():
            raise FileNotFoundError(f"BIO file not found: {p}")

    existing_texts = set(df["text"].astype(str).tolist())
    next_id = int(pd.to_numeric(df["id"], errors="coerce").max()) + 1
    new_rows: List[Dict] = []

    for p in bio_paths:
        text, event_bio = parse_bio_file(p)
        if text in existing_texts:
            print(f"Skip duplicate text by content: {p.name}")
            continue
        new_rows.append(
            {
                "id": next_id,
                "text": text,
                "event_bio": event_bio,
                "sentiment": infer_sentiment_from_name(p.name),
            }
        )
        next_id += 1

    if not new_rows:
        print("No new rows appended.")
        return

    new_df = pd.DataFrame(new_rows, columns=["id", "text", "event_bio", "sentiment"])
    merged_df = pd.concat([df, new_df], axis=0, ignore_index=True)
    merged_df["id"] = merged_df["id"].astype(int)
    merged_df["sentiment"] = merged_df["sentiment"].astype(int)
    merged_df["text"] = merged_df["text"].astype(str)
    merged_df["event_bio"] = merged_df["event_bio"].astype(str).map(normalize_event_bio)

    unknown_tags = find_unknown_tags(merged_df["event_bio"])
    if unknown_tags:
        raise ValueError(
            "Found tags outside StoryWell schema after merge: "
            + ", ".join(unknown_tags[:20])
        )

    if args.backup:
        backup_path = target_csv.with_suffix(target_csv.suffix + ".bak_merge")
        target_csv.replace(backup_path)
        print(f"Backup created: {backup_path}")

    merged_df.to_csv(target_csv, index=False, encoding="utf-8-sig")

    print(f"Merged rows: {len(new_rows)}")
    print(f"New total rows: {len(merged_df)}")
    print(f"Sentiment distribution: {merged_df['sentiment'].value_counts().sort_index().to_dict()}")
    print("Added ids:", [int(r["id"]) for r in new_rows])
    for row in new_rows:
        print(
            f"  id={row['id']}, sentiment={row['sentiment']}, "
            f"text_preview={str(row['text'])[:42]}"
        )


if __name__ == "__main__":
    main()
