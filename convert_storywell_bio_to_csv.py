import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

from elder_portrait.tag_schema import (
    STORYWELL_ALLOWED_TAG_SET,
    join_bio_tokens,
    split_bio_tokens,
)


LINE_PATTERN = re.compile(r"^(.*?)[ \t]+(.+?)\s*$")


def parse_line(line: str) -> Tuple[str, str]:
    match = LINE_PATTERN.match(line)
    if not match:
        raise ValueError(f"Invalid BIO line: {line!r}")
    char = match.group(1)
    tag = match.group(2)
    if not char:
        # Keep alignment for broken whitespace-only char lines.
        char = " "
    return char, tag


def flush_record(
    records: List[Tuple[int, str, str, int]],
    chars: List[str],
    tags: List[str],
    sentiment: int,
) -> None:
    if not chars:
        return
    text = "".join(chars)
    if len(text) != len(tags):
        raise ValueError(
            f"text/tag length mismatch: text={len(text)} tags={len(tags)}"
        )
    event_bio = join_bio_tokens(tags, prefer_tab=True)
    records.append((len(records) + 1, text, event_bio, sentiment))


def convert_bio_to_csv(src: Path, out: Path, sentiment: int) -> int:
    records: List[Tuple[int, str, str, int]] = []
    chars: List[str] = []
    tags: List[str] = []

    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.strip():
                flush_record(records, chars, tags, sentiment)
                chars, tags = [], []
                continue

            ch, tag = parse_line(line)
            chars.append(ch)
            tags.append(tag)

    flush_record(records, chars, tags, sentiment)

    unknown = sorted(
        {
            t
            for _, _, bio, _ in records
            for t in split_bio_tokens(bio)
            if t not in STORYWELL_ALLOWED_TAG_SET
        }
    )
    if unknown:
        preview = ", ".join(unknown[:12])
        raise ValueError(
            "Found tags outside StoryWell schema after conversion: "
            f"{preview}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "event_bio", "sentiment"])
        writer.writerows(records)
    return len(records)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert StoryWell .txt.bio file into project CSV format."
    )
    parser.add_argument("--input_bio", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument(
        "--default_sentiment",
        type=int,
        default=1,
        help="Default sentiment label used for converted rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.input_bio)
    out = Path(args.output_csv)
    count = convert_bio_to_csv(src, out, sentiment=args.default_sentiment)
    print(f"Saved {count} records to {out}")


if __name__ == "__main__":
    main()
