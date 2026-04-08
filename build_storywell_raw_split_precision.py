import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from elder_portrait.tag_schema import (
    STORYWELL_ALLOWED_TAG_SET,
    join_bio_tokens,
    normalize_event_bio,
    split_bio_tokens,
    split_tag,
)


@dataclass
class RowPack:
    row_id: int
    text: str
    tags: List[str]
    sentiment: int
    source: str


EVENT_TYPES_FOR_SINGLE_CHAR_REPAIR = {
    "Health_pro",
    "Health_par",
    "Activity_pro",
    "Social Activity_pro",
    "Interest_pro",
    "Achievement_pro",
}

LEGAL_SINGLE_CHAR_EVENT_TOKENS = {
    "\u764c",  # 癌
    "\u762b",  # 瘫
    "\u804b",  # 聋
    "\u76f2",  # 盲
    "\u75db",  # 痛
}

MOJIBAKE_MARKERS = {
    "\ufffd",  # replacement char
    "Ã",
    "Â",
    "â",
    "€",
    "™",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build precision/balance oriented training set from storywell_raw_split_new.csv. "
            "Adds data quality gates, hardset_v1 merge, hard negatives, and hard positives."
        )
    )
    parser.add_argument("--input_csv", type=str, default="dataset/storywell_raw_split_new.csv")
    parser.add_argument("--hardset_csv", type=str, default="dataset/hardset_v1.csv")
    parser.add_argument("--output_csv", type=str, default="dataset/storywell_raw_split_new_precision.csv")
    parser.add_argument("--report_json", type=str, default="dataset/data_qc_report.json")
    parser.add_argument("--min_char_freq", type=int, default=30)
    parser.add_argument("--max_non_o_ratio", type=float, default=0.01)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--min_window_chars", type=int, default=8)
    parser.add_argument("--max_hard_negative_ratio", type=float, default=0.15)
    parser.add_argument("--max_hard_positive_rows", type=int, default=100)
    parser.add_argument("--hardset_limit", type=int, default=500)
    parser.add_argument(
        "--single_char_event_repair",
        dest="single_char_event_repair",
        action="store_true",
        help="Enable single-char event span repair.",
    )
    parser.add_argument(
        "--disable_single_char_event_repair",
        dest="single_char_event_repair",
        action="store_false",
        help="Disable single-char event span repair.",
    )
    parser.set_defaults(single_char_event_repair=True)
    return parser.parse_args()


def _is_probably_garbled(text: str) -> bool:
    if not text:
        return False
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        return True
    # Typical mojibake patterns after wrong utf8/gbk conversion.
    if re.search(r"(?:\u951f|\u953f|\u9365|\u95c2|\u6a82|\u9357|\u9350|\u95c4){3,}", text):
        return True
    return False


def _repair_single_char_event_spans(text: str, tags: List[str]) -> Tuple[List[str], List[Dict]]:
    repaired = list(tags)
    changes: List[Dict] = []
    n = len(repaired)
    i = 0
    while i < n:
        tag = str(repaired[i])
        if tag == "O" or "-" not in tag:
            i += 1
            continue
        prefix, etype = split_tag(tag)
        if prefix not in {"B", "I"} or not etype:
            i += 1
            continue
        start = i
        j = i + 1
        while j < n and str(repaired[j]) == f"I-{etype}":
            j += 1
        span_text = str(text[start:j])
        span_len = int(j - start)
        if (
            span_len == 1
            and etype in EVENT_TYPES_FOR_SINGLE_CHAR_REPAIR
            and span_text not in LEGAL_SINGLE_CHAR_EVENT_TOKENS
        ):
            repaired[start] = "O"
            changes.append(
                {
                    "start": int(start),
                    "end": int(j),
                    "text": span_text,
                    "type": etype,
                }
            )
        i = j
    return repaired, changes


def _count_spans(tags: List[str]) -> Tuple[int, int]:
    total = 0
    single_char = 0
    i = 0
    n = len(tags)
    while i < n:
        tag = str(tags[i])
        if tag == "O" or "-" not in tag:
            i += 1
            continue
        prefix, etype = split_tag(tag)
        if prefix not in {"B", "I"} or not etype:
            i += 1
            continue
        start = i
        i += 1
        while i < n and str(tags[i]) == f"I-{etype}":
            i += 1
        total += 1
        if i - start == 1:
            single_char += 1
    return single_char, total


def _build_bio_from_phrase_spans(text: str, spans: List[Tuple[str, str]]) -> str:
    tags = ["O"] * len(text)
    occupied = [False] * len(text)
    for phrase, etype in spans:
        phrase = str(phrase)
        s = text.find(phrase)
        if s < 0:
            continue
        e = s + len(phrase)
        if e > len(text):
            continue
        if any(occupied[k] for k in range(s, e)):
            continue
        tags[s] = f"B-{etype}"
        for k in range(s + 1, e):
            tags[k] = f"I-{etype}"
        for k in range(s, e):
            occupied[k] = True
    return join_bio_tokens(tags)


def _build_hard_positive_rows(max_rows: int) -> List[Dict]:
    names = [
        "\u5f20\u82b3",
        "\u674e\u6842\u5170",
        "\u738b\u79cb\u9999",
        "\u8d75\u7d20\u73cd",
        "\u5468\u79c0\u82f1",
        "\u5434\u7f8e\u82b3",
        "\u90d1\u6dd1\u73cd",
        "\u9648\u79cb\u82f1",
        "\u5218\u7389\u73cd",
        "\u8c22\u5fb7\u82f1",
    ]
    times = [
        "\u6700\u8fd1",
        "\u4e0a\u5468",
        "\u8fd9\u4e24\u4e2a\u6708",
        "\u524d\u6bb5\u65f6\u95f4",
        "\u53bb\u5e74",
    ]
    health_pairs = [
        ("\u590d\u67e5\u8840\u538b", "\u6309\u65f6\u5403\u836f", 1),
        ("\u7761\u7720\u53d8\u6d45", "\u591c\u91cc\u5e38\u9192", 0),
        ("\u8170\u9178\u80cc\u75db", "\u8d77\u5e8a\u9700\u8981\u6276\u7740\u5e8a\u8fb9", 0),
        ("\u542c\u529b\u4e0b\u964d", "\u770b\u7535\u89c6\u9700\u8981\u5b57\u5e55", 1),
        ("\u89c6\u529b\u6a21\u7cca", "\u5fc5\u987b\u4f69\u6234\u8001\u82b1\u955c", 1),
        ("\u819d\u76d6\u75bc", "\u9634\u96e8\u5929\u53d1\u9178\u53d1\u6c89", 0),
        ("\u590d\u5065\u8bad\u7ec3", "\u6b65\u6001\u8f83\u524d\u7a33\u5b9a", 2),
        ("\u8840\u7cd6\u6ce2\u52a8", "\u533b\u751f\u5efa\u8bae\u63a7\u5236\u996e\u98df", 1),
        ("\u5934\u6655", "\u9700\u8981\u5750\u4e0b\u4f11\u606f", 0),
        ("\u5931\u7720", "\u767d\u5929\u7cbe\u795e\u4e0d\u592a\u597d", 0),
    ]
    participants = [
        "\u5973\u513f",
        "\u513f\u5b50",
        "\u8001\u4f34",
        "\u5b59\u5973",
        "\u90bb\u5c45",
        "\u793e\u5de5",
    ]
    rows: List[Dict] = []
    seen = set()

    for name in names:
        for time_word in times:
            for symptom, action, sentiment in health_pairs:
                text = (
                    f"{name}{time_word}\u8bf4{symptom}\uff0c"
                    f"\u793e\u533a\u533b\u751f\u5efa\u8bae{action}\u3002"
                )
                spans = [
                    (name, "protagonist"),
                    (symptom, "Health_pro"),
                    (action, "Health_pro"),
                ]
                bio = _build_bio_from_phrase_spans(text, spans)
                key = (text, bio, sentiment)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "text": text,
                        "event_bio": bio,
                        "sentiment": int(sentiment),
                        "source_row": -1,
                        "hard_negative": 0,
                        "hard_positive": 1,
                        "source": "hard_positive_generated",
                    }
                )
                if len(rows) >= max_rows:
                    return rows

    for name in names:
        for role in participants:
            text = (
                f"{name}\u7684{role}\u6bcf\u5468\u6765\u966a\u5979\u6563\u6b65\uff0c"
                "\u793e\u533a\u793e\u5de5\u4e5f\u4f1a\u63d0\u9192\u5979\u6309\u65f6\u5403\u836f\u3002"
            )
            spans = [
                (name, "protagonist"),
                (role, "participant_par"),
                ("\u6563\u6b65", "Social Activity_pro"),
                ("\u6309\u65f6\u5403\u836f", "Health_pro"),
            ]
            bio = _build_bio_from_phrase_spans(text, spans)
            key = (text, bio, 2)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "text": text,
                    "event_bio": bio,
                    "sentiment": 2,
                    "source_row": -1,
                    "hard_negative": 0,
                    "hard_positive": 1,
                    "source": "hard_positive_generated",
                }
            )
            if len(rows) >= max_rows:
                return rows

    return rows[:max_rows]


def _load_external_hardset(path: Path, limit: int) -> Tuple[List[Dict], Dict]:
    if not path.exists():
        return [], {"hardset_exists": False, "hardset_rows_loaded": 0}

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"text", "event_bio", "sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in hardset file: {sorted(missing)}")

    rows: List[Dict] = []
    invalid_alignment = 0
    invalid_tags = 0
    garbled_rows = 0
    unknown_tags = Counter()

    for idx, row in df.reset_index(drop=True).iterrows():
        text = str(row["text"])
        if _is_probably_garbled(text):
            garbled_rows += 1
            continue
        event_bio = normalize_event_bio(str(row["event_bio"]))
        tags = split_bio_tokens(event_bio)
        if len(tags) != len(text):
            invalid_alignment += 1
            continue
        bad = [t for t in tags if t not in STORYWELL_ALLOWED_TAG_SET]
        if bad:
            invalid_tags += 1
            unknown_tags.update(bad)
            continue
        rows.append(
            {
                "text": text,
                "event_bio": event_bio,
                "sentiment": int(row["sentiment"]),
                "source_row": int(idx),
                "hard_negative": 0,
                "hard_positive": 1,
                "source": "hardset_v1",
            }
        )
        if len(rows) >= int(limit):
            break

    return rows, {
        "hardset_exists": True,
        "hardset_rows_loaded": int(len(rows)),
        "hardset_invalid_alignment_rows": int(invalid_alignment),
        "hardset_invalid_tag_rows": int(invalid_tags),
        "hardset_garbled_rows": int(garbled_rows),
        "hardset_unknown_tags_top": unknown_tags.most_common(30),
    }


def _load_and_filter(
    df: pd.DataFrame,
    source_name: str,
    enable_single_char_event_repair: bool = True,
) -> Tuple[List[RowPack], Dict]:
    required = {"text", "event_bio", "sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    valid_rows: List[RowPack] = []
    invalid_len_rows: List[int] = []
    invalid_tag_rows: List[int] = []
    garbled_rows: List[int] = []
    unknown_tags = Counter()

    single_char_repair_rows = 0
    single_char_repair_count = 0
    single_char_repair_by_type = Counter()
    single_char_repair_tokens = Counter()

    single_char_before = 0
    span_total_before = 0
    single_char_after = 0
    span_total_after = 0

    for idx, row in df.reset_index(drop=True).iterrows():
        text = str(row["text"])
        if _is_probably_garbled(text):
            garbled_rows.append(int(idx))
            continue

        event_bio = normalize_event_bio(str(row["event_bio"]))
        tags = split_bio_tokens(event_bio)
        if len(tags) != len(text):
            invalid_len_rows.append(int(idx))
            continue

        bad = [t for t in tags if t not in STORYWELL_ALLOWED_TAG_SET]
        if bad:
            invalid_tag_rows.append(int(idx))
            unknown_tags.update(bad)
            continue

        sc_before, total_before = _count_spans(tags)
        single_char_before += int(sc_before)
        span_total_before += int(total_before)

        if enable_single_char_event_repair:
            tags, changes = _repair_single_char_event_spans(text=text, tags=tags)
            if changes:
                single_char_repair_rows += 1
                single_char_repair_count += len(changes)
                for ch in changes:
                    single_char_repair_by_type[str(ch["type"])] += 1
                    single_char_repair_tokens[str(ch["text"])] += 1

        sc_after, total_after = _count_spans(tags)
        single_char_after += int(sc_after)
        span_total_after += int(total_after)

        valid_rows.append(
            RowPack(
                row_id=int(idx),
                text=text,
                tags=tags,
                sentiment=int(row["sentiment"]),
                source=source_name,
            )
        )

    stats = {
        "source_name": source_name,
        "input_rows": int(len(df)),
        "valid_rows": int(len(valid_rows)),
        "garbled_rows": int(len(garbled_rows)),
        "garbled_row_examples": garbled_rows[:20],
        "invalid_alignment_rows": int(len(invalid_len_rows)),
        "invalid_tag_rows": int(len(invalid_tag_rows)),
        "invalid_alignment_row_examples": invalid_len_rows[:20],
        "invalid_tag_row_examples": invalid_tag_rows[:20],
        "unknown_tags": sorted(unknown_tags.keys()),
        "unknown_tags_top": unknown_tags.most_common(20),
        "single_char_span_in_gold_rate_before": round(
            single_char_before / max(1, span_total_before), 6
        ),
        "single_char_span_in_gold_rate": round(
            single_char_after / max(1, span_total_after), 6
        ),
        "single_char_event_repair_enabled": bool(enable_single_char_event_repair),
        "single_char_event_repair_rows": int(single_char_repair_rows),
        "single_char_event_repair_count": int(single_char_repair_count),
        "single_char_event_repair_by_type": {
            str(k): int(v) for k, v in sorted(single_char_repair_by_type.items())
        },
        "single_char_event_repair_tokens_top": single_char_repair_tokens.most_common(50),
    }
    return valid_rows, stats


def _mine_high_conf_o_chars(
    rows: List[RowPack],
    min_char_freq: int,
    max_non_o_ratio: float,
) -> Tuple[set[str], List[Dict]]:
    char_total = Counter()
    char_non_o = Counter()
    for row in rows:
        for ch, tag in zip(row.text, row.tags):
            if ch.strip() == "":
                continue
            if re.fullmatch(r"[\W_]+", ch):
                continue
            if not re.fullmatch(r"[0-9A-Za-z\u4e00-\u9fff]+", ch):
                continue
            char_total[ch] += 1
            if tag != "O":
                char_non_o[ch] += 1

    selected = set()
    detail: List[Dict] = []
    for ch, total in char_total.items():
        if total < int(min_char_freq):
            continue
        non_o = int(char_non_o.get(ch, 0))
        ratio = float(non_o / max(1, total))
        if ratio <= float(max_non_o_ratio):
            selected.add(ch)
            detail.append(
                {
                    "char": ch,
                    "total": int(total),
                    "non_o": int(non_o),
                    "non_o_ratio": round(ratio, 6),
                }
            )
    detail.sort(key=lambda x: (-x["total"], x["non_o_ratio"], x["char"]))
    return selected, detail


def _extract_hard_negatives(
    rows: List[RowPack],
    o_chars: set[str],
    window_size: int,
    min_window_chars: int,
    max_rows: int,
) -> List[Dict]:
    if not o_chars or max_rows <= 0:
        return []
    seen = set()
    out: List[Dict] = []
    for row in rows:
        n = len(row.text)
        if n < int(min_window_chars):
            continue
        for i, ch in enumerate(row.text):
            if ch not in o_chars:
                continue
            if row.tags[i] != "O":
                continue
            left = max(0, i - window_size // 2)
            right = min(n, left + int(window_size))
            left = max(0, right - int(window_size))
            span_text = row.text[left:right]
            span_tags = row.tags[left:right]
            if len(span_text) < int(min_window_chars):
                continue
            if any(t != "O" for t in span_tags):
                continue
            key = (span_text, row.sentiment)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "text": span_text,
                    "event_bio": join_bio_tokens(span_tags),
                    "sentiment": int(row.sentiment),
                    "source_row": int(row.row_id),
                    "hard_negative": 1,
                    "hard_positive": 0,
                    "source": "hard_negative_window",
                }
            )
            if len(out) >= max_rows:
                return out
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    hardset_path = Path(args.hardset_csv)
    output_path = Path(args.output_csv)
    report_path = Path(args.report_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    valid_rows, base_stats = _load_and_filter(
        df=df,
        source_name="base_dataset",
        enable_single_char_event_repair=bool(args.single_char_event_repair),
    )

    tag_dist = Counter()
    sent_dist = Counter()
    for row in valid_rows:
        tag_dist.update(row.tags)
        sent_dist.update([int(row.sentiment)])

    o_chars, o_char_detail = _mine_high_conf_o_chars(
        valid_rows,
        min_char_freq=int(args.min_char_freq),
        max_non_o_ratio=float(args.max_non_o_ratio),
    )

    max_hard_rows = int(len(valid_rows) * float(args.max_hard_negative_ratio))
    hard_rows = _extract_hard_negatives(
        rows=valid_rows,
        o_chars=o_chars,
        window_size=int(args.window_size),
        min_window_chars=int(args.min_window_chars),
        max_rows=max_hard_rows,
    )

    hard_positive_rows = _build_hard_positive_rows(
        max_rows=int(max(0, args.max_hard_positive_rows))
    )
    hardset_rows, hardset_stats = _load_external_hardset(
        path=hardset_path,
        limit=int(max(0, args.hardset_limit)),
    )

    out_rows: List[Dict] = []
    for i, row in enumerate(valid_rows, start=1):
        out_rows.append(
            {
                "id": int(i),
                "text": row.text,
                "event_bio": join_bio_tokens(row.tags),
                "sentiment": int(row.sentiment),
                "source": row.source,
                "hard_negative": 0,
                "hard_positive": 0,
                "hardset_v1": 0,
            }
        )
    for extra in hard_rows:
        out_rows.append(
            {
                "id": int(len(out_rows) + 1),
                "text": str(extra["text"]),
                "event_bio": str(extra["event_bio"]),
                "sentiment": int(extra["sentiment"]),
                "source": str(extra["source"]),
                "hard_negative": 1,
                "hard_positive": 0,
                "hardset_v1": 0,
            }
        )
    for extra in hard_positive_rows:
        out_rows.append(
            {
                "id": int(len(out_rows) + 1),
                "text": str(extra["text"]),
                "event_bio": str(extra["event_bio"]),
                "sentiment": int(extra["sentiment"]),
                "source": str(extra["source"]),
                "hard_negative": 0,
                "hard_positive": 1,
                "hardset_v1": 0,
            }
        )
    for extra in hardset_rows:
        out_rows.append(
            {
                "id": int(len(out_rows) + 1),
                "text": str(extra["text"]),
                "event_bio": str(extra["event_bio"]),
                "sentiment": int(extra["sentiment"]),
                "source": str(extra["source"]),
                "hard_negative": 0,
                "hard_positive": 1,
                "hardset_v1": 1,
            }
        )

    out_df = pd.DataFrame(out_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    report = {
        "input_csv": str(input_path),
        "hardset_csv": str(hardset_path),
        "output_csv": str(output_path),
        "report_json": str(report_path),
        **base_stats,
        **hardset_stats,
        "output_rows": int(len(out_df)),
        "hard_negative_rows": int(len(hard_rows)),
        "hard_positive_rows": int(len(hard_positive_rows)),
        "hardset_v1_rows": int(len(hardset_rows)),
        "hard_negative_ratio": round(len(hard_rows) / max(1, len(valid_rows)), 6),
        "selected_o_char_count": int(len(o_chars)),
        "selected_o_chars_top": o_char_detail[:80],
        "tag_distribution_top": tag_dist.most_common(60),
        "sentiment_distribution": {str(k): int(v) for k, v in sorted(sent_dist.items())},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved precision dataset: {output_path}")
    print(f"Saved QC report: {report_path}")
    print(
        "Summary:",
        {
            "valid_rows": report["valid_rows"],
            "hard_negative_rows": report["hard_negative_rows"],
            "hard_positive_rows": report["hard_positive_rows"],
            "hardset_v1_rows": report["hardset_v1_rows"],
            "output_rows": report["output_rows"],
            "invalid_alignment_rows": report["invalid_alignment_rows"],
            "garbled_rows": report["garbled_rows"],
            "single_char_span_in_gold_rate": report["single_char_span_in_gold_rate"],
        },
    )


if __name__ == "__main__":
    main()
