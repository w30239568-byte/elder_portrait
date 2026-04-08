import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from elder_portrait.tag_schema import split_bio_tokens


# Guideline lexicons from user rules.
NEGATIVE_WORDS: List[str] = [
    "住院",
    "疼痛",
    "失眠",
    "焦虑",
    "无法行动",
    "孤独",
    "孤单",
    "恶化",
    "摔伤",
    "患病",
    "隔离",
    "加重",
    "无法",
    "不能",
    "不便",
    "受限",
    "痛苦",
]

POSITIVE_WORDS: List[str] = [
    "康复",
    "好转",
    "陪伴",
    "开心",
    "获奖",
    "团聚",
    "支持",
    "和睦",
    "活跃",
    "满足",
    "知足",
    "恢复",
    "改善",
    "稳定",
    "正常",
]

NEUTRAL_OBJECTIVE_WORDS: List[str] = [
    "出生",
    "就读",
    "任职",
    "搬家",
    "籍贯",
    "职业",
    "学历",
    "经历",
    "居住",
]

RECOVERY_TERMS: List[str] = ["康复", "恢复", "好转", "改善", "已能", "能够", "可以"]
SUPPORT_TERMS: List[str] = ["探望", "陪伴", "照顾", "支持", "关心", "子女", "家人", "社区", "问诊"]
SOCIAL_IMPAIR_TERMS: List[str] = ["独居", "孤单", "孤独", "无法出门", "几乎无法出门", "缺少交流", "社交减少"]
TREATMENT_OBJECTIVE_TERMS: List[str] = ["调理", "复查", "治疗", "服药", "吃药", "中药"]


def _count_hits(text: str, words: Iterable[str]) -> int:
    content = str(text or "")
    return sum(content.count(w) for w in words)


def _has_any(text: str, words: Iterable[str]) -> bool:
    content = str(text or "")
    return any(w in content for w in words)


def _has_health_tag(tags: List[str]) -> bool:
    tag_set = set(tags)
    return any(
        t in tag_set for t in {"B-Health_pro", "I-Health_pro", "B-Health_par", "I-Health_par"}
    )


def _compute_scores(text: str, event_bio: str) -> Tuple[float, float, dict]:
    content = str(text or "")
    tags = split_bio_tokens(str(event_bio or ""))

    pos_hits = _count_hits(content, POSITIVE_WORDS)
    neg_hits = _count_hits(content, NEGATIVE_WORDS)
    neutral_hits = _count_hits(content, NEUTRAL_OBJECTIVE_WORDS)

    pos_score = float(pos_hits)
    neg_score = float(neg_hits)

    # Phrase-level cues.
    if _has_any(content, RECOVERY_TERMS):
        pos_score += 0.9
    if _has_any(content, SUPPORT_TERMS):
        pos_score += 0.6
    if _has_any(content, SOCIAL_IMPAIR_TERMS):
        neg_score += 1.0

    # Pattern emphasis.
    if re.search(r"(几乎|长期|一直).{0,4}(无法|不能|不便|受限)", content):
        neg_score += 1.2
    if re.search(r"(术后|经过).{0,10}(康复|恢复|好转|已能|能够|可以)", content):
        pos_score += 1.5
    if re.search(r"(恶化|加重|反复|持续).{0,6}(疼|痛|不适|失眠|焦虑)", content):
        neg_score += 1.2

    # Health BIO prior: mild and conditional.
    health_tag = _has_health_tag(tags)
    if health_tag:
        if _has_any(content, RECOVERY_TERMS):
            pos_score += 0.4
        elif _has_any(content, ["恶化", "加重", "无法", "不能", "疼痛", "摔伤"]):
            neg_score += 0.5
        elif _has_any(content, TREATMENT_OBJECTIVE_TERMS):
            # Objective treatment facts should not be forced to negative.
            pos_score += 0.0
            neg_score += 0.0

    # Achievement prior.
    tag_set = set(tags)
    if any(t in tag_set for t in {"B-Achievement_pro", "I-Achievement_pro"}):
        pos_score += 0.8

    meta = {
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
        "neutral_hits": neutral_hits,
        "health_tag": health_tag,
    }
    return pos_score, neg_score, meta


def infer_base(text: str, event_bio: str) -> dict:
    content = str(text or "")
    pos_score, neg_score, meta = _compute_scores(content, event_bio)
    diff = float(pos_score - neg_score)
    has_pos = pos_score > 0
    has_neg = neg_score > 0

    objective_only = (
        meta["neutral_hits"] > 0 and meta["pos_hits"] == 0 and meta["neg_hits"] == 0 and abs(diff) < 0.6
    )
    mixed_close = has_pos and has_neg and abs(diff) <= 0.9

    if objective_only:
        base_label = 1
    elif mixed_close:
        # User rule: conflicting polarity with similar strength -> neutral.
        base_label = 1
    elif has_pos and has_neg:
        # Stronger side wins.
        base_label = 2 if diff > 0 else 0
    elif has_pos:
        base_label = 2 if pos_score >= 1.2 else 1
    elif has_neg:
        base_label = 0 if neg_score >= 1.2 else 1
    else:
        base_label = 1

    strong_extreme = abs(diff) >= 2.0 and base_label in {0, 2}
    lock_neutral = objective_only or mixed_close
    return {
        "base_label": int(base_label),
        "diff": diff,
        "lock_neutral": bool(lock_neutral),
        "strong_extreme": bool(strong_extreme),
    }


def calibrate_labels(base_rows: List[dict], target_neg_ratio: float, target_pos_ratio: float) -> List[int]:
    """
    Keep guideline rules as hard constraints, then softly calibrate class ratio:
    - lock_neutral rows stay 1
    - strong_extreme rows keep 0/2
    - remaining rows are assigned by score quantile to approach target ratios
    """
    n = len(base_rows)
    if n == 0:
        return []

    target_neg = int(round(n * max(0.0, min(0.49, float(target_neg_ratio)))))
    target_pos = int(round(n * max(0.0, min(0.49, float(target_pos_ratio)))))

    labels = np.ones(n, dtype=np.int64)

    for i, row in enumerate(base_rows):
        if row["lock_neutral"]:
            labels[i] = 1
        elif row["strong_extreme"]:
            labels[i] = int(row["base_label"])

    current_neg = int(np.sum(labels == 0))
    current_pos = int(np.sum(labels == 2))

    candidate_idx = [
        i
        for i, row in enumerate(base_rows)
        if not row["lock_neutral"] and not row["strong_extreme"]
    ]
    if candidate_idx:
        # Fill negatives from lowest diff, positives from highest diff.
        sorted_low = sorted(candidate_idx, key=lambda i: float(base_rows[i]["diff"]))
        sorted_high = sorted(candidate_idx, key=lambda i: float(base_rows[i]["diff"]), reverse=True)

        need_neg = max(0, target_neg - current_neg)
        need_pos = max(0, target_pos - current_pos)

        used = set()
        for i in sorted_low:
            if need_neg <= 0:
                break
            labels[i] = 0
            used.add(i)
            need_neg -= 1

        for i in sorted_high:
            if need_pos <= 0:
                break
            if i in used:
                continue
            labels[i] = 2
            used.add(i)
            need_pos -= 1

    return labels.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel sentiment for storywell_raw_split_new.csv by guideline + calibration."
    )
    parser.add_argument("--csv_path", type=str, default="dataset/storywell_raw_split_new.csv")
    parser.add_argument(
        "--target_neg_ratio",
        type=float,
        default=0.22,
        help="Soft target ratio of label 0.",
    )
    parser.add_argument(
        "--target_pos_ratio",
        type=float,
        default=0.22,
        help="Soft target ratio of label 2.",
    )
    parser.add_argument("--backup", action="store_true", help="Create backup before overwrite.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_cols = {"text", "event_bio", "sentiment"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    before = df["sentiment"].astype(int).value_counts().sort_index().to_dict()
    old_vals = df["sentiment"].astype(int).tolist()

    base_rows = [
        infer_base(text=t, event_bio=b)
        for t, b in zip(df["text"].astype(str), df["event_bio"].astype(str))
    ]
    new_vals = calibrate_labels(
        base_rows=base_rows,
        target_neg_ratio=float(args.target_neg_ratio),
        target_pos_ratio=float(args.target_pos_ratio),
    )

    changed = sum(int(a != b) for a, b in zip(old_vals, new_vals))
    df["sentiment"] = pd.Series(new_vals, dtype="int64")

    if args.backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak_sentiment_guideline")
        csv_path.replace(backup_path)
        print(f"Backup created: {backup_path}")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    after = df["sentiment"].value_counts().sort_index().to_dict()
    print(f"Saved: {csv_path}")
    print(f"Changed labels: {changed}/{len(df)}")
    print(f"Sentiment before: {before}")
    print(f"Sentiment after : {after}")


if __name__ == "__main__":
    main()
