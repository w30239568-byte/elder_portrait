import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


NEG_WORDS: Iterable[str] = [
    "侵略", "侵占", "战乱", "困难", "艰苦", "饥饿", "挨饿", "尸体", "病", "疼", "痛",
    "去世", "逝世", "受伤", "牺牲", "痛苦", "批斗", "下放", "文革", "贫困", "不幸",
    "住院", "复查", "失败", "压迫", "压榨", "关门", "无法", "不能", "卧床", "骨折", "感染",
]

POS_WORDS: Iterable[str] = [
    "欢迎", "温馨", "幸福", "荣誉", "获奖", "成就", "突出", "贡献", "先进", "优秀",
    "顺利", "毕业", "团结", "喜欢", "热爱", "结婚", "钻石婚", "坚持", "康复", "改善",
    "解放", "合作", "成功", "担任", "主任", "委员", "教授", "支持", "帮助", "恢复",
    "好转", "满意", "便利", "称号", "突破", "自豪",
]


def _count_hits(text: str, words: Iterable[str]) -> int:
    return sum(len(re.findall(re.escape(w), text)) for w in words)


def infer_sentiment(text: str, event_bio: str) -> int:
    t = str(text)
    bio = str(event_bio)
    pos = _count_hits(t, POS_WORDS)
    neg = _count_hits(t, NEG_WORDS)

    # BIO-based weak priors.
    if "B-Health_pro" in bio or "I-Health_pro" in bio or "B-Health_par" in bio or "I-Health_par" in bio:
        neg += 1
    if "B-Achievement_pro" in bio or "I-Achievement_pro" in bio:
        pos += 1

    # Simple negation around positive words.
    if re.search(r"(不|没|无|未).{0,2}(好|顺利|健康|成功|恢复|满意)", t):
        neg += 1

    score = pos - neg
    if score >= 2:
        return 2
    if score <= -1:
        return 0
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel sentence-level sentiment for storywell_2.")
    parser.add_argument("--csv_path", type=str, default="dataset/storywell_2.csv")
    parser.add_argument(
        "--id_min",
        type=int,
        default=38,
        help="Only relabel rows with id >= id_min.",
    )
    parser.add_argument("--backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"id", "text", "event_bio", "sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    mask = df["id"].astype(int) >= int(args.id_min)
    before_counts = df["sentiment"].value_counts().sort_index().to_dict()

    old_vals = df.loc[mask, "sentiment"].astype(int).tolist()
    new_vals = [
        infer_sentiment(text=t, event_bio=b)
        for t, b in zip(df.loc[mask, "text"].astype(str), df.loc[mask, "event_bio"].astype(str))
    ]
    changed = sum(int(a != b) for a, b in zip(old_vals, new_vals))
    df.loc[mask, "sentiment"] = new_vals
    df["sentiment"] = df["sentiment"].astype(int)

    if args.backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak_sentiment")
        csv_path.replace(backup_path)
        print(f"Backup created: {backup_path}")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    after_counts = df["sentiment"].value_counts().sort_index().to_dict()

    print(f"Relabeled rows (id >= {args.id_min}): {int(mask.sum())}")
    print(f"Changed labels: {changed}")
    print(f"Sentiment before: {before_counts}")
    print(f"Sentiment after : {after_counts}")


if __name__ == "__main__":
    main()
