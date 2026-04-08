import argparse
from pathlib import Path

import pandas as pd


"""
Relabel sentiment for dataset/storywell_2.csv.

Label convention (same as project):
- 0: negative
- 1: neutral
- 2: positive
"""


NEGATIVE_IDS = {
    9,   # 三年自然灾害、挨饿、看到尸体
    11,  # 文革开始，研究搁置
    12,  # 文革冲击
    13,  # 下矿井再教育、生活艰苦
    14,  # 眼睛受损、治疗恢复有限
    26,  # 爱人骨折住院，长期卧床
    27,  # 眼睛不好，生活不便
    28,  # 照护压力大
}

POSITIVE_IDS = {
    1,   # 人生身份与履历亮点
    3,   # 学生会主席、受欢迎
    4,   # 大学阶段活跃
    6,   # 婚礼温馨、钻石婚
    8,   # 带队科研响应国家号召
    10,  # 突破性发现
    15,  # 科研工作重启推进
    17,  # 发表第一篇相关文章
    18,  # 国际反响大
    19,  # 媒体采访与国内外关注
    20,  # 开课成功、教学影响大
    24,  # 协助获得国家地质公园称号
    25,  # 高龄仍坚持科研
    29,  # 养老机构生活满意
    30,  # 餐饮服务积极评价
    31,  # 护理服务积极评价
    32,  # 客服服务积极评价
    33,  # 机构管理反馈积极
    34,  # 组织建设与持续活动
    37,  # 节日家庭团聚
}


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV must contain `id` and `sentiment` columns.")

    overlap = NEGATIVE_IDS.intersection(POSITIVE_IDS)
    if overlap:
        raise ValueError(f"IDs overlap in positive/negative sets: {sorted(overlap)}")

    out = df.copy()
    out["sentiment"] = 1
    out.loc[out["id"].isin(NEGATIVE_IDS), "sentiment"] = 0
    out.loc[out["id"].isin(POSITIVE_IDS), "sentiment"] = 2
    out["sentiment"] = out["sentiment"].astype(int)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel storywell_2 sentiment.")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/storywell_2.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/storywell_2.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup file before overwriting output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    out_df = relabel(df)

    if args.backup and output_path.exists():
        backup_path = output_path.with_suffix(output_path.suffix + ".bak")
        output_path.replace(backup_path)
        print(f"Backup created: {backup_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    counts = out_df["sentiment"].value_counts().sort_index().to_dict()
    print(f"Saved relabeled file to: {output_path}")
    print(f"Sentiment distribution: {counts}")
    for label in [0, 1, 2]:
        ids = out_df.loc[out_df["sentiment"] == label, "id"].tolist()
        print(f"label={label}, ids={ids}")


if __name__ == "__main__":
    main()
