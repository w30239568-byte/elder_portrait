import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from elder_portrait.tag_schema import join_bio_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hardset_v1 (300-500 rows) for event boundary and role disambiguation."
    )
    parser.add_argument("--output_csv", type=str, default="dataset/hardset_v1.csv")
    parser.add_argument("--target_rows", type=int, default=360)
    return parser.parse_args()


def build_bio(text: str, spans: List[Tuple[str, str]]) -> str:
    tags = ["O"] * len(text)
    occupied = [False] * len(text)
    for phrase, etype in spans:
        start = text.find(phrase)
        if start < 0:
            continue
        end = start + len(phrase)
        if end > len(text):
            continue
        if any(occupied[i] for i in range(start, end)):
            continue
        tags[start] = f"B-{etype}"
        for i in range(start + 1, end):
            tags[i] = f"I-{etype}"
        for i in range(start, end):
            occupied[i] = True
    return join_bio_tokens(tags)


def build_rows(target_rows: int) -> List[Dict]:
    names = [
        "\u5f20\u82b3",
        "\u674e\u6842\u82b3",
        "\u5468\u79c0\u82f1",
        "\u738b\u79cb\u9999",
        "\u8d75\u7d20\u73cd",
        "\u9648\u7f8e\u82f1",
        "\u5434\u5170\u82f1",
        "\u5218\u7389\u73cd",
        "\u8c22\u534e\u82f1",
        "\u90d1\u6dd1\u73cd",
        "\u5510\u6842\u5170",
        "\u674e\u79cb\u73cd",
    ]
    relations = [
        "\u5973\u513f",
        "\u513f\u5b50",
        "\u8001\u4f34",
        "\u5b59\u5973",
        "\u5b59\u5b50",
        "\u90bb\u5c45",
    ]
    time_words = [
        "\u6700\u8fd1",
        "\u4e0a\u5468",
        "\u8fd9\u4e24\u4e2a\u6708",
        "\u4e0a\u4e2a\u6708",
        "\u53bb\u5e74",
        "\u524d\u6bb5\u65f6\u95f4",
    ]
    health_items = [
        ("\u590d\u67e5\u8840\u538b", "\u533b\u751f\u5efa\u8bae\u6309\u65f6\u5403\u836f", 1),
        ("\u8170\u9178\u80cc\u75db", "\u65e9\u4e0a\u8d77\u5e8a\u9700\u8981\u6276\u7740\u5e8a\u8fb9", 0),
        ("\u7761\u7720\u53d8\u6d45", "\u591c\u91cc\u5e38\u9192", 0),
        ("\u542c\u529b\u4e0b\u964d", "\u770b\u7535\u89c6\u9700\u8981\u5b57\u5e55", 1),
        ("\u89c6\u529b\u6a21\u7cca", "\u5fc5\u987b\u4f69\u6234\u8001\u82b1\u955c", 1),
        ("\u819d\u76d6\u75bc", "\u9634\u96e8\u5929\u53d1\u9178\u53d1\u6c89", 0),
        ("\u8840\u7cd6\u6ce2\u52a8", "\u63a7\u5236\u996e\u98df\u540e\u6709\u6240\u7f13\u89e3", 2),
        ("\u590d\u5065\u8bad\u7ec3", "\u6b65\u6001\u6bd4\u4e4b\u524d\u66f4\u7a33", 2),
        ("\u5931\u7720", "\u767d\u5929\u7cbe\u795e\u4e0d\u592a\u597d", 0),
        ("\u5934\u6655", "\u9700\u8981\u5750\u4e0b\u4f11\u606f", 0),
    ]
    social_items = [
        ("\u6563\u6b65", "\u793e\u4ea4\u652f\u6301", 2),
        ("\u6253\u592a\u6781", "\u793e\u4ea4\u6d3b\u8dc3", 2),
        ("\u53c2\u52a0\u793e\u533a\u6d3b\u52a8", "\u793e\u4ea4\u6d3b\u8dc3", 2),
        ("\u5f88\u5c11\u51fa\u95e8", "\u793e\u4ea4\u53d7\u635f", 0),
        ("\u51e0\u4e4e\u4e0d\u53c2\u4e0e\u793e\u533a\u6d3b\u52a8", "\u793e\u4ea4\u53d7\u635f", 0),
    ]

    rows: List[Dict] = []
    seen = set()

    for name in names:
        for time_word in time_words:
            for symptom, action, sentiment in health_items:
                text = f"{name}{time_word}\u603b\u8bf4{symptom}\uff0c\u793e\u533a\u533b\u9662\u590d\u67e5\u540e{action}\u3002"
                spans = [
                    (name, "protagonist"),
                    (symptom, "Health_pro"),
                    (action, "Health_pro"),
                ]
                bio = build_bio(text, spans)
                key = (text, bio, sentiment)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "text": text,
                        "event_bio": bio,
                        "sentiment": int(sentiment),
                        "source": "hardset_v1_health",
                    }
                )
                if len(rows) >= target_rows:
                    return rows

    for name in names:
        for relation in relations:
            for activity, _, sentiment in social_items:
                text = (
                    f"{name}\u7684{relation}\u6bcf\u5468\u6765\u966a\u5979{activity}\uff0c"
                    "\u793e\u5de5\u4e5f\u4f1a\u53ca\u65f6\u8ddf\u8fdb\u3002"
                )
                spans = [
                    (name, "protagonist"),
                    (relation, "participant_par"),
                    ("\u793e\u5de5", "participant_par"),
                    (activity, "Social Activity_pro"),
                ]
                bio = build_bio(text, spans)
                key = (text, bio, sentiment)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "text": text,
                        "event_bio": bio,
                        "sentiment": int(sentiment),
                        "source": "hardset_v1_social",
                    }
                )
                if len(rows) >= target_rows:
                    return rows

    return rows[:target_rows]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(max(300, min(500, int(args.target_rows))))
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved hardset_v1: {output_path} rows={len(df)}")


if __name__ == "__main__":
    main()
