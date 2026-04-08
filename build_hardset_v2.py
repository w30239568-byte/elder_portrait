import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from elder_portrait.tag_schema import STORYWELL_ALLOWED_TAG_SET, join_bio_tokens, split_tag


META_REPORT_KEYS = {"accuracy", "macro avg", "weighted avg"}
PERSON_ENTITY_TYPES = {"protagonist", "participant_par"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build hardset_v2 from low-F1 classes in test_report. "
            "Generates 8 classes x 25 rows = 200 rows with strict QC."
        )
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="runs/elder_portrait_balance_v2_exp1/test_report.json",
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="runs/elder_portrait_balance_v2_exp1/label_mapping.json",
    )
    parser.add_argument("--output_csv", type=str, default="dataset/hardset_v2.csv")
    parser.add_argument(
        "--review_csv", type=str, default="dataset/hardset_v2_review_checklist.csv"
    )
    parser.add_argument(
        "--qc_json", type=str, default="dataset/hardset_v2_qc_report.json"
    )
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--per_tag", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_low_f1_tags(
    report_path: Path,
    mapping_path: Path,
    top_k: int,
) -> List[Dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    id2tag = {str(k): str(v) for k, v in mapping.get("id2tag", {}).items()}
    cls_report = report.get("event_token_classification_report", {})
    rows: List[Dict] = []
    for class_id, metrics in cls_report.items():
        if class_id in META_REPORT_KEYS:
            continue
        if class_id not in id2tag:
            continue
        tag = id2tag[class_id]
        if tag in {"O", "[PAD]"}:
            continue
        if not (tag.startswith("B-") or tag.startswith("I-")):
            continue
        support = float(metrics.get("support", 0.0) or 0.0)
        if support <= 0:
            continue
        f1 = float(metrics.get("f1-score", 0.0) or 0.0)
        rows.append(
            {
                "class_id": int(class_id),
                "tag": tag,
                "f1": f1,
                "support": support,
            }
        )
    rows.sort(key=lambda x: (x["f1"], x["support"], x["class_id"]))
    return rows[: int(top_k)]


def normalize_span_type(span_type: str) -> str:
    if span_type.startswith("B-") or span_type.startswith("I-"):
        _, etype = split_tag(span_type)
        return etype
    return span_type


def build_bio(text: str, spans: List[Tuple[str, str]]) -> Optional[str]:
    tags = ["O"] * len(text)
    occupied = [False] * len(text)
    for phrase, span_type in spans:
        phrase = str(phrase)
        etype = normalize_span_type(str(span_type))
        if not phrase or not etype:
            return None
        if f"B-{etype}" not in STORYWELL_ALLOWED_TAG_SET:
            return None
        start = -1
        search_from = 0
        while True:
            pos = text.find(phrase, search_from)
            if pos < 0:
                break
            end = pos + len(phrase)
            if not any(occupied[i] for i in range(pos, end)):
                start = pos
                break
            search_from = pos + 1
        if start < 0:
            return None
        end = start + len(phrase)
        tags[start] = f"B-{etype}"
        for i in range(start + 1, end):
            tags[i] = f"I-{etype}"
        for i in range(start, end):
            occupied[i] = True
    return join_bio_tokens(tags)


def iter_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
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
        spans.append((start, i, etype))
    return spans


def pick_sentiment(etype: str) -> int:
    if etype.startswith("Achievement"):
        return 2
    if etype.startswith("Health"):
        return 0
    return 1


def phrase_pool_by_type() -> Dict[str, List[str]]:
    return {
        "Achievement_par": [
            "\u83b7\u5f97\u793e\u533a\u8868\u5f70",
            "\u62ff\u5230\u4f18\u79c0\u5458\u5de5\u5956",
            "\u5f97\u5230\u5fd7\u613f\u670d\u52a1\u5956\u7ae0",
            "\u5b8c\u6210\u56f0\u96be\u9879\u76ee\u653b\u5173",
            "\u83b7\u5f97\u8868\u6f14\u4e00\u7b49\u5956",
            "\u5728\u6280\u80fd\u6bd4\u8d5b\u83b7\u5956",
            "\u88ab\u8bc4\u4e3a\u5148\u8fdb\u4e2a\u4eba",
            "\u516c\u5f00\u53d1\u8868\u7814\u7a76\u6210\u679c",
        ],
        "Achievement_pro": [
            "\u83b7\u5f97\u793e\u533a\u5eb7\u590d\u4e4b\u661f",
            "\u53c2\u52a0\u8001\u5e74\u4e66\u753b\u5c55\u83b7\u5956",
            "\u5728\u5408\u5531\u961f\u8868\u6f14\u53d7\u5230\u8868\u626c",
            "\u5b8c\u6210\u533b\u751f\u5efa\u8bae\u7684\u590d\u5065\u8ba1\u5212",
            "\u5728\u793e\u533a\u6f14\u8bb2\u4e2d\u5f97\u5230\u597d\u8bc4",
            "\u575a\u6301\u8fd0\u52a8\u540e\u5065\u5eb7\u8bc4\u4f30\u63d0\u5347",
            "\u88ab\u793e\u5de5\u8bb0\u5f55\u4e3a\u79ef\u6781\u6837\u672c",
            "\u5b8c\u6210\u5bb6\u5ead\u62a4\u7406\u57f9\u8bad\u8003\u6838",
        ],
        "Education background_pro": [
            "\u6bd5\u4e1a\u4e8e\u5e08\u8303\u5b66\u6821",
            "\u53c2\u52a0\u591c\u6821\u8bfe\u7a0b",
            "\u5b8c\u6210\u62a4\u7406\u57f9\u8bad",
            "\u5b66\u4e60\u8fc7\u8001\u5e74\u5b66\u6821\u8bfe\u7a0b",
            "\u5728\u6210\u4eba\u6559\u80b2\u4e2d\u8fdb\u4fee",
            "\u8bfb\u8fc7\u804c\u4e1a\u6280\u672f\u5b66\u9662",
            "\u53c2\u52a0\u5065\u5eb7\u6559\u80b2\u8bb2\u5ea7",
            "\u5728\u793e\u533a\u5927\u5b66\u5b66\u4e60",
        ],
        "Occupation_par": [
            "\u62c5\u4efb\u62a4\u58eb\u957f",
            "\u5728\u836f\u623f\u5de5\u4f5c",
            "\u4ece\u4e8b\u7269\u6d41\u7ba1\u7406",
            "\u505a\u793e\u533a\u793e\u5de5",
            "\u5728\u5b66\u6821\u4efb\u6559",
            "\u5728\u65b0\u95fb\u673a\u6784\u4efb\u804c",
            "\u4ece\u4e8b\u8bbe\u5907\u7ef4\u62a4",
            "\u8d1f\u8d23\u57fa\u5c42\u7ba1\u7406",
        ],
        "location_par": [
            "\u4f4f\u5728\u4e0a\u6d77\u6d66\u4e1c",
            "\u642c\u5230\u5408\u80a5\u5305\u6cb3",
            "\u957f\u671f\u5728\u5357\u4eac\u5de5\u4f5c",
            "\u76ee\u524d\u5728\u676d\u5dde\u5b9a\u5c45",
            "\u8fd1\u5e74\u79fb\u5c45\u82cf\u5dde",
            "\u4e00\u76f4\u5728\u9752\u5c9b\u751f\u6d3b",
            "\u521a\u4ece\u6df1\u5733\u8c03\u56de",
            "\u5728\u6b66\u6c49\u7ec4\u5efa\u5bb6\u5ead",
        ],
    }


def build_text_and_spans(
    etype: str,
    phrase: str,
    idx: int,
    rng: random.Random,
) -> Tuple[str, List[Tuple[str, str]], int]:
    protagonists = [
        "\u5f20\u82b3",
        "\u674e\u6842\u5170",
        "\u738b\u79cb\u9999",
        "\u5468\u79c0\u82f1",
        "\u8d75\u7d20\u73cd",
        "\u9648\u79e6\u82f1",
    ]
    participant_roles = [
        "\u5973\u513f",
        "\u513f\u5b50",
        "\u8001\u4f34",
        "\u90bb\u5c45",
        "\u793e\u5de5",
    ]
    participant_names = [
        "\u674e\u6885",
        "\u738b\u5f3a",
        "\u5468\u654f",
        "\u9648\u4e3d",
        "\u5f20\u5b81",
    ]
    time_words = [
        "\u4e0a\u5468",
        "\u4e0a\u4e2a\u6708",
        "\u53bb\u5e74",
        "\u8fd9\u4e24\u4e2a\u6708",
        "\u6700\u8fd1",
    ]
    health_left = [
        "\u590d\u67e5\u8840\u538b",
        "\u7761\u7720\u53d8\u6d45",
        "\u8170\u9178\u80cc\u75db",
        "\u591c\u91cc\u5e38\u9192",
    ]
    health_right = [
        "\u65e9\u4e0a\u8d77\u5e8a\u9700\u8981\u6276\u7740\u5e8a\u8fb9",
        "\u767d\u5929\u7cbe\u795e\u4e0d\u592a\u597d",
        "\u533b\u751f\u5efa\u8bae\u6309\u65f6\u5403\u836f",
        "\u793e\u533a\u968f\u8bbf\u65f6\u63d0\u9192\u5c11\u76d0\u996e\u98df",
    ]

    pro = protagonists[idx % len(protagonists)]
    role = participant_roles[idx % len(participant_roles)]
    pname = participant_names[idx % len(participant_names)]
    time_word = time_words[idx % len(time_words)]
    h1 = health_left[idx % len(health_left)]
    h2 = health_right[(idx + 1) % len(health_right)]
    sentiment = pick_sentiment(etype)

    participant_span = f"{role}{pname}"

    if etype.endswith("_par"):
        template_mode = idx % 4
        if template_mode == 0:
            text = (
                f"{time_word}{pro}\u7684{participant_span}{phrase}\uff0c"
                f"{pro}\u6700\u8fd1{h1}\uff0c{h2}\u3002"
            )
        elif template_mode == 1:
            text = (
                f"{time_word}{participant_span}\u5728\u7535\u8bdd\u4e2d\u8bf4\u81ea\u5df1{phrase}\uff0c"
                f"{pro}\u8fd8\u53bb\u793e\u533a\u533b\u9662{h1}\u3002"
            )
        elif template_mode == 2:
            text = (
                f"{time_word}\u793e\u5de5\u8bb0\u5f55\uff1a{pro}\u7684{participant_span}{phrase}\uff0c"
                f"\u800c{pro}\u8fd9\u9635\u5b50{h1}\u3002"
            )
        else:
            text = (
                f"{time_word}{pro}\u548c{participant_span}\u89c1\u9762\u65f6\u63d0\u5230{participant_span}{phrase}\uff0c"
                f"{pro}\u6b64\u5916{h1}\uff0c{h2}\u3002"
            )
        spans = [
            (pro, "protagonist"),
            (participant_span, "participant_par"),
            (phrase, etype),
            (h1, "Health_pro"),
            (h2, "Health_pro"),
        ]
    else:
        template_mode = idx % 4
        if template_mode == 0:
            text = (
                f"{time_word}{pro}{phrase}\uff0c"
                f"{pro}\u6700\u8fd1{h1}\uff0c{h2}\u3002"
            )
        elif template_mode == 1:
            text = (
                f"{time_word}{pro}\u7684{participant_span}\u8bf4{pro}{phrase}\uff0c"
                f"{pro}\u8fd8\u9700\u8981\u5904\u7406{h1}\u3002"
            )
        elif template_mode == 2:
            text = (
                f"{time_word}\u793e\u5de5\u8bb0\u5f55\uff1a{pro}{phrase}\uff0c"
                f"{participant_span}\u966a\u5979\u590d\u8bca\u65f6\u53d1\u73b0{h1}\u3002"
            )
        else:
            text = (
                f"{time_word}{pro}\u5728\u5bb6\u5ead\u4ea4\u6d41\u4e2d\u63d0\u5230{phrase}\uff0c"
                f"{participant_span}\u5efa\u8bae\u5979\u5173\u6ce8{h1}\uff0c{h2}\u3002"
            )
        spans = [
            (pro, "protagonist"),
            (participant_span, "participant_par"),
            (phrase, etype),
            (h1, "Health_pro"),
            (h2, "Health_pro"),
        ]

    if rng.random() < 0.2:
        text = text.replace("\u6700\u8fd1", "\u524d\u6bb5\u65f6\u95f4", 1)
    return text, spans, sentiment


def validate_rows(df: pd.DataFrame) -> Dict:
    invalid_len = 0
    invalid_bio = 0
    forbidden_single_char = 0
    missing_target_tag = 0
    source_counter = Counter()
    target_counter = Counter()
    errors: List[Dict] = []
    for i, row in df.iterrows():
        text = str(row["text"])
        tags = str(row["event_bio"]).split("\t") if "\t" in str(row["event_bio"]) else str(row["event_bio"]).split()
        source = str(row.get("source", ""))
        source_counter[source] += 1
        target_tag = str(row.get("target_tag", ""))
        target_counter[target_tag] += 1
        if len(tags) != len(text):
            invalid_len += 1
            errors.append({"row": int(i), "type": "invalid_len"})
            continue
        prev_prefix = "O"
        prev_type = ""
        legal = True
        for tok in tags:
            if tok == "O":
                prev_prefix = "O"
                prev_type = ""
                continue
            if tok not in STORYWELL_ALLOWED_TAG_SET:
                legal = False
                break
            prefix, etype = split_tag(tok)
            if prefix == "I":
                if not (prev_prefix in {"B", "I"} and prev_type == etype):
                    legal = False
                    break
            prev_prefix, prev_type = prefix, etype
        if not legal:
            invalid_bio += 1
            errors.append({"row": int(i), "type": "invalid_bio"})
            continue
        if target_tag and target_tag not in tags:
            missing_target_tag += 1
            errors.append({"row": int(i), "type": "missing_target_tag", "target_tag": target_tag})
        for s, e, etype in iter_spans(tags):
            if (e - s) == 1 and etype not in PERSON_ENTITY_TYPES:
                forbidden_single_char += 1
                errors.append(
                    {
                        "row": int(i),
                        "type": "forbidden_single_char",
                        "etype": etype,
                        "text": text[s:e],
                    }
                )
    return {
        "rows": int(len(df)),
        "invalid_len_rows": int(invalid_len),
        "invalid_bio_rows": int(invalid_bio),
        "missing_target_tag_rows": int(missing_target_tag),
        "forbidden_single_char_spans": int(forbidden_single_char),
        "target_tag_distribution": dict(sorted(target_counter.items())),
        "source_distribution": dict(sorted(source_counter.items())),
        "error_examples": errors[:50],
    }


def build_hardset_v2(
    selected_tags: List[Dict],
    per_tag: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(int(seed))
    pools = phrase_pool_by_type()
    rows: List[Dict] = []
    seen = set()
    low_tag_report: Dict[str, Dict] = {}
    for item in selected_tags:
        target_tag = str(item["tag"])
        class_id = int(item["class_id"])
        f1 = float(item["f1"])
        support = float(item["support"])
        low_tag_report[target_tag] = {
            "class_id": class_id,
            "f1": f1,
            "support": support,
        }
        _, etype = split_tag(target_tag)
        phrases = pools.get(etype, [])
        if not phrases:
            phrases = [f"{etype}\u76f8\u5173\u4e8b\u9879"]
        built = 0
        idx = 0
        max_try = per_tag * 40
        while built < per_tag and idx < max_try:
            phrase = phrases[idx % len(phrases)]
            if target_tag.startswith("I-") and len(phrase) < 2:
                idx += 1
                continue
            text, spans, sentiment = build_text_and_spans(
                etype=etype,
                phrase=phrase,
                idx=idx,
                rng=rng,
            )
            bio = build_bio(text, spans)
            if not bio:
                idx += 1
                continue
            tokens = bio.split("\t") if "\t" in bio else bio.split()
            if target_tag not in tokens:
                idx += 1
                continue
            key = (text, bio, sentiment, target_tag)
            if key in seen:
                idx += 1
                continue
            seen.add(key)
            rows.append(
                {
                    "text": text,
                    "event_bio": bio,
                    "sentiment": int(sentiment),
                    "source": f"hardset_v2_{target_tag}",
                    "target_tag": target_tag,
                    "target_class_id": class_id,
                }
            )
            built += 1
            idx += 1
        if built < per_tag:
            raise RuntimeError(
                f"Failed to build enough rows for {target_tag}: {built}/{per_tag}"
            )
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_path)
    mapping_path = Path(args.mapping_path)
    output_csv = Path(args.output_csv)
    review_csv = Path(args.review_csv)
    qc_json = Path(args.qc_json)
    top_k = int(args.top_k)
    per_tag = int(args.per_tag)
    selected = load_low_f1_tags(
        report_path=report_path,
        mapping_path=mapping_path,
        top_k=top_k,
    )
    if len(selected) < top_k:
        raise RuntimeError(
            f"Only {len(selected)} eligible tags found, expected at least {top_k}."
        )
    df = build_hardset_v2(selected_tags=selected, per_tag=per_tag, seed=int(args.seed))
    qc = validate_rows(df)
    if (
        qc["invalid_len_rows"] != 0
        or qc["invalid_bio_rows"] != 0
        or qc["missing_target_tag_rows"] != 0
        or qc["forbidden_single_char_spans"] != 0
    ):
        raise RuntimeError(f"hardset_v2 QC failed: {qc}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_df = df[["text", "event_bio", "sentiment", "source"]].copy()
    save_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    review_df = df[
        ["target_tag", "target_class_id", "text", "event_bio", "sentiment", "source"]
    ].copy()
    review_df["manual_review_status"] = "AUTO_QC_PASS"
    review_df["manual_review_notes"] = ""
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(review_csv, index=False, encoding="utf-8-sig")

    qc_payload = {
        "selected_low_tags": selected,
        "requested_top_k": top_k,
        "requested_per_tag": per_tag,
        "generated_rows": int(len(df)),
        "strict_qc": qc,
    }
    qc_json.parent.mkdir(parents=True, exist_ok=True)
    qc_json.write_text(
        json.dumps(qc_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        "Saved hardset_v2:",
        {
            "output_csv": str(output_csv),
            "review_csv": str(review_csv),
            "qc_json": str(qc_json),
            "rows": int(len(df)),
            "targets": qc["target_tag_distribution"],
        },
    )


if __name__ == "__main__":
    main()
