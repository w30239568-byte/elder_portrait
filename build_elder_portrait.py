import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from elder_portrait.data import ElderPortraitDataset, load_dataset
from elder_portrait.event_parser import parse_bio_entities, summarize_entities
from elder_portrait.model import ElderPortraitMultiTaskModel, EventFusionClassifier
from elder_portrait.portrait import build_portrait_summary, build_timeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate elderly portrait from narratives and event BIO tags."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/elder_portrait_profile")
    parser.add_argument("--use_gold_sentiment", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--mapping_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_sentiment(
    df: pd.DataFrame,
    checkpoint_path: str,
    mapping_path: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> List[int]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    tag2id = mapping["tag2id"]
    id2label = {int(k): int(v) for k, v in mapping["id2label"].items()}

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    label2id = checkpoint["label2id"]
    trigger2id = mapping.get("trigger2id") or checkpoint.get("trigger2id") or {}
    event_type2id = mapping.get("event_type2id") or checkpoint.get("event_type2id") or {}

    tokenizer_path = Path(mapping_path).parent / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    dataset = ElderPortraitDataset(
        dataframe=df,
        tokenizer=tokenizer,
        tag2id=tag2id,
        label2id=None,
        max_length=max_length,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    architecture = str(checkpoint.get("architecture", "fusion_v1"))
    is_multitask = architecture in {"multitask_v2", "multitask_joint_v3"}
    if is_multitask:
        state_dict = checkpoint["model_state_dict"]
        has_trigger_head = any(k.startswith("trigger_classifier.") for k in state_dict.keys())
        has_event_type_head = any(
            k.startswith("event_type_classifier.") for k in state_dict.keys()
        )
        num_trigger_tags = len(trigger2id) if has_trigger_head else 0
        num_event_types = len(event_type2id) if has_event_type_head else 0
        if has_trigger_head and num_trigger_tags <= 0 and "trigger_classifier.weight" in state_dict:
            num_trigger_tags = int(state_dict["trigger_classifier.weight"].shape[0])
        if has_event_type_head and num_event_types <= 0 and "event_type_classifier.weight" in state_dict:
            num_event_types = int(state_dict["event_type_classifier.weight"].shape[0])
        model = ElderPortraitMultiTaskModel(
            model_name=model_name,
            num_labels=len(label2id),
            num_event_tags=len(tag2id),
            num_trigger_tags=num_trigger_tags,
            num_event_types=num_event_types,
            dropout=float(checkpoint["config"].get("dropout", 0.2)),
        ).to(device)
    else:
        model = EventFusionClassifier(
            model_name=model_name,
            num_labels=len(label2id),
            num_event_tags=len(tag2id),
            event_embed_dim=checkpoint["config"]["event_embed_dim"],
            dropout=checkpoint["config"]["dropout"],
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds: List[int] = []
    with torch.no_grad():
        for batch in dataloader:
            if is_multitask:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )["sentiment_logits"]
            else:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    event_ids=batch["event_ids"].to(device),
                )
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend([id2label[x] for x in pred_ids])
    return preds


def enrich_records(df: pd.DataFrame, sentiments: List[int]) -> List[Dict]:
    records = []
    for row, sentiment in zip(df.to_dict(orient="records"), sentiments):
        entities = parse_bio_entities(row["text"], row["event_bio"])
        summary = summarize_entities(entities)
        records.append(
            {
                "id": row.get("id", ""),
                "text": row["text"],
                "event_bio": row["event_bio"],
                "sentiment": int(sentiment),
                "entities": entities,
                "primary_event_type": summary["primary_event_type"],
                "protagonist_mentions": summary["protagonist_mentions"],
                "participant_mentions": summary["participant_mentions"],
                "entity_mentions_by_type": summary["entity_mentions_by_type"],
                "entity_type_counts": summary["entity_type_counts"],
            }
        )
    return records


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    df = load_dataset(args.data_path, strict_storywell_schema=True)

    if args.use_gold_sentiment:
        if "sentiment" not in df.columns:
            raise ValueError("`--use_gold_sentiment` requires `sentiment` column.")
        sentiments = [int(x) for x in df["sentiment"].tolist()]
    else:
        if not args.checkpoint or not args.mapping_path:
            raise ValueError(
                "When not using gold sentiment, `--checkpoint` and `--mapping_path` are required."
            )
        sentiments = predict_sentiment(
            df=df,
            checkpoint_path=args.checkpoint,
            mapping_path=args.mapping_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )

    records = enrich_records(df, sentiments)
    summary = build_portrait_summary(records)
    timeline = build_timeline(records)

    records_df = pd.DataFrame(
        [
            {
                "id": r["id"],
                "text": r["text"],
                "sentiment": r["sentiment"],
                "primary_event_type": r["primary_event_type"],
                "protagonist_mentions": "；".join(r["protagonist_mentions"]),
                "participant_mentions": "；".join(r["participant_mentions"]),
                "entity_mentions_by_type": json.dumps(
                    r["entity_mentions_by_type"], ensure_ascii=False
                ),
                "entity_type_counts": json.dumps(
                    r["entity_type_counts"], ensure_ascii=False
                ),
            }
            for r in records
        ]
    )
    records_df.to_csv(output_dir / "portrait_records.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "portrait_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(output_dir / "portrait_timeline.json", "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    with open(output_dir / "portrait_records_full.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved portrait outputs to: {output_dir}")


if __name__ == "__main__":
    main()
