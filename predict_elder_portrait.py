import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from elder_portrait.data import ElderPortraitDataset, load_dataset
from elder_portrait.model import ElderPortraitMultiTaskModel, EventFusionClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict sentiment labels for elderly narratives."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mapping_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    device = get_device(args.device)

    with open(args.mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    tag2id = mapping["tag2id"]
    id2label = {int(k): int(v) for k, v in mapping["id2label"].items()}

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint["model_name"]
    label2id = checkpoint["label2id"]
    architecture = str(checkpoint.get("architecture", "fusion_v1"))
    is_multitask = architecture in {"multitask_v2", "multitask_joint_v3"}
    trigger2id = mapping.get("trigger2id") or checkpoint.get("trigger2id") or {}
    event_type2id = mapping.get("event_type2id") or checkpoint.get("event_type2id") or {}
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

    tokenizer_path = Path(args.mapping_path).parent / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    df = load_dataset(args.data_path, strict_storywell_schema=True)
    dataset = ElderPortraitDataset(
        dataframe=df,
        tokenizer=tokenizer,
        tag2id=tag2id,
        label2id=None,
        max_length=args.max_length,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    preds = []
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

    out_df = pd.read_csv(args.data_path, encoding="utf-8")
    out_df["pred_sentiment"] = preds
    out_df.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
