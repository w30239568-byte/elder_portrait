import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from elder_portrait.config import TrainConfig
from elder_portrait.data import (
    ElderPortraitDataset,
    build_event_type_vocab,
    build_event_vocab,
    build_label_vocab,
    build_trigger_vocab,
)
from elder_portrait.data import load_dataset, split_train_val_test
from elder_portrait.decode_utils import (
    aggregate_char_probs,
    apply_span_confidence_filter,
    decode_bio_constrained,
)
from elder_portrait.model import ElderPortraitMultiTaskModel
from elder_portrait.tag_schema import split_bio_tokens

IGNORE_INDEX = -100


def resolve_default_data_path() -> str:
    candidates = [
        Path("dataset/storywell_raw_split_new.csv"),
        Path("dataset/storywell_2.csv"),
        Path("data/storywell_2.csv"),
        Path("dataset/elderly_narrative_with_labels.csv"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train joint event extraction + sentiment model for elderly portrait."
    )
    parser.add_argument("--data_path", type=str, default="dataset/storywell_raw_split_new.csv")
    parser.add_argument("--output_dir", type=str, default="runs/elder_portrait")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--event_embed_dim", type=int, default=64)  # legacy arg
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--event_loss_weight", type=float, default=2.0)
    parser.add_argument("--sentiment_loss_weight", type=float, default=1.0)
    parser.add_argument("--event_o_weight_scale", type=float, default=1.4)
    parser.add_argument("--event_weight_clip_max", type=float, default=8.0)
    parser.add_argument("--event_weight_power", type=float, default=0.5)
    parser.add_argument("--event_focal_gamma", type=float, default=2.0)
    parser.add_argument("--event_sampler_alpha", type=float, default=0.4)
    parser.add_argument("--trigger_loss_weight", type=float, default=1.0)
    parser.add_argument("--event_type_loss_weight", type=float, default=0.5)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--entity_o_aux_weight", type=float, default=0.4)
    parser.add_argument(
        "--decode_objective",
        type=str,
        default="precision",
        choices=["precision", "balance", "recall"],
    )
    parser.add_argument(
        "--decode_precision_floor",
        type=float,
        default=0.85,
        help="Minimum precision when selecting decode thresholds for balance objective.",
    )
    parser.add_argument("--init_checkpoint", type=str, default="")
    args = parser.parse_args()

    if not args.data_path:
        args.data_path = resolve_default_data_path()
    if not Path(args.data_path).exists():
        parser.error(
            "Cannot find dataset file. "
            f"Current path: {args.data_path}. "
            "Please set --data_path explicitly."
        )
    return TrainConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        dev = str(device_name).lower()
        if dev.startswith("cuda") and not torch.cuda.is_available():
            print("Warning: CUDA requested but current PyTorch has no CUDA support. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = np.power(weights, 0.5)
    weights = weights / max(1e-8, float(weights.mean()))
    return torch.tensor(weights, dtype=torch.float32)


def compute_event_tag_weights(
    event_bio_series: pd.Series,
    tag2id: Dict[str, int],
    o_weight_scale: float = 1.0,
    clip_max: float = 8.0,
    weight_power: float = 0.5,
) -> torch.Tensor:
    num_tags = len(tag2id)
    counts = np.ones(num_tags, dtype=np.float32)
    default_o_id = int(tag2id.get("O", 0))
    for bio in event_bio_series.tolist():
        for tok in split_bio_tokens(str(bio)):
            tid = int(tag2id.get(tok, default_o_id))
            counts[tid] += 1.0

    weights = counts.sum() / (max(1, num_tags) * counts)
    if weight_power > 0:
        weights = np.power(weights, float(weight_power))
    if "[PAD]" in tag2id:
        weights[int(tag2id["[PAD]"])] = 0.0
    if "O" in tag2id:
        weights[int(tag2id["O"])] *= float(max(0.0, o_weight_scale))
    if clip_max > 0:
        weights = np.clip(weights, 0.0, float(clip_max))
    non_zero = weights > 0
    if np.any(non_zero):
        weights[non_zero] = weights[non_zero] / max(
            1e-8, float(weights[non_zero].mean())
        )
    return torch.tensor(weights, dtype=torch.float32)


def _safe_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _safe_macro_f1_excluding_label(
    y_true: List[int],
    y_pred: List[int],
    excluded_label: int,
) -> float:
    if not y_true:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    labels = [x for x in labels if x != int(excluded_label)]
    if not labels:
        return 0.0
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def _compute_sentiment_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if not y_true:
        return {"acc": 0.0, "macro_f1": 0.0}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": _safe_macro_f1(y_true, y_pred),
    }


def _compute_event_metrics(
    y_true: List[int],
    y_pred: List[int],
    o_tag_id: int,
) -> Dict[str, float]:
    if not y_true:
        return {
            "token_acc": 0.0,
            "token_macro_f1": 0.0,
            "token_macro_f1_no_o": 0.0,
        }
    return {
        "token_acc": float(accuracy_score(y_true, y_pred)),
        "token_macro_f1": _safe_macro_f1(y_true, y_pred),
        "token_macro_f1_no_o": _safe_macro_f1_excluding_label(
            y_true=y_true,
            y_pred=y_pred,
            excluded_label=o_tag_id,
        ),
    }


def _compute_event_type_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if not y_true:
        return {"acc": 0.0, "macro_f1": 0.0}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": _safe_macro_f1(y_true, y_pred),
    }


class EventTokenLoss(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        ignore_index: int = IGNORE_INDEX,
        focal_gamma: float = 1.5,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.ignore_index = int(ignore_index)
        self.focal_gamma = float(max(0.0, focal_gamma))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid_mask = labels.ne(self.ignore_index)
        if not torch.any(valid_mask):
            return logits.sum() * 0.0
        loss = loss[valid_mask]
        if self.focal_gamma > 0:
            prob = torch.exp(-loss).clamp(min=1e-8, max=1.0)
            loss = ((1.0 - prob) ** self.focal_gamma) * loss
        return loss.mean()


def compute_entity_vs_o_aux_loss(
    event_logits: torch.Tensor,
    event_label_ids: torch.Tensor,
    o_tag_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Auxiliary binary loss: token is entity (non-O) vs O.
    This helps reduce noisy non-O over-prediction under precision-first setting.
    """
    num_tags = int(event_logits.size(-1))
    mask = torch.ones(num_tags, dtype=torch.bool, device=event_logits.device)
    mask[int(o_tag_id)] = False
    non_o_logits = torch.logsumexp(event_logits[..., mask], dim=-1)
    o_logits = event_logits[..., int(o_tag_id)]
    bin_logits = torch.stack([o_logits, non_o_logits], dim=-1)

    valid = event_label_ids.ne(int(ignore_index))
    if not torch.any(valid):
        return event_logits.sum() * 0.0
    bin_labels = event_label_ids.ne(int(o_tag_id)).long()
    return F.cross_entropy(
        bin_logits[valid],
        bin_labels[valid],
        weight=torch.tensor([1.15, 0.85], dtype=torch.float32, device=event_logits.device),
    )


def objective_to_beta(objective: str) -> float:
    obj = str(objective or "precision").lower()
    if obj == "recall":
        return 2.0
    if obj == "balance":
        return 1.0
    return 0.5


def _fbeta_binary_non_o(
    y_true: List[int],
    y_pred: List[int],
    o_tag_id: int,
    beta: float,
) -> Dict[str, float]:
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f_beta": 0.0, "f0_5": 0.0}
    gold = [0 if int(x) == int(o_tag_id) else 1 for x in y_true]
    pred = [0 if int(x) == int(o_tag_id) else 1 for x in y_pred]
    p, r, f, _ = precision_recall_fscore_support(
        gold,
        pred,
        average="binary",
        beta=float(beta),
        zero_division=0,
    )
    _, _, f05, _ = precision_recall_fscore_support(
        gold,
        pred,
        average="binary",
        beta=0.5,
        zero_division=0,
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f_beta": float(f),
        "f0_5": float(f05),
    }


def _decode_text_event_tags(
    model: ElderPortraitMultiTaskModel,
    tokenizer,
    text: str,
    max_length: int,
    device: torch.device,
    id2tag: Dict[int, str],
    o_tag_id: int,
    token_non_o_min_prob: float,
    span_conf_min: float,
) -> List[str]:
    chars = list(str(text))
    if not chars:
        return []
    enc = tokenizer(
        [chars],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        probs = torch.softmax(outputs["event_logits"].cpu(), dim=-1)[0]
    word_ids = enc.word_ids(batch_index=0)
    char_probs, _ = aggregate_char_probs(
        text_len=len(chars),
        word_ids=word_ids,
        event_prob_row=probs,
        o_tag_id=int(o_tag_id),
    )
    tags, confs = decode_bio_constrained(
        char_probs=char_probs,
        id2tag=id2tag,
        token_non_o_min_prob=float(token_non_o_min_prob),
    )
    tags = apply_span_confidence_filter(
        tags=tags,
        confs=confs,
        span_conf_min=float(span_conf_min),
    )
    return tags


def scan_decode_thresholds(
    model: ElderPortraitMultiTaskModel,
    tokenizer,
    val_df: pd.DataFrame,
    max_length: int,
    device: torch.device,
    id2tag: Dict[int, str],
    o_tag_id: int,
    decode_objective: str,
    precision_floor: float = 0.0,
) -> Dict[str, float]:
    objective = str(decode_objective or "precision").lower()
    beta = objective_to_beta(objective)
    if objective == "recall":
        token_grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
        span_grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    elif objective == "balance":
        token_grid = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18]
        span_grid = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18]
    else:
        token_grid = [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
        span_grid = [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]

    samples: List[Tuple[str, List[str]]] = []
    for _, row in val_df.iterrows():
        text = str(row["text"])
        gold_tags = split_bio_tokens(str(row["event_bio"]))
        if len(text) != len(gold_tags):
            continue
        samples.append((text, gold_tags))
    if not samples:
        return {
            "token_non_o_min_prob": 0.12,
            "span_conf_min": 0.12,
            "decode_objective": objective,
            "precision": 0.0,
            "recall": 0.0,
            "f_beta": 0.0,
            "f0_5": 0.0,
            "beta": float(beta),
            "num_eval_samples": 0,
        }

    best = None
    best_any = None
    model.eval()
    for tok_th in token_grid:
        for span_th in span_grid:
            y_true: List[int] = []
            y_pred: List[int] = []
            for text, gold_tags in samples:
                pred_tags = _decode_text_event_tags(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    max_length=max_length,
                    device=device,
                    id2tag=id2tag,
                    o_tag_id=o_tag_id,
                    token_non_o_min_prob=tok_th,
                    span_conf_min=span_th,
                )
                if len(pred_tags) != len(gold_tags):
                    m = min(len(pred_tags), len(gold_tags))
                    pred_tags = pred_tags[:m]
                    gold = gold_tags[:m]
                else:
                    gold = gold_tags
                for gt, pt in zip(gold, pred_tags):
                    y_true.append(0 if gt == "O" else 1)
                    y_pred.append(0 if pt == "O" else 1)
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", beta=float(beta), zero_division=0
            )
            _, _, f05, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", beta=0.5, zero_division=0
            )
            cur = {
                "token_non_o_min_prob": float(tok_th),
                "span_conf_min": float(span_th),
                "decode_objective": objective,
                "precision": float(p),
                "recall": float(r),
                "f_beta": float(f),
                "f0_5": float(f05),
                "beta": float(beta),
                "num_eval_samples": int(len(samples)),
            }
            if objective == "recall":
                key = (cur["f_beta"], cur["recall"], cur["precision"])
            else:
                key = (cur["f_beta"], cur["precision"], cur["recall"])

            if best_any is None:
                best_any = cur
            else:
                if objective == "recall":
                    best_any_key = (best_any["f_beta"], best_any["recall"], best_any["precision"])
                else:
                    best_any_key = (best_any["f_beta"], best_any["precision"], best_any["recall"])
                if key > best_any_key:
                    best_any = cur

            if objective == "balance" and float(cur["precision"]) < float(precision_floor):
                continue

            if objective == "recall":
                prev = (
                    best["f_beta"],
                    best["recall"],
                    best["precision"],
                ) if best else None
            else:
                prev = (
                    best["f_beta"],
                    best["precision"],
                    best["recall"],
                ) if best else None
            if best is None or key > prev:
                best = cur

    if best is None:
        best = best_any
        if best is None:
            best = {
                "token_non_o_min_prob": 0.12,
                "span_conf_min": 0.12,
                "decode_objective": objective,
                "precision": 0.0,
                "recall": 0.0,
                "f_beta": 0.0,
                "f0_5": 0.0,
                "beta": float(beta),
                "num_eval_samples": int(len(samples)),
            }
        best["precision_floor_satisfied"] = False
    else:
        best["precision_floor_satisfied"] = True

    best["precision_floor"] = float(precision_floor)
    return best


def build_event_row_sampling_weights(
    event_bio_series: pd.Series,
    alpha: float,
) -> torch.Tensor:
    alpha = float(max(0.0, alpha))
    if alpha <= 0:
        return torch.ones(len(event_bio_series), dtype=torch.double)

    token_counts: Dict[str, int] = {}
    rows = event_bio_series.astype(str).tolist()
    for bio in rows:
        for tok in split_bio_tokens(bio):
            if tok == "O":
                continue
            token_counts[tok] = token_counts.get(tok, 0) + 1

    sample_weights: List[float] = []
    for bio in rows:
        tags = [t for t in split_bio_tokens(bio) if t != "O"]
        if not tags:
            sample_weights.append(1.0)
            continue
        uniq_tags = sorted(set(tags))
        rarity = float(np.mean([1.0 / np.sqrt(token_counts.get(t, 1)) for t in uniq_tags]))
        sample_weights.append(1.0 + alpha * rarity)

    weights = np.asarray(sample_weights, dtype=np.float64)
    weights = weights / max(1e-8, float(weights.mean()))
    return torch.tensor(weights, dtype=torch.double)


def _collect_token_labels(
    pred_ids: torch.Tensor,
    gold_ids: torch.Tensor,
) -> Tuple[List[int], List[int]]:
    pred_flat = pred_ids.detach().cpu().view(-1).tolist()
    gold_flat = gold_ids.detach().cpu().view(-1).tolist()
    filtered_pred: List[int] = []
    filtered_gold: List[int] = []
    for p, g in zip(pred_flat, gold_flat):
        if int(g) == IGNORE_INDEX:
            continue
        filtered_pred.append(int(p))
        filtered_gold.append(int(g))
    return filtered_gold, filtered_pred


def _iter_predicted_spans_from_ids(tag_ids: List[int], id2tag: Dict[int, str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cur_start = -1
    cur_type = ""

    def flush(curr_idx: int) -> None:
        nonlocal cur_start, cur_type
        if cur_start >= 0:
            spans.append((cur_start, curr_idx))
        cur_start = -1
        cur_type = ""

    for idx, tid in enumerate(tag_ids):
        tag = str(id2tag.get(int(tid), "O"))
        if tag == "O" or tag == "[PAD]":
            flush(idx)
            continue
        if "-" not in tag:
            flush(idx)
            continue
        prefix, etype = tag.split("-", 1)
        prefix = str(prefix)
        etype = str(etype)
        if prefix == "B":
            flush(idx)
            cur_start = idx
            cur_type = etype
            continue
        if prefix == "I":
            if cur_start >= 0 and cur_type == etype:
                continue
            flush(idx)
            cur_start = idx
            cur_type = etype
            continue
        flush(idx)
    flush(len(tag_ids))
    return spans


def _single_char_entity_stats_from_batch(
    pred_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    id2tag: Dict[int, str],
) -> Tuple[int, int]:
    single_char = 0
    total_spans = 0
    pred_np = pred_ids.detach().cpu()
    gold_np = gold_ids.detach().cpu()
    batch_size = int(pred_np.size(0))
    for i in range(batch_size):
        seq_pred = pred_np[i].tolist()
        seq_gold = gold_np[i].tolist()
        valid_pred = [int(p) for p, g in zip(seq_pred, seq_gold) if int(g) != IGNORE_INDEX]
        spans = _iter_predicted_spans_from_ids(valid_pred, id2tag=id2tag)
        total_spans += int(len(spans))
        single_char += int(sum(1 for s, e in spans if int(e - s) <= 1))
    return single_char, total_spans


def evaluate(
    model: ElderPortraitMultiTaskModel,
    dataloader: DataLoader,
    criterion_sentiment: torch.nn.Module,
    criterion_trigger: torch.nn.Module,
    criterion_event: torch.nn.Module,
    criterion_event_type: torch.nn.Module,
    device: torch.device,
    sentiment_loss_weight: float,
    trigger_loss_weight: float,
    event_loss_weight: float,
    event_type_loss_weight: float,
    entity_o_aux_weight: float,
    trigger_o_tag_id: int,
    o_tag_id: int,
    id2tag: Dict[int, str],
) -> Dict[str, float]:
    if len(dataloader) == 0:
        return {
            "loss": 0.0,
            "sent_loss": 0.0,
            "trigger_loss": 0.0,
            "event_loss": 0.0,
            "event_type_loss": 0.0,
            "event_aux_loss": 0.0,
            "sent_acc": 0.0,
            "sent_macro_f1": 0.0,
            "trigger_token_acc": 0.0,
            "trigger_token_macro_f1": 0.0,
            "trigger_token_macro_f1_no_o": 0.0,
            "event_token_acc": 0.0,
            "event_token_macro_f1": 0.0,
            "event_token_macro_f1_no_o": 0.0,
            "event_type_acc": 0.0,
            "event_type_macro_f1": 0.0,
            "event_single_char_entity_rate": 0.0,
            "joint_score": 0.0,
        }

    model.eval()
    losses: List[float] = []
    sent_losses: List[float] = []
    trigger_losses: List[float] = []
    event_losses: List[float] = []
    event_type_losses: List[float] = []
    aux_losses: List[float] = []
    sent_preds: List[int] = []
    sent_labels: List[int] = []
    trigger_preds: List[int] = []
    trigger_labels: List[int] = []
    event_preds: List[int] = []
    event_labels: List[int] = []
    event_type_preds: List[int] = []
    event_type_labels: List[int] = []
    event_single_char_count = 0
    event_span_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            trigger_label_ids = batch["trigger_label_ids"].to(device)
            event_label_ids = batch["event_label_ids"].to(device)
            event_type_label = batch["event_type_label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            sent_logits = outputs["sentiment_logits"]
            trigger_logits = outputs["trigger_logits"]
            event_logits = outputs["event_logits"]
            event_type_logits = outputs["event_type_logits"]

            sent_loss = criterion_sentiment(sent_logits, labels)
            trigger_loss = criterion_trigger(
                trigger_logits.view(-1, trigger_logits.size(-1)),
                trigger_label_ids.view(-1),
            )
            event_loss = criterion_event(
                event_logits.view(-1, event_logits.size(-1)),
                event_label_ids.view(-1),
            )
            event_type_loss = criterion_event_type(event_type_logits, event_type_label)
            aux_loss = compute_entity_vs_o_aux_loss(
                event_logits=event_logits,
                event_label_ids=event_label_ids,
                o_tag_id=o_tag_id,
                ignore_index=IGNORE_INDEX,
            )
            loss = (
                sentiment_loss_weight * sent_loss
                + trigger_loss_weight * trigger_loss
                + event_loss_weight * event_loss
                + event_type_loss_weight * event_type_loss
                + entity_o_aux_weight * aux_loss
            ) / max(
                1e-8,
                sentiment_loss_weight
                + trigger_loss_weight
                + event_loss_weight
                + event_type_loss_weight
                + entity_o_aux_weight,
            )

            losses.append(float(loss.item()))
            sent_losses.append(float(sent_loss.item()))
            trigger_losses.append(float(trigger_loss.item()))
            event_losses.append(float(event_loss.item()))
            event_type_losses.append(float(event_type_loss.item()))
            aux_losses.append(float(aux_loss.item()))

            sent_batch_pred = torch.argmax(sent_logits, dim=-1)
            sent_preds.extend(sent_batch_pred.detach().cpu().tolist())
            sent_labels.extend(labels.detach().cpu().tolist())

            trigger_batch_pred = torch.argmax(trigger_logits, dim=-1)
            trig_gold_flat, trig_pred_flat = _collect_token_labels(
                pred_ids=trigger_batch_pred,
                gold_ids=trigger_label_ids,
            )
            trigger_labels.extend(trig_gold_flat)
            trigger_preds.extend(trig_pred_flat)

            event_batch_pred = torch.argmax(event_logits, dim=-1)
            gold_flat, pred_flat = _collect_token_labels(
                pred_ids=event_batch_pred,
                gold_ids=event_label_ids,
            )
            event_labels.extend(gold_flat)
            event_preds.extend(pred_flat)
            sc, total = _single_char_entity_stats_from_batch(
                pred_ids=event_batch_pred,
                gold_ids=event_label_ids,
                id2tag=id2tag,
            )
            event_single_char_count += int(sc)
            event_span_count += int(total)

            type_batch_pred = torch.argmax(event_type_logits, dim=-1)
            event_type_preds.extend(type_batch_pred.detach().cpu().tolist())
            event_type_labels.extend(event_type_label.detach().cpu().tolist())

    sent_metrics = _compute_sentiment_metrics(sent_labels, sent_preds)
    trigger_metrics = _compute_event_metrics(
        trigger_labels,
        trigger_preds,
        o_tag_id=trigger_o_tag_id,
    )
    event_metrics = _compute_event_metrics(
        event_labels,
        event_preds,
        o_tag_id=o_tag_id,
    )
    event_type_metrics = _compute_event_type_metrics(
        event_type_labels,
        event_type_preds,
    )
    joint_score = (
        sentiment_loss_weight * sent_metrics["macro_f1"]
        + trigger_loss_weight * trigger_metrics["token_macro_f1_no_o"]
        + event_loss_weight * event_metrics["token_macro_f1_no_o"]
        + event_type_loss_weight * event_type_metrics["macro_f1"]
    ) / max(
        1e-8,
        sentiment_loss_weight + trigger_loss_weight + event_loss_weight + event_type_loss_weight,
    )

    return {
        "loss": float(np.mean(losses)),
        "sent_loss": float(np.mean(sent_losses)),
        "trigger_loss": float(np.mean(trigger_losses)),
        "event_loss": float(np.mean(event_losses)),
        "event_type_loss": float(np.mean(event_type_losses)),
        "event_aux_loss": float(np.mean(aux_losses)),
        "sent_acc": sent_metrics["acc"],
        "sent_macro_f1": sent_metrics["macro_f1"],
        "trigger_token_acc": trigger_metrics["token_acc"],
        "trigger_token_macro_f1": trigger_metrics["token_macro_f1"],
        "trigger_token_macro_f1_no_o": trigger_metrics["token_macro_f1_no_o"],
        "event_token_acc": event_metrics["token_acc"],
        "event_token_macro_f1": event_metrics["token_macro_f1"],
        "event_token_macro_f1_no_o": event_metrics["token_macro_f1_no_o"],
        "event_type_acc": event_type_metrics["acc"],
        "event_type_macro_f1": event_type_metrics["macro_f1"],
        "event_single_char_entity_rate": float(event_single_char_count / max(1, event_span_count)),
        "joint_score": float(joint_score),
    }


def train(config: TrainConfig) -> Tuple[Dict[str, float], Dict[str, float]]:
    set_seed(config.seed)
    output_dir = config.resolve_output_dir()
    device = get_device(config.device)

    df = load_dataset(
        config.data_path,
        strict_storywell_schema=config.strict_storywell_schema,
        strict_alignment=True,
    )
    sentiment_classes = sorted(int(x) for x in df["sentiment"].unique().tolist())
    if len(sentiment_classes) < 2:
        print(
            "Warning: sentiment labels in dataset contain only one class "
            f"{sentiment_classes}. Sentiment head will not learn polarity."
        )
    if config.max_samples > 0:
        sampled_parts = []
        class_count = max(1, df["sentiment"].nunique())
        per_class = max(1, config.max_samples // class_count)
        for _, group in df.groupby("sentiment"):
            sampled_parts.append(
                group.sample(n=min(len(group), per_class), random_state=config.seed)
            )
        df = (
            pd.concat(sampled_parts, axis=0)
            .sample(frac=1.0, random_state=config.seed)
            .reset_index(drop=True)
        )

    train_df, val_df, test_df = split_train_val_test(
        df=df,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tag2id = build_event_vocab(train_df["event_bio"])
    trigger2id = build_trigger_vocab()
    event_type2id = build_event_type_vocab(train_df["event_type"])
    label2id = build_label_vocab(train_df["sentiment"])
    id2label = {v: k for k, v in label2id.items()}
    id2tag = {v: k for k, v in tag2id.items()}
    id2trigger = {v: k for k, v in trigger2id.items()}
    id2event_type = {v: k for k, v in event_type2id.items()}
    o_tag_id = int(tag2id.get("O", 0))
    trigger_o_tag_id = int(trigger2id.get("O", 1))

    train_ds = ElderPortraitDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        tag2id=tag2id,
        trigger2id=trigger2id,
        label2id=label2id,
        event_type2id=event_type2id,
        max_length=config.max_length,
        strict_tag_alignment=True,
    )
    val_ds = ElderPortraitDataset(
        dataframe=val_df,
        tokenizer=tokenizer,
        tag2id=tag2id,
        trigger2id=trigger2id,
        label2id=label2id,
        event_type2id=event_type2id,
        max_length=config.max_length,
        strict_tag_alignment=True,
    )
    test_ds = ElderPortraitDataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        tag2id=tag2id,
        trigger2id=trigger2id,
        label2id=label2id,
        event_type2id=event_type2id,
        max_length=config.max_length,
        strict_tag_alignment=True,
    )

    sampler = None
    if config.event_sampler_alpha > 0:
        sample_weights = build_event_row_sampling_weights(
            event_bio_series=train_df["event_bio"],
            alpha=config.event_sampler_alpha,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = ElderPortraitMultiTaskModel(
        model_name=config.model_name,
        num_labels=len(label2id),
        num_event_tags=len(tag2id),
        num_trigger_tags=len(trigger2id),
        num_event_types=len(event_type2id),
        dropout=config.dropout,
    ).to(device)
    if str(config.init_checkpoint or "").strip():
        init_ckpt_path = Path(str(config.init_checkpoint))
        if not init_ckpt_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_ckpt_path}")
        init_ckpt = torch.load(init_ckpt_path, map_location=device)
        init_state = init_ckpt.get("model_state_dict", {})
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        print(
            "Warm start from checkpoint:",
            str(init_ckpt_path),
            {"missing_keys": len(missing), "unexpected_keys": len(unexpected)},
        )

    class_weights = compute_class_weights(
        labels=[label2id[int(v)] for v in train_df["sentiment"].tolist()],
        num_classes=len(label2id),
    ).to(device)
    criterion_sentiment = torch.nn.CrossEntropyLoss(weight=class_weights)

    trigger_class_weights = compute_event_tag_weights(
        event_bio_series=train_df["trigger_bio"],
        tag2id=trigger2id,
        o_weight_scale=config.event_o_weight_scale,
        clip_max=config.event_weight_clip_max,
        weight_power=config.event_weight_power,
    ).to(device)
    criterion_trigger = EventTokenLoss(
        weight=trigger_class_weights,
        ignore_index=IGNORE_INDEX,
        focal_gamma=config.event_focal_gamma,
    )

    event_class_weights = compute_event_tag_weights(
        event_bio_series=train_df["event_bio"],
        tag2id=tag2id,
        o_weight_scale=config.event_o_weight_scale,
        clip_max=config.event_weight_clip_max,
        weight_power=config.event_weight_power,
    ).to(device)
    criterion_event = EventTokenLoss(
        weight=event_class_weights,
        ignore_index=IGNORE_INDEX,
        focal_gamma=config.event_focal_gamma,
    )
    event_type_class_weights = compute_class_weights(
        labels=[event_type2id[str(v)] for v in train_df["event_type"].tolist()],
        num_classes=len(event_type2id),
    ).to(device)
    criterion_event_type = torch.nn.CrossEntropyLoss(weight=event_type_class_weights)
    print(
        "Event loss weights configured:",
        {
            "event_o_weight_scale": float(config.event_o_weight_scale),
            "event_weight_clip_max": float(config.event_weight_clip_max),
            "event_weight_power": float(config.event_weight_power),
            "event_focal_gamma": float(config.event_focal_gamma),
            "event_sampler_alpha": float(config.event_sampler_alpha),
            "trigger_loss_weight": float(config.trigger_loss_weight),
            "event_type_loss_weight": float(config.event_type_loss_weight),
            "entity_o_aux_weight": float(config.entity_o_aux_weight),
            "grad_accum_steps": int(config.grad_accum_steps),
            "fp16": bool(config.fp16),
            "decode_objective": str(config.decode_objective),
            "decode_precision_floor": float(config.decode_precision_floor),
        },
    )
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    grad_accum_steps = max(1, int(config.grad_accum_steps))
    steps_per_epoch = int(np.ceil(len(train_loader) / grad_accum_steps))
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    use_amp = bool(config.fp16 and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast_ctx = torch.amp.autocast
        autocast_kwargs = {"device_type": "cuda", "enabled": use_amp}
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = torch.cuda.amp.autocast
        autocast_kwargs = {"enabled": use_amp}

    history: List[Dict[str, float]] = []
    best_val_joint = -1.0
    best_checkpoint = output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: List[float] = []
        sent_losses: List[float] = []
        trigger_losses: List[float] = []
        event_losses: List[float] = []
        event_type_losses: List[float] = []
        aux_losses: List[float] = []
        sent_preds: List[int] = []
        sent_labels: List[int] = []
        trigger_preds: List[int] = []
        trigger_labels: List[int] = []
        event_preds: List[int] = []
        event_labels: List[int] = []
        event_type_preds: List[int] = []
        event_type_labels: List[int] = []
        event_single_char_count = 0
        event_span_count = 0

        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"),
            start=1,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            trigger_label_ids = batch["trigger_label_ids"].to(device)
            event_label_ids = batch["event_label_ids"].to(device)
            event_type_label = batch["event_type_label"].to(device)

            with autocast_ctx(**autocast_kwargs):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                sent_logits = outputs["sentiment_logits"]
                trigger_logits = outputs["trigger_logits"]
                event_logits = outputs["event_logits"]
                event_type_logits = outputs["event_type_logits"]

                sent_loss = criterion_sentiment(sent_logits, labels)
                trigger_loss = criterion_trigger(
                    trigger_logits.view(-1, trigger_logits.size(-1)),
                    trigger_label_ids.view(-1),
                )
                event_loss = criterion_event(
                    event_logits.view(-1, event_logits.size(-1)),
                    event_label_ids.view(-1),
                )
                event_type_loss = criterion_event_type(
                    event_type_logits,
                    event_type_label,
                )
                aux_loss = compute_entity_vs_o_aux_loss(
                    event_logits=event_logits,
                    event_label_ids=event_label_ids,
                    o_tag_id=o_tag_id,
                    ignore_index=IGNORE_INDEX,
                )
                loss = (
                    config.sentiment_loss_weight * sent_loss
                    + config.trigger_loss_weight * trigger_loss
                    + config.event_loss_weight * event_loss
                    + config.event_type_loss_weight * event_type_loss
                    + config.entity_o_aux_weight * aux_loss
                ) / max(
                    1e-8,
                    config.sentiment_loss_weight
                    + config.trigger_loss_weight
                    + config.event_loss_weight
                    + config.event_type_loss_weight
                    + config.entity_o_aux_weight,
                )

            scaled = loss / grad_accum_steps
            scaler.scale(scaled).backward()

            should_step = (step_idx % grad_accum_steps == 0) or (step_idx == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.item()))
            sent_losses.append(float(sent_loss.item()))
            trigger_losses.append(float(trigger_loss.item()))
            event_losses.append(float(event_loss.item()))
            event_type_losses.append(float(event_type_loss.item()))
            aux_losses.append(float(aux_loss.item()))

            sent_batch_pred = torch.argmax(sent_logits, dim=-1)
            sent_preds.extend(sent_batch_pred.detach().cpu().tolist())
            sent_labels.extend(labels.detach().cpu().tolist())

            trigger_batch_pred = torch.argmax(trigger_logits, dim=-1)
            trig_gold_flat, trig_pred_flat = _collect_token_labels(
                pred_ids=trigger_batch_pred,
                gold_ids=trigger_label_ids,
            )
            trigger_labels.extend(trig_gold_flat)
            trigger_preds.extend(trig_pred_flat)

            event_batch_pred = torch.argmax(event_logits, dim=-1)
            gold_flat, pred_flat = _collect_token_labels(
                pred_ids=event_batch_pred,
                gold_ids=event_label_ids,
            )
            event_labels.extend(gold_flat)
            event_preds.extend(pred_flat)
            sc, total = _single_char_entity_stats_from_batch(
                pred_ids=event_batch_pred,
                gold_ids=event_label_ids,
                id2tag=id2tag,
            )
            event_single_char_count += int(sc)
            event_span_count += int(total)

            event_type_batch_pred = torch.argmax(event_type_logits, dim=-1)
            event_type_preds.extend(event_type_batch_pred.detach().cpu().tolist())
            event_type_labels.extend(event_type_label.detach().cpu().tolist())

        train_sent_metrics = _compute_sentiment_metrics(sent_labels, sent_preds)
        train_trigger_metrics = _compute_event_metrics(
            trigger_labels,
            trigger_preds,
            o_tag_id=trigger_o_tag_id,
        )
        train_event_metrics = _compute_event_metrics(
            event_labels,
            event_preds,
            o_tag_id=o_tag_id,
        )
        train_event_type_metrics = _compute_event_type_metrics(
            event_type_labels,
            event_type_preds,
        )
        train_joint_score = (
            config.sentiment_loss_weight * train_sent_metrics["macro_f1"]
            + config.trigger_loss_weight * train_trigger_metrics["token_macro_f1_no_o"]
            + config.event_loss_weight * train_event_metrics["token_macro_f1_no_o"]
            + config.event_type_loss_weight * train_event_type_metrics["macro_f1"]
        ) / max(
            1e-8,
            config.sentiment_loss_weight
            + config.trigger_loss_weight
            + config.event_loss_weight
            + config.event_type_loss_weight,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion_sentiment=criterion_sentiment,
            criterion_trigger=criterion_trigger,
            criterion_event=criterion_event,
            criterion_event_type=criterion_event_type,
            device=device,
            sentiment_loss_weight=config.sentiment_loss_weight,
            trigger_loss_weight=config.trigger_loss_weight,
            event_loss_weight=config.event_loss_weight,
            event_type_loss_weight=config.event_type_loss_weight,
            entity_o_aux_weight=config.entity_o_aux_weight,
            trigger_o_tag_id=trigger_o_tag_id,
            o_tag_id=o_tag_id,
            id2tag=id2tag,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_sent_loss": float(np.mean(sent_losses)),
            "train_trigger_loss": float(np.mean(trigger_losses)),
            "train_event_loss": float(np.mean(event_losses)),
            "train_event_type_loss": float(np.mean(event_type_losses)),
            "train_event_aux_loss": float(np.mean(aux_losses)),
            "train_sent_acc": train_sent_metrics["acc"],
            "train_sent_macro_f1": train_sent_metrics["macro_f1"],
            "train_trigger_token_acc": train_trigger_metrics["token_acc"],
            "train_trigger_token_macro_f1": train_trigger_metrics["token_macro_f1"],
            "train_trigger_token_macro_f1_no_o": train_trigger_metrics["token_macro_f1_no_o"],
            "train_event_token_acc": train_event_metrics["token_acc"],
            "train_event_token_macro_f1": train_event_metrics["token_macro_f1"],
            "train_event_token_macro_f1_no_o": train_event_metrics[
                "token_macro_f1_no_o"
            ],
            "train_event_single_char_entity_rate": float(
                event_single_char_count / max(1, event_span_count)
            ),
            "train_event_type_acc": train_event_type_metrics["acc"],
            "train_event_type_macro_f1": train_event_type_metrics["macro_f1"],
            "train_joint_score": float(train_joint_score),
            "val_loss": val_metrics["loss"],
            "val_sent_loss": val_metrics["sent_loss"],
            "val_trigger_loss": val_metrics["trigger_loss"],
            "val_event_loss": val_metrics["event_loss"],
            "val_event_type_loss": val_metrics["event_type_loss"],
            "val_event_aux_loss": val_metrics["event_aux_loss"],
            "val_sent_acc": val_metrics["sent_acc"],
            "val_sent_macro_f1": val_metrics["sent_macro_f1"],
            "val_trigger_token_acc": val_metrics["trigger_token_acc"],
            "val_trigger_token_macro_f1": val_metrics["trigger_token_macro_f1"],
            "val_trigger_token_macro_f1_no_o": val_metrics["trigger_token_macro_f1_no_o"],
            "val_event_token_acc": val_metrics["event_token_acc"],
            "val_event_token_macro_f1": val_metrics["event_token_macro_f1"],
            "val_event_token_macro_f1_no_o": val_metrics["event_token_macro_f1_no_o"],
            "val_event_single_char_entity_rate": val_metrics["event_single_char_entity_rate"],
            "val_event_type_acc": val_metrics["event_type_acc"],
            "val_event_type_macro_f1": val_metrics["event_type_macro_f1"],
            "val_joint_score": val_metrics["joint_score"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if val_metrics["joint_score"] >= best_val_joint:
            best_val_joint = val_metrics["joint_score"]
            torch.save(
                {
                    "architecture": "multitask_joint_v3",
                    "model_state_dict": model.state_dict(),
                    "model_name": config.model_name,
                    "config": asdict(config),
                    "tag2id": tag2id,
                    "trigger2id": trigger2id,
                    "event_type2id": event_type2id,
                    "label2id": label2id,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    decode_config = scan_decode_thresholds(
        model=model,
        tokenizer=tokenizer,
        val_df=val_df,
        max_length=config.max_length,
        device=device,
        id2tag=id2tag,
        o_tag_id=o_tag_id,
        decode_objective=config.decode_objective,
        precision_floor=config.decode_precision_floor,
    )
    checkpoint["decode_config"] = decode_config
    torch.save(checkpoint, best_checkpoint)

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion_sentiment=criterion_sentiment,
        criterion_trigger=criterion_trigger,
        criterion_event=criterion_event,
        criterion_event_type=criterion_event_type,
        device=device,
        sentiment_loss_weight=config.sentiment_loss_weight,
        trigger_loss_weight=config.trigger_loss_weight,
        event_loss_weight=config.event_loss_weight,
        event_type_loss_weight=config.event_type_loss_weight,
        entity_o_aux_weight=config.entity_o_aux_weight,
        trigger_o_tag_id=trigger_o_tag_id,
        o_tag_id=o_tag_id,
        id2tag=id2tag,
    )

    sent_test_preds: List[int] = []
    sent_test_labels: List[int] = []
    trigger_test_preds: List[int] = []
    trigger_test_labels: List[int] = []
    event_test_preds: List[int] = []
    event_test_labels: List[int] = []
    event_type_test_preds: List[int] = []
    event_type_test_labels: List[int] = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            sent_logits = outputs["sentiment_logits"]
            trigger_logits = outputs["trigger_logits"]
            event_logits = outputs["event_logits"]
            event_type_logits = outputs["event_type_logits"]
            sent_pred_ids = torch.argmax(sent_logits, dim=-1).cpu().tolist()
            sent_gold_ids = batch["labels"].cpu().tolist()
            sent_test_preds.extend(sent_pred_ids)
            sent_test_labels.extend(sent_gold_ids)

            trigger_pred_ids = torch.argmax(trigger_logits, dim=-1)
            trig_gold_flat, trig_pred_flat = _collect_token_labels(
                pred_ids=trigger_pred_ids,
                gold_ids=batch["trigger_label_ids"],
            )
            trigger_test_labels.extend(trig_gold_flat)
            trigger_test_preds.extend(trig_pred_flat)

            event_pred_ids = torch.argmax(event_logits, dim=-1)
            gold_flat, pred_flat = _collect_token_labels(
                pred_ids=event_pred_ids,
                gold_ids=batch["event_label_ids"],
            )
            event_test_labels.extend(gold_flat)
            event_test_preds.extend(pred_flat)

            event_type_pred_ids = torch.argmax(event_type_logits, dim=-1).cpu().tolist()
            event_type_gold_ids = batch["event_type_label"].cpu().tolist()
            event_type_test_preds.extend(event_type_pred_ids)
            event_type_test_labels.extend(event_type_gold_ids)

    if sent_test_labels:
        sentiment_report = classification_report(
            sent_test_labels,
            sent_test_preds,
            output_dict=True,
            zero_division=0,
        )
    else:
        sentiment_report = {}

    if event_test_labels:
        event_report = classification_report(
            event_test_labels,
            event_test_preds,
            labels=sorted(list(set(event_test_labels + event_test_preds))),
            output_dict=True,
            zero_division=0,
        )
    else:
        event_report = {}

    if trigger_test_labels:
        trigger_report = classification_report(
            trigger_test_labels,
            trigger_test_preds,
            labels=sorted(list(set(trigger_test_labels + trigger_test_preds))),
            output_dict=True,
            zero_division=0,
        )
    else:
        trigger_report = {}

    if event_type_test_labels:
        event_type_report = classification_report(
            event_type_test_labels,
            event_type_test_preds,
            labels=sorted(list(set(event_type_test_labels + event_type_test_preds))),
            output_dict=True,
            zero_division=0,
        )
    else:
        event_type_report = {}

    decode_eval = _fbeta_binary_non_o(
        y_true=event_test_labels,
        y_pred=event_test_preds,
        o_tag_id=o_tag_id,
        beta=objective_to_beta(config.decode_objective),
    )

    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): int(v) for k, v in id2label.items()},
                "tag2id": tag2id,
                "id2tag": {str(k): v for k, v in id2tag.items()},
                "trigger2id": trigger2id,
                "id2trigger": {str(k): v for k, v in id2trigger.items()},
                "event_type2id": event_type2id,
                "id2event_type": {str(k): v for k, v in id2event_type.items()},
                "decode_config": decode_config,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(output_dir / "decode_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(decode_config, f, ensure_ascii=False, indent=2)
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(output_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": test_metrics,
                "decode_eval_binary_non_o": decode_eval,
                "decode_config": decode_config,
                "sentiment_classification_report": sentiment_report,
                "trigger_token_classification_report": trigger_report,
                "event_token_classification_report": event_report,
                "event_type_classification_report": event_type_report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Best checkpoint:", best_checkpoint)
    print("Test metrics:", test_metrics)
    return history[-1], test_metrics


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
