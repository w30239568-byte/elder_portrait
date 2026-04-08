from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from elder_portrait.tag_schema import (
    find_unknown_tags,
    join_bio_tokens,
    normalize_event_bio,
    split_bio_tokens,
    split_tag,
)
from elder_portrait.tag_schema import STORYWELL_ALLOWED_TAGS


PAD_TAG = "[PAD]"
DEFAULT_TAG = "O"
IGNORE_INDEX = -100
DEFAULT_EVENT_TYPE = "NONE"
TRIGGER_O = "O"
TRIGGER_B = "B-Trigger"
TRIGGER_I = "I-Trigger"


def derive_trigger_tags_from_event_tags(event_tags: List[str]) -> List[str]:
    trigger_tags: List[str] = []
    prev_inside = False
    prev_type: Optional[str] = None
    for tag in event_tags:
        if tag == "O":
            trigger_tags.append(TRIGGER_O)
            prev_inside = False
            prev_type = None
            continue
        prefix, etype = split_tag(tag)
        is_new_span = (
            (not prev_inside)
            or prefix == "B"
            or (prev_type is not None and etype is not None and prev_type != etype)
        )
        if is_new_span:
            trigger_tags.append(TRIGGER_B)
        else:
            trigger_tags.append(TRIGGER_I)
        prev_inside = True
        prev_type = etype
    return trigger_tags


def infer_primary_event_type_from_tags(event_tags: List[str]) -> str:
    counts: Dict[str, int] = {}
    for tag in event_tags:
        if tag == "O":
            continue
        prefix, etype = split_tag(tag)
        if prefix in {"B", "I"} and etype:
            counts[etype] = counts.get(etype, 0) + 1
    if not counts:
        return DEFAULT_EVENT_TYPE
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def load_dataset(
    csv_path: str,
    encoding: str = "utf-8",
    strict_storywell_schema: bool = True,
    strict_alignment: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding=encoding)
    required_cols = {"text", "event_bio"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(int)
    df["text"] = df["text"].astype(str)
    df["event_bio"] = df["event_bio"].astype(str).map(normalize_event_bio)
    if strict_storywell_schema:
        unknown_tags = find_unknown_tags(df["event_bio"])
        if unknown_tags:
            examples = ", ".join(unknown_tags[:12])
            raise ValueError(
                "Found tags outside StoryWell schema: "
                f"{examples}. "
                "Please convert BIO tags to StoryWell schema before training."
            )
    if strict_alignment:
        validate_text_bio_alignment(df, raise_on_error=True)
    if "trigger_bio" not in df.columns:
        df["trigger_bio"] = df["event_bio"].map(
            lambda x: join_bio_tokens(
                derive_trigger_tags_from_event_tags(split_bio_tokens(str(x)))
            )
        )
    else:
        df["trigger_bio"] = df["trigger_bio"].astype(str)
    if "event_type" not in df.columns:
        df["event_type"] = df["event_bio"].map(
            lambda x: infer_primary_event_type_from_tags(split_bio_tokens(str(x)))
        )
    else:
        df["event_type"] = df["event_type"].astype(str)
    return df


def split_train_val_test(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "sentiment" not in df.columns:
        raise ValueError("Column `sentiment` is required for training.")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("`val_ratio` and `test_ratio` must be >=0 and sum to <1.")

    if val_ratio == 0 and test_ratio == 0:
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()

    temp_ratio = val_ratio + test_ratio
    stratify_full = None
    sentiment_counts = df["sentiment"].value_counts()
    if sentiment_counts.size >= 2 and int(sentiment_counts.min()) >= 2:
        stratify_full = df["sentiment"]

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=stratify_full,
    )

    if test_ratio == 0:
        return train_df.reset_index(drop=True), temp_df.reset_index(drop=True), temp_df.iloc[0:0].copy()
    if val_ratio == 0:
        return train_df.reset_index(drop=True), temp_df.iloc[0:0].copy(), temp_df.reset_index(drop=True)

    rel_test_ratio = test_ratio / temp_ratio
    stratify_temp = None
    temp_counts = temp_df["sentiment"].value_counts()
    if temp_counts.size >= 2 and int(temp_counts.min()) >= 2:
        stratify_temp = temp_df["sentiment"]

    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test_ratio,
        random_state=seed,
        stratify=stratify_temp,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_event_vocab(series: pd.Series) -> Dict[str, int]:
    tags = set()
    for bio in series:
        tags.update(split_bio_tokens(str(bio)))

    # Keep a stable full StoryWell vocabulary so train/infer checkpoints always
    # share the same tag space even when current training data misses some tags.
    tags.update(STORYWELL_ALLOWED_TAGS)
    tags.discard(PAD_TAG)
    tags = sorted(tags)
    if DEFAULT_TAG not in tags:
        tags.insert(0, DEFAULT_TAG)
    id2tag = [PAD_TAG] + tags
    return {tag: idx for idx, tag in enumerate(id2tag)}


def build_label_vocab(series: pd.Series) -> Dict[int, int]:
    labels = sorted(int(v) for v in series.unique().tolist())
    return {label: idx for idx, label in enumerate(labels)}


def build_trigger_vocab() -> Dict[str, int]:
    id2tag = [PAD_TAG, TRIGGER_O, TRIGGER_B, TRIGGER_I]
    return {tag: idx for idx, tag in enumerate(id2tag)}


def build_event_type_vocab(series: pd.Series) -> Dict[str, int]:
    types = sorted(set(str(x) for x in series.tolist() if str(x).strip()))
    if DEFAULT_EVENT_TYPE not in types:
        types.insert(0, DEFAULT_EVENT_TYPE)
    return {t: i for i, t in enumerate(types)}


def align_tags(text: str, tags: List[str]) -> List[str]:
    if len(tags) == len(text):
        return tags
    if len(tags) < len(text):
        return tags + [DEFAULT_TAG] * (len(text) - len(tags))
    return tags[: len(text)]


def validate_text_bio_alignment(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> List[Dict[str, int]]:
    mismatches: List[Dict[str, int]] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        text = str(row.get("text", ""))
        tags = split_bio_tokens(str(row.get("event_bio", "")))
        if len(text) != len(tags):
            mismatches.append(
                {
                    "row_index": int(idx),
                    "text_len": int(len(text)),
                    "tag_len": int(len(tags)),
                }
            )
    if mismatches and raise_on_error:
        sample = mismatches[:8]
        raise ValueError(
            "Found text/event_bio length mismatches. "
            f"Examples: {sample}"
        )
    return mismatches


@dataclass
class Example:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    event_ids: torch.Tensor
    event_label_ids: torch.Tensor
    trigger_label_ids: Optional[torch.Tensor]
    event_type_label: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]


class ElderPortraitDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        tag2id: Dict[str, int],
        trigger2id: Optional[Dict[str, int]] = None,
        label2id: Optional[Dict[int, int]] = None,
        event_type2id: Optional[Dict[str, int]] = None,
        max_length: int = 128,
        strict_tag_alignment: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.trigger2id = trigger2id
        self.label2id = label2id
        self.event_type2id = event_type2id
        self.max_length = max_length
        self.strict_tag_alignment = bool(strict_tag_alignment)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row["text"])
        chars = list(text)
        raw_bio = str(row.get("event_bio", "")).strip()
        if not raw_bio:
            raw_tags = [DEFAULT_TAG] * len(text)
        else:
            raw_tags = split_bio_tokens(raw_bio)
        if len(raw_tags) != len(text):
            if self.strict_tag_alignment:
                raise ValueError(
                    f"text/event_bio length mismatch at idx={idx}: "
                    f"text_len={len(text)}, tag_len={len(raw_tags)}"
                )
            tags = align_tags(text, raw_tags)
        else:
            tags = raw_tags
        trigger_tags = derive_trigger_tags_from_event_tags(tags)
        event_type = infer_primary_event_type_from_tags(tags)

        default_tag_id = self.tag2id.get(DEFAULT_TAG, 1)
        pad_tag_id = self.tag2id[PAD_TAG]
        trigger_pad_id = 0
        trigger_default_id = 1
        if self.trigger2id:
            trigger_pad_id = int(self.trigger2id.get(PAD_TAG, 0))
            trigger_default_id = int(self.trigger2id.get(TRIGGER_O, 1))

        if getattr(self.tokenizer, "is_fast", False):
            encoding = self.tokenizer(
                chars,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_attention_mask=True,
            )
            word_ids = encoding.word_ids()
            if word_ids is None:
                raise RuntimeError("Tokenizer failed to produce word ids.")

            event_ids = []
            event_label_ids = []
            trigger_label_ids = []
            for wid in word_ids:
                if wid is None:
                    event_ids.append(pad_tag_id)
                    event_label_ids.append(IGNORE_INDEX)
                    trigger_label_ids.append(IGNORE_INDEX)
                elif wid < len(tags):
                    tag_id = self.tag2id.get(tags[wid], default_tag_id)
                    event_ids.append(tag_id)
                    event_label_ids.append(tag_id)
                    if self.trigger2id:
                        trig_id = int(
                            self.trigger2id.get(trigger_tags[wid], trigger_default_id)
                        )
                        trigger_label_ids.append(trig_id)
                    else:
                        trigger_label_ids.append(IGNORE_INDEX)
                else:
                    event_ids.append(default_tag_id)
                    event_label_ids.append(default_tag_id)
                    if self.trigger2id:
                        trigger_label_ids.append(trigger_default_id)
                    else:
                        trigger_label_ids.append(IGNORE_INDEX)

            sample = {
                "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    encoding["attention_mask"], dtype=torch.long
                ),
                "event_ids": torch.tensor(event_ids, dtype=torch.long),
                "event_label_ids": torch.tensor(event_label_ids, dtype=torch.long),
                "trigger_label_ids": torch.tensor(trigger_label_ids, dtype=torch.long),
            }
        else:
            cls_token = self.tokenizer.cls_token or "[CLS]"
            sep_token = self.tokenizer.sep_token or "[SEP]"
            unk_token = self.tokenizer.unk_token or "[UNK]"

            tokens = [cls_token]
            event_ids = [pad_tag_id]
            event_label_ids = [IGNORE_INDEX]
            trigger_label_ids = [IGNORE_INDEX]
            for ch, tag, trig_tag in zip(chars, tags, trigger_tags):
                sub_tokens = self.tokenizer.tokenize(ch)
                if not sub_tokens:
                    sub_tokens = [unk_token]
                tag_id = self.tag2id.get(tag, default_tag_id)
                if self.trigger2id:
                    trig_id = int(self.trigger2id.get(trig_tag, trigger_default_id))
                else:
                    trig_id = IGNORE_INDEX
                for st in sub_tokens:
                    tokens.append(st)
                    event_ids.append(tag_id)
                    event_label_ids.append(tag_id)
                    trigger_label_ids.append(trig_id)
            tokens.append(sep_token)
            event_ids.append(pad_tag_id)
            event_label_ids.append(IGNORE_INDEX)
            trigger_label_ids.append(IGNORE_INDEX)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)[: self.max_length]
            event_ids = event_ids[: self.max_length]
            event_label_ids = event_label_ids[: self.max_length]
            trigger_label_ids = trigger_label_ids[: self.max_length]
            attention_mask = [1] * len(input_ids)

            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = 0
                input_ids += [pad_id] * pad_len
                attention_mask += [0] * pad_len
                event_ids += [pad_tag_id] * pad_len
                event_label_ids += [IGNORE_INDEX] * pad_len
                trigger_label_ids += [IGNORE_INDEX] * pad_len

            sample = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "event_ids": torch.tensor(event_ids, dtype=torch.long),
                "event_label_ids": torch.tensor(event_label_ids, dtype=torch.long),
                "trigger_label_ids": torch.tensor(trigger_label_ids, dtype=torch.long),
            }
        if self.label2id is not None and "sentiment" in row:
            sample["labels"] = torch.tensor(
                self.label2id[int(row["sentiment"])], dtype=torch.long
            )
        if self.event_type2id is not None:
            event_type_id = int(
                self.event_type2id.get(
                    str(row.get("event_type", event_type)),
                    self.event_type2id.get(DEFAULT_EVENT_TYPE, 0),
                )
            )
            sample["event_type_label"] = torch.tensor(event_type_id, dtype=torch.long)
        return sample
