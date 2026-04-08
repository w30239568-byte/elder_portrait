import math
from typing import Dict, List, Sequence, Tuple

import torch

from elder_portrait.tag_schema import split_tag


NEG_INF = -1e18


def _is_valid_transition(prev_tag: str, curr_tag: str) -> bool:
    if curr_tag in {"[PAD]", ""}:
        return False
    if curr_tag == "O":
        return True
    curr_prefix, curr_type = split_tag(curr_tag)
    if curr_prefix == "B":
        return bool(curr_type)
    if curr_prefix != "I" or not curr_type:
        return False
    prev_prefix, prev_type = split_tag(prev_tag)
    return prev_prefix in {"B", "I"} and prev_type == curr_type


def aggregate_char_probs(
    text_len: int,
    word_ids: Sequence[int | None],
    event_prob_row: torch.Tensor,
    o_tag_id: int,
) -> Tuple[List[List[float]], List[float]]:
    num_tags = int(event_prob_row.size(-1))
    probs = [[0.0] * num_tags for _ in range(max(0, int(text_len)))]
    best_token_conf = [0.0] * max(0, int(text_len))

    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid < 0 or wid >= text_len:
            continue
        tok_probs = event_prob_row[tok_idx].tolist()
        tok_top = float(max(tok_probs))
        if tok_top >= best_token_conf[wid]:
            probs[wid] = [float(x) for x in tok_probs]
            best_token_conf[wid] = tok_top

    for i in range(text_len):
        if best_token_conf[i] <= 0:
            probs[i][o_tag_id] = 1.0
            best_token_conf[i] = 1.0
        else:
            s = float(sum(probs[i]))
            if s <= 0:
                probs[i][o_tag_id] = 1.0
                best_token_conf[i] = 1.0
            else:
                probs[i] = [float(x / s) for x in probs[i]]
    return probs, best_token_conf


def decode_bio_constrained(
    char_probs: Sequence[Sequence[float]],
    id2tag: Dict[int, str],
    token_non_o_min_prob: float = 0.0,
) -> Tuple[List[str], List[float]]:
    seq_len = len(char_probs)
    if seq_len == 0:
        return [], []
    num_tags = len(char_probs[0])
    valid_tag_ids = [i for i in range(num_tags) if id2tag.get(i, "") not in {"[PAD]", ""}]
    if not valid_tag_ids:
        return ["O"] * seq_len, [1.0] * seq_len

    dp = [[NEG_INF] * num_tags for _ in range(seq_len)]
    back = [[-1] * num_tags for _ in range(seq_len)]

    for tid in valid_tag_ids:
        tag = id2tag.get(tid, "O")
        if tag != "O":
            if char_probs[0][tid] < float(token_non_o_min_prob):
                continue
            prefix, _ = split_tag(tag)
            if prefix == "I":
                continue
        p = max(1e-12, float(char_probs[0][tid]))
        dp[0][tid] = math.log(p)
        back[0][tid] = -1

    for t in range(1, seq_len):
        for curr in valid_tag_ids:
            curr_tag = id2tag.get(curr, "O")
            if curr_tag != "O" and char_probs[t][curr] < float(token_non_o_min_prob):
                continue
            best_score = NEG_INF
            best_prev = -1
            emit = math.log(max(1e-12, float(char_probs[t][curr])))
            for prev in valid_tag_ids:
                if dp[t - 1][prev] <= NEG_INF / 2:
                    continue
                prev_tag = id2tag.get(prev, "O")
                if not _is_valid_transition(prev_tag, curr_tag):
                    continue
                score = dp[t - 1][prev] + emit
                if score > best_score:
                    best_score = score
                    best_prev = prev
            if best_prev >= 0:
                dp[t][curr] = best_score
                back[t][curr] = best_prev

    last = max(valid_tag_ids, key=lambda tid: dp[seq_len - 1][tid])
    if dp[seq_len - 1][last] <= NEG_INF / 2:
        return ["O"] * seq_len, [1.0] * seq_len

    path = [last]
    for t in range(seq_len - 1, 0, -1):
        prev = back[t][path[-1]]
        if prev < 0:
            prev = valid_tag_ids[0]
        path.append(prev)
    path.reverse()

    tags: List[str] = []
    confs: List[float] = []
    for t, tid in enumerate(path):
        tag = id2tag.get(int(tid), "O")
        if tag == "[PAD]":
            tag = "O"
        tags.append(tag)
        confs.append(float(char_probs[t][int(tid)]))
    return tags, confs


def apply_span_confidence_filter(
    tags: Sequence[str],
    confs: Sequence[float],
    span_conf_min: float,
) -> List[str]:
    out = list(tags)
    n = len(out)
    i = 0
    while i < n:
        tag = out[i]
        if tag == "O":
            i += 1
            continue
        prefix, etype = split_tag(tag)
        if prefix not in {"B", "I"} or not etype:
            out[i] = "O"
            i += 1
            continue
        j = i + 1
        while j < n:
            pfx, tp = split_tag(out[j])
            if pfx == "I" and tp == etype:
                j += 1
                continue
            break
        span_conf = float(sum(confs[i:j]) / max(1, j - i))
        if span_conf < float(span_conf_min):
            for k in range(i, j):
                out[k] = "O"
        i = j
    return out
