from typing import Dict, List

from elder_portrait.tag_schema import (
    is_participant_type,
    split_tag,
    split_bio_tokens,
)


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for val in values:
        if val in seen:
            continue
        seen.add(val)
        result.append(val)
    return result


def parse_bio_entities(text: str, event_bio: str) -> List[Dict]:
    chars = list(str(text))
    tags = split_bio_tokens(event_bio)
    if len(tags) < len(chars):
        tags += ["O"] * (len(chars) - len(tags))
    elif len(tags) > len(chars):
        tags = tags[: len(chars)]

    entities: List[Dict] = []
    cur_type = None
    cur_start = None
    cur_chars: List[str] = []

    def flush() -> None:
        nonlocal cur_type, cur_start, cur_chars
        if cur_type is None or cur_start is None or not cur_chars:
            cur_type = None
            cur_start = None
            cur_chars = []
            return
        entities.append(
            {
                "type": cur_type,
                "text": "".join(cur_chars),
                "start": cur_start,
                "end": cur_start + len(cur_chars),
            }
        )
        cur_type = None
        cur_start = None
        cur_chars = []

    for idx, (ch, tag) in enumerate(zip(chars, tags)):
        prefix, etype = split_tag(tag)

        if tag == "O":
            flush()
            continue

        if prefix == "B":
            flush()
            cur_type = etype
            cur_start = idx
            cur_chars = [ch]
            continue

        if prefix == "I":
            if cur_type == etype and cur_start is not None:
                cur_chars.append(ch)
            else:
                flush()
                cur_type = etype
                cur_start = idx
                cur_chars = [ch]
            continue

        # For non-BIO tags like Activity_pro/background_pro, merge contiguous
        # same-tag spans into one entity.
        if cur_type == tag and cur_start is not None:
            cur_chars.append(ch)
        else:
            flush()
            cur_type = tag
            cur_start = idx
            cur_chars = [ch]

    flush()
    return entities


def summarize_entities(entities: List[Dict]) -> Dict:
    type_counts: Dict[str, int] = {}
    mentions_by_type: Dict[str, List[str]] = {}
    protagonist_mentions: List[str] = []
    participant_mentions: List[str] = []

    for ent in entities:
        etype = ent["type"]
        text = ent["text"]
        type_counts[etype] = type_counts.get(etype, 0) + 1
        mentions_by_type.setdefault(etype, []).append(text)

        if etype == "protagonist":
            protagonist_mentions.append(text)
        elif is_participant_type(etype):
            participant_mentions.append(text)

    entity_mentions_by_type = {
        etype: _dedupe_keep_order(values) for etype, values in mentions_by_type.items()
    }

    primary_event = "GENERAL"
    if type_counts:
        primary_event = sorted(
            type_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[0][0]

    return {
        "entity_type_counts": type_counts,
        "entity_mentions_by_type": entity_mentions_by_type,
        "protagonist_mentions": _dedupe_keep_order(protagonist_mentions),
        "participant_mentions": _dedupe_keep_order(participant_mentions),
        "primary_event_type": primary_event,
    }
