import re
from typing import Dict, Iterable, List, Set, Tuple


STORYWELL_ALLOWED_TAGS: List[str] = [
    "O",
    "B-protagonist",
    "I-protagonist",
    "B-participant_par",
    "I-participant_par",
    "B-location_pro",
    "I-location_pro",
    "B-location_par",
    "I-location_par",
    "B-Health_pro",
    "I-Health_pro",
    "B-Health_par",
    "I-Health_par",
    "B-Identity_pro",
    "I-Identity_pro",
    "B-Identity_par",
    "I-Identity_par",
    "B-Achievement_pro",
    "I-Achievement_pro",
    "B-Achievement_par",
    "I-Achievement_par",
    "B-Interest_pro",
    "I-Interest_pro",
    "B-Interest_par",
    "I-Interest_par",
    "B-Occupation_pro",
    "I-Occupation_pro",
    "B-Occupation_par",
    "I-Occupation_par",
    "B-Education background_pro",
    "I-Education background_pro",
    "B-Education background_par",
    "I-Education background_par",
    "B-Social Activity_pro",
    "I-Social Activity_pro",
]

STORYWELL_ALLOWED_TAG_SET: Set[str] = set(STORYWELL_ALLOWED_TAGS)

# Legacy aliases used by early versions of this project. We normalize them to
# the current StoryWell schema so old CSV files can still be trained directly.
LEGACY_TAG_ALIASES: Dict[str, str] = {
    "B-PERSON": "B-protagonist",
    "I-PERSON": "I-protagonist",
    "B-FAMILY": "B-participant_par",
    "I-FAMILY": "I-participant_par",
    "B-HEALTH": "B-Health_pro",
    "I-HEALTH": "I-Health_pro",
    "B-LEISURE": "B-Interest_pro",
    "I-LEISURE": "I-Interest_pro",
    "B-TRAVEL": "B-Social Activity_pro",
    "I-TRAVEL": "I-Social Activity_pro",
    # Legacy non-BIO tags used by older rule modules.
    "Activity_pro": "B-Social Activity_pro",
    "background_pro": "B-Education background_pro",
    "B-PENSION": "B-Identity_pro",
    "I-PENSION": "I-Identity_pro",
    # Current schema has no dedicated TIME tag in event_bio.
    "B-TIME": "O",
    "I-TIME": "O",
}


def split_bio_tokens(event_bio: str) -> List[str]:
    raw = str(event_bio or "").strip()
    if not raw:
        return []
    if "\t" in raw:
        # Preserve tags containing spaces, e.g. "B-Education background_pro".
        return [tok.strip() for tok in raw.split("\t") if tok.strip()]
    return raw.split()


def join_bio_tokens(tokens: List[str], prefer_tab: bool = False) -> str:
    if not tokens:
        return ""
    has_space_tag = any(re.search(r"\s", t) for t in tokens)
    if prefer_tab or has_space_tag:
        return "\t".join(tokens)
    return " ".join(tokens)


def split_tag(tag: str) -> Tuple[str, str]:
    if "-" not in tag:
        return tag, ""
    prefix, etype = tag.split("-", 1)
    return prefix, etype


def normalize_tag(tag: str) -> str:
    return LEGACY_TAG_ALIASES.get(tag, tag)


def normalize_event_bio(event_bio: str) -> str:
    prefer_tab = "\t" in str(event_bio or "")
    tokens = [normalize_tag(tok) for tok in split_bio_tokens(event_bio)]
    if not tokens:
        return ""

    # Fix orphan I-xxx into B-xxx so downstream BIO parsing remains stable.
    normalized: List[str] = []
    prev_prefix = ""
    prev_type = ""
    for tok in tokens:
        prefix, etype = split_tag(tok)
        if prefix == "I" and etype:
            if not (prev_prefix in {"B", "I"} and prev_type == etype):
                tok = f"B-{etype}"
                prefix = "B"
        normalized.append(tok)
        prev_prefix, prev_type = split_tag(tok)
    return join_bio_tokens(normalized, prefer_tab=prefer_tab)


def is_protagonist_type(entity_type: str) -> bool:
    return (
        entity_type == "protagonist"
        or entity_type.endswith("_pro")
        or entity_type in {
            "Activity_pro",
            "background_pro",
            "Social Activity_pro",
            "Education background_pro",
        }
    )


def is_participant_type(entity_type: str) -> bool:
    return entity_type.endswith("_par")


def find_unknown_tags(event_bio_series: Iterable[str]) -> List[str]:
    unknown = set()
    for bio in event_bio_series:
        normalized = normalize_event_bio(str(bio))
        for tag in split_bio_tokens(normalized):
            if tag not in STORYWELL_ALLOWED_TAG_SET:
                unknown.add(tag)
    return sorted(unknown)


def collect_entity_type_counts(tag_series: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tag in tag_series:
        if not tag or tag == "O":
            continue
        prefix, etype = split_tag(tag)
        if prefix in {"B", "I"} and etype:
            entity_type = etype
        elif prefix == "O":
            continue
        else:
            entity_type = tag
        counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts
