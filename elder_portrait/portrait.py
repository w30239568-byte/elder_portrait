from typing import Dict, List

from elder_portrait.tag_schema import is_participant_type, is_protagonist_type


SENTIMENT_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return a / b


def _top_k_from_mentions(mentions: List[str], top_k: int = 20) -> List[str]:
    counts: Dict[str, int] = {}
    for m in mentions:
        counts[m] = counts.get(m, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [x[0] for x in ranked[:top_k]]


def build_portrait_summary(records: List[Dict]) -> Dict:
    total = len(records)
    sentiment_counts = {"negative": 0, "neutral": 0, "positive": 0}
    primary_event_counts: Dict[str, int] = {}
    entity_type_counts: Dict[str, int] = {}
    all_protagonist_mentions: List[str] = []
    all_participant_mentions: List[str] = []

    for rec in records:
        senti_name = SENTIMENT_LABEL.get(int(rec["sentiment"]), "neutral")
        sentiment_counts[senti_name] += 1

        primary_event = rec.get("primary_event_type", "GENERAL")
        primary_event_counts[primary_event] = primary_event_counts.get(primary_event, 0) + 1

        for etype, cnt in rec.get("entity_type_counts", {}).items():
            entity_type_counts[etype] = entity_type_counts.get(etype, 0) + int(cnt)

        all_protagonist_mentions.extend(rec.get("protagonist_mentions", []))
        all_participant_mentions.extend(rec.get("participant_mentions", []))

    negative_ratio = _safe_ratio(sentiment_counts["negative"], total)
    positive_ratio = _safe_ratio(sentiment_counts["positive"], total)
    neutral_ratio = _safe_ratio(sentiment_counts["neutral"], total)

    total_entity_mentions = sum(entity_type_counts.values())
    protagonist_entity_mentions = sum(
        cnt for t, cnt in entity_type_counts.items() if is_protagonist_type(t)
    )
    participant_entity_mentions = sum(
        cnt for t, cnt in entity_type_counts.items() if is_participant_type(t)
    )

    achievement_mentions = entity_type_counts.get("Achievement_pro", 0)
    health_mentions = entity_type_counts.get("Health_pro", 0) + entity_type_counts.get(
        "Health_par", 0
    )

    portrait_dims = {
        "emotion_stability": round((positive_ratio + neutral_ratio) * 100, 2),
        "emotion_risk": round(negative_ratio * 100, 2),
        "protagonist_feature_focus": round(
            _safe_ratio(protagonist_entity_mentions, total_entity_mentions) * 100, 2
        ),
        "participant_feature_focus": round(
            _safe_ratio(participant_entity_mentions, total_entity_mentions) * 100, 2
        ),
        "achievement_attention": round(
            _safe_ratio(achievement_mentions, total_entity_mentions) * 100, 2
        ),
        "health_attention": round(
            _safe_ratio(health_mentions, total_entity_mentions) * 100, 2
        ),
    }

    return {
        "record_count": total,
        "sentiment_distribution": sentiment_counts,
        "primary_event_distribution": primary_event_counts,
        "entity_type_distribution": entity_type_counts,
        "portrait_dimensions": portrait_dims,
        "top_protagonist_mentions": _top_k_from_mentions(all_protagonist_mentions, 20),
        "top_participant_mentions": _top_k_from_mentions(all_participant_mentions, 20),
    }


def build_timeline(records: List[Dict]) -> List[Dict]:
    timeline = []
    for idx, rec in enumerate(records, start=1):
        timeline.append(
            {
                "index": idx,
                "id": int(rec["id"]) if str(rec.get("id", "")).isdigit() else rec.get("id"),
                "text": rec["text"],
                "sentiment": int(rec["sentiment"]),
                "sentiment_name": SENTIMENT_LABEL.get(int(rec["sentiment"]), "neutral"),
                "primary_event_type": rec.get("primary_event_type", "GENERAL"),
                "protagonist_mentions": rec.get("protagonist_mentions", []),
                "participant_mentions": rec.get("participant_mentions", []),
                "entity_mentions_by_type": rec.get("entity_mentions_by_type", {}),
                "entities": rec.get("entities", []),
            }
        )
    return timeline

