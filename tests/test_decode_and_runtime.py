import unittest
from datetime import date
from types import SimpleNamespace

from elder_portrait.decode_utils import decode_bio_constrained
from elder_portrait.runtime import (
    PortraitModelRuntime,
    _normalize_time_expr,
    enrich_entities,
    extract_time_spans,
    sample_unmatched_time_candidates,
)
from elder_portrait.tag_schema import split_bio_tokens, split_tag


class TestDecodeAndRuntime(unittest.TestCase):
    def test_bio_constrained_decode_has_no_illegal_i(self) -> None:
        id2tag = {
            0: "[PAD]",
            1: "O",
            2: "B-Health_pro",
            3: "I-Health_pro",
            4: "B-protagonist",
            5: "I-protagonist",
        }
        char_probs = [
            [0.0, 0.05, 0.10, 0.80, 0.03, 0.02],
            [0.0, 0.02, 0.90, 0.04, 0.02, 0.02],
            [0.0, 0.04, 0.03, 0.02, 0.04, 0.87],
            [0.0, 0.80, 0.05, 0.05, 0.05, 0.05],
        ]
        tags, _ = decode_bio_constrained(
            char_probs=char_probs,
            id2tag=id2tag,
            token_non_o_min_prob=0.0,
        )
        self.assertTrue(tags)
        for i, tag in enumerate(tags):
            if tag == "O":
                continue
            pfx, etype = split_tag(tag)
            if pfx != "I":
                continue
            self.assertGreater(i, 0)
            prev_prefix, prev_type = split_tag(tags[i - 1])
            self.assertIn(prev_prefix, {"B", "I"})
            self.assertEqual(prev_type, etype)

    def test_precision_mode_rejects_low_confidence(self) -> None:
        text = "abcdef"
        runtime = PortraitModelRuntime.__new__(PortraitModelRuntime)
        runtime.config = SimpleNamespace(
            enable_rule_fallback=False,
            event_confidence_threshold=0.45,
            event_confidence_threshold_has_entity=0.12,
        )
        runtime.decode_objective = "precision"
        runtime._validate_tags = lambda _: None
        runtime._is_noisy_event_bio = lambda **_: False

        event_bio, source, reason = runtime._select_event_bio_for_request(
            text=text,
            request={},
            model_event_bio="B-Health_pro I-Health_pro O O O O",
            model_event_conf=0.2,
        )
        tokens = split_bio_tokens(event_bio)
        self.assertEqual(len(tokens), len(text))
        self.assertTrue(all(t == "O" for t in tokens))
        self.assertEqual(source, "model_v2_rejected_low_confidence")
        self.assertEqual(reason, "low_confidence")

    def test_balance_mode_rejects_low_confidence(self) -> None:
        text = "abcdef"
        runtime = PortraitModelRuntime.__new__(PortraitModelRuntime)
        runtime.config = SimpleNamespace(
            enable_rule_fallback=False,
            event_confidence_threshold=0.45,
            event_confidence_threshold_has_entity=0.12,
        )
        runtime.decode_objective = "balance"
        runtime._validate_tags = lambda _: None
        runtime._is_noisy_event_bio = lambda **_: False

        event_bio, source, reason = runtime._select_event_bio_for_request(
            text=text,
            request={},
            model_event_bio="B-Health_pro I-Health_pro O O O O",
            model_event_conf=0.2,
        )
        tokens = split_bio_tokens(event_bio)
        self.assertEqual(len(tokens), len(text))
        self.assertTrue(all(t == "O" for t in tokens))
        self.assertEqual(source, "model_v2_rejected_low_confidence")
        self.assertEqual(reason, "low_confidence")

    def test_recall_mode_keeps_low_confidence_model_output(self) -> None:
        text = "abcdef"
        runtime = PortraitModelRuntime.__new__(PortraitModelRuntime)
        runtime.config = SimpleNamespace(
            enable_rule_fallback=False,
            event_confidence_threshold=0.45,
            event_confidence_threshold_has_entity=0.12,
        )
        runtime.decode_objective = "recall"
        runtime._validate_tags = lambda _: None
        runtime._is_noisy_event_bio = lambda **_: False

        model_bio = "B-Health_pro I-Health_pro O O O O"
        event_bio, source, reason = runtime._select_event_bio_for_request(
            text=text,
            request={},
            model_event_bio=model_bio,
            model_event_conf=0.05,
        )
        self.assertEqual(source, "model_v2_low_confidence")
        self.assertEqual(event_bio, model_bio)
        self.assertEqual(reason, "low_confidence")

    def test_proposal_view_suppresses_single_char_and_keeps_quote(self) -> None:
        runtime = PortraitModelRuntime.__new__(PortraitModelRuntime)
        text = "第一句。第二句包含关键事件，后面还有描述。第三句。"
        key_start = text.index("关键事件")
        key_end = key_start + len("关键事件")
        entities = [
            {"type": "Health_pro", "text": "b", "normalized_text": "b", "start": 1, "end": 2},
            {
                "type": "Health_pro",
                "text": "关键事件",
                "normalized_text": "sleep_issue",
                "start": key_start,
                "end": key_end,
            },
        ]
        summary = {
            "entity_type_counts": {"Health_pro": 2},
            "protagonist_mentions": ["A"],
            "participant_mentions": [],
            "entity_mentions_by_type": {"Health_pro": ["b", "关键事件"]},
        }
        sentiment = {"label": 1, "label_name": "neutral", "confidence": 0.8}

        proposal = runtime._build_proposal_view(text, entities, summary, sentiment)
        event_elements = proposal["event_elements"]
        self.assertEqual(event_elements["suppressed_single_char_events"], 1)
        self.assertEqual(len(event_elements["event_triplets"]), 1)
        quote = str(event_elements["event_triplets"][0].get("quote", "")).strip()
        self.assertEqual(quote, "第二句包含关键事件，后面还有描述。")
        self.assertNotIn("【", quote)

    def test_normalize_recent_n_months(self) -> None:
        ref = date(2026, 4, 8)
        self.assertEqual(
            _normalize_time_expr("近两个月", ref),
            "2026-02-01 ~ 2026-04-08",
        )
        self.assertEqual(
            _normalize_time_expr("最近三个月", ref),
            "2026-01-01 ~ 2026-04-08",
        )
        self.assertEqual(
            _normalize_time_expr("近2个月", ref),
            "2026-02-01 ~ 2026-04-08",
        )
        self.assertEqual(
            _normalize_time_expr("近3个月", ref),
            "2026-01-01 ~ 2026-04-08",
        )
        self.assertEqual(
            _normalize_time_expr("这两个月", ref),
            "2026-02-01 ~ 2026-04-08",
        )

    def test_normalize_last_month_remains_unchanged(self) -> None:
        ref = date(2026, 4, 8)
        self.assertEqual(_normalize_time_expr("上个月", ref), "2026-03")

    def test_extract_time_spans_hits_recent_n_months_and_weekly(self) -> None:
        ref = date(2026, 4, 8)
        text = "近两个月睡眠变浅，每周复查血压。"
        spans = extract_time_spans(text, ref_date=ref)
        span_map = {str(s.get("text")): str(s.get("normalized")) for s in spans}
        self.assertEqual(span_map.get("近两个月"), "2026-02-01 ~ 2026-04-08")
        self.assertEqual(span_map.get("每周"), "RRULE:FREQ=WEEKLY;WKST=MO")

    def test_normalize_recent_n_units_day_week_year(self) -> None:
        ref = date(2026, 4, 8)
        self.assertEqual(
            _normalize_time_expr("近10天", ref),
            "2026-03-30 ~ 2026-04-08",
        )
        self.assertEqual(
            _normalize_time_expr("近6周", ref),
            "2026-03-02 ~ 2026-04-08",
        )
        self.assertEqual(
            _normalize_time_expr("近1年", ref),
            "2025-01-01 ~ 2026-04-08",
        )

    def test_normalize_relative_week_variants(self) -> None:
        ref = date(2026, 4, 8)  # Wednesday
        self.assertEqual(_normalize_time_expr("上星期", ref), "2026-03-30 ~ 2026-04-05")
        self.assertEqual(_normalize_time_expr("这周五", ref), "2026-04-10")
        self.assertEqual(_normalize_time_expr("上周末", ref), "2026-04-04 ~ 2026-04-05")

    def test_normalize_recurrent_interval(self) -> None:
        ref = date(2026, 4, 8)
        self.assertEqual(
            _normalize_time_expr("每隔3天", ref),
            "RRULE:FREQ=DAILY;INTERVAL=3",
        )
        self.assertEqual(
            _normalize_time_expr("每隔两周", ref),
            "RRULE:FREQ=WEEKLY;INTERVAL=2;WKST=MO",
        )
        self.assertEqual(
            _normalize_time_expr("每星期日", ref),
            "RRULE:FREQ=WEEKLY;BYDAY=SU;WKST=MO",
        )

    def test_extract_time_spans_prefers_longer_overlap(self) -> None:
        ref = date(2026, 4, 8)
        text = "最近三个月她睡眠差。"
        spans = extract_time_spans(text, ref_date=ref)
        mentions = [str(s.get("text", "")) for s in spans]
        self.assertIn("最近三个月", mentions)
        self.assertNotIn("最近", mentions)

    def test_enrich_entities_picks_same_sentence_time(self) -> None:
        ref = date(2026, 4, 8)
        text = "近两个月她晚上总醒很多次并且白天精神不太好。下一句没有时间。"
        spans = extract_time_spans(text, ref_date=ref)
        entities = [
            {
                "type": "Health_pro",
                "text": "白天精神不太好",
                "start": text.index("白天"),
                "end": text.index("白天") + len("白天精神不太好"),
            }
        ]
        enriched = enrich_entities(text, entities, spans)
        self.assertEqual(enriched[0]["time_text"], "近两个月")
        self.assertEqual(enriched[0]["time_normalized"], "2026-02-01 ~ 2026-04-08")

    def test_unknown_time_candidate_sampling(self) -> None:
        text = "她前阵子有点焦虑，每隔两周复查，之后三天后再电话回访。"
        spans = extract_time_spans(text, ref_date=date(2026, 4, 8))
        sampled = sample_unmatched_time_candidates(text, spans)
        sampled_texts = {str(x.get("text", "")) for x in sampled}
        self.assertIn("三天后", sampled_texts)

    def test_time_challenge_set_v1_recall(self) -> None:
        ref = date(2026, 4, 8)
        challenge_cases = [
            {"text": "张芳近两个月睡眠变浅。", "expects": ["近两个月"]},
            {"text": "最近3个月她夜里常醒。", "expects": ["最近3个月"]},
            {"text": "近10天她头晕。", "expects": ["近10天"]},
            {"text": "近6周体重下降。", "expects": ["近6周"]},
            {"text": "近1年活动减少。", "expects": ["近1年"]},
            {"text": "上周三去复查血压。", "expects": ["上周三"]},
            {"text": "这周五再去门诊。", "expects": ["这周五"]},
            {"text": "上星期情绪不稳。", "expects": ["上星期"]},
            {"text": "上个月她住院治疗。", "expects": ["上个月"]},
            {"text": "去年做过手术。", "expects": ["去年"]},
            {"text": "每天早上散步。", "expects": ["每天", "早上"]},
            {"text": "每周复查一次。", "expects": ["每周"]},
            {"text": "每星期二上门访视。", "expects": ["每星期二"]},
            {"text": "每月15号测血压。", "expects": ["每月15号"]},
            {"text": "每年3月1日体检。", "expects": ["每年3月1日"]},
            {"text": "近期她总说腰痛。", "expects": ["近期"]},
            {"text": "这段时间食欲下降。", "expects": ["这段时间"]},
            {"text": "前阵子她常失眠。", "expects": ["前阵子"]},
            {"text": "春节前后家人来探望。", "expects": ["春节"]},
            {"text": "晚上她容易惊醒。", "expects": ["晚上"]},
        ]
        hit = 0
        total = 0
        for case in challenge_cases:
            spans = extract_time_spans(case["text"], ref_date=ref)
            mentions = {str(s.get("text", "")) for s in spans}
            for expected in case["expects"]:
                total += 1
                if expected in mentions:
                    hit += 1
        recall = hit / max(1, total)
        self.assertGreaterEqual(recall, 0.90)


if __name__ == "__main__":
    unittest.main()
