import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from elder_portrait.auto_bio import RuleBasedBioTagger
from elder_portrait.data import ElderPortraitDataset
from elder_portrait.decode_utils import (
    aggregate_char_probs,
    apply_span_confidence_filter,
    decode_bio_constrained,
)
from elder_portrait.event_parser import parse_bio_entities, summarize_entities
from elder_portrait.model import ElderPortraitMultiTaskModel, EventFusionClassifier
from elder_portrait.portrait import SENTIMENT_LABEL, build_portrait_summary
from elder_portrait.tag_schema import (
    find_unknown_tags,
    join_bio_tokens,
    normalize_event_bio,
    split_bio_tokens,
)


@dataclass
class RuntimeConfig:
    checkpoint_path: str
    mapping_path: str
    max_length: int = 128
    batch_size: int = 32
    device: str = "auto"
    strict_storywell_schema: bool = True
    event_confidence_threshold: float = 0.45
    event_confidence_threshold_has_entity: float = 0.12
    enable_rule_fallback: bool = True
    token_non_o_min_prob: Optional[float] = None
    span_conf_min: Optional[float] = None
    decode_objective: Optional[str] = None


def _get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        dev = str(device_name).lower()
        if dev.startswith("cuda") and not torch.cuda.is_available():
            print("Warning: CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TimeRule:
    name: str
    kind: str
    pattern: re.Pattern
    priority: int = 5
    normalizer: Optional[Callable[[str, date], str]] = None

WEEKDAY_MAP = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}

WEEKDAY_RRULE_MAP = {
    0: "MO",
    1: "TU",
    2: "WE",
    3: "TH",
    4: "FR",
    5: "SA",
    6: "SU",
}

RECENT_ALIAS = {
    "平时": "RANGE:USUAL",
    "日常": "RANGE:USUAL",
    "近期": "RANGE:RECENT",
    "最近": "RANGE:RECENT",
    "近来": "RANGE:VAGUE_PERIOD",
    "最近一段时间": "RANGE:VAGUE_PERIOD",
    "这段时间": "RANGE:VAGUE_PERIOD",
    "前阵子": "RANGE:VAGUE_PERIOD",
    "这阵子": "RANGE:VAGUE_PERIOD",
}

HEALTH_NORM_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(眼花|看不清|白内障)"), "视力问题"),
    (re.compile(r"(耳背|听不清)"), "听力问题"),
    (re.compile(r"(高血压|低血压|血压)"), "血压问题"),
    (re.compile(r"(糖尿病|血糖)"), "血糖问题"),
    (re.compile(r"(关节炎|膝|腿脚|腰|骨折|骨质疏松)"), "骨关节问题"),
    (re.compile(r"(失眠|睡眠)"), "睡眠问题"),
    (re.compile(r"(头晕|头痛|脑梗|中风)"), "神经系统问题"),
    (re.compile(r"(咳嗽|慢阻肺|肺)"), "呼吸系统问题"),
    (re.compile(r"(住院|手术|复查|体检|康复|理疗|吃药|服药)"), "医疗行为"),
]

LOCATION_NORM_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(家里|家中|家)"), "家庭住所"),
    (re.compile(r"(养老院|敬老院|护理院|养亲苑)"), "养老机构"),
    (re.compile(r"(社区|社区医院|社区卫生服务中心)"), "社区服务场所"),
    (re.compile(r"(医院|门诊|急诊)"), "医疗机构"),
    (re.compile(r"(公园|广场|酒店)"), "公共活动场所"),
]

ACTIVITY_NORM_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(晚会|聚会|交流会|团圆饭|聚餐|午餐|过年)"), "社交活动"),
    (re.compile(r"(体检|复查)"), "医疗随访活动"),
    (re.compile(r"(义工|志愿)"), "志愿活动"),
]

IDENTITY_NORM_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(退休)"), "退休身份"),
    (re.compile(r"(老师|教授)"), "教育工作者"),
    (re.compile(r"(工程师|研究员)"), "专业技术人员"),
    (re.compile(r"(医生|护士)"), "医护人员"),
]

FREQUENCY_REGEX_RULES: List[Tuple[re.Pattern, str, int]] = [
    (re.compile(r"(每天|每日|天天|每晚|每晨|每早)"), "daily", 4),
    (re.compile(r"(每周|每星期|每周末)"), "weekly", 3),
    (re.compile(r"(每月|月月|每隔一月)"), "monthly", 2),
    (re.compile(r"(每年|年年)"), "yearly", 1),
    (re.compile(r"(长期|一直|多年|反复|经常|老是|总是)"), "chronic", 4),
    (re.compile(r"(偶尔|有时|时不时|偶发|间或)"), "occasional", 1),
    (re.compile(r"(昨天|今日|今天|明天|后天|刚刚|这次|本次|一次)"), "once", 1),
]

INTENSITY_REGEX_RULES: List[Tuple[re.Pattern, str, int]] = [
    (re.compile(r"(剧烈|难忍|严重|无法|不能|卧床|很厉害|非常严重)"), "severe", 4),
    (re.compile(r"(加重|恶化|越来越|愈发|明显)"), "worsening", 4),
    (re.compile(r"(持续|反复|频繁|影响)"), "moderate", 2),
    (re.compile(r"(有点|轻微|稍微|偶有|隐隐)"), "mild", 1),
]

FREQUENCY_FROM_RRULE = {
    "DAILY": ("daily", 4),
    "WEEKLY": ("weekly", 3),
    "MONTHLY": ("monthly", 2),
    "YEARLY": ("yearly", 1),
}

TIME_RULES: List[TimeRule] = [
    # Relative N-unit ranges: near/recent + count + unit.
    TimeRule(
        name="recent_n_unit",
        kind="relative_range",
        pattern=re.compile(r"(?:近|最近|这|过去)\s*[一二两三四五六七八九十百零\d]{1,4}\s*个?\s*(?:天|周|星期|月|年)"),
        priority=1,
    ),
    # Absolute time expressions.
    TimeRule(
        name="absolute_ymd_cn",
        kind="absolute",
        pattern=re.compile(r"\d{4}年\d{1,2}月\d{1,2}[日号]?"),
        priority=1,
    ),
    TimeRule(
        name="absolute_ymd_sep",
        kind="absolute",
        pattern=re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"),
        priority=1,
    ),
    TimeRule(
        name="absolute_ym_cn",
        kind="absolute",
        pattern=re.compile(r"\d{4}年\d{1,2}月"),
        priority=2,
    ),
    TimeRule(
        name="absolute_ym_sep",
        kind="absolute",
        pattern=re.compile(r"\d{4}[-/.]\d{1,2}"),
        priority=2,
    ),
    TimeRule(
        name="absolute_md_cn",
        kind="absolute",
        pattern=re.compile(r"\d{1,2}月\d{1,2}[日号]?"),
        priority=3,
    ),
    TimeRule(
        name="absolute_year_cn",
        kind="absolute",
        pattern=re.compile(r"\d{4}年"),
        priority=4,
    ),
    # Relative day/week/month/year.
    TimeRule(
        name="relative_day",
        kind="relative_day",
        pattern=re.compile(r"(前天|昨天|昨日|今天|明天|后天)"),
        priority=2,
    ),
    TimeRule(
        name="relative_week_day",
        kind="relative_week_day",
        pattern=re.compile(r"(?:上周|下周|本周|这周|上星期|下星期|本星期|这星期)[一二三四五六日天末]"),
        priority=2,
    ),
    TimeRule(
        name="relative_week",
        kind="relative_week",
        pattern=re.compile(r"(上周|下周|本周|这周|上星期|下星期|本星期|这星期)"),
        priority=3,
    ),
    TimeRule(
        name="relative_month",
        kind="relative_month",
        pattern=re.compile(r"(上个月|本月|这个月|下个月)"),
        priority=3,
    ),
    TimeRule(
        name="relative_year",
        kind="relative_year",
        pattern=re.compile(r"(前年|去年|今年|明年)"),
        priority=3,
    ),
    # Recurrent expressions.
    TimeRule(
        name="rrule_weekly_detail",
        kind="recurrent",
        pattern=re.compile(r"(每周末|每星期天|每星期日|每周[一二三四五六日天]|每星期[一二三四五六日天])"),
        priority=2,
    ),
    TimeRule(
        name="rrule_weekly_plain",
        kind="recurrent",
        pattern=re.compile(r"(每周|每星期)"),
        priority=4,
    ),
    TimeRule(
        name="rrule_daily",
        kind="recurrent",
        pattern=re.compile(r"(每天|每日|每晚|每早|每晨)"),
        priority=3,
    ),
    TimeRule(
        name="rrule_every_interval",
        kind="recurrent",
        pattern=re.compile(r"每隔\s*[一二两三四五六七八九十百零\d]{1,4}\s*(?:天|周|星期|月|年)"),
        priority=3,
    ),
    TimeRule(
        name="rrule_monthly",
        kind="recurrent",
        pattern=re.compile(r"每月\d{1,2}(?:日|号)?"),
        priority=2,
    ),
    TimeRule(
        name="rrule_yearly",
        kind="recurrent",
        pattern=re.compile(r"每年\d{1,2}月\d{1,2}(?:日|号)?"),
        priority=2,
    ),
    # Vague period words.
    TimeRule(
        name="vague_recent",
        kind="vague_period",
        pattern=re.compile(r"(近期|最近|近来|最近一段时间|这段时间|前阵子|这阵子|平时|日常)"),
        priority=6,
    ),
    # Festival / day-part words.
    TimeRule(
        name="festival",
        kind="festival",
        pattern=re.compile(r"(春节|除夕)"),
        priority=5,
    ),
    TimeRule(
        name="daypart",
        kind="day_part",
        pattern=re.compile(r"(凌晨|清晨|早上|上午|中午|下午|傍晚|晚上|夜里|夜间)"),
        priority=6,
    ),
    # Clock time.
    TimeRule(
        name="clock_time",
        kind="time_of_day",
        pattern=re.compile(r"\d{1,2}:\d{1,2}"),
        priority=5,
    ),
]

UNKNOWN_TIME_CANDIDATE_PATTERNS: List[re.Pattern] = [
    re.compile(r"(近来|前阵子|这阵子|最近一段时间|这段时间)"),
    re.compile(r"每隔\s*[一二两三四五六七八九十百零\d]{1,4}\s*(?:天|周|月|年)"),
    re.compile(r"[一二两三四五六七八九十百零\d]{1,4}\s*(?:天|周|星期|个月|月|年)\s*(?:前|后)"),
]


def _month_start_n_months_ago(ref_date: date, months: int) -> date:
    months = max(0, int(months))
    total = ref_date.year * 12 + (ref_date.month - 1) - months
    year = total // 12
    month = total % 12 + 1
    return date(year, month, 1)


def _parse_count(raw: str) -> Optional[int]:
    val = str(raw or "").strip()
    if not val:
        return None
    if val.isdigit():
        return int(val)

    digit_map = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if val in digit_map:
        return digit_map[val]

    unit_map = {"十": 10, "百": 100, "千": 1000}
    total = 0
    current = 0
    has_unit = False
    for ch in val:
        if ch in digit_map:
            current = digit_map[ch]
            continue
        if ch in unit_map:
            has_unit = True
            unit = unit_map[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
            continue
        return None
    if has_unit:
        return total + current
    return None


def _start_of_week(ref_date: date) -> date:
    return ref_date - timedelta(days=ref_date.weekday())


def _range_for_recent_unit(ref_date: date, count: int, unit: str) -> str:
    n = max(1, int(count))
    if unit == "天":
        start = ref_date - timedelta(days=n - 1)
        return f"{start.isoformat()} ~ {ref_date.isoformat()}"
    if unit in {"周", "星期"}:
        start = _start_of_week(ref_date) - timedelta(weeks=n - 1)
        return f"{start.isoformat()} ~ {ref_date.isoformat()}"
    if unit == "月":
        start = _month_start_n_months_ago(ref_date, n)
        return f"{start.isoformat()} ~ {ref_date.isoformat()}"
    if unit == "年":
        start = date(ref_date.year - n, 1, 1)
        return f"{start.isoformat()} ~ {ref_date.isoformat()}"
    return ""


def _weekday_target_date(ref_date: date, weekday_idx: int, week_shift: int = 0) -> date:
    monday = ref_date - timedelta(days=ref_date.weekday())
    target = monday + timedelta(days=weekday_idx + 7 * week_shift)
    return target


def _normalize_time_expr(expr: str, ref_date: date) -> str:
    expr_clean = str(expr or "").strip()
    anchor_date = _start_of_week(ref_date)

    if not expr_clean:
        return ""

    recent_n_unit_match = re.fullmatch(
        r"(?:近|最近|这|过去)\s*([一二两三四五六七八九十百零\d]{1,4})\s*个?\s*(天|周|星期|月|年)",
        expr_clean,
    )
    if recent_n_unit_match:
        count = _parse_count(recent_n_unit_match.group(1))
        if count and 1 <= count <= 120:
            return _range_for_recent_unit(ref_date, count, recent_n_unit_match.group(2))

    rel_days = {
        "前天": -2,
        "昨天": -1,
        "昨日": -1,
        "今天": 0,
        "明天": 1,
        "后天": 2,
    }
    if expr_clean in rel_days:
        return (ref_date + timedelta(days=rel_days[expr_clean])).isoformat()

    m = re.fullmatch(r"(上周|下周|本周|这周|上星期|下星期|本星期|这星期)([一二三四五六日天末])?", expr_clean)
    if m:
        week_word = m.group(1)
        weekday = m.group(2)
        week_shift = {
            "上周": -1,
            "上星期": -1,
            "下周": 1,
            "下星期": 1,
            "本周": 0,
            "这周": 0,
            "本星期": 0,
            "这星期": 0,
        }[week_word]
        base_monday = anchor_date + timedelta(weeks=week_shift)
        if not weekday:
            return f"{base_monday.isoformat()} ~ {(base_monday + timedelta(days=6)).isoformat()}"
        if weekday == "末":
            saturday = base_monday + timedelta(days=5)
            sunday = base_monday + timedelta(days=6)
            return f"{saturday.isoformat()} ~ {sunday.isoformat()}"
        weekday_idx = WEEKDAY_MAP[weekday]
        return (base_monday + timedelta(days=weekday_idx)).isoformat()

    m = re.fullmatch(r"每(?:周|星期)([一二三四五六日天])", expr_clean)
    if m:
        weekday_idx = WEEKDAY_MAP[m.group(1)]
        return f"RRULE:FREQ=WEEKLY;BYDAY={WEEKDAY_RRULE_MAP[weekday_idx]};WKST=MO"

    if expr_clean == "每周末":
        return "RRULE:FREQ=WEEKLY;BYDAY=SA,SU;WKST=MO"
    if expr_clean in {"每星期天", "每星期日"}:
        return "RRULE:FREQ=WEEKLY;BYDAY=SU;WKST=MO"

    if expr_clean in {"每周", "每星期"}:
        return "RRULE:FREQ=WEEKLY;WKST=MO"

    if re.fullmatch(r"(每天|每日|每晚|每早|每晨)", expr_clean):
        return "RRULE:FREQ=DAILY"

    m = re.fullmatch(r"每隔\s*([一二两三四五六七八九十百零\d]{1,4})\s*(天|周|星期|月|年)", expr_clean)
    if m:
        count = _parse_count(m.group(1))
        unit = m.group(2)
        if count and 1 <= count <= 120:
            if unit == "天":
                return f"RRULE:FREQ=DAILY;INTERVAL={count}"
            if unit in {"周", "星期"}:
                return f"RRULE:FREQ=WEEKLY;INTERVAL={count};WKST=MO"
            if unit == "月":
                return f"RRULE:FREQ=MONTHLY;INTERVAL={count}"
            if unit == "年":
                return f"RRULE:FREQ=YEARLY;INTERVAL={count}"

    m = re.fullmatch(r"每月(\d{1,2})(?:日|号)?", expr_clean)
    if m:
        day_of_month = max(1, min(31, int(m.group(1))))
        return f"RRULE:FREQ=MONTHLY;BYMONTHDAY={day_of_month}"

    m = re.fullmatch(r"每年(\d{1,2})月(\d{1,2})(?:日|号)?", expr_clean)
    if m:
        month = max(1, min(12, int(m.group(1))))
        day_of_month = max(1, min(31, int(m.group(2))))
        return f"RRULE:FREQ=YEARLY;BYMONTH={month};BYMONTHDAY={day_of_month}"

    year_map = {
        "前年": ref_date.year - 2,
        "去年": ref_date.year - 1,
        "今年": ref_date.year,
        "明年": ref_date.year + 1,
    }
    if expr_clean in year_map:
        return str(year_map[expr_clean])

    month_map = {
        "上个月": (ref_date.replace(day=1) - timedelta(days=1)).strftime("%Y-%m"),
        "本月": ref_date.strftime("%Y-%m"),
        "这个月": ref_date.strftime("%Y-%m"),
        "下个月": (
            ref_date.replace(day=28) + timedelta(days=4)
        ).replace(day=1).strftime("%Y-%m"),
    }
    if expr_clean in month_map:
        return month_map[expr_clean]

    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", expr_clean)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})", expr_clean)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return f"{y:04d}-{mo:02d}"

    m = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})[日号]?", expr_clean)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    m = re.fullmatch(r"(\d{4})年(\d{1,2})月", expr_clean)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return f"{y:04d}-{mo:02d}"

    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})[日号]?", expr_clean)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        return f"{ref_date.year:04d}-{mo:02d}-{d:02d}"

    m = re.fullmatch(r"(\d{4})年", expr_clean)
    if m:
        return m.group(1)

    if expr_clean == "春节":
        return "FESTIVAL:SPRING_FESTIVAL"
    if expr_clean == "除夕":
        return "FESTIVAL:CHUXI"

    day_part_map = {
        "凌晨": "DAYPART:DAWN",
        "清晨": "DAYPART:MORNING_EARLY",
        "早上": "DAYPART:MORNING",
        "上午": "DAYPART:FORENOON",
        "中午": "DAYPART:NOON",
        "下午": "DAYPART:AFTERNOON",
        "傍晚": "DAYPART:EVENING",
        "晚上": "DAYPART:NIGHT",
        "夜里": "DAYPART:NIGHT",
        "夜间": "DAYPART:NIGHT",
    }
    if expr_clean in day_part_map:
        return day_part_map[expr_clean]

    if expr_clean in RECENT_ALIAS:
        return RECENT_ALIAS[expr_clean]

    return ""


def extract_time_spans(text: str, ref_date: date | None = None) -> List[Dict]:
    ref_date = ref_date or date.today()
    candidates: List[Dict] = []
    seen = set()

    for rule_idx, rule in enumerate(TIME_RULES):
        for m in rule.pattern.finditer(text):
            mention = str(m.group(0) or "").strip()
            if not mention:
                continue
            key = (m.start(), m.end(), mention, rule.name)
            if key in seen:
                continue
            seen.add(key)
            normalizer = rule.normalizer or _normalize_time_expr
            normalized = str(normalizer(mention, ref_date) or "")
            candidates.append(
                {
                    "text": mention,
                    "start": m.start(),
                    "end": m.end(),
                    "normalized": normalized,
                    "kind": rule.kind,
                    "priority": int(rule.priority),
                    "rule_idx": int(rule_idx),
                }
            )

    # Resolve overlaps by longer match first, then semantic priority, then rule order.
    candidates.sort(
        key=lambda x: (
            x["start"],
            -(x["end"] - x["start"]),
            x["priority"],
            x["rule_idx"],
        )
    )
    accepted: List[Dict] = []
    occupied = [False] * max(1, len(text))
    for item in candidates:
        if any(occupied[i] for i in range(item["start"], item["end"])):
            continue
        for i in range(item["start"], item["end"]):
            occupied[i] = True
        accepted.append(item)

    accepted.sort(key=lambda x: (x["start"], x["end"]))
    for item in accepted:
        item.pop("priority", None)
        item.pop("rule_idx", None)
    return accepted


def sample_unmatched_time_candidates(
    text: str,
    time_spans: List[Dict],
    max_items: int = 12,
) -> List[Dict]:
    if not text:
        return []

    occupied = [False] * max(1, len(text))
    for span in time_spans:
        s = max(0, int(span.get("start", 0)))
        e = min(len(text), int(span.get("end", 0)))
        for i in range(s, max(s, e)):
            occupied[i] = True

    candidates: List[Dict] = []
    seen = set()
    for pat in UNKNOWN_TIME_CANDIDATE_PATTERNS:
        for m in pat.finditer(text):
            s, e = int(m.start()), int(m.end())
            if any(occupied[i] for i in range(s, e)):
                continue
            mention = str(m.group(0) or "").strip()
            if not mention:
                continue
            key = (s, e, mention)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"text": mention, "start": s, "end": e})

    candidates.sort(key=lambda x: (x["start"], x["end"]))
    return candidates[:max_items]


def extract_time_mentions(text: str) -> List[str]:
    return [x["text"] for x in extract_time_spans(text)]


def _apply_norm_rules(text: str, rules: List[Tuple[re.Pattern, str]]) -> str:
    for pat, canonical in rules:
        if pat.search(text):
            return canonical
    return text


def normalize_subject_mention(text: str) -> str:
    val = str(text)
    if re.search(
        r"(\u513f\u5b50|\u5973\u513f|\u5b50\u5973|\u5b69\u5b50|\u5b59\u5b50|\u5b59\u5973|\u8001\u4f34|\u5bb6\u4eba)",
        val,
    ):
        return "\u5bb6\u5c5e"
    if re.search(r"(\u533b\u751f|\u62a4\u58eb)", val):
        return "\u533b\u62a4\u4eba\u5458"
    if re.search(r"(\u670b\u53cb|\u90bb\u5c45|\u4eb2\u5bb6)", val):
        return "\u793e\u4f1a\u5173\u7cfb"
    return val


def infer_protagonist_name(text: str) -> str:
    text = str(text or "")
    if not text:
        return ""
    surname_chars = (
        "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华"
        "金魏陶姜戚谢邹喻柏水窦章云苏潘葛范彭郎鲁韦昌马苗凤花方俞任"
        "袁柳鲍史唐费廉岑薛雷贺倪汤殷罗毕郝邬安常乐于时傅皮卞齐康伍余"
        "元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞"
        "熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐"
        "邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓"
        "郁单杭洪包诸左石崔吉龚程邢滑裴陆荣翁荀羊於惠甄麴家封芮羿储靳"
        "汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇"
        "栾暴甘斜厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲台从鄂"
        "索咸籍赖卓蔺屠蒙池乔阴郁胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰"
        "郦雍郤璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习"
        "宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳"
        "沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰"
        "巢关蒯相查后荆红游竺权逯盖益桓公万俟司马上官欧阳夏侯诸葛闻人"
        "东方赫连皇甫尉迟公羊澹台公冶宗政濮阳淳于单于太叔申屠公孙仲孙"
        "轩辕令狐钟离宇文长孙慕容司徒司空"
    )
    pattern = re.compile(
        rf"([{surname_chars}][\u4e00-\u9fa5]{{1,2}})(?=(?:阿姨|大爷|奶奶|爷爷|叔叔|先生|女士|老人|夫妻|今年|在|，|。|、|$))"
    )
    m = pattern.search(text)
    if m:
        return m.group(1)
    relaxed = re.compile(rf"([{surname_chars}][\u4e00-\u9fa5]{{1,2}})")
    m2 = relaxed.search(text[:24])
    return m2.group(1) if m2 else ""


def normalize_entity_text(entity_type: str, text: str) -> str:
    val = str(text)
    if entity_type in {"Health_pro", "Health_par"}:
        return _apply_norm_rules(val, HEALTH_NORM_RULES)
    if entity_type in {"location_pro", "location_par"}:
        return _apply_norm_rules(val, LOCATION_NORM_RULES)
    if entity_type in {"Activity_pro", "Social Activity_pro"}:
        return _apply_norm_rules(val, ACTIVITY_NORM_RULES)
    if entity_type in {"Identity_pro", "Occupation_pro", "Education background_pro"}:
        return _apply_norm_rules(val, IDENTITY_NORM_RULES)
    if entity_type == "participant_par":
        return normalize_subject_mention(val)
    return val


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for val in values:
        if val in seen:
            continue
        seen.add(val)
        result.append(val)
    return result


def normalize_mentions_by_type(entity_summary: Dict) -> Dict[str, List[str]]:
    mentions_by_type = entity_summary.get("entity_mentions_by_type", {})
    normalized: Dict[str, List[str]] = {}
    for etype, mentions in mentions_by_type.items():
        normalized_vals = [normalize_entity_text(etype, m) for m in mentions]
        normalized[etype] = _dedupe_keep_order(normalized_vals)
    return normalized


def _parse_rrule_freq(value: str) -> Tuple[str, int]:
    m = re.search(r"FREQ=([A-Z]+)", value or "")
    if not m:
        return "unknown", 0
    freq_key = m.group(1)
    return FREQUENCY_FROM_RRULE.get(freq_key, ("unknown", 0))


def _normalized_time_sort_key(normalized: str, ref_date: date | None = None) -> Tuple[int, int]:
    ref_date = ref_date or date.today()
    value = str(normalized or "").strip()
    if not value:
        return 9, 0

    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", value)
    if m:
        try:
            return 0, date.fromisoformat(m.group(1)).toordinal()
        except ValueError:
            return 9, 0

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return 0, date.fromisoformat(value).toordinal()
        except ValueError:
            return 9, 0

    m = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if m:
        try:
            return 0, date(int(m.group(1)), int(m.group(2)), 1).toordinal()
        except ValueError:
            return 9, 0

    m = re.fullmatch(r"(\d{4})", value)
    if m:
        return 0, date(int(m.group(1)), 1, 1).toordinal()

    if value.startswith("RANGE:USUAL"):
        return 6, 0
    if value.startswith("RANGE:RECENT"):
        return 7, ref_date.toordinal()
    if value.startswith("RANGE:VAGUE_PERIOD"):
        return 7, ref_date.toordinal()
    if value.startswith("DAYPART:"):
        return 8, 0
    if value.startswith("FESTIVAL:"):
        return 8, 0
    if value.startswith("RRULE:"):
        return 8, 0
    return 9, 0


def _event_priority_label(score: int) -> str:
    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    if score >= 1:
        return "low"
    return "none"


def _time_span_distance(ent_start: int, ent_end: int, span: Dict) -> int:
    s, e = int(span["start"]), int(span["end"])
    if e < ent_start:
        return ent_start - e
    if s > ent_end:
        return s - ent_end
    return 0


def _time_specificity_rank(normalized: str) -> int:
    value = str(normalized or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}\s*~\s*\d{4}-\d{2}-\d{2}", value):
        return 0
    if re.fullmatch(r"\d{4}(-\d{2}){0,2}", value):
        return 1
    if value.startswith("RRULE:"):
        return 2
    if value.startswith("RANGE:"):
        return 3
    if value.startswith("DAYPART:") or value.startswith("FESTIVAL:"):
        return 4
    return 5


def _sentence_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    s = max(0, min(int(start), len(text)))
    e = max(s, min(int(end), len(text)))
    left = s
    while left > 0 and text[left - 1] not in "。！？!?；;\n\r":
        left -= 1
    right = e
    while right < len(text) and text[right] not in "。！？!?；;\n\r":
        right += 1
    if right < len(text):
        right += 1
    return left, right


def _best_span_by_distance(
    ent_start: int,
    ent_end: int,
    spans: List[Dict],
) -> Optional[Dict]:
    if not spans:
        return None
    ranked = sorted(
        spans,
        key=lambda x: (
            _time_span_distance(ent_start, ent_end, x),
            _time_specificity_rank(str(x.get("normalized", ""))),
            int(x.get("start", 0)),
        ),
    )
    return ranked[0] if ranked else None


def _best_span_by_specificity(
    ent_start: int,
    ent_end: int,
    spans: List[Dict],
) -> Optional[Dict]:
    if not spans:
        return None
    ranked = sorted(
        spans,
        key=lambda x: (
            _time_specificity_rank(str(x.get("normalized", ""))),
            _time_span_distance(ent_start, ent_end, x),
            int(x.get("start", 0)),
        ),
    )
    return ranked[0] if ranked else None


def _nearest_time_span(
    text: str,
    ent_start: int,
    ent_end: int,
    time_spans: List[Dict],
) -> Dict | None:
    if not time_spans:
        return None

    sent_left, sent_right = _sentence_bounds(text, ent_start, ent_end)
    same_sentence = [
        x for x in time_spans
        if int(x.get("start", 0)) >= sent_left and int(x.get("end", 0)) <= sent_right
    ]

    # Tier 1: nearby mentions.
    near = [x for x in time_spans if _time_span_distance(ent_start, ent_end, x) <= 28]
    if near:
        picked = _best_span_by_distance(ent_start, ent_end, near)
        if picked is not None:
            # If only day-part/festival is nearby but sentence has stronger date/range signal,
            # prefer the stronger same-sentence span for stable timeline anchoring.
            picked_rank = _time_specificity_rank(str(picked.get("normalized", "")))
            if picked_rank >= 4 and same_sentence:
                sentence_best = _best_span_by_specificity(ent_start, ent_end, same_sentence)
                if sentence_best is not None:
                    sentence_rank = _time_specificity_rank(str(sentence_best.get("normalized", "")))
                    if sentence_rank < picked_rank:
                        return sentence_best
            return picked

    # Tier 2: same sentence mentions.
    if same_sentence:
        picked = _best_span_by_distance(ent_start, ent_end, same_sentence)
        if picked is not None:
            return picked

    # Tier 3: nearest mention in paragraph/range.
    in_paragraph = [x for x in time_spans if _time_span_distance(ent_start, ent_end, x) <= 120]
    if in_paragraph:
        picked = _best_span_by_distance(ent_start, ent_end, in_paragraph)
        if picked is not None:
            return picked

    # Final fallback: nearest in entire text.
    return _best_span_by_distance(ent_start, ent_end, time_spans)


def _context_window(text: str, start: int, end: int, size: int = 14) -> str:
    s = max(0, int(start) - size)
    e = min(len(text), int(end) + size)
    return text[s:e]


def _build_quote_fragment(text: str, start: int, end: int, size: int = 16) -> str:
    if not text:
        return ""
    s = max(0, min(int(start), len(text)))
    e = max(s, min(int(end), len(text)))
    sent_s, sent_e = _sentence_bounds(text, s, e)
    sentence = str(text[sent_s:sent_e]).strip()
    if sentence:
        return sentence
    # Fallback to context window for unexpected malformed offsets.
    window_s = max(0, s - size)
    window_e = min(len(text), e + size)
    return str(text[window_s:window_e]).strip()


def _effective_event_text(normalized: str, raw: str) -> str:
    n = str(normalized or "").strip()
    if n:
        return n
    return str(raw or "").strip()


def infer_event_frequency(
    text: str,
    ent_start: int,
    ent_end: int,
    time_spans: List[Dict],
) -> Dict:
    nearest = _nearest_time_span(text, ent_start, ent_end, time_spans)
    if nearest:
        normalized = str(nearest.get("normalized", ""))
        if normalized.startswith("RRULE:"):
            label, score = _parse_rrule_freq(normalized)
            return {
                "label": label,
                "score": score,
                "evidence": nearest.get("text", ""),
                "source": "time_rrule",
            }
        if normalized.startswith("RANGE:USUAL"):
            return {"label": "chronic", "score": 3, "evidence": nearest.get("text", ""), "source": "time_range"}
        if normalized.startswith("RANGE:RECENT"):
            return {"label": "occasional", "score": 1, "evidence": nearest.get("text", ""), "source": "time_range"}
        if re.fullmatch(r"\d{4}(-\d{2}){0,2}", normalized):
            return {"label": "once", "score": 1, "evidence": nearest.get("text", ""), "source": "time_absolute"}

    ctx = _context_window(text, ent_start, ent_end, size=20)
    for pat, label, score in FREQUENCY_REGEX_RULES:
        m = pat.search(ctx)
        if m:
            return {"label": label, "score": score, "evidence": m.group(0), "source": "context_regex"}
    return {"label": "unknown", "score": 0, "evidence": "", "source": "none"}


def infer_event_intensity(text: str, ent_start: int, ent_end: int, entity_type: str) -> Dict:
    if entity_type not in {"Health_pro", "Health_par"}:
        return {"label": "unknown", "score": 0, "evidence": ""}
    event_text = text[int(ent_start):int(ent_end)]
    if re.search(r"(浣忛櫌|鎵嬫湳|澶嶆煡|浣撴|搴峰|鐞嗙枟|鍚冭嵂|鏈嶈嵂)", event_text):
        return {"label": "unknown", "score": 0, "evidence": ""}
    ctx = _context_window(text, ent_start, ent_end, size=20)
    for pat, label, score in INTENSITY_REGEX_RULES:
        m = pat.search(ctx)
        if m:
            return {"label": label, "score": score, "evidence": m.group(0)}
    return {"label": "unknown", "score": 0, "evidence": ""}


def enrich_entities(text: str, entities: List[Dict], time_spans: List[Dict]) -> List[Dict]:
    enriched: List[Dict] = []
    for ent in entities:
        nearest = _nearest_time_span(text, ent["start"], ent["end"], time_spans)
        freq = infer_event_frequency(text, ent["start"], ent["end"], time_spans)
        intensity = infer_event_intensity(text, ent["start"], ent["end"], ent["type"])
        dimension = map_entity_dimension(ent["type"])
        priority_score = int(freq["score"]) + int(intensity["score"]) * 2
        if dimension == "health" and int(intensity["score"]) > 0:
            priority_score += 1
        enriched.append(
            {
                **ent,
                "time_text": str(nearest.get("text", "")) if nearest else "",
                "time_normalized": str(nearest.get("normalized", "")) if nearest else "",
                "time_kind": str(nearest.get("kind", "")) if nearest else "",
                "normalized_text": normalize_entity_text(ent["type"], ent["text"]),
                "frequency": freq["label"],
                "frequency_score": freq["score"],
                "frequency_evidence": freq["evidence"],
                "intensity": intensity["label"],
                "intensity_score": intensity["score"],
                "intensity_evidence": intensity["evidence"],
                "priority_score": priority_score,
                "priority_level": _event_priority_label(priority_score),
            }
        )
    return enriched


def map_entity_dimension(entity_type: str) -> str:
    if entity_type in {"Health_pro", "Health_par"}:
        return "health"
    if entity_type in {
        "participant_par",
        "Activity_pro",
        "Social Activity_pro",
        "Interest_pro",
        "location_par",
    }:
        return "social"
    return "profile"


class PortraitModelRuntime:
    """
    Stable inference runtime for backend integration.

    Keep this class API stable so Java/SpringBoot can call it without being
    affected by model-internal refactors.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.device = _get_device(config.device)
        self.auto_tagger = RuleBasedBioTagger()

        checkpoint = torch.load(config.checkpoint_path, map_location=self.device)
        self.model_name = checkpoint["model_name"]
        self.architecture = str(checkpoint.get("architecture", "fusion_v1"))
        self.is_multitask = self.architecture in {"multitask_v2", "multitask_joint_v3"}
        self.label2id: Dict[int, int] = {
            int(k): int(v) for k, v in checkpoint["label2id"].items()
        }

        with open(config.mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.tag2id: Dict[str, int] = {str(k): int(v) for k, v in mapping["tag2id"].items()}
        self.trigger2id: Dict[str, int] = {
            str(k): int(v)
            for k, v in (mapping.get("trigger2id") or checkpoint.get("trigger2id") or {}).items()
        }
        self.event_type2id: Dict[str, int] = {
            str(k): int(v)
            for k, v in (mapping.get("event_type2id") or checkpoint.get("event_type2id") or {}).items()
        }
        self.id2label: Dict[int, int] = {
            int(k): int(v) for k, v in mapping["id2label"].items()
        }
        self.id2tag: Dict[int, str] = {
            int(k): str(v) for k, v in mapping.get("id2tag", {}).items()
        }
        if not self.id2tag:
            self.id2tag = {int(v): str(k) for k, v in self.tag2id.items()}
        decode_cfg = dict(mapping.get("decode_config", {}) or {})
        if not decode_cfg:
            decode_cfg = dict(checkpoint.get("decode_config", {}) or {})
        explicit_objective = str(config.decode_objective or "").strip().lower()
        if explicit_objective:
            objective = explicit_objective
        else:
            objective = str(decode_cfg.get("decode_objective", "balance")).strip().lower()
            if objective == "recall":
                objective = "balance"
        if objective not in {"precision", "balance", "recall"}:
            objective = "balance"
        self.decode_objective = objective

        decode_token_th = float(decode_cfg.get("token_non_o_min_prob", 0.12))
        decode_span_th = float(decode_cfg.get("span_conf_min", 0.12))
        if config.token_non_o_min_prob is not None:
            self.token_non_o_min_prob = float(config.token_non_o_min_prob)
        else:
            self.token_non_o_min_prob = float(decode_token_th)
            if self.decode_objective in {"precision", "balance"}:
                self.token_non_o_min_prob = float(max(0.12, self.token_non_o_min_prob))
        if config.span_conf_min is not None:
            self.span_conf_min = float(config.span_conf_min)
        else:
            self.span_conf_min = float(decode_span_th)
            if self.decode_objective in {"precision", "balance"}:
                self.span_conf_min = float(max(0.12, self.span_conf_min))

        tokenizer_path = Path(config.mapping_path).parent / "tokenizer"
        if tokenizer_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), use_fast=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if self.is_multitask:
            state_dict = checkpoint["model_state_dict"]
            has_trigger_head = any(k.startswith("trigger_classifier.") for k in state_dict.keys())
            has_event_type_head = any(
                k.startswith("event_type_classifier.") for k in state_dict.keys()
            )
            num_trigger_tags = int(len(self.trigger2id)) if has_trigger_head else 0
            num_event_types = int(len(self.event_type2id)) if has_event_type_head else 0
            if has_trigger_head and num_trigger_tags <= 0 and "trigger_classifier.weight" in state_dict:
                num_trigger_tags = int(state_dict["trigger_classifier.weight"].shape[0])
            if has_event_type_head and num_event_types <= 0 and "event_type_classifier.weight" in state_dict:
                num_event_types = int(state_dict["event_type_classifier.weight"].shape[0])
            self.model = ElderPortraitMultiTaskModel(
                model_name=self.model_name,
                num_labels=len(self.label2id),
                num_event_tags=len(self.tag2id),
                num_trigger_tags=num_trigger_tags,
                num_event_types=num_event_types,
                dropout=float(checkpoint["config"].get("dropout", 0.2)),
            ).to(self.device)
        else:
            self.model = EventFusionClassifier(
                model_name=self.model_name,
                num_labels=len(self.label2id),
                num_event_tags=len(self.tag2id),
                event_embed_dim=checkpoint["config"]["event_embed_dim"],
                dropout=checkpoint["config"]["dropout"],
            ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.model_version = checkpoint.get(
            "model_version", Path(config.checkpoint_path).stem
        )

    @staticmethod
    def _emotion_score(sentiment: Dict) -> float:
        label = int(sentiment["label"])
        conf = float(sentiment["confidence"])
        if label == 2:
            return round(50 + 50 * conf, 2)
        if label == 0:
            return round(50 - 50 * conf, 2)
        return 50.0

    @staticmethod
    def _pick_subjects(entity_type: str, entity_summary: Dict) -> List[str]:
        protagonist = entity_summary.get("protagonist_mentions", [])
        participants = entity_summary.get("participant_mentions", [])
        if entity_type.endswith("_par"):
            return participants or protagonist
        return protagonist or participants

    def _build_proposal_view(
        self,
        text: str,
        entities: List[Dict],
        entity_summary: Dict,
        sentiment: Dict,
    ) -> Dict:
        type_counts = entity_summary["entity_type_counts"]
        total_entities = max(1, sum(int(v) for v in type_counts.values()))
        health_count = int(type_counts.get("Health_pro", 0)) + int(
            type_counts.get("Health_par", 0)
        )
        social_count = sum(
            int(c)
            for t, c in type_counts.items()
            if map_entity_dimension(t) == "social"
        )

        time_spans = extract_time_spans(text)
        unknown_time_candidates = sample_unmatched_time_candidates(text, time_spans)
        time_mentions = [x["text"] for x in time_spans]
        normalized_times = [x["normalized"] for x in time_spans if x["normalized"]]
        normalized_entity_mentions = normalize_mentions_by_type(entity_summary)
        ref_date = date.today()

        event_items = []
        event_triplets = []
        timeline = []
        frequency_summary: Dict[str, int] = {}
        intensity_summary: Dict[str, int] = {}
        priority_summary: Dict[str, int] = {}
        suppressed_single_char_events = 0
        event_candidate_types = {
            "Activity_pro",
            "Social Activity_pro",
            "Health_pro",
            "Health_par",
            "Achievement_pro",
            "Interest_pro",
        }
        for idx, ent in enumerate(entities, start=1):
            if str(ent.get("type", "")) not in event_candidate_types:
                continue
            dimension = map_entity_dimension(ent["type"])
            subjects = self._pick_subjects(ent["type"], entity_summary)
            normalized_subjects = [normalize_subject_mention(x) for x in subjects]
            normalized_what = ent.get("normalized_text", normalize_entity_text(ent["type"], ent["text"]))
            effective_what = _effective_event_text(normalized_what, ent.get("text", ""))
            if len(effective_what) <= 1:
                suppressed_single_char_events += 1
                continue
            frequency_label = str(ent.get("frequency", "unknown"))
            frequency_score = int(ent.get("frequency_score", 0))
            intensity_label = str(ent.get("intensity", "unknown"))
            intensity_score = int(ent.get("intensity_score", 0))
            time_text = str(ent.get("time_text", ""))
            time_normalized = str(ent.get("time_normalized", ""))
            time_kind = str(ent.get("time_kind", ""))
            quote = _build_quote_fragment(text, int(ent.get("start", 0)), int(ent.get("end", 0)))
            if not time_text and not time_normalized:
                nearest = _nearest_time_span(text, ent["start"], ent["end"], time_spans)
                if nearest:
                    time_text = str(nearest.get("text", ""))
                    time_normalized = str(nearest.get("normalized", ""))
                    time_kind = str(nearest.get("kind", ""))
            when_values = [x for x in [time_normalized, time_text] if x]
            sort_bucket, sort_value = _normalized_time_sort_key(
                time_normalized or (when_values[0] if when_values else ""),
                ref_date=ref_date,
            )

            priority_score = frequency_score + intensity_score * 2
            if dimension == "health" and intensity_score > 0:
                priority_score += 1
            priority_level = _event_priority_label(priority_score)

            frequency_summary[frequency_label] = frequency_summary.get(frequency_label, 0) + 1
            intensity_summary[intensity_label] = intensity_summary.get(intensity_label, 0) + 1
            priority_summary[priority_level] = priority_summary.get(priority_level, 0) + 1

            event_item = {
                "type": ent["type"],
                "text": ent["text"],
                "normalized_text": normalized_what,
                "time_text": time_text,
                "time_normalized": time_normalized,
                "time_kind": time_kind,
                "frequency": frequency_label,
                "frequency_score": frequency_score,
                "frequency_evidence": ent.get("frequency_evidence", ""),
                "intensity": intensity_label,
                "intensity_score": intensity_score,
                "intensity_evidence": ent.get("intensity_evidence", ""),
                "priority_score": priority_score,
                "priority_level": priority_level,
                "quote": quote,
                "start": ent["start"],
                "end": ent["end"],
                "dimension": dimension,
            }
            event_items.append(event_item)

            triplet = {
                "event_type": ent["type"],
                "when": when_values,
                "time_text": time_text,
                "time_normalized": time_normalized,
                "time_kind": time_kind,
                "who": subjects,
                "who_normalized": _dedupe_keep_order(normalized_subjects),
                "what": ent["text"],
                "what_normalized": normalized_what,
                "frequency": frequency_label,
                "frequency_score": frequency_score,
                "intensity": intensity_label,
                "intensity_score": intensity_score,
                "priority_score": priority_score,
                "priority_level": priority_level,
                "quote": quote,
                "dimension": dimension,
                "start": ent["start"],
                "end": ent["end"],
            }
            event_triplets.append(triplet)

            timeline.append(
                {
                    "order": idx,
                    "when": when_values,
                    "time_text": time_text,
                    "time_normalized": time_normalized,
                    "time_kind": time_kind,
                    "who": subjects,
                    "who_normalized": _dedupe_keep_order(normalized_subjects),
                    "what": ent["text"],
                    "what_normalized": normalized_what,
                    "frequency": frequency_label,
                    "frequency_score": frequency_score,
                    "intensity": intensity_label,
                    "intensity_score": intensity_score,
                    "priority_score": priority_score,
                    "priority_level": priority_level,
                    "event_type": ent["type"],
                    "quote": quote,
                    "dimension": dimension,
                    "timeline_bucket": sort_bucket,
                    "timeline_value": sort_value,
                }
            )

        timeline_sorted = [
            dict(item)
            for item in sorted(
                timeline,
                key=lambda x: (
                    int(x.get("timeline_bucket", 9)),
                    int(x.get("timeline_value", 0)),
                    int(x.get("order", 0)),
                ),
            )
        ]
        for pos, item in enumerate(timeline_sorted, start=1):
            item["order_sorted"] = pos

        return {
            "event_elements": {
                "time_mentions": time_mentions,
                "time_spans": time_spans,
                "time_parse_stats": {
                    "recognized_count": int(len(time_spans)),
                    "unknown_candidate_count": int(len(unknown_time_candidates)),
                    "unknown_candidates": unknown_time_candidates,
                },
                "subject_mentions": {
                    "protagonist": entity_summary["protagonist_mentions"],
                    "participant": entity_summary["participant_mentions"],
                },
                "subject_mentions_normalized": {
                    "protagonist": _dedupe_keep_order(
                        [normalize_subject_mention(x) for x in entity_summary["protagonist_mentions"]]
                    ),
                    "participant": _dedupe_keep_order(
                        [normalize_subject_mention(x) for x in entity_summary["participant_mentions"]]
                    ),
                },
                "events": event_items,
                "event_triplets": event_triplets,
                "timeline": timeline,
                "timeline_sorted": timeline_sorted,
                "timeline_sort_order": "old_to_new",
                "entity_mentions_by_type_normalized": normalized_entity_mentions,
                "frequency_summary": frequency_summary,
                "intensity_summary": intensity_summary,
                "priority_summary": priority_summary,
                "suppressed_single_char_events": int(suppressed_single_char_events),
            },
            "core_dimensions": {
                "health": {
                    "score": round(health_count * 100 / total_entities, 2),
                    "evidence_count": health_count,
                },
                "emotion": {
                    "label": sentiment["label_name"],
                    "score": self._emotion_score(sentiment),
                    "confidence": sentiment["confidence"],
                },
                "social": {
                    "score": round(social_count * 100 / total_entities, 2),
                    "evidence_count": social_count,
                },
            },
        }

    def _validate_tags(self, event_bio: str) -> None:
        if not self.config.strict_storywell_schema:
            return
        unknown = find_unknown_tags([event_bio])
        if unknown:
            raise ValueError(
                "Found tags outside StoryWell schema: " + ", ".join(unknown[:12])
            )

    def _prepare_event_bio_legacy(self, text: str, request: Dict) -> tuple[str, str]:
        provided_event_bio = request.get("event_bio")
        if provided_event_bio is None or str(provided_event_bio).strip() == "":
            protagonist_name = str(request.get("protagonist_name", "")).strip()
            event_bio = self.auto_tagger.tag(text, protagonist_name=protagonist_name)
            source = "auto_rule_v1"
        else:
            event_bio = normalize_event_bio(str(provided_event_bio))
            source = "provided"
        self._validate_tags(event_bio)
        return event_bio, source

    def _predict_batch_legacy(self, dataframe: pd.DataFrame) -> List[Dict]:
        dataset = ElderPortraitDataset(
            dataframe=dataframe,
            tokenizer=self.tokenizer,
            tag2id=self.tag2id,
            label2id=None,
            max_length=self.config.max_length,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False
        )

        logits_list: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in dataloader:
                logits = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    event_ids=batch["event_ids"].to(self.device),
                )
                logits_list.append(logits.cpu())

        logits_all = torch.cat(logits_list, dim=0)
        probs_all = torch.softmax(logits_all, dim=-1).numpy()
        outputs = []
        for probs in probs_all:
            pred_idx = int(probs.argmax())
            pred_label = int(self.id2label[pred_idx])
            outputs.append(
                {
                    "label": pred_label,
                    "label_name": SENTIMENT_LABEL.get(pred_label, "unknown"),
                    "confidence": float(probs[pred_idx]),
                    "probabilities": {
                        str(self.id2label[i]): float(probs[i])
                        for i in range(len(self.id2label))
                    },
                }
            )
        return outputs

    def _decode_event_bio(
        self,
        text: str,
        word_ids: List[int | None],
        event_prob_row: torch.Tensor,
    ) -> tuple[str, float, str]:
        o_tag_id = int(self.tag2id.get("O", 0))
        char_probs, _ = aggregate_char_probs(
            text_len=len(text),
            word_ids=word_ids,
            event_prob_row=event_prob_row,
            o_tag_id=o_tag_id,
        )
        tags, confs = decode_bio_constrained(
            char_probs=char_probs,
            id2tag=self.id2tag,
            token_non_o_min_prob=float(self.token_non_o_min_prob),
        )
        tags = apply_span_confidence_filter(
            tags=tags,
            confs=confs,
            span_conf_min=float(self.span_conf_min),
        )
        event_bio = normalize_event_bio(join_bio_tokens(tags))
        valid_confs = [x for x in confs if x > 0]
        non_o_confs = [c for t, c in zip(tags, confs) if t != "O" and c > 0]
        if non_o_confs:
            confidence = float(sum(non_o_confs) / len(non_o_confs))
            confidence_mode = "non_o_mean"
        else:
            confidence = float(sum(valid_confs) / len(valid_confs)) if valid_confs else 0.0
            confidence_mode = "all_token_mean"
        return event_bio, confidence, confidence_mode

    def _predict_batch_multitask(self, texts: List[str]) -> List[Dict]:
        if not texts:
            return []
        chars_batch = [list(str(t)) for t in texts]
        enc = self.tokenizer(
            chars_batch,
            is_split_into_words=True,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            sent_logits = outputs["sentiment_logits"].cpu()
            event_logits = outputs["event_logits"].cpu()
            event_type_logits = outputs.get("event_type_logits")
            if event_type_logits is not None:
                event_type_logits = event_type_logits.cpu()

        sent_probs_all = torch.softmax(sent_logits, dim=-1)
        event_probs_all = torch.softmax(event_logits, dim=-1)
        event_type_probs_all = (
            torch.softmax(event_type_logits, dim=-1) if event_type_logits is not None else None
        )
        id2event_type = {v: k for k, v in self.event_type2id.items()}
        predictions: List[Dict] = []
        for i, text in enumerate(texts):
            sent_probs = sent_probs_all[i]
            pred_idx = int(torch.argmax(sent_probs).item())
            pred_label = int(self.id2label[pred_idx])
            sentiment = {
                "label": pred_label,
                "label_name": SENTIMENT_LABEL.get(pred_label, "unknown"),
                "confidence": float(sent_probs[pred_idx].item()),
                "probabilities": {
                    str(self.id2label[j]): float(sent_probs[j].item())
                    for j in range(len(self.id2label))
                },
            }
            word_ids = enc.word_ids(batch_index=i)
            event_bio, event_conf, event_conf_mode = self._decode_event_bio(
                text=str(text),
                word_ids=word_ids,
                event_prob_row=event_probs_all[i],
            )
            event_type_pred = None
            if event_type_probs_all is not None:
                probs = event_type_probs_all[i]
                et_idx = int(torch.argmax(probs).item())
                event_type_pred = {
                    "label": id2event_type.get(et_idx, str(et_idx)),
                    "confidence": float(probs[et_idx].item()),
                }
            predictions.append(
                {
                    "sentiment": sentiment,
                    "event_bio": event_bio,
                    "event_confidence": event_conf,
                    "event_confidence_mode": event_conf_mode,
                    "event_type": event_type_pred,
                }
            )
        return predictions

    @staticmethod
    def _has_non_o_tag(event_bio: str) -> bool:
        return any(tok != "O" for tok in split_bio_tokens(event_bio))

    @staticmethod
    def _all_o_event_bio(text: str) -> str:
        return join_bio_tokens(["O"] * len(str(text)))

    @staticmethod
    def _is_noisy_event_bio(text: str, event_bio: str) -> bool:
        entities = parse_bio_entities(text, event_bio)
        if not entities:
            return False
        text_len = max(1, len(text))
        max_reasonable_entities = max(10, text_len // 4)
        if len(entities) > max_reasonable_entities:
            return True
        total_span_len = sum(max(0, int(e["end"]) - int(e["start"])) for e in entities)
        avg_span_len = total_span_len / max(1, len(entities))
        if len(entities) >= 8 and avg_span_len < 1.8:
            return True
        single_spans = [e for e in entities if int(e["end"]) - int(e["start"]) <= 1]
        if single_spans:
            punct_like = 0
            for e in single_spans:
                t = str(e.get("text", ""))
                if re.fullmatch(r"[\W_]+", t) or re.fullmatch(r"[，。！？；：、,.!?;:\"“”‘’（）()《》【】\[\]—…-]", t):
                    punct_like += 1
            if punct_like >= max(1, int(0.3 * len(entities))):
                return True
            if len(single_spans) / max(1, len(entities)) > 0.7:
                return True
        coverage = total_span_len / text_len
        if coverage > 0.85 and avg_span_len < 2.2:
            return True
        return False

    def _select_event_bio_for_request(
        self,
        text: str,
        request: Dict,
        model_event_bio: str,
        model_event_conf: float,
    ) -> tuple[str, str, str]:
        provided_event_bio = request.get("event_bio")
        if provided_event_bio is not None and str(provided_event_bio).strip() != "":
            event_bio = normalize_event_bio(str(provided_event_bio))
            self._validate_tags(event_bio)
            return event_bio, "provided", ""

        event_bio_model = normalize_event_bio(str(model_event_bio))
        self._validate_tags(event_bio_model)
        has_entity = self._has_non_o_tag(event_bio_model)
        noisy = self._is_noisy_event_bio(text=text, event_bio=event_bio_model)
        base_threshold = float(self.config.event_confidence_threshold)
        has_entity_threshold = float(self.config.event_confidence_threshold_has_entity)
        objective = str(getattr(self, "decode_objective", "precision")).lower()
        if has_entity:
            if objective in {"precision", "balance"}:
                threshold = max(base_threshold, has_entity_threshold)
            else:
                threshold = min(base_threshold, has_entity_threshold)
        else:
            threshold = base_threshold
        conf_ok = float(model_event_conf) >= threshold
        if has_entity and conf_ok and not noisy:
            return event_bio_model, "model_v2", ""
        if (not has_entity) and conf_ok and not noisy:
            return event_bio_model, "model_v2", ""

        if self.config.enable_rule_fallback:
            protagonist_name = str(request.get("protagonist_name", "")).strip()
            if not protagonist_name:
                protagonist_name = infer_protagonist_name(text)
            event_bio_rule = self.auto_tagger.tag(text, protagonist_name=protagonist_name)
            self._validate_tags(event_bio_rule)
            if noisy:
                fallback_reason = "noisy_extraction"
            else:
                fallback_reason = "low_confidence" if has_entity else "empty_extraction"
            return event_bio_rule, "model_v2_rule_fallback", fallback_reason

        if noisy:
            fallback_reason = "noisy_extraction"
        else:
            fallback_reason = "low_confidence" if has_entity else "empty_extraction"
        # Precision/balance modes use strict rejection for low-confidence/noisy outputs.
        if objective in {"precision", "balance"}:
            return (
                self._all_o_event_bio(text),
                "model_v2_rejected_low_confidence",
                fallback_reason,
            )
        # Recall mode keeps low-confidence model output to maximize event recall.
        return event_bio_model, "model_v2_low_confidence", fallback_reason

    def analyze(self, request: Dict) -> Dict:
        results = self.analyze_batch([request])
        return results[0] if results else {}

    def analyze_batch(self, requests: List[Dict]) -> List[Dict]:
        if not requests:
            return []

        texts: List[str] = []
        for req in requests:
            text = str(req.get("text", ""))
            if not text:
                raise ValueError("Each item in batch must have non-empty `text`.")
            texts.append(text)

        prepared = []
        if self.is_multitask:
            predicted = self._predict_batch_multitask(texts)
            for req, text, pred in zip(requests, texts, predicted):
                event_bio, source, fallback_reason = self._select_event_bio_for_request(
                    text=text,
                    request=req,
                    model_event_bio=pred["event_bio"],
                    model_event_conf=float(pred["event_confidence"]),
                )
                prepared.append(
                    {
                        "request": req,
                        "text": text,
                        "event_bio": event_bio,
                        "event_bio_source": source,
                        "event_bio_confidence": float(pred["event_confidence"]),
                        "event_bio_confidence_mode": str(pred.get("event_confidence_mode", "")),
                        "fallback_reason": fallback_reason,
                        "sentiment": pred["sentiment"],
                        "event_type": pred.get("event_type"),
                    }
                )
        else:
            rows = []
            for req, text in zip(requests, texts):
                event_bio, source = self._prepare_event_bio_legacy(text, req)
                rows.append({"id": 1, "text": text, "event_bio": event_bio})
                prepared.append(
                    {
                        "request": req,
                        "text": text,
                        "event_bio": event_bio,
                        "event_bio_source": source,
                        "event_bio_confidence": 1.0,
                        "event_bio_confidence_mode": "legacy_rule_or_provided",
                        "fallback_reason": "",
                    }
                )
            sentiments = self._predict_batch_legacy(pd.DataFrame(rows, dtype=object))
            for prep, sentiment in zip(prepared, sentiments):
                prep["sentiment"] = sentiment

        results = []
        for prep in prepared:
            sentiment = prep["sentiment"]
            entities = parse_bio_entities(prep["text"], prep["event_bio"])
            time_spans = extract_time_spans(prep["text"])
            normalized_entities = enrich_entities(prep["text"], entities, time_spans)
            entity_summary = summarize_entities(entities)
            if not entity_summary.get("protagonist_mentions"):
                inferred_name = infer_protagonist_name(prep["text"])
                if inferred_name:
                    entity_summary["protagonist_mentions"] = [inferred_name]
                    entity_summary["entity_mentions_by_type"].setdefault(
                        "protagonist", [inferred_name]
                    )
                    entity_summary["entity_type_counts"]["protagonist"] = max(
                        1, int(entity_summary["entity_type_counts"].get("protagonist", 0))
                    )
            portrait_summary = build_portrait_summary(
                [
                    {
                        "sentiment": sentiment["label"],
                        "primary_event_type": entity_summary["primary_event_type"],
                        "entity_type_counts": entity_summary["entity_type_counts"],
                        "protagonist_mentions": entity_summary["protagonist_mentions"],
                        "participant_mentions": entity_summary["participant_mentions"],
                    }
                ]
            )
            proposal_view = self._build_proposal_view(
                text=prep["text"],
                entities=normalized_entities,
                entity_summary=entity_summary,
                sentiment=sentiment,
            )

            results.append(
                {
                    "schema_version": "v1",
                    "request_id": prep["request"].get("request_id", ""),
                    "user_id": prep["request"].get("user_id", ""),
                    "model_version": self.model_version,
                    "input": {
                        "text": prep["text"],
                        "event_bio_source": prep["event_bio_source"],
                        "event_bio_confidence": prep.get("event_bio_confidence", 1.0),
                        "event_bio_confidence_mode": prep.get("event_bio_confidence_mode", ""),
                        "fallback_reason": prep.get("fallback_reason", ""),
                        "architecture": self.architecture,
                    },
                    "sentiment": sentiment,
                    "event": {
                        "primary_event_type": entity_summary["primary_event_type"],
                        "entity_type_counts": entity_summary["entity_type_counts"],
                        "entity_mentions_by_type": entity_summary["entity_mentions_by_type"],
                        "protagonist_mentions": entity_summary["protagonist_mentions"],
                        "participant_mentions": entity_summary["participant_mentions"],
                        "entities": normalized_entities,
                    },
                    "portrait_features": portrait_summary["portrait_dimensions"],
                    "proposal_view": proposal_view,
                }
            )

        return results


