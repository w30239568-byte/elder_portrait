import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from elder_portrait.runtime import PortraitModelRuntime, RuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stable model API entrypoint for backend integration."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mapping_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--event_confidence_threshold", type=float, default=0.45)
    parser.add_argument("--event_confidence_threshold_has_entity", type=float, default=0.12)
    parser.add_argument("--token_non_o_min_prob", type=float, default=None)
    parser.add_argument("--span_conf_min", type=float, default=None)
    parser.add_argument("--decode_objective", type=str, default="")
    parser.add_argument("--disable_rule_fallback", action="store_true")
    return parser.parse_args()


def _read_input(input_json: str) -> Any:
    if input_json:
        raw = Path(input_json).read_text(encoding="utf-8-sig")
        return json.loads(raw)
    data = sys.stdin.read().strip()
    if not data:
        raise ValueError("No input provided. Use --input_json or stdin JSON.")
    return json.loads(data.lstrip("\ufeff"))


def _normalize_requests(payload: Any) -> List[Dict]:
    if isinstance(payload, dict) and "items" in payload:
        items = payload["items"]
        if not isinstance(items, list):
            raise ValueError("`items` must be a list.")
        return items
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Input JSON must be an object, list, or object with `items`.")


def _write_output(output_json: str, data: Dict) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if output_json:
        Path(output_json).write_text(text, encoding="utf-8")
    else:
        print(text)


def main() -> None:
    args = parse_args()
    runtime = PortraitModelRuntime(
        RuntimeConfig(
            checkpoint_path=args.checkpoint,
            mapping_path=args.mapping_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            strict_storywell_schema=True,
            event_confidence_threshold=args.event_confidence_threshold,
            event_confidence_threshold_has_entity=args.event_confidence_threshold_has_entity,
            enable_rule_fallback=not args.disable_rule_fallback,
            token_non_o_min_prob=args.token_non_o_min_prob,
            span_conf_min=args.span_conf_min,
            decode_objective=args.decode_objective,
        )
    )

    payload = _read_input(args.input_json)
    requests = _normalize_requests(payload)
    results = runtime.analyze_batch(requests)
    response = {
        "schema_version": "v1",
        "model_version": runtime.model_version,
        "count": len(results),
        "results": results,
    }
    _write_output(args.output_json, response)


if __name__ == "__main__":
    main()
