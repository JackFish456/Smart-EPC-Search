from __future__ import annotations

import argparse
import json

from epc_smart_search.assistant import ContractAssistant
from epc_smart_search.behavior_eval import DisabledGemma, evaluate_behavior_suite, format_suite_summary, load_behavior_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EPC Smart Search behavior evaluations against the live contract store.")
    parser.add_argument(
        "--cases",
        help="Optional path to a JSON behavior case file. Defaults to assets/behavior_eval_cases.json.",
    )
    parser.add_argument(
        "--only",
        action="append",
        dest="only_names",
        help="Case name to run. Repeat to run multiple named cases.",
    )
    parser.add_argument(
        "--use-gemma",
        action="store_true",
        help="Require Gemma to be available for this evaluation run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full suite result as JSON after the text summary.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    assistant = ContractAssistant()
    status = assistant.get_index_status()
    print(
        json.dumps(
            {
                "index_ready": status.ready,
                "document_id": status.document_id,
                "chunk_count": status.chunk_count,
                "feature_count": status.feature_count,
            },
            ensure_ascii=True,
        )
    )
    if not status.ready:
        return 1

    cases = load_behavior_cases(args.cases)
    if args.only_names:
        selected = set(args.only_names)
        cases = [case for case in cases if case.name in selected]
    if not cases:
        print("No behavior cases selected.")
        return 1

    if args.use_gemma:
        availability = assistant.gemma.get_availability()
        print(
            json.dumps(
                {
                    "gemma_available": availability.available,
                    "gemma_mode": availability.mode,
                    "gemma_message": availability.message,
                },
                ensure_ascii=True,
            )
        )
        if not availability.available:
            return 2
        gemma_client = assistant.gemma
    else:
        gemma_client = DisabledGemma()

    result = evaluate_behavior_suite(cases, assistant.answer_policy, assistant.retriever, gemma_client)
    print(format_suite_summary(result))
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=True))
    return 0 if result.failed_cases == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
