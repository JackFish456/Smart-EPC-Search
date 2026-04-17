from __future__ import annotations

import argparse
import json
import sys
import time

from epc_smart_search.assistant import ContractAssistant


DEFAULT_QUESTIONS = (
    "what is the closed cooling water pump configuration",
    "what are the design temperatures",
    "what is the fire water pump horse power",
    "what are my emission guarantees",
    "what are my emission guarentees in appendix e",
    "What does the contract say about steam bypass valves?",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a live EPC Smart Search sanity pass.")
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to run. Repeat to provide multiple prompts.",
    )
    parser.add_argument(
        "--use-gemma",
        action="store_true",
        help="Require Gemma to be available and run the live pass through the managed Gemma client.",
    )
    parser.add_argument(
        "--deep-think",
        action="store_true",
        help="Run questions through deep-think mode.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    assistant = ContractAssistant()
    status = assistant.get_index_status()
    payload = {
        "index_ready": status.ready,
        "document_id": status.document_id,
        "chunk_count": status.chunk_count,
        "feature_count": status.feature_count,
    }
    print(json.dumps(payload, ensure_ascii=True))
    if not status.ready:
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
        class NoGemma:
            def ask(self, *args, **kwargs):
                raise RuntimeError("Gemma disabled for this sanity pass")

        gemma_client = NoGemma()

    questions = tuple(args.questions or DEFAULT_QUESTIONS)
    for question in questions:
        started = time.perf_counter()
        answer = assistant.answer_policy.answer(
            question,
            None,
            gemma_client,
            deep_think=bool(args.deep_think),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        print(
            json.dumps(
                {
                    "question": question,
                    "deep_think": bool(args.deep_think),
                    "used_gemma": bool(args.use_gemma),
                    "elapsed_ms": round(elapsed_ms, 1),
                    "refused": answer.refused,
                    "answer": answer.text,
                    "citations": [
                        {
                            "section": citation.section_number,
                            "heading": citation.heading,
                            "attachment": citation.attachment,
                            "page_start": citation.page_start,
                            "page_end": citation.page_end,
                        }
                        for citation in answer.citations
                    ],
                },
                ensure_ascii=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
