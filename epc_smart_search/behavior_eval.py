from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from epc_smart_search.answer_policy import AnswerPolicy, AssistantAnswer


@dataclass(slots=True, frozen=True)
class BehaviorCase:
    name: str
    question: str
    history: tuple[dict[str, str], ...] = ()
    deep_think: bool = False
    expand_answer: bool = False
    previous_answer: str | None = None
    expected_request_shape: str | None = None
    expected_answer_mode: str | None = None
    expected_refused: bool | None = None
    required_snippets: tuple[str, ...] = ()
    banned_snippets: tuple[str, ...] = ()
    expected_sections: tuple[str, ...] = ()
    banned_headings: tuple[str, ...] = ()
    min_citations: int | None = None


@dataclass(slots=True, frozen=True)
class BehaviorCaseResult:
    case_name: str
    question: str
    effective_question: str
    request_shape: str
    answer_mode: str
    refused: bool
    passed: bool
    failures: tuple[str, ...]
    answer_text: str
    selected_bundle_id: str | None
    selected_section: str | None
    selected_heading: str | None
    top_sections: tuple[str, ...]
    top_headings: tuple[str, ...]
    citation_sections: tuple[str, ...]
    citation_headings: tuple[str, ...]
    used_gemma_disambiguation: bool


@dataclass(slots=True, frozen=True)
class BehaviorSuiteResult:
    case_results: tuple[BehaviorCaseResult, ...]

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def passed_cases(self) -> int:
        return sum(1 for case in self.case_results if case.passed)

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases

    def to_dict(self) -> dict[str, object]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "results": [asdict(case) for case in self.case_results],
        }


class DisabledGemma:
    def ask(self, *args, **kwargs):
        raise RuntimeError("Gemma disabled for behavior evaluation")


def default_behavior_case_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "behavior_eval_cases.json"


def load_behavior_cases(path: str | Path | None = None) -> list[BehaviorCase]:
    case_path = Path(path) if path is not None else default_behavior_case_path()
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    return [behavior_case_from_dict(item) for item in payload]


def behavior_case_from_dict(payload: dict[str, object]) -> BehaviorCase:
    return BehaviorCase(
        name=str(payload["name"]),
        question=str(payload["question"]),
        history=tuple(dict(item) for item in payload.get("history", [])),
        deep_think=bool(payload.get("deep_think", False)),
        expand_answer=bool(payload.get("expand_answer", False)),
        previous_answer=str(payload["previous_answer"]) if payload.get("previous_answer") is not None else None,
        expected_request_shape=str(payload["expected_request_shape"]) if payload.get("expected_request_shape") is not None else None,
        expected_answer_mode=str(payload["expected_answer_mode"]) if payload.get("expected_answer_mode") is not None else None,
        expected_refused=bool(payload["expected_refused"]) if payload.get("expected_refused") is not None else None,
        required_snippets=tuple(str(item) for item in payload.get("required_snippets", [])),
        banned_snippets=tuple(str(item) for item in payload.get("banned_snippets", [])),
        expected_sections=tuple(str(item) for item in payload.get("expected_sections", [])),
        banned_headings=tuple(str(item) for item in payload.get("banned_headings", [])),
        min_citations=int(payload["min_citations"]) if payload.get("min_citations") is not None else None,
    )


def evaluate_behavior_case(case: BehaviorCase, answer_policy: AnswerPolicy, retriever, gemma_client) -> BehaviorCaseResult:
    effective_question = answer_policy.resolve_question(case.question, case.history)
    plan = answer_policy._plan_query(effective_question)
    trace = answer_policy._retrieve_trace(effective_question, gemma_client, deep_think=case.deep_think)
    answer = answer_policy.answer(
        case.question,
        case.history,
        gemma_client,
        deep_think=case.deep_think,
        expand_answer=case.expand_answer,
        previous_answer=case.previous_answer,
    )
    return build_case_result(case, effective_question, plan.request_shape, trace, answer)


def evaluate_behavior_suite(cases: Sequence[BehaviorCase], answer_policy: AnswerPolicy, retriever, gemma_client) -> BehaviorSuiteResult:
    results = [evaluate_behavior_case(case, answer_policy, retriever, gemma_client) for case in cases]
    return BehaviorSuiteResult(tuple(results))


def build_case_result(case: BehaviorCase, effective_question: str, request_shape: str, trace, answer: AssistantAnswer) -> BehaviorCaseResult:
    selected_bundle = trace.selected_bundle if trace is not None else None
    merged_ranked = list(trace.merged_ranked) if trace is not None else []
    citation_sections = tuple(citation.section_number or "" for citation in answer.citations)
    citation_headings = tuple(citation.heading for citation in answer.citations)
    top_sections = tuple(chunk.section_number or "" for chunk in merged_ranked[:5])
    top_headings = tuple(chunk.heading for chunk in merged_ranked[:5])
    answer_mode = infer_answer_mode(answer)
    failures = validate_behavior_case(case, request_shape, answer_mode, answer, selected_bundle, top_sections, top_headings)
    return BehaviorCaseResult(
        case_name=case.name,
        question=case.question,
        effective_question=effective_question,
        request_shape=request_shape,
        answer_mode=answer_mode,
        refused=answer.refused,
        passed=not failures,
        failures=tuple(failures),
        answer_text=answer.text,
        selected_bundle_id=getattr(selected_bundle, "bundle_id", None),
        selected_section=(selected_bundle.ranked_chunks[0].section_number if selected_bundle and selected_bundle.ranked_chunks else None),
        selected_heading=(selected_bundle.ranked_chunks[0].heading if selected_bundle and selected_bundle.ranked_chunks else None),
        top_sections=top_sections,
        top_headings=top_headings,
        citation_sections=citation_sections,
        citation_headings=citation_headings,
        used_gemma_disambiguation=bool(getattr(trace, "used_gemma_disambiguation", False)),
    )


def infer_answer_mode(answer: AssistantAnswer) -> str:
    text = answer.text.strip()
    if answer.refused or text == "I can't verify that from the contract.":
        return "refusal"
    if text.startswith("Here are the strongest contract-supported points about "):
        return "broad_topic_summary"
    if text.startswith("Answer:") or text.startswith("Direct contract text:"):
        return "grounded_fact"
    if text.startswith("- ") and "Section " in text:
        return "grouped_list"
    if "Section " in text and "Pages:" in text:
        return "grounded_fact"
    return "generated"


def validate_behavior_case(
    case: BehaviorCase,
    request_shape: str,
    answer_mode: str,
    answer: AssistantAnswer,
    selected_bundle,
    top_sections: tuple[str, ...],
    top_headings: tuple[str, ...],
) -> list[str]:
    failures: list[str] = []
    lowered_answer = answer.text.lower()
    lowered_headings = tuple(heading.lower() for heading in top_headings)
    if case.expected_request_shape is not None and request_shape != case.expected_request_shape:
        failures.append(f"expected request_shape={case.expected_request_shape!r}, got {request_shape!r}")
    if case.expected_answer_mode is not None and answer_mode != case.expected_answer_mode:
        failures.append(f"expected answer_mode={case.expected_answer_mode!r}, got {answer_mode!r}")
    if case.expected_refused is not None and answer.refused != case.expected_refused:
        failures.append(f"expected refused={case.expected_refused!r}, got {answer.refused!r}")
    for snippet in case.required_snippets:
        if snippet.lower() not in lowered_answer:
            failures.append(f"missing required snippet {snippet!r}")
    for snippet in case.banned_snippets:
        if snippet.lower() in lowered_answer:
            failures.append(f"found banned snippet {snippet!r}")
    if case.expected_sections:
        if not any(section in top_sections or section in tuple(citation.section_number or "" for citation in answer.citations) for section in case.expected_sections):
            failures.append(f"missing expected section from {case.expected_sections!r}")
    for banned_heading in case.banned_headings:
        lowered_banned = banned_heading.lower()
        selected_heading = getattr(selected_bundle.ranked_chunks[0], "heading", "") if selected_bundle and selected_bundle.ranked_chunks else ""
        if lowered_banned in selected_heading.lower():
            failures.append(f"selected bundle used banned heading {banned_heading!r}")
        if any(lowered_banned in citation.heading.lower() for citation in answer.citations):
            failures.append(f"citations included banned heading {banned_heading!r}")
    if case.min_citations is not None and len(answer.citations) < case.min_citations:
        failures.append(f"expected at least {case.min_citations} citations, got {len(answer.citations)}")
    return failures


def format_suite_summary(result: BehaviorSuiteResult) -> str:
    lines = [f"Behavior eval: {result.passed_cases}/{result.total_cases} passed"]
    failed = [case for case in result.case_results if not case.passed]
    for case in failed:
        lines.append(f"- {case.case_name}: " + "; ".join(case.failures))
    return "\n".join(lines)
