from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from epc_smart_search.answer_policy import AnswerPolicy, AssistantAnswer
from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
from epc_smart_search.retrieval import ExactPageHit, HashingEmbedder, HybridRetriever
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore, pack_vector


@dataclass(slots=True, frozen=True)
class RegressionBenchmarkCase:
    name: str
    category: str
    question: str
    expected_top_chunk_ids: tuple[str, ...] = ()
    expected_answer_contains: tuple[str, ...] = ()
    expected_section: str | None = None
    expected_page: int | None = None
    expected_exact_hit_page: int | None = None
    require_citations: bool = False
    expect_refusal: bool = False


@dataclass(slots=True, frozen=True)
class RegressionBenchmarkResult:
    case: RegressionBenchmarkCase
    effective_question: str
    answer: AssistantAnswer
    exact_hits: tuple[ExactPageHit, ...]
    trace: object | None
    selected_chunk_id: str | None
    top_chunk_ids: tuple[str, ...]


class DisabledBenchmarkGemma:
    def ask(self, *args, **kwargs):
        raise RuntimeError("Gemma is disabled for the regression benchmark suite")


def default_benchmark_corpus_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "regression_benchmark_corpus.json"


def default_benchmark_cases_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "regression_benchmark_cases.json"


def load_regression_benchmark_chunks(path: str | Path | None = None) -> list[ChunkRecord]:
    corpus_path = Path(path) if path is not None else default_benchmark_corpus_path()
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [chunk_from_dict(item) for item in payload]


def load_regression_benchmark_cases(path: str | Path | None = None) -> list[RegressionBenchmarkCase]:
    case_path = Path(path) if path is not None else default_benchmark_cases_path()
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    return [regression_case_from_dict(item) for item in payload]


def regression_case_from_dict(payload: dict[str, object]) -> RegressionBenchmarkCase:
    return RegressionBenchmarkCase(
        name=str(payload["name"]),
        category=str(payload["category"]),
        question=str(payload["question"]),
        expected_top_chunk_ids=tuple(str(item) for item in payload.get("expected_top_chunk_ids", [])),
        expected_answer_contains=tuple(str(item) for item in payload.get("expected_answer_contains", [])),
        expected_section=str(payload["expected_section"]) if payload.get("expected_section") is not None else None,
        expected_page=int(payload["expected_page"]) if payload.get("expected_page") is not None else None,
        expected_exact_hit_page=(
            int(payload["expected_exact_hit_page"]) if payload.get("expected_exact_hit_page") is not None else None
        ),
        require_citations=bool(payload.get("require_citations", False)),
        expect_refusal=bool(payload.get("expect_refusal", False)),
    )


def chunk_from_dict(payload: dict[str, object]) -> ChunkRecord:
    page_start = int(payload["page_start"])
    page_end = int(payload["page_end"]) if payload.get("page_end") is not None else page_start
    return ChunkRecord(
        chunk_id=str(payload["chunk_id"]),
        document_id="benchmark-doc",
        chunk_type=str(payload.get("chunk_type", "section")),
        section_number=str(payload["section_number"]),
        heading=str(payload["heading"]),
        full_text=str(payload["full_text"]),
        page_start=page_start,
        page_end=page_end,
        parent_chunk_id=str(payload["parent_chunk_id"]) if payload.get("parent_chunk_id") is not None else None,
        ordinal_in_document=int(payload.get("ordinal_in_document", page_start)),
    )


class RegressionBenchmarkHarness:
    def __init__(self, retriever: HybridRetriever, answer_policy: AnswerPolicy) -> None:
        self.retriever = retriever
        self.answer_policy = answer_policy
        self.gemma = DisabledBenchmarkGemma()

    @classmethod
    def from_defaults(cls) -> RegressionBenchmarkHarness:
        chunks = load_regression_benchmark_chunks()
        retriever = seed_benchmark_retriever(chunks)
        return cls(retriever, AnswerPolicy(retriever.store, retriever))

    def evaluate_case(self, case: RegressionBenchmarkCase) -> RegressionBenchmarkResult:
        effective_question = self.answer_policy.resolve_question(case.question)
        exact_hits = tuple(self.retriever.find_exact_page_hits(effective_question))
        trace = self.answer_policy._retrieve_trace(effective_question, self.gemma, deep_think=False)
        answer = self.answer_policy.answer(case.question, None, self.gemma)
        selected_chunk_id = None
        top_chunk_ids: tuple[str, ...] = ()
        if trace is not None:
            selected_bundle = getattr(trace, "selected_bundle", None)
            if selected_bundle is not None and getattr(selected_bundle, "ranked_chunks", None):
                selected_chunk_id = selected_bundle.ranked_chunks[0].chunk_id
            merged_ranked = tuple(getattr(trace, "merged_ranked", ()) or ())
            if merged_ranked:
                top_chunk_ids = tuple(chunk.chunk_id for chunk in merged_ranked[:5])
        return RegressionBenchmarkResult(
            case=case,
            effective_question=effective_question,
            answer=answer,
            exact_hits=exact_hits,
            trace=trace,
            selected_chunk_id=selected_chunk_id,
            top_chunk_ids=top_chunk_ids,
        )


def seed_benchmark_retriever(chunks: list[ChunkRecord]) -> HybridRetriever:
    db_path = "file:regression_benchmark_suite?mode=memory&cache=shared"
    store = ContractStore(db_path)
    embedder = HashingEmbedder(dimension=24)
    pages = [PageText(page_num=chunk.page_start, text=chunk.full_text, ocr_used=False) for chunk in chunks]
    store.replace_document(
        document_id="benchmark-doc",
        display_name="Synthetic EPC Contract.pdf",
        version_label="benchmark-v1",
        file_path="Synthetic EPC Contract.pdf",
        sha256="synthetic-benchmark",
        page_count=max((page.page_num for page in pages), default=0),
        chunks=chunks,
        pages=pages,
        features=build_chunk_features(chunks),
        embeddings={chunk.chunk_id: pack_vector(embedder.embed(chunk.full_text)) for chunk in chunks},
        model_name=embedder.model_name,
        dimension=embedder.dimension,
    )
    return HybridRetriever(store, embedder)


def cases_for_category(cases: list[RegressionBenchmarkCase], category: str) -> list[RegressionBenchmarkCase]:
    return [case for case in cases if case.category == category]


def assert_expected_top_hit(case: RegressionBenchmarkCase, result: RegressionBenchmarkResult) -> None:
    if not case.expected_top_chunk_ids:
        return
    actual_top = result.selected_chunk_id or (result.top_chunk_ids[0] if result.top_chunk_ids else None)
    assert actual_top in case.expected_top_chunk_ids, (
        f"{case.name}: expected top chunk in {case.expected_top_chunk_ids!r}, "
        f"got {actual_top!r} with ranked {result.top_chunk_ids!r}"
    )


def assert_answer_contains_exact_value(case: RegressionBenchmarkCase, result: RegressionBenchmarkResult) -> None:
    for snippet in case.expected_answer_contains:
        assert snippet in result.answer.text, f"{case.name}: missing {snippet!r} in {result.answer.text!r}"


def assert_expected_citations_exist(case: RegressionBenchmarkCase, result: RegressionBenchmarkResult) -> None:
    if not case.require_citations:
        return
    assert result.answer.citations, f"{case.name}: expected citations in answer"
    assert all(citation.page_start >= 1 for citation in result.answer.citations), f"{case.name}: citation pages were incomplete"
    assert all(citation.heading.strip() for citation in result.answer.citations), f"{case.name}: citation headings were incomplete"


def assert_expected_refusal(case: RegressionBenchmarkCase, result: RegressionBenchmarkResult) -> None:
    if case.expect_refusal:
        assert result.answer.refused is True, f"{case.name}: expected refusal, got {result.answer.text!r}"
        assert result.answer.text == "I can't verify that from the contract.", (
            f"{case.name}: refusal text changed to {result.answer.text!r}"
        )
        return
    assert result.answer.refused is False, f"{case.name}: unexpected refusal {result.answer.text!r}"


def assert_expected_section(case: RegressionBenchmarkCase, result: RegressionBenchmarkResult) -> None:
    if case.expected_section is None:
        return
    selected_bundle = getattr(result.trace, "selected_bundle", None) if result.trace is not None else None
    selected_section = None
    if selected_bundle is not None and getattr(selected_bundle, "ranked_chunks", None):
        selected_section = selected_bundle.ranked_chunks[0].section_number
    cited_sections = {citation.section_number or "" for citation in result.answer.citations}
    assert (
        f"Section {case.expected_section}" in result.answer.text
        or selected_section == case.expected_section
        or case.expected_section in cited_sections
    ), f"{case.name}: expected section {case.expected_section!r}, got answer {result.answer.text!r}"


def assert_expected_page_hit(case: RegressionBenchmarkCase, result: RegressionBenchmarkResult) -> None:
    if case.expected_page is None:
        return
    exact_pages = tuple(hit.page_num for hit in result.exact_hits)
    citation_pages = {citation.page_start for citation in result.answer.citations}
    page_text = f"Page {case.expected_page}"
    pages_text = f"Pages: {case.expected_page}"
    assert (
        case.expected_page in exact_pages
        or case.expected_page in citation_pages
        or page_text in result.answer.text
        or pages_text in result.answer.text
    ), f"{case.name}: expected page {case.expected_page!r}, got exact hits {exact_pages!r} and answer {result.answer.text!r}"
    if case.expected_exact_hit_page is not None:
        actual_first = exact_pages[0] if exact_pages else None
        assert actual_first == case.expected_exact_hit_page, (
            f"{case.name}: expected first exact page hit {case.expected_exact_hit_page!r}, got {actual_first!r}"
        )
