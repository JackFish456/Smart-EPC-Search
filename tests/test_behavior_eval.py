from epc_smart_search.answer_policy import AnswerPolicy
from epc_smart_search.behavior_eval import (
    BehaviorCase,
    DisabledGemma,
    behavior_case_from_dict,
    evaluate_behavior_case,
    format_suite_summary,
)
from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
from epc_smart_search.retrieval import HashingEmbedder, HybridRetriever
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore, pack_vector


def test_behavior_case_from_dict_reads_expectations() -> None:
    case = behavior_case_from_dict(
        {
            "name": "air_permits_broad_topic",
            "question": "give me information about air permits",
            "expected_request_shape": "broad_topic",
            "expected_answer_mode": "broad_topic_summary",
            "expected_refused": False,
            "required_snippets": ["air permit"],
            "banned_headings": ["Common Acronyms in Air Permits"],
            "min_citations": 1,
        }
    )

    assert case.name == "air_permits_broad_topic"
    assert case.expected_request_shape == "broad_topic"
    assert case.expected_answer_mode == "broad_topic_summary"
    assert case.expected_refused is False
    assert case.required_snippets == ("air permit",)
    assert case.banned_headings == ("Common Acronyms in Air Permits",)
    assert case.min_citations == 1


def test_evaluate_behavior_case_passes_for_broad_topic_contract_summary() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "acronyms",
                "A.1",
                "Common Acronyms in Air Permits",
                "APD = Air Permits. AMOC = alternate means of control. acfm = actual cubic feet per minute.",
                14,
                chunk_type="definition",
            ),
            _chunk(
                "permit_clause",
                "8.5",
                "Air Permit Compliance",
                "Contractor shall obtain air permits, perform required emissions testing, and demonstrate ongoing compliance with air permit conditions.",
                15,
            ),
            _chunk(
                "testing_clause",
                "8.6",
                "Environmental Testing",
                "Contractor shall perform environmental testing, maintain records, and provide supporting calculations demonstrating compliance with permit limits.",
                16,
            ),
        ]
    )
    policy = AnswerPolicy(retriever.store, retriever)
    case = BehaviorCase(
        name="air_permits_broad_topic",
        question="give me information about air permits",
        expected_request_shape="broad_topic",
        expected_answer_mode="broad_topic_summary",
        expected_refused=False,
        required_snippets=("air permit",),
        banned_headings=("Common Acronyms in Air Permits",),
        min_citations=1,
    )

    result = evaluate_behavior_case(case, policy, retriever, DisabledGemma())

    assert result.passed is True
    assert result.answer_mode == "broad_topic_summary"
    assert result.request_shape == "broad_topic"
    assert "Common Acronyms in Air Permits" not in result.citation_headings


def test_evaluate_behavior_case_reports_refusal_expectation() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "random_schedule",
                "6.9",
                "KV - 480 V",
                "Closed cooling water pump motor 1200 HP. Boiler feedwater pump motor 4750 HP.",
                359,
            )
        ]
    )
    policy = AnswerPolicy(retriever.store, retriever)
    case = BehaviorCase(
        name="fire_water_pump_refusal",
        question="what is the fire water pump horse power",
        expected_request_shape="scalar",
        expected_answer_mode="refusal",
        expected_refused=True,
    )

    result = evaluate_behavior_case(case, policy, retriever, DisabledGemma())

    assert result.passed is True
    assert result.refused is True
    assert "I can't verify that from the contract." in result.answer_text


def test_format_suite_summary_lists_failures() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "permit_clause",
                "8.5",
                "Air Permit Compliance",
                "Contractor shall obtain air permits and maintain compliance with air permit conditions.",
                15,
            )
        ]
    )
    policy = AnswerPolicy(retriever.store, retriever)
    case = BehaviorCase(
        name="wrong_expectation",
        question="give me information about air permits",
        expected_request_shape="scalar",
    )

    result = evaluate_behavior_case(case, policy, retriever, DisabledGemma())
    summary = format_suite_summary(type("Suite", (), {"passed_cases": 0, "total_cases": 1, "case_results": (result,)})())

    assert result.passed is False
    assert "wrong_expectation" in summary
    assert "expected request_shape='scalar'" in summary


def _seed_retriever(chunks: list[ChunkRecord]) -> HybridRetriever:
    db_path = "file:behavior_eval_suite?mode=memory&cache=shared"
    store = ContractStore(db_path)
    embedder = HashingEmbedder(dimension=16)
    pages = [PageText(page_num=chunk.page_start, text=chunk.full_text, ocr_used=False) for chunk in chunks]
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc",
        page_count=len(pages),
        chunks=chunks,
        pages=pages,
        features=build_chunk_features(chunks),
        embeddings={chunk.chunk_id: pack_vector(embedder.embed(chunk.full_text)) for chunk in chunks},
        model_name=embedder.model_name,
        dimension=embedder.dimension,
    )
    return HybridRetriever(store, embedder)


def _chunk(
    chunk_id: str,
    section_number: str,
    heading: str,
    full_text: str,
    page_num: int,
    *,
    chunk_type: str = "section",
    parent_chunk_id: str | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc1",
        chunk_type=chunk_type,
        section_number=section_number,
        heading=heading,
        full_text=full_text,
        page_start=page_num,
        page_end=page_num,
        parent_chunk_id=parent_chunk_id,
        ordinal_in_document=page_num,
    )
