from __future__ import annotations

from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
from epc_smart_search.retrieval import HybridRetriever, SearchCoverageCase
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore


GOLDEN_CASES = (
    {
        "label": "exact_section_lookup",
        "query": "section 14.2.1",
        "query_class": "section_lookup",
        "expected_top1": "due_org",
    },
    {
        "label": "definition_lookup",
        "query": "What is substantial completion?",
        "query_class": "definition",
        "expected_top1": "definition_sc",
    },
    {
        "label": "responsibility_lookup",
        "query": "Who is responsible for permitting?",
        "query_class": "responsibility",
        "expected_top1": "permit_clause",
    },
    {
        "label": "payment_liability_lookup",
        "query": "Who pays if the project finishes late?",
        "query_class": "payment_liability",
        "expected_top1": "delay_payment",
    },
    {
        "label": "termination_lookup",
        "query": "Can the owner end the contract for convenience?",
        "query_class": "termination",
        "expected_top1": "termination",
    },
    {
        "label": "delay_schedule_lookup",
        "query": "What happens if weather delays the work?",
        "query_class": "schedule_delay",
        "expected_top1": "weather",
    },
    {
        "label": "heading_lookup",
        "query": "What does the contract say about electric motors?",
        "query_class": "heading_lookup",
        "expected_top1": "motors",
    },
    {
        "label": "warranty_lookup",
        "query": "What is the warranty period?",
        "query_class": "heading_lookup",
        "expected_top3": "warranty",
    },
    {
        "label": "appendix_lookup",
        "query": "Show me appendix A fuel gas summary",
        "query_class": "heading_lookup",
        "expected_top3": "appendix_a",
    },
    {
        "label": "numeric_fuel_gas_value",
        "query": "45 MMSCFD",
        "query_class": "numeric_lookup",
        "expected_top3": "appendix_a",
    },
    {
        "label": "typo_permitting",
        "query": "permiting",
        "query_class": "typo_fuzzy",
        "expected_top3": "permit_clause",
    },
    {
        "label": "typo_termination",
        "query": "terminat for convenience",
        "query_class": "typo_fuzzy",
        "expected_top3": "termination",
    },
    {
        "label": "typo_substantial_completion",
        "query": "substantive completion",
        "query_class": "typo_fuzzy",
        "expected_top3": "definition_sc",
    },
    {
        "label": "typo_motors",
        "query": "motorss",
        "query_class": "typo_fuzzy",
        "expected_top3": "motors",
    },
)


def test_frozen_golden_set_relevance() -> None:
    retriever = _seed_retriever()

    for case in GOLDEN_CASES:
        ranked = retriever.retrieve(case["query"])
        assert ranked, case["label"]
        top_ids = [row.chunk_id for row in ranked[:3]]

        expected_top1 = case.get("expected_top1")
        if expected_top1 is not None:
            assert ranked[0].chunk_id == expected_top1, case

        expected_top3 = case.get("expected_top3")
        if expected_top3 is not None:
            assert expected_top3 in top_ids, case


def test_typo_rescue_precision_does_not_promote_known_wrong_clauses() -> None:
    retriever = _seed_retriever()

    ranked = retriever.retrieve("permiting")
    assert ranked[0].chunk_id != "loto_clause"

    ranked = retriever.retrieve("terminat for convenience")
    assert ranked[0].chunk_id != "convenience_receptacles"

    ranked = retriever.retrieve("substantive completion")
    assert ranked[0].chunk_id != "schedule_clause"


def test_coverage_case_evaluation_tracks_stage_outcomes() -> None:
    retriever = _seed_retriever()

    results = retriever.evaluate_coverage_cases(
        [
            SearchCoverageCase(
                label="permiting",
                query="permiting",
                expected_chunk_id="permit_clause",
            )
        ]
    )

    assert results[0].found is True
    assert results[0].retrieval_stage in {"chunk_fts", "block_fts", "trigram_rescue"}


def _seed_retriever() -> HybridRetriever:
    db_path = "file:frozen_golden_set?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunks = [
        _chunk(
            "due_org",
            "14.2.1",
            "Due Organization of Owner",
            "Owner is a limited liability company duly organized and validly existing under applicable law.",
            12,
        ),
        _chunk(
            "definition_sc",
            "1.1",
            "Substantial Completion",
            '"Substantial Completion" means the stage at which the Work is capable of commercial operation.',
            2,
            chunk_type="definition",
        ),
        _chunk(
            "permit_clause",
            "4.2",
            "Cooperation with Permitting Authorities",
            "Contractor shall obtain and maintain all permits and approvals required for the Work.",
            4,
        ),
        _chunk(
            "loto_clause",
            "6.13",
            "Independent LOTO Authority",
            "A second LOTO Authority is responsible for reviewing permit paperwork during lockout procedures.",
            6,
        ),
        _chunk(
            "delay_payment",
            "6.2.4.4",
            "Payment of Late Substantial Completion Payments",
            "Contractor shall pay liquidated damages if Substantial Completion is achieved late.",
            12,
        ),
        _chunk(
            "schedule_clause",
            "6.2.4.1",
            "Anticipated Substantial Completion Date",
            "The anticipated substantial completion date must be updated monthly in the project schedule.",
            11,
        ),
        _chunk(
            "termination",
            "12.1",
            "Owner Termination for Convenience",
            "Owner may terminate this Contract for convenience at any time upon written notice.",
            50,
        ),
        _chunk(
            "convenience_receptacles",
            "4.4.8.10",
            "Convenience Receptacles",
            "Convenience receptacles shall be installed in temporary facilities.",
            88,
        ),
        _chunk(
            "weather",
            "9.9",
            "Severe Weather Conditions",
            "Severe weather may entitle Contractor to schedule relief for delay to the Work.",
            70,
        ),
        _chunk(
            "motors",
            "4.4.3",
            "Electric Motors",
            "Electric motors must meet the design requirements for the project.",
            12,
        ),
        _chunk(
            "warranty",
            "8.2",
            "Warranty Period",
            "The warranty period begins on the date of substantial completion for the applicable unit.",
            78,
        ),
        _chunk(
            "appendix_a",
            "A",
            "Appendix A Fuel Gas Summary",
            "Appendix A provides the fuel gas system summary and interface data for the project. The summary flow is 45 MMSCFD.",
            205,
            chunk_type="exhibit",
        ),
    ]
    pages_by_number = {
        chunk.page_start: PageText(page_num=chunk.page_start, text=chunk.full_text, ocr_used=False)
        for chunk in chunks
    }
    pages = [pages_by_number[page_num] for page_num in sorted(pages_by_number)]
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
    )
    return HybridRetriever(store)


def _chunk(
    chunk_id: str,
    section_number: str,
    heading: str,
    full_text: str,
    page_num: int,
    *,
    chunk_type: str = "section",
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
        parent_chunk_id=None,
        ordinal_in_document=page_num,
    )
