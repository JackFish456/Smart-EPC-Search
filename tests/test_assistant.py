from epc_smart_search.assistant import AssistantAnswer, ContractAssistant
from epc_smart_search.retrieval import Citation
from epc_smart_search.retrieval import RankedChunk


def test_limit_citations_prefers_distinct_pages() -> None:
    citations = [
        Citation("chunk1", "5.23", "Fuel Gas Supply", None, 270, 271, "quote"),
        Citation("chunk2", "5.23", "Fuel Gas Supply", None, 271, 271, "quote"),
        Citation("chunk3", "11", "Permit", None, 3131, 3131, "quote"),
    ]

    limited = ContractAssistant._limit_citations(citations, limit=2)

    assert [(citation.page_start, citation.page_end) for citation in limited] == [(270, 271), (3131, 3131)]


def test_build_extractive_answer_returns_direct_contract_text() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="5.23",
            heading="Fuel Gas Supply",
            full_text=(
                "Contractor shall provide the fuel gas supply equipment for the Work. "
                "Fuel gas pressure must be maintained at 450 psi during testing."
            ),
            page_start=270,
            page_end=271,
            ordinal_in_document=1,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]
    citations = [Citation("chunk1", "5.23", "Fuel Gas Supply", None, 270, 271, "quote")]

    answer = ContractAssistant._build_extractive_answer(
        "What does the contract say about fuel gas supply?",
        ranked,
        citations,
    )

    assert isinstance(answer, AssistantAnswer)
    assert answer is not None
    assert "Most relevant contract text:" in answer.text
    assert "Section 5.23 - Fuel Gas Supply" in answer.text
    assert 'Contract text: "Contractor shall provide the fuel gas supply equipment for the Work.' in answer.text
    assert answer.citations == citations
    assert not answer.refused


def test_build_extractive_answer_keeps_numeric_requirement_language() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk2",
            section_number="7.5.3",
            heading="Air Permit Test Result",
            full_text=(
                "Within twenty-four (24) hours after completion of a test, Contractor shall deliver the results. "
                "The submission must include all emissions values and supporting calculations."
            ),
            page_start=83,
            page_end=83,
            ordinal_in_document=1,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]

    answer = ContractAssistant._build_extractive_answer(
        "What are the air permit testing requirements?",
        ranked,
        [],
    )

    assert answer is not None
    assert "twenty-four (24) hours" in answer.text
    assert "shall deliver the results" in answer.text


def test_prefers_generated_answer_only_for_summary_style_prompts() -> None:
    assert not ContractAssistant._prefers_generated_answer("What does the contract say about fuel gas supply?")
    assert ContractAssistant._prefers_generated_answer("Summarize the fuel gas supply requirements in plain English.")
    assert ContractAssistant._prefers_generated_answer("Can you explain the fuel gas system?")
    assert not ContractAssistant._prefers_generated_answer("Give me an overview of the fuel gas system.")
    assert not ContractAssistant._prefers_generated_answer("Put the fuel gas system in plain English.")


def test_build_extractive_answer_skips_thin_toc_style_blocks() -> None:
    ranked = [
        RankedChunk(
            chunk_id="toc",
            section_number="5.22",
            heading="Fuel Gas Supply ...................................................................................................................... 5-91",
            full_text="5.22",
            page_start=146,
            page_end=146,
            ordinal_in_document=1,
            total_score=5.0,
            lexical_score=1.0,
            semantic_score=0.0,
        ),
        RankedChunk(
            chunk_id="real",
            section_number="5.23",
            heading="Fuel Gas Supply",
            full_text="The Fuel Gas System receives, conditions and transports natural gas supplied from an offsite pipeline.",
            page_start=270,
            page_end=271,
            ordinal_in_document=2,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        ),
    ]

    answer = ContractAssistant._build_extractive_answer(
        "What does the contract say about fuel gas supply?",
        ranked,
        [],
    )

    assert answer is not None
    assert "Section 5.23 - Fuel Gas Supply" in answer.text
    assert "Section 5.22" not in answer.text


def test_build_extractive_answer_windows_to_matching_term_in_long_standards_list() -> None:
    ranked = [
        RankedChunk(
            chunk_id="codes",
            section_number="29",
            heading="CFR",
            full_text=(
                "29 CFR Part 1900 Series Occupational Safety and Health Administration (OSHA) 2018 "
                "AIHA Z10 American Industrial Hygiene Association 2012 ICC - IFC International Fire Code 2018 "
                "NFPA 101 National Fire Protection Association - Life Safety Code 2018 "
                "NFPA 70 - NEC National Fire Protection Association - National Electrical Code 2023"
            ),
            page_start=1424,
            page_end=1424,
            ordinal_in_document=1,
            total_score=3.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]

    answer = ContractAssistant._build_extractive_answer(
        "Does the contract mention NFPA?",
        ranked,
        [],
    )

    assert answer is not None
    assert "NFPA 101 National Fire Protection Association" in answer.text


def test_build_extractive_answer_skips_definition_only_text_for_requirement_questions() -> None:
    ranked = [
        RankedChunk(
            chunk_id="definition",
            section_number="2",
            heading="Air Permit Test",
            full_text="Air Permit Test means those tests identified in the Project's air permit and provided in the testing procedures.",
            page_start=9,
            page_end=9,
            ordinal_in_document=1,
            total_score=5.0,
            lexical_score=1.0,
            semantic_score=0.0,
        ),
        RankedChunk(
            chunk_id="requirement",
            section_number="7.5.1",
            heading="Air Permit Tests",
            full_text=(
                "Contractor shall perform each of the Air Permit Tests for such Unit in accordance with the air permit "
                "and the Detailed Testing Procedures."
            ),
            page_start=83,
            page_end=83,
            ordinal_in_document=2,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        ),
    ]

    answer = ContractAssistant._build_extractive_answer(
        "What are the air permit testing requirements?",
        ranked,
        [],
    )

    assert answer is not None
    assert "Section 7.5.1 - Air Permit Tests" in answer.text
    assert "Section 2 - Air Permit Test" not in answer.text
