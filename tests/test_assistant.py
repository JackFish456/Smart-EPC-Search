from epc_smart_search.assistant import AssistantAnswer, ContractAssistant
from epc_smart_search.assistant import DEEP_MAX_NEW_TOKENS
from epc_smart_search.assistant import SUMMARY_ENABLE_THINKING
from epc_smart_search.assistant import SUMMARY_MAX_NEW_TOKENS
from epc_smart_search.retrieval import Citation, ExactPageHit
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
    assert "Most relevant contract text:" not in answer.text
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


def test_build_summary_prompt_context_uses_ranked_chunks_and_filters_noise() -> None:
    ranked = [
        RankedChunk(
            chunk_id="toc",
            section_number="5.22",
            heading="Fuel Gas Supply ...................................................................................................................... 5-91",
            full_text="5.22",
            page_start=146,
            page_end=146,
            ordinal_in_document=1,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        ),
        RankedChunk(
            chunk_id="main",
            section_number="5.23",
            heading="Fuel Gas Supply",
            full_text=(
                "The Fuel Gas System receives, conditions and transports natural gas supplied from an offsite pipeline. "
                "Contractor shall supply and install the main fuel gas regulating station."
            ),
            page_start=270,
            page_end=271,
            ordinal_in_document=2,
            total_score=3.0,
            lexical_score=1.0,
            semantic_score=0.0,
        ),
        RankedChunk(
            chunk_id="requirements",
            section_number="13.2.2.1",
            heading="Customer Gas Fuel System Supply Requirements",
            full_text=(
                "The allowable minimum and maximum fuel gas supply pressures are referenced to the inlet of the gas module. "
                "The plant designer shall account for a pressure drop of 51 psi."
            ),
            page_start=1955,
            page_end=1955,
            ordinal_in_document=3,
            total_score=2.8,
            lexical_score=1.0,
            semantic_score=0.0,
        ),
    ]

    context = ContractAssistant._build_summary_prompt_context("summarize the fuel gas system", ranked)

    assert "Heading: Fuel Gas Supply" in context
    assert "Heading: Customer Gas Fuel System Supply Requirements" in context
    assert "5.22" not in context


def test_summary_requests_use_reasoning_and_fall_back_to_extractive() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="5.23",
            heading="Fuel Gas Supply",
            full_text=(
                "The Fuel Gas System receives, conditions and transports natural gas supplied from an offsite pipeline. "
                "Contractor shall supply and install the main fuel gas regulating station."
            ),
            page_start=270,
            page_end=271,
            ordinal_in_document=1,
            total_score=3.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]
    citations = [Citation("chunk1", "5.23", "Fuel Gas Supply", None, 270, 271, "quote")]

    class FakeRetriever:
        def find_exact_page_hits(self, question: str):
            return []

        def retrieve(self, question: str):
            return ranked

        def expand_with_context(self, incoming_ranked):
            return citations

    class FakeGemma:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def ask(self, question: str, context: str, *, enable_thinking=None, max_new_tokens=None, response_style=None):
            self.calls.append(
                {
                    "question": question,
                    "context": context,
                    "enable_thinking": enable_thinking,
                    "max_new_tokens": max_new_tokens,
                    "response_style": response_style,
                }
            )
            return "I can't verify that from the contract."

    assistant = ContractAssistant.__new__(ContractAssistant)
    assistant.retriever = FakeRetriever()
    assistant.gemma = FakeGemma()

    answer = assistant.ask("summarize the fuel gas system")

    assert "Most relevant contract text:" not in answer.text
    assert assistant.gemma.calls[0]["enable_thinking"] is SUMMARY_ENABLE_THINKING
    assert assistant.gemma.calls[0]["max_new_tokens"] == SUMMARY_MAX_NEW_TOKENS
    assert assistant.gemma.calls[0]["response_style"] == "detailed_summary"


def test_summary_request_does_not_change_following_normal_request_behavior() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="5.23",
            heading="Fuel Gas Supply",
            full_text=(
                "The Fuel Gas System receives, conditions and transports natural gas supplied from an offsite pipeline. "
                "Contractor shall supply and install the main fuel gas regulating station."
            ),
            page_start=270,
            page_end=271,
            ordinal_in_document=1,
            total_score=3.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]
    citations = [Citation("chunk1", "5.23", "Fuel Gas Supply", None, 270, 271, "quote")]

    class FakeRetriever:
        def find_exact_page_hits(self, question: str):
            return []

        def retrieve(self, question: str):
            return ranked

        def expand_with_context(self, incoming_ranked):
            return citations

    class FakeGemma:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def ask(self, question: str, context: str, *, enable_thinking=None, max_new_tokens=None, response_style=None):
            self.calls.append(
                {
                    "question": question,
                    "enable_thinking": enable_thinking,
                    "max_new_tokens": max_new_tokens,
                    "response_style": response_style,
                }
            )
            return "Grounded summary."

    assistant = ContractAssistant.__new__(ContractAssistant)
    assistant.retriever = FakeRetriever()
    assistant.gemma = FakeGemma()

    summary_answer = assistant.ask("explain the fuel gas system")
    normal_answer = assistant.ask("what does the contract say about fuel gas supply?")

    assert summary_answer.text == "Grounded summary."
    assert "Most relevant contract text:" not in normal_answer.text
    assert len(assistant.gemma.calls) == 1
    assert assistant.gemma.calls[0]["response_style"] == "detailed_summary"


def test_deep_think_routes_non_summary_questions_through_gemma() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="5.23",
            heading="Fuel Gas Supply",
            full_text=(
                "The Fuel Gas System receives, conditions and transports natural gas supplied from an offsite pipeline. "
                "Contractor shall supply and install the main fuel gas regulating station."
            ),
            page_start=270,
            page_end=271,
            ordinal_in_document=1,
            total_score=3.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]
    citations = [Citation("chunk1", "5.23", "Fuel Gas Supply", None, 270, 271, "quote")]
    exact_hits = [ExactPageHit(page_num=270, snippet="fuel gas", page_text=ranked[0].full_text)]

    class FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def find_exact_page_hits(self, question: str):
            self.calls.append({"kind": "exact", "question": question})
            return exact_hits

        def retrieve(self, question: str, profile: str = "normal"):
            self.calls.append({"kind": "retrieve", "question": question, "profile": profile})
            return ranked

        def expand_with_context(self, incoming_ranked):
            return citations

        def resolve_document_id(self) -> str:
            return "doc1"

    class FakeStore:
        def fetch_page_window(self, document_id: str, page_start: int, page_end: int, *, padding: int = 0, limit: int = 2):
            return [{"page_num": 270, "page_text": ranked[0].full_text}]

    class FakeGemma:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def ask(self, question: str, context: str, *, enable_thinking=None, max_new_tokens=None, response_style=None):
            self.calls.append(
                {
                    "question": question,
                    "context": context,
                    "enable_thinking": enable_thinking,
                    "max_new_tokens": max_new_tokens,
                    "response_style": response_style,
                }
            )
            return "## Answer\n- **Contractor** supplies the main regulating station."

    assistant = ContractAssistant.__new__(ContractAssistant)
    assistant.store = FakeStore()
    assistant.retriever = FakeRetriever()
    assistant.gemma = FakeGemma()

    answer = assistant.ask("What does the contract say about fuel gas supply?", deep_think=True)

    assert answer.text.startswith("## Answer")
    assert assistant.retriever.calls[1]["profile"] == "deep"
    assert assistant.gemma.calls[0]["enable_thinking"] is True
    assert assistant.gemma.calls[0]["max_new_tokens"] == DEEP_MAX_NEW_TOKENS
    assert assistant.gemma.calls[0]["response_style"] == "deep_answer"
    assert "Ranked contract sections:" in assistant.gemma.calls[0]["context"]
    assert "Exact page hits:" in assistant.gemma.calls[0]["context"]


def test_deep_think_falls_back_to_extractive_when_gemma_refuses() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="5.23",
            heading="Fuel Gas Supply",
            full_text=(
                "The Fuel Gas System receives, conditions and transports natural gas supplied from an offsite pipeline. "
                "Contractor shall supply and install the main fuel gas regulating station."
            ),
            page_start=270,
            page_end=271,
            ordinal_in_document=1,
            total_score=3.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]
    citations = [Citation("chunk1", "5.23", "Fuel Gas Supply", None, 270, 271, "quote")]

    class FakeRetriever:
        def find_exact_page_hits(self, question: str):
            return []

        def retrieve(self, question: str, profile: str = "normal"):
            return ranked

        def expand_with_context(self, incoming_ranked):
            return citations

        def resolve_document_id(self) -> str:
            return "doc1"

    class FakeStore:
        def fetch_page_window(self, document_id: str, page_start: int, page_end: int, *, padding: int = 0, limit: int = 2):
            return [{"page_num": 270, "page_text": ranked[0].full_text}]

    class FakeGemma:
        def __init__(self) -> None:
            self.calls = 0

        def ask(self, question: str, context: str, *, enable_thinking=None, max_new_tokens=None, response_style=None):
            self.calls += 1
            return "I can't verify that from the contract."

    assistant = ContractAssistant.__new__(ContractAssistant)
    assistant.store = FakeStore()
    assistant.retriever = FakeRetriever()
    assistant.gemma = FakeGemma()

    answer = assistant.ask("What does the contract say about fuel gas supply?", deep_think=True)

    assert assistant.gemma.calls == 1
    assert "Section 5.23 - Fuel Gas Supply" in answer.text
    assert not answer.refused


def test_follow_up_after_summary_reuses_last_user_topic_for_retrieval() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="5.23",
            heading="Fuel Gas Supply",
            full_text=(
                "The Fuel Gas System receives, conditions and transports natural gas supplied from an offsite pipeline. "
                "Contractor shall supply and install the main fuel gas regulating station."
            ),
            page_start=270,
            page_end=271,
            ordinal_in_document=1,
            total_score=3.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]
    citations = [Citation("chunk1", "5.23", "Fuel Gas Supply", None, 270, 271, "quote")]

    class FakeRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def find_exact_page_hits(self, question: str):
            self.queries.append(question)
            return []

        def retrieve(self, question: str):
            self.queries.append(question)
            return ranked

        def expand_with_context(self, incoming_ranked):
            return citations

    class FakeGemma:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def ask(self, question: str, context: str, *, enable_thinking=None, max_new_tokens=None, response_style=None):
            self.calls.append({"question": question, "context": context, "response_style": response_style})
            return "Grounded summary."

    assistant = ContractAssistant.__new__(ContractAssistant)
    assistant.retriever = FakeRetriever()
    assistant.gemma = FakeGemma()

    answer = assistant.ask(
        "what section is that in?",
        history=[
            {"role": "user", "content": "summarize the fuel gas system"},
            {"role": "assistant", "content": "Grounded summary."},
        ],
    )

    assert any("fuel gas system" in query.lower() for query in assistant.retriever.queries)
    assert "Section 5.23 - Fuel Gas Supply" in answer.text
    assert assistant.gemma.calls == []


def test_expand_answer_routes_through_gemma_with_previous_answer() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="5.23",
            heading="Fuel Gas Supply",
            full_text=(
                "The Fuel Gas System receives, conditions and transports natural gas supplied from an offsite pipeline. "
                "Contractor shall supply and install the main fuel gas regulating station."
            ),
            page_start=270,
            page_end=271,
            ordinal_in_document=1,
            total_score=3.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]
    citations = [Citation("chunk1", "5.23", "Fuel Gas Supply", None, 270, 271, "quote")]
    exact_hits = [ExactPageHit(page_num=270, snippet="fuel gas", page_text=ranked[0].full_text)]

    class FakeRetriever:
        def find_exact_page_hits(self, question: str):
            return exact_hits

        def retrieve(self, question: str):
            return ranked

        def expand_with_context(self, incoming_ranked):
            return citations

    class FakeGemma:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def ask(
            self,
            question: str,
            context: str,
            *,
            enable_thinking=None,
            max_new_tokens=None,
            response_style=None,
            previous_answer=None,
        ):
            self.calls.append(
                {
                    "question": question,
                    "context": context,
                    "enable_thinking": enable_thinking,
                    "max_new_tokens": max_new_tokens,
                    "response_style": response_style,
                    "previous_answer": previous_answer,
                }
            )
            return "Expanded answer."

    assistant = ContractAssistant.__new__(ContractAssistant)
    assistant.retriever = FakeRetriever()
    assistant.gemma = FakeGemma()

    answer = assistant.ask(
        "show me the fuel gas supply clause",
        expand_answer=True,
        previous_answer="Short grounded answer.",
    )

    assert answer.text == "Expanded answer."
    assert assistant.gemma.calls[0]["response_style"] == "expand_answer"
    assert assistant.gemma.calls[0]["previous_answer"] == "Short grounded answer."
    assert "Previous answer shown to the user:" in assistant.gemma.calls[0]["context"]
