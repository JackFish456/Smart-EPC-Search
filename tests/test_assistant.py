from epc_smart_search.assistant import AssistantAnswer, ContractAssistant
from epc_smart_search.assistant import DEEP_MAX_NEW_TOKENS
from epc_smart_search.assistant import SUMMARY_ENABLE_THINKING
from epc_smart_search.assistant import SUMMARY_MAX_NEW_TOKENS
from epc_smart_search.answer_policy import AnswerPolicy
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


def test_build_extractive_answer_prefers_exact_model_sentence() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk_model",
            section_number="9.2",
            heading="Selected Turbine Generator",
            full_text=(
                "General turbine requirements are listed in this section. "
                "The selected turbine model shall be Siemens SGT6-5000F for the project. "
                "Accessories shall be delivered with the package."
            ),
            page_start=19,
            page_end=19,
            ordinal_in_document=1,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]

    answer = ContractAssistant._build_extractive_answer(
        "What is the turbine we are using?",
        ranked,
        [],
    )

    assert answer is not None
    assert "Siemens SGT6-5000F" in answer.text
    assert "Accessories shall be delivered" not in answer.text


def test_build_extractive_answer_prefers_configuration_sentence() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk_config",
            section_number="7.4.2",
            heading="Fuel Gas Dew Point Configuration",
            full_text=(
                "Dew point monitoring shall be included in the controls package. "
                "The fuel gas dew point configuration shall use a duplex analyzer arrangement with automatic switchover. "
                "Commissioning checks shall verify the analyzer alarms."
            ),
            page_start=41,
            page_end=41,
            ordinal_in_document=1,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]

    answer = ContractAssistant._build_extractive_answer(
        "What is the dew point configuration?",
        ranked,
        [],
    )

    assert answer is not None
    assert "duplex analyzer arrangement with automatic switchover" in answer.text
    assert "Commissioning checks shall verify the analyzer alarms." not in answer.text


def test_build_extractive_answer_keeps_design_condition_values_together() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk_conditions",
            section_number="12.4.1",
            heading="Compressor Design Conditions",
            full_text=(
                "The compressor design conditions shall be 1250 psig discharge pressure and 105 degF inlet temperature. "
                "Normal operating pressure is expected to be lower during startup."
            ),
            page_start=28,
            page_end=28,
            ordinal_in_document=1,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]

    answer = ContractAssistant._build_extractive_answer(
        "What are the compressor design conditions?",
        ranked,
        [],
    )

    assert answer is not None
    assert "1250 psig discharge pressure" in answer.text
    assert "105 degF inlet temperature" in answer.text


def test_build_extractive_answer_prefers_exact_power_value() -> None:
    ranked = [
        RankedChunk(
            chunk_id="fire_water_pump",
            section_number="8.4.2",
            heading="Fire Water Pump",
            full_text=(
                "General pump arrangement details are provided elsewhere. "
                "Each fire water pump shall be rated at 350 HP for the project fire water service. "
                "The driver shall be suitable for emergency operation."
            ),
            page_start=412,
            page_end=412,
            ordinal_in_document=1,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]

    answer = ContractAssistant._build_extractive_answer(
        "what is the fire water pump horse power",
        ranked,
        [],
    )

    assert answer is not None
    assert "350 HP" in answer.text
    assert "driver shall be suitable" not in answer.text.lower()


def test_build_extractive_answer_stops_after_first_high_confidence_exact_block() -> None:
    ranked = [
        RankedChunk(
            chunk_id="steam_turbine",
            section_number="8.2.1",
            heading="Steam Turbine",
            full_text=(
                "Each steam turbine unit is of the model STF-D600 and includes the following equipment: "
                "One combined HP/IP turbine and one double flow LP turbine."
            ),
            page_start=1762,
            page_end=1762,
            ordinal_in_document=1,
            total_score=4.5,
            lexical_score=1.0,
            semantic_score=0.0,
        ),
        RankedChunk(
            chunk_id="manual_trip",
            section_number="5",
            heading="Manual trip - turbine is tripped manually",
            full_text="Manual trip logic for a steam turbine is described in this appendix section.",
            page_start=2716,
            page_end=2716,
            ordinal_in_document=2,
            total_score=2.5,
            lexical_score=0.7,
            semantic_score=0.0,
        ),
    ]

    answer = ContractAssistant._build_extractive_answer(
        "what turbine are we using",
        ranked,
        [],
    )

    assert answer is not None
    assert "STF-D600" in answer.text
    assert "Manual trip" not in answer.text
    assert answer.text.count("Section ") == 1


def test_build_exact_answer_prefers_quantity_value_for_how_many_question() -> None:
    assistant = ContractAssistant.__new__(ContractAssistant)
    hits = [
        ExactPageHit(
            page_num=290,
            snippet="Demineralized water pumps",
            page_text=(
                "The Demineralized Water System includes two (2) x 100% demineralized water pumps. "
                "These pumps deliver water to the storage tank."
            ),
        )
    ]

    answer = assistant._build_exact_answer("how many demineralized water pumps do we have", hits)

    assert answer is not None
    assert "Answer:" in answer.text
    assert "two (2) x 100%" in answer.text.lower()


def test_build_extractive_answer_rejects_system_overview_for_how_many_question_without_quantity() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk_count_fail",
            section_number="5.50",
            heading="Demineralized Water",
            full_text=(
                "The Demineralized Water System provides and stores demineralized water in the storage tank. "
                "Demineralized water is used for makeup and wash services."
            ),
            page_start=290,
            page_end=290,
            ordinal_in_document=1,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]

    answer = ContractAssistant._build_extractive_answer(
        "how many demineralized water pumps do we have",
        ranked,
        [],
    )

    assert answer is None


def test_ask_refuses_wrong_but_related_power_clause() -> None:
    wrong = RankedChunk(
        chunk_id="random_schedule",
        section_number="6.9",
        heading="KV - 480 V",
        full_text="Closed cooling water pump motor 1200 HP. Boiler feedwater pump motor 4750 HP.",
        page_start=359,
        page_end=359,
        ordinal_in_document=1,
        total_score=0.62,
        lexical_score=0.3,
        semantic_score=0.0,
    )
    citations = [Citation("random_schedule", "6.9", "KV - 480 V", None, 359, 359, "quote")]

    class FakeRetriever:
        def find_exact_page_hits(self, question: str):
            return []

        def retrieve(self, question: str, profile: str = "normal"):
            return [wrong]

        def expand_with_context(self, incoming_ranked):
            return citations

    class FakeGemma:
        def ask(self, *args, **kwargs):
            raise AssertionError("Gemma should not be called when the top clause fails grounding.")

    assistant = ContractAssistant.__new__(ContractAssistant)
    assistant.retriever = FakeRetriever()
    assistant.gemma = FakeGemma()
    assistant.answer_policy = AnswerPolicy(None, assistant.retriever)

    answer = assistant.ask("what is the fire water pump horse power")

    assert answer.refused is True
    assert answer.text == "I can't verify that from the contract."


def test_ask_returns_compact_answer_for_strong_power_match() -> None:
    correct = RankedChunk(
        chunk_id="fire_water_pump",
        section_number="8.4.2",
        heading="Fire Water Pump",
        full_text="Each fire water pump shall be rated at 350 HP for the project fire water service.",
        page_start=412,
        page_end=412,
        ordinal_in_document=1,
        total_score=1.2,
        lexical_score=0.7,
        semantic_score=0.0,
    )
    citations = [Citation("fire_water_pump", "8.4.2", "Fire Water Pump", None, 412, 412, "quote")]

    class FakeRetriever:
        def find_exact_page_hits(self, question: str):
            return []

        def retrieve(self, question: str, profile: str = "normal"):
            return [correct]

        def expand_with_context(self, incoming_ranked):
            return citations

    class FakeGemma:
        def ask(self, *args, **kwargs):
            raise AssertionError("Gemma should not be called for a strong compact answer.")

    assistant = ContractAssistant.__new__(ContractAssistant)
    assistant.retriever = FakeRetriever()
    assistant.gemma = FakeGemma()
    assistant.answer_policy = AnswerPolicy(None, assistant.retriever)

    answer = assistant.ask("what is the fire water pump horse power")

    assert answer.refused is False
    assert "350 HP" in answer.text
    assert answer.text.count("Section ") == 1


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


def test_build_compact_answer_returns_exact_model_line_for_type_question() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="9.2",
            heading="Selected Turbine Generator",
            full_text="The selected turbine model shall be Siemens SGT6-5000F for the project.",
            page_start=19,
            page_end=19,
            ordinal_in_document=1,
            total_score=4.2,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]

    answer = AnswerPolicy.build_compact_answer(
        "What is the turbine we are using?",
        ranked,
        [],
    )

    assert answer is not None
    assert '"The selected turbine model shall be Siemens SGT6-5000F for the project."' in answer.text
    assert "Section 9.2 - Selected Turbine Generator" in answer.text
    assert "Pages: 19" in answer.text


def test_build_compact_answer_returns_exact_function_line_for_how_it_works_question() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="5.2",
            heading="Fuel Gas System Description",
            full_text=(
                "The fuel gas system receives natural gas from the pipeline, conditions the gas, "
                "and distributes it to the combustion turbines. Contractor shall install the skids on the foundations."
            ),
            page_start=21,
            page_end=21,
            ordinal_in_document=1,
            total_score=4.0,
            lexical_score=1.0,
            semantic_score=0.0,
        )
    ]

    answer = AnswerPolicy.build_compact_answer(
        "How does the fuel gas system work?",
        ranked,
        [],
    )

    assert answer is not None
    assert '"The fuel gas system receives natural gas from the pipeline, conditions the gas, and distributes it to the combustion turbines."' in answer.text
    assert "Section 5.2 - Fuel Gas System Description" in answer.text


def test_system_attribute_question_refuses_when_best_match_is_blurry() -> None:
    ranked = [
        RankedChunk(
            chunk_id="chunk1",
            section_number="3.1",
            heading="Dew Point",
            full_text="Dew point testing shall be completed during commissioning.",
            page_start=10,
            page_end=10,
            ordinal_in_document=1,
            total_score=0.8,
            lexical_score=0.3,
            semantic_score=0.0,
        )
    ]
    citations = [Citation("chunk1", "3.1", "Dew Point", None, 10, 10, "quote")]

    class FakeRetriever:
        def find_exact_page_hits(self, question: str):
            return []

        def retrieve(self, question: str):
            return ranked

        def expand_with_context(self, incoming_ranked):
            return citations

    class FakeGemma:
        def ask(self, *args, **kwargs):
            raise AssertionError("Gemma should not be called for a blurry exact query")

    assistant = ContractAssistant.__new__(ContractAssistant)
    assistant.retriever = FakeRetriever()
    assistant.gemma = FakeGemma()

    answer = assistant.ask("What is the dew point configuration?")

    assert answer.refused
    assert answer.text == "I can't verify that from the contract."


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
