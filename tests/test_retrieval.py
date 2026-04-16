from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
from epc_smart_search.query_planner import build_like_fallback, plan_query
from epc_smart_search.retrieval import HashingEmbedder, HybridRetriever
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore, pack_vector


def test_direct_section_lookup_wins() -> None:
    db_path = "file:retrieval_contract?mode=memory&cache=shared"
    store = ContractStore(db_path)
    embedder = HashingEmbedder(dimension=16)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="subsection",
        section_number="14.2.1",
        heading="Liquidated Damages",
        full_text="This clause covers liquidated damages.",
        page_start=12,
        page_end=12,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc",
        page_count=1,
        chunks=[chunk],
        pages=[PageText(page_num=12, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
        embeddings={"chunk1": pack_vector(embedder.embed(chunk.full_text))},
        model_name=embedder.model_name,
        dimension=embedder.dimension,
    )
    retriever = HybridRetriever(store, embedder)
    ranked = retriever.retrieve("section 14.2.1")
    assert ranked
    assert ranked[0].section_number == "14.2.1"


def test_context_clues_prefer_permitting_responsibility_clause() -> None:
    retriever = _seed_retriever(
        [
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
        ]
    )
    ranked = retriever.retrieve("Who is responsible for permitting?")
    assert ranked
    assert ranked[0].chunk_id == "permit_clause"


def test_context_clues_prefer_late_completion_payment_clause() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "delay_payment",
                "6.2.4.4",
                "Payment of Late Substantial Completion Payments",
                "Contractor shall pay liquidated damages if Substantial Completion is achieved late.",
                12,
            ),
            _chunk(
                "documents",
                "18.6.3.1",
                "Documents",
                "Contractor shall submit documents to support progress payments.",
                40,
            ),
        ]
    )
    ranked = retriever.retrieve("Who pays if the project finishes late?")
    assert ranked
    assert ranked[0].chunk_id == "delay_payment"


def test_context_clues_prefer_owner_termination_for_convenience() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "termination",
                "12.1",
                "Owner Termination for Convenience",
                "Owner may terminate this Contract for convenience at any time upon written notice.",
                50,
            ),
            _chunk(
                "receptacles",
                "4.4.8.10",
                "Convenience Receptacles",
                "Convenience receptacles shall be installed in temporary facilities.",
                88,
            ),
        ]
    )
    ranked = retriever.retrieve("Can the owner end the contract for convenience?")
    assert ranked
    assert ranked[0].chunk_id == "termination"


def test_context_clues_prefer_weather_delay_clause() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "weather",
                "9.9",
                "Severe Weather Conditions",
                "Severe weather may entitle Contractor to schedule relief for delay to the Work.",
                70,
            ),
            _chunk(
                "suspension",
                "9.3.5.1",
                "Delays Due to Suspension of Work",
                "If Owner suspends the Work, Contractor may seek time relief under this Section.",
                65,
            ),
        ]
    )
    ranked = retriever.retrieve("What happens if weather delays the work?")
    assert ranked
    assert ranked[0].chunk_id == "weather"


def test_definition_questions_prefer_definition_chunks() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "def_sc",
                "1.1",
                "Substantial Completion",
                '"Substantial Completion" means the stage at which the Work is capable of commercial operation.',
                2,
                chunk_type="definition",
            ),
            _chunk(
                "misc",
                "8.1",
                "Schedule",
                "Contractor shall update the construction schedule monthly.",
                22,
            ),
        ]
    )
    ranked = retriever.retrieve("What is substantial completion?")
    assert ranked
    assert ranked[0].chunk_id == "def_sc"


def test_direct_text_page_hits_find_exact_phrase() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "permits",
                "4.2",
                "Cooperation with Permitting Authorities",
                "Contractor shall obtain all permits and approvals required for the Work.",
                5,
            ),
        ]
    )
    hits = retriever.find_exact_page_hits("show me permits and approvals")
    assert hits
    assert hits[0].page_num == 5


def test_contract_say_about_queries_strip_boilerplate_for_focus_terms() -> None:
    plan = plan_query("What does the contract say about electric motors?")

    assert plan.intent == "direct_text"
    assert plan.content_query == "electric motors"
    assert plan.focus_terms == ("electric", "motors")
    assert build_like_fallback(plan) == "electric motors"


def test_query_plan_extracts_system_and_attribute_for_configuration_question() -> None:
    plan = plan_query("What is the dew point configuration?")

    assert plan.system_phrase == "dew point"
    assert plan.system_terms == ("dew", "point")
    assert plan.attribute_label == "configuration"
    assert "configuration" in plan.attribute_terms


def test_direct_text_phrasing_prefers_equipment_heading_over_generic_match() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "generic_qna",
                "4",
                "Q: What insulation or trace heating is provided on pipework to ensure water does not freeze",
                "A generic appendix answer about tracing and water tubing.",
                99,
            ),
            _chunk(
                "motors",
                "4.4.3",
                "Electric Motors",
                "Electric motors must meet the design requirements for the project.",
                12,
            ),
        ]
    )

    ranked = retriever.retrieve("What does the contract say about electric motors?")

    assert ranked
    assert ranked[0].chunk_id == "motors"


def test_hierarchical_search_prefers_exact_system_and_configuration_clause() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "generic_dew",
                "3.1",
                "Dew Point",
                "Dew point testing shall be completed during commissioning.",
                10,
            ),
            _chunk(
                "config_clause",
                "7.4.2",
                "Fuel Gas Dew Point Configuration",
                "The fuel gas dew point configuration shall use a duplex analyzer arrangement with automatic switchover.",
                41,
            ),
        ]
    )

    ranked = retriever.retrieve("What is the dew point configuration?")

    assert ranked
    assert ranked[0].chunk_id == "config_clause"


def test_hierarchical_search_prefers_exact_model_clause_over_generic_equipment_text() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "generic_turbine",
                "9.1",
                "Steam Turbine",
                "Turbine components shall comply with the applicable standards for rotating equipment.",
                18,
            ),
            _chunk(
                "selected_turbine",
                "9.2",
                "Selected Turbine Generator",
                "The selected turbine model shall be Siemens SGT6-5000F for the project.",
                19,
            ),
        ]
    )

    ranked = retriever.retrieve("What is the turbine we are using?")

    assert ranked
    assert ranked[0].chunk_id == "selected_turbine"


def test_hierarchical_search_prefers_exact_design_conditions_clause() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "generic_design",
                "12.1",
                "General Design Philosophy",
                "Design reviews and operating conditions shall be coordinated with the project team.",
                27,
            ),
            _chunk(
                "compressor_conditions",
                "12.4.1",
                "Compressor Design Conditions",
                "The compressor design conditions shall be 1250 psig discharge pressure and 105 degF inlet temperature.",
                28,
            ),
        ]
    )

    ranked = retriever.retrieve("What are the compressor design conditions?")

    assert ranked
    assert ranked[0].chunk_id == "compressor_conditions"


def test_deep_profile_prefers_specific_clause_over_generic_match() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "generic_qna",
                "4",
                "Q: What insulation or trace heating is provided on pipework to ensure water does not freeze",
                "A generic appendix answer about tracing and water tubing.",
                99,
            ),
            _chunk(
                "motors",
                "4.4.3",
                "Electric Motors",
                "Electric motors must meet the design requirements for the project.",
                12,
            ),
        ]
    )

    ranked = retriever.retrieve("What does the contract say about electric motors?", profile="deep")

    assert ranked
    assert ranked[0].chunk_id == "motors"


def test_deep_profile_keeps_direct_evidence_queries_stable() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "schedule_line",
                "0",
                "30-Jun-28",
                "GEV - ST #1 Steam Bypass Valves - Delivered to Site",
                132,
            ),
            _chunk(
                "steam_valves",
                "11.1",
                "Steam Turbine Bypass Valves, Dump devices",
                "Steam turbine bypass valves include one set of HP bypass valve per HRSG.",
                1952,
            ),
        ]
    )

    ranked = retriever.retrieve("What does the contract say about steam bypass valves?", profile="deep")

    assert ranked
    assert ranked[0].chunk_id == "steam_valves"


def test_focus_terms_beat_generic_responsibility_language_for_equipment_requests() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "qa_program",
                "2.2.23",
                "Quality Assurance Program",
                "Contractor has sole responsibility for the quality assurance program and shall provide all required documentation.",
                43,
            ),
            _chunk(
                "lifting",
                "4.1.3.1",
                "Lifting Equipment",
                "The following table summarizes the lifting equipment to be provided by Contractor for the Work.",
                168,
            ),
        ]
    )

    ranked = retriever.retrieve("What lifting equipment is Contractor required to provide?")

    assert ranked
    assert ranked[0].chunk_id == "lifting"


def test_date_like_schedule_noise_is_penalized_for_equipment_queries() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "schedule_line",
                "0",
                "30-Jun-28",
                "GEV - ST #1 Steam Bypass Valves - Delivered to Site",
                132,
            ),
            _chunk(
                "steam_valves",
                "11.1",
                "Steam Turbine Bypass Valves, Dump devices",
                "Steam turbine bypass valves include one set of HP bypass valve per HRSG.",
                1952,
            ),
        ]
    )

    ranked = retriever.retrieve("What does the contract say about steam bypass valves?")

    assert ranked
    assert ranked[0].chunk_id == "steam_valves"


def _seed_retriever(chunks: list[ChunkRecord]) -> HybridRetriever:
    db_path = "file:retrieval_suite?mode=memory&cache=shared"
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
