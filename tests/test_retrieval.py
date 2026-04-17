from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.fact_extraction import extract_contract_facts
from epc_smart_search.ocr_support import PageText
from epc_smart_search.query_planner import (
    REQUEST_SHAPE_BROAD_TOPIC,
    RETRIEVAL_MODE_FACT_LOOKUP,
    RETRIEVAL_MODE_GROUPED_LIST,
    RETRIEVAL_MODE_SECTION_LOOKUP,
    RETRIEVAL_MODE_TOPIC_SUMMARY,
    build_like_fallback,
    plan_query,
)
from epc_smart_search.retrieval import ContextEnrichmentSettings, HashingEmbedder, HybridRetriever, RankedChunk, format_trace_debug
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractFactRow, ContractStore, pack_vector


def test_plan_query_does_not_bind_site_as_system_for_site_design_conditions() -> None:
    plan = plan_query("what are the site design conditions")

    assert plan.attribute_label == "design_conditions"
    assert plan.system_phrase == ""
    assert "site" in plan.focus_terms


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
    assert retriever.plan_query("section 14.2.1").retrieval_mode == RETRIEVAL_MODE_SECTION_LOOKUP
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


def test_query_plan_normalizes_explicit_system_aliases_for_fact_lookup() -> None:
    plan = plan_query("What is the CCW configuration?")

    assert plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP
    assert plan.system_phrase == "closed cooling water"
    assert "ccw" in plan.system_aliases
    assert "closed cooling water system" in plan.system_aliases
    assert plan.attribute_label == "configuration"
    assert plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP


def test_query_plan_normalizes_emission_guarantees_typo_and_keeps_general_intent() -> None:
    plan = plan_query("what are my emission guarentees")

    assert plan.intent == "general_topic"
    assert plan.request_shape == "grouped_list"
    assert plan.answer_family == "guarantee_or_limit"
    assert plan.aggregate_requested is True
    assert plan.content_query == "my emission guarantees"
    assert plan.focus_terms == ("emission", "guarantees")
    assert plan.concept_terms == ("emission", "guarantees")
    assert plan.system_terms == ()


def test_query_plan_tracks_appendix_scope_for_grouped_guarantee_question() -> None:
    plan = plan_query("show me all emission guarantees in Appendix E")

    assert plan.request_shape == "grouped_list"
    assert plan.retrieval_mode == RETRIEVAL_MODE_GROUPED_LIST
    assert plan.scope_terms == ("appendix e", "appendix")
    assert plan.concept_terms == ("emission", "guarantees")


def test_query_plan_marks_environmental_requirements_as_broad_topic() -> None:
    plan = plan_query("do we have any environmental requirements?")

    assert plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC
    assert "environmental" in plan.focus_terms
    assert "requirements" in plan.focus_terms


def test_query_plan_marks_air_permits_prompt_as_broad_topic() -> None:
    plan = plan_query("give me information about air permits")

    assert plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC
    assert plan.content_query == "air permits"
    assert plan.retrieval_mode == RETRIEVAL_MODE_TOPIC_SUMMARY


def test_query_plan_marks_system_summary_prompt_as_broad_topic() -> None:
    plan = plan_query("Summarize the closed cooling water system")

    assert plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC
    assert plan.retrieval_mode == RETRIEVAL_MODE_TOPIC_SUMMARY
    assert plan.system_phrase == ""


def test_query_plan_marks_describe_prompt_as_topic_summary() -> None:
    plan = plan_query("Describe the closed cooling water system")

    assert plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC
    assert plan.retrieval_mode == RETRIEVAL_MODE_TOPIC_SUMMARY
    assert plan.content_query == "closed cooling water system"


def test_query_plan_keeps_attribute_queries_on_fact_lookup_even_with_describe_wording() -> None:
    plan = plan_query("Describe the CCW configuration")

    assert plan.attribute_label == "configuration"
    assert plan.system_phrase == "closed cooling water"
    assert plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP


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


def test_hierarchical_search_prefers_site_design_conditions_table_over_appendix_mentions() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "appendix_note",
                "1.1",
                "Design Basis for Guarantees",
                (
                    "The performance guarantees are stated for the following operating conditions and parameters. "
                    "Additional design conditions are referenced in the guarantee appendix."
                ),
                3035,
            ),
            _chunk(
                "site_design_conditions",
                "2.1",
                "Site Design Conditions",
                (
                    "Characteristic Specification GTG Equipment Location Outdoor ST/STG Location Outdoor "
                    "Elevation 455 ft Ambient Pressure 14.457 psia Minimum Outdoor Ambient Temperature -5°F "
                    "Maximum Outdoor Ambient Temperature 110°F Design Ambient Temperature 87°F"
                ),
                1591,
            ),
        ]
    )

    ranked = retriever.retrieve("what are the site design conditions?")

    assert ranked
    assert ranked[0].chunk_id == "site_design_conditions"


def test_hierarchical_search_prefers_exact_power_clause_over_random_pump_schedule() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "random_schedule",
                "6.9",
                "KV - 480 V",
                "02-CCW-PMP-01A TRAIN 2 CLOSED COOLING WATER PUMP MOTOR 1A 1200 01-BFW-PMP-01A TRAIN 1 BOILER FEEDWATER PUMP MOTOR 1A 4750 HP.",
                359,
            ),
            _chunk(
                "fire_water_pump",
                "8.4.2",
                "Fire Water Pump",
                "Each fire water pump shall be rated at 350 HP for the project fire water service.",
                412,
            ),
        ]
    )

    ranked = retriever.retrieve("what is the fire water pump horse power")

    assert ranked
    assert ranked[0].chunk_id == "fire_water_pump"


def test_contract_vocabulary_prefers_exact_multiword_system_over_related_sibling() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "cooling_water_pump",
                "8.4.1",
                "Cooling Water Pump",
                "Each cooling water pump shall be rated at 425 HP for circulating cooling water service.",
                401,
            ),
            _chunk(
                "fire_water_pump",
                "8.4.2",
                "Fire Water Pump",
                "Each fire water pump shall be rated at 350 HP for the project fire water service.",
                412,
            ),
        ]
    )

    ranked = retriever.retrieve("what is the fire water pumps horse power")

    assert ranked
    assert ranked[0].chunk_id == "fire_water_pump"


def test_contract_vocabulary_uses_heading_alias_without_promoting_wrong_related_match() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "combustion_controls",
                "9.2",
                "Combustion Turbine Controls",
                "Combustion turbine controls shall coordinate startup sequencing and alarms.",
                55,
            ),
            _chunk(
                "fuel_gas_system",
                "9.3",
                "Fuel Gas System (FGS)",
                "FGS shall include coalescing filters and a gas heater upstream of the turbine skids.",
                56,
            ),
        ]
    )

    ranked = retriever.retrieve("What does the contract say about the fuel gas system?")

    assert ranked
    assert ranked[0].chunk_id == "fuel_gas_system"


def test_contract_vocabulary_does_not_match_partial_water_pump_modifier_as_exact_system() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "fire_water_pump",
                "8.4.2",
                "Fire Water Pump",
                "Each fire water pump shall be rated at 350 HP for the project fire water service.",
                412,
            ),
            _chunk(
                "cooling_water_pump",
                "8.4.3",
                "Cooling Water Pump",
                "Each cooling water pump shall be rated at 425 HP for the circulating cooling water service.",
                413,
            ),
        ]
    )

    ranked = retriever.retrieve("what is the cooling water pump horse power")

    assert ranked
    assert ranked[0].chunk_id == "cooling_water_pump"


def test_hierarchical_search_prefers_exact_function_clause() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "install_clause",
                "5.1",
                "Fuel Gas System Installation",
                "Contractor shall install the fuel gas system in accordance with the project drawings.",
                20,
            ),
            _chunk(
                "function_clause",
                "5.2",
                "Fuel Gas System Description",
                "The fuel gas system receives natural gas from the pipeline, conditions the gas, and distributes it to the combustion turbines.",
                21,
            ),
        ]
    )

    ranked = retriever.retrieve("How does the fuel gas system work?")

    assert ranked
    assert ranked[0].chunk_id == "function_clause"


def test_grouped_question_prefers_appendix_guarantees_over_generic_emissions_clause() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "generic_emissions",
                "7.5.3",
                "Air Permit Test Result",
                "The submission must include all emissions values and supporting calculations.",
                83,
            ),
            _chunk(
                "appendix_e",
                "E",
                "APPENDIX E - Emission Guarantees",
                "Emission Guarantees. Seller guarantees NOx emissions shall not exceed 2.0 ppmvd at 15% oxygen.",
                2686,
                chunk_type="exhibit",
            ),
        ]
    )

    ranked = retriever.retrieve("what are my emission guarantees")

    assert ranked
    assert ranked[0].chunk_id == "appendix_e"


def test_grouped_question_prefers_performance_guarantee_wording_for_emissions() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "generic_environmental",
                "13.2.7",
                "Environmental",
                "For emissions guarantees, refer to the Performance Guarantees tab of this proposal.",
                1965,
            ),
            _chunk(
                "performance_guarantees",
                "H",
                "EXHIBIT H - PERFORMANCE GUARANTEES",
                "Seller guarantees as a Minimum Performance Guarantee that NOx emissions shall not exceed 2.0 ppmvd and CO emissions shall not exceed 4.0 ppmvd.",
                3031,
                chunk_type="exhibit",
            ),
        ]
    )

    ranked = retriever.retrieve("what are my emission guarantees")

    assert ranked
    assert ranked[0].chunk_id == "performance_guarantees"


def test_grouped_question_returns_related_appendix_results_as_bundle_candidates() -> None:
    appendix = _chunk(
        "appendix_e",
        "E",
        "APPENDIX E - Emission Guarantees",
        "Emission Guarantees.",
        2686,
        chunk_type="exhibit",
    )
    sibling = _chunk(
        "appendix_e_limits",
        "E.1",
        "Guarantee Limits",
        "Seller guarantees NOx emissions shall not exceed 2.0 ppmvd at 15% oxygen. CO emissions shall not exceed 4.0 ppmvd at 15% oxygen.",
        2687,
        parent_chunk_id="appendix_e",
    )
    retriever = _seed_retriever([appendix, sibling])

    ranked = retriever.retrieve("show me all emission guarantees in appendix e")

    assert ranked
    assert {ranked[0].chunk_id, ranked[1].chunk_id} == {"appendix_e", "appendix_e_limits"}


def test_appendix_anchor_bundle_reaches_nearby_guarantee_sections_without_parent_links() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "appendix_e",
                "E",
                "APPENDIX E - Facility Guarantees",
                "Facility Guarantees.",
                100,
                chunk_type="exhibit",
            ),
            _chunk(
                "appendix_e_emissions",
                "1.9",
                "Auxiliary Boiler Emissions Guarantees",
                "NOx shall not exceed 0.036 lb/mmbtu HHV and CO shall not exceed 50 ppmvd corrected to 3% O2.",
                102,
            ),
            _chunk(
                "generic_environmental",
                "13.2.7",
                "Environmental",
                "For emissions guarantees, refer to the performance guarantees tab of this proposal.",
                300,
            ),
        ]
    )

    ranked = retriever.retrieve("what are my emission guarantees in appendix e")

    assert ranked
    assert ranked[0].chunk_id == "appendix_e_emissions"


def test_specific_appendix_scope_does_not_collapse_to_any_appendix() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "appendix_aa_noise",
                "AA",
                "APPENDIX AA - Air Permit",
                "Opacity of emissions from the combustion sources shall not exceed 5 percent.",
                300,
                chunk_type="exhibit",
            ),
            _chunk(
                "appendix_e",
                "E",
                "APPENDIX E - Facility Guarantees",
                "Facility Guarantees.",
                100,
                chunk_type="exhibit",
            ),
            _chunk(
                "appendix_e_emissions",
                "1.9",
                "Auxiliary Boiler Emissions Guarantees",
                "NOx shall not exceed 0.036 lb/mmbtu HHV and CO shall not exceed 50 ppmvd corrected to 3% O2.",
                102,
            ),
        ]
    )

    ranked = retriever.retrieve("what are my emission guarantees in appendix e")

    assert ranked
    assert ranked[0].chunk_id == "appendix_e_emissions"


def test_retrieve_trace_prefers_semantic_system_clause_over_heading_only_noise() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "ccw_heading_only",
                "3",
                "Closed Cooling Water Pump",
                "3",
                1485,
            ),
            _chunk(
                "ccw_system",
                "5.18",
                "Closed Cooling Water",
                "The Closed Cooling Water System major components include two 100% capacity, horizontal, closed cooling water pumps.",
                261,
            ),
        ]
    )

    trace = retriever.retrieve_trace("what is the closed cooling water pump configuration")

    assert trace.selected_bundle is not None
    assert trace.selected_bundle.ranked_chunks[0].chunk_id == "ccw_system"
    assert any(chunk.chunk_id == "ccw_heading_only" for chunk in trace.recall_sources["raw_fts"])
    assert any(chunk.chunk_id == "ccw_system" for chunk in trace.recall_sources["semantic"])


def test_fact_lookup_uses_shared_system_and_attribute_normalization() -> None:
    db_path = "file:retrieval_fact_alias?mode=memory&cache=shared"
    store = ContractStore(db_path)
    embedder = HashingEmbedder(dimension=16)
    chunk = _chunk(
        "ccw_fact",
        "5.18",
        "Auxiliary Summary",
        "The system summary table lists two parallel trains.",
        261,
    )
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc",
        page_count=1,
        chunks=[chunk],
        pages=[PageText(page_num=261, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
        facts=[
            ContractFactRow(
                document_id="doc1",
                system="Closed Cooling Water System",
                attribute="Configuration / Arrangement",
                value="2 x 100%",
                evidence_text="The closed cooling water system shall be arranged as 2 x 100%.",
                source_chunk_id=chunk.chunk_id,
                page_start=261,
                page_end=261,
            )
        ],
        embeddings={chunk.chunk_id: pack_vector(embedder.embed(chunk.full_text))},
        model_name=embedder.model_name,
        dimension=embedder.dimension,
    )
    retriever = HybridRetriever(store, embedder)

    trace = retriever.retrieve_trace("What is the CCW configuration?")

    assert trace.plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP
    assert trace.recall_sources["fact_lookup"]
    assert trace.selected_bundle is not None
    assert trace.selected_bundle.ranked_chunks[0].chunk_id == "ccw_fact"
    assert trace.normalized_system == "closed cooling water"
    assert trace.normalized_attribute == "configuration"
    assert trace.fact_lookup_attempted is True
    assert trace.chunk_fallback_used is False
    assert "ccw" in trace.system_aliases_used
    assert trace.fts_hits == []
    assert trace.embedding_hits == []
    assert trace.keyword_hits == []
    assert len(trace.candidate_chunks) == 1
    assert trace.candidate_chunks[0].chunk_id == "ccw_fact"
    assert trace.candidate_chunks[0].source_type == "fact_lookup"
    assert [chunk.chunk_id for chunk in trace.selected_chunks] == ["ccw_fact"]
    assert [row.value for row in trace.fact_rows] == ["2 x 100%"]
    assert trace.fact_hit is not None
    assert trace.fact_hit.value == "2 x 100%"
    assert trace.fact_rows_returned == 1
    assert trace.fact_fallback_reason is None
    assert trace.fallback_reason is None
    debug = trace.to_debug_dict()
    assert debug["query"] == "What is the CCW configuration?"
    assert debug["retrieval_mode"] == "fact_lookup"
    assert debug["request_shape"] == "scalar"
    assert debug["normalized_system"] == "closed cooling water"
    assert debug["normalized_attribute"] == "configuration"
    assert "ccw" in debug["system_aliases_used"]
    assert debug["fact_lookup_attempted"] is True
    assert debug["chunk_fallback_used"] is False
    assert debug["fact_rows_returned"] == 1
    assert debug["fact_fallback_reason"] is None
    assert debug["fallback_reason"] is None
    assert debug["selected_bundle_id"] == "ccw_fact"
    assert debug["recall_source_counts"] == {"fact_lookup": 1}
    assert debug["fts_hits"] == []
    assert debug["embedding_hits"] == []
    assert debug["keyword_hits"] == []
    assert debug["candidate_chunks"][0]["source_type"] == "fact_lookup"
    assert debug["selected_chunks"][0]["chunk_id"] == "ccw_fact"
    assert len(debug["fact_rows"]) == 1
    assert debug["fact_rows"][0]["system_normalized"] == "closed cooling water"
    assert debug["fact_rows"][0]["attribute_normalized"] == "configuration"
    assert debug["fact_rows"][0]["value"] == "2 x 100%"
    assert debug["fact_hit"]["value"] == "2 x 100%"


def test_exact_attribute_fact_lookup_returns_direct_fact_hit_for_dew_point_configuration() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "dew_point_fact",
                "7.4.2",
                "Dew Point Heaters",
                "Dew point heaters shall be furnished in a 4 x 50% configuration.",
                41,
            ),
            _chunk(
                "dew_point_overview",
                "7.4",
                "Fuel Gas Heating",
                "The fuel gas heating system shall maintain gas temperature ahead of the conditioning package.",
                40,
            ),
        ],
        facts=[
            ContractFactRow(
                document_id="doc1",
                system="Dew Point Heaters",
                system_normalized="dew point heater",
                attribute="Configuration / Arrangement",
                attribute_normalized="configuration",
                value="4 x 50%",
                evidence_text="Dew point heaters shall be furnished in a 4 x 50% configuration.",
                source_chunk_id="dew_point_fact",
                page_start=41,
                page_end=41,
            )
        ],
    )

    trace = retriever.retrieve_trace("What is the configuration of the dew point heaters?")

    assert trace.plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP
    assert list(trace.recall_sources) == ["fact_lookup"]
    assert trace.fact_hit is not None
    assert trace.fact_hit.value == "4 x 50%"
    assert trace.selected_bundle is not None
    assert trace.selected_bundle.primary_chunk_id == "dew_point_fact"


def test_topic_summary_keeps_bundle_path_for_closed_cooling_water_summary() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "ccw_summary",
                "5.18",
                "Closed Cooling Water",
                "The closed cooling water system includes pumps, air coolers, and glycol-water circulation for unit cooling.",
                261,
            ),
            _chunk(
                "ccw_support",
                "5.18.1",
                "Closed Cooling Water Pumps",
                "Closed cooling water pumps circulate coolant through the unitized closed cooling water system.",
                262,
                parent_chunk_id="ccw_summary",
            ),
        ]
    )

    trace = retriever.retrieve_trace("Summarize the closed cooling water system")

    assert trace.plan.request_shape == REQUEST_SHAPE_BROAD_TOPIC
    assert trace.plan.retrieval_mode == RETRIEVAL_MODE_TOPIC_SUMMARY
    assert "fact_lookup" not in trace.recall_sources
    assert trace.fact_lookup_attempted is False
    assert trace.chunk_fallback_used is False
    assert trace.selected_bundle is not None
    assert trace.selected_bundle.primary_chunk_id == "ccw_summary"


def test_retrieve_trace_can_use_gemma_to_break_close_bundle_ties() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "candidate_a",
                "9.1",
                "Selected Turbine",
                "The selected turbine model shall be Siemens SGT6-5000F for the project.",
                18,
            ),
            _chunk(
                "candidate_b",
                "9.2",
                "Selected Turbine Generator",
                "The selected turbine model shall be Mitsubishi M701F for the project.",
                19,
            ),
        ]
    )

    class FakeGemma:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def ask(self, question: str, context: str, **kwargs):
            self.calls.append({"question": question, "context": context, **kwargs})
            return '{"candidate_id":"candidate_b","supporting_quote":"Mitsubishi M701F","insufficient_support":false}'

    fake_gemma = FakeGemma()
    retriever._should_use_gemma_disambiguation = lambda bundles: True  # type: ignore[method-assign]

    trace = retriever.retrieve_trace("What is the turbine we are using?", gemma_client=fake_gemma)

    assert trace.used_gemma_disambiguation is True
    assert trace.selected_bundle is not None
    assert trace.selected_bundle.bundle_id == "candidate_b"
    assert trace.selected_bundle.supporting_quote == "Mitsubishi M701F"
    assert fake_gemma.calls[0]["response_style"] == "candidate_select"
    assert "Candidate ID: candidate_a" in fake_gemma.calls[0]["context"]
    assert "Candidate ID: candidate_b" in fake_gemma.calls[0]["context"]


def test_fact_lookup_short_circuits_generic_recall_for_exact_value_questions() -> None:
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
                "The selected turbine generator model shall be Siemens SGT6-5000F for the project.",
                19,
            ),
        ],
        facts=[
            ContractFactRow(
                document_id="doc1",
                system="turbine generator",
                system_normalized="turbine generator",
                attribute="model",
                attribute_normalized="model",
                value="Siemens SGT6-5000F for the project",
                evidence_text="The selected turbine generator model shall be Siemens SGT6-5000F for the project.",
                source_chunk_id="selected_turbine",
                page_start=19,
                page_end=19,
            )
        ],
    )

    trace = retriever.retrieve_trace("What model turbine generator is selected?")

    assert trace.plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP
    assert list(trace.recall_sources) == ["fact_lookup"]
    assert trace.recall_sources["fact_lookup"]
    assert trace.merged_ranked
    assert trace.merged_ranked[0].chunk_id == "selected_turbine"


def test_topic_summary_keeps_broader_bundle_path_for_summary_questions() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "permit_overview",
                "8.5",
                "Air Permit Compliance",
                "Contractor shall obtain air permits and maintain compliance with permit conditions.",
                15,
            ),
            _chunk(
                "permit_testing",
                "8.5.1",
                "Air Permit Testing",
                "Contractor shall perform required emissions testing and submit reports demonstrating air permit compliance.",
                16,
                parent_chunk_id="permit_overview",
            ),
        ]
    )

    trace = retriever.retrieve_trace("give me information about air permits")

    assert trace.plan.retrieval_mode == RETRIEVAL_MODE_TOPIC_SUMMARY
    assert "fact_lookup" not in trace.recall_sources
    assert trace.selected_bundle is not None
    assert trace.selected_bundle.primary_chunk_id == "permit_overview"
    assert {chunk.chunk_id for chunk in trace.selected_bundle.ranked_chunks[:2]} == {"permit_overview", "permit_testing"}
    assert trace.fact_lookup_attempted is False
    assert trace.chunk_fallback_used is False
    assert trace.fact_rows_returned == 0
    assert trace.fallback_reason == "topic_summary_mode"
    assert trace.to_debug_dict()["fact_lookup_attempted"] is False
    assert trace.to_debug_dict()["recall_source_counts"]["planner_hints"] > 0


def test_low_confidence_fact_lookup_falls_back_to_generic_recall() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "service_water_overview",
                "6.1",
                "Service Water Tank",
                "The service water tank stores treated water for startup and shutdown operations.",
                10,
            ),
            _chunk(
                "service_water_capacity",
                "6.2",
                "Service Water Tank Data",
                "The service water tank capacity shall be 500000 gallons.",
                11,
            ),
        ],
        facts=[
            ContractFactRow(
                document_id="doc1",
                system="service water tank",
                system_normalized="service water tank",
                attribute="capacity",
                attribute_normalized="capacity",
                value="startup water",
                evidence_text="The service water tank stores treated water for startup and shutdown operations.",
                source_chunk_id="service_water_overview",
                page_start=10,
                page_end=10,
            ),
            ContractFactRow(
                document_id="doc1",
                system="service water tank",
                system_normalized="service water tank",
                attribute="capacity",
                attribute_normalized="capacity",
                value="500000 gallons",
                evidence_text="The service water tank capacity shall be 500000 gallons.",
                source_chunk_id="service_water_capacity",
                page_start=11,
                page_end=11,
            ),
        ],
    )

    trace = retriever.retrieve_trace("What is the capacity of the service water tank?")

    assert trace.plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP
    assert "fact_lookup" in trace.recall_sources
    assert "planner_hints" in trace.recall_sources
    assert trace.merged_ranked
    assert trace.merged_ranked[0].chunk_id == "service_water_capacity"
    assert trace.normalized_system == "service water tank"
    assert trace.normalized_attribute == "capacity"
    assert trace.fact_lookup_attempted is True
    assert trace.chunk_fallback_used is True
    assert trace.fts_hits
    assert trace.embedding_hits
    assert isinstance(trace.keyword_hits, list)
    assert trace.candidate_chunks
    assert any("fact_lookup" in candidate.source_names for candidate in trace.candidate_chunks)
    assert any(len(candidate.source_names) > 1 for candidate in trace.candidate_chunks)
    assert trace.selected_chunks
    assert trace.fact_rows_returned == 2
    assert trace.fact_hit is None
    assert trace.fact_fallback_reason == "conflicting_fact_values"
    assert trace.fallback_reason == "conflicting_fact_values"
    debug = trace.to_debug_dict()
    assert debug["fact_fallback_reason"] == "conflicting_fact_values"
    assert debug["fallback_reason"] == "conflicting_fact_values"
    assert debug["chunk_fallback_used"] is True
    assert len(debug["fact_rows"]) == 2
    assert debug["fts_hits"]
    assert debug["embedding_hits"]
    assert isinstance(debug["keyword_hits"], list)
    assert debug["candidate_chunks"]
    assert debug["selected_chunks"]
    assert debug["recall_source_counts"]["fact_lookup"] == 2
    assert debug["recall_source_counts"]["planner_hints"] > 0


def test_format_trace_debug_distinguishes_fact_hit_and_fallback_paths() -> None:
    fact_trace = _seed_retriever(
        [
            _chunk(
                "dew_point_fact",
                "7.4.2",
                "Dew Point Heaters",
                "Dew point heaters shall be furnished in a 4 x 50% configuration.",
                41,
            )
        ],
        facts=[
            ContractFactRow(
                document_id="doc1",
                system="Dew Point Heaters",
                system_normalized="dew point heater",
                attribute="Configuration / Arrangement",
                attribute_normalized="configuration",
                value="4 x 50%",
                evidence_text="Dew point heaters shall be furnished in a 4 x 50% configuration.",
                source_chunk_id="dew_point_fact",
                page_start=41,
                page_end=41,
            )
        ],
    ).retrieve_trace("What is the configuration of the dew point heaters?")

    fallback_trace = _seed_retriever(
        [
            _chunk(
                "service_water_overview",
                "6.1",
                "Service Water Tank",
                "The service water tank stores treated water for startup and shutdown operations.",
                10,
            ),
            _chunk(
                "service_water_capacity",
                "6.2",
                "Service Water Tank Data",
                "The service water tank capacity shall be 500000 gallons.",
                11,
            ),
        ],
        facts=[
            ContractFactRow(
                document_id="doc1",
                system="service water tank",
                system_normalized="service water tank",
                attribute="capacity",
                attribute_normalized="capacity",
                value="startup water",
                evidence_text="The service water tank stores treated water for startup and shutdown operations.",
                source_chunk_id="service_water_overview",
                page_start=10,
                page_end=10,
            ),
            ContractFactRow(
                document_id="doc1",
                system="service water tank",
                system_normalized="service water tank",
                attribute="capacity",
                attribute_normalized="capacity",
                value="500000 gallons",
                evidence_text="The service water tank capacity shall be 500000 gallons.",
                source_chunk_id="service_water_capacity",
                page_start=11,
                page_end=11,
            ),
        ],
    ).retrieve_trace("What is the capacity of the service water tank?")

    fact_debug = format_trace_debug(fact_trace)
    fallback_debug = format_trace_debug(fallback_trace)

    assert "Fact Hit: Dew Point Heaters | Configuration / Arrangement = 4 x 50%" in fact_debug
    assert "Chunk Fallback Used: no" in fact_debug
    assert "Fallback Reason: none" in fact_debug
    assert "Fact Hit: none" in fallback_debug
    assert "Chunk Fallback Used: yes" in fallback_debug
    assert "Fallback Reason: conflicting_fact_values" in fallback_debug
    assert "Candidate Chunks:" in fact_debug
    assert "Selected Chunks:" in fallback_debug


def test_broad_topic_environmental_requirements_prefers_requirement_clause_over_appendix_header() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "appendix_header",
                "AA",
                "APPENDIX AA",
                "Appendix AA contains administrative materials and reference exhibits for the project.",
                9,
                chunk_type="exhibit",
            ),
            _chunk(
                "definition_only",
                "1.2",
                "Applicable Legal Requirements",
                '"Applicable Legal Requirements" means all laws, Environmental Laws, permits, approvals, and similar legal requirements.',
                10,
                chunk_type="definition",
            ),
            _chunk(
                "env_requirements",
                "7.4",
                "Environmental Requirements",
                "Contractor shall obtain required environmental permits, maintain compliance with permit conditions, and submit testing results that demonstrate emissions compliance.",
                11,
            ),
        ]
    )

    ranked = retriever.retrieve("do we have any environmental requirements?")

    assert ranked
    assert ranked[0].chunk_id == "env_requirements"


def test_broad_topic_air_permits_prefers_operational_clause_over_acronyms() -> None:
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
        ]
    )

    ranked = retriever.retrieve("give me information about air permits")

    assert ranked
    assert ranked[0].chunk_id == "permit_clause"


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


def test_context_enrichment_includes_parent_and_neighbors_in_order() -> None:
    chunks = [
        _chunk(
            "parent_sec",
            "1",
            "Article Scope",
            "Parent section establishes scope and obligations for the work described herein.",
            10,
            chunk_type="section",
            ordinal_in_document=10,
        ),
        _chunk(
            "mid",
            "1.1",
            "Detail clause",
            "Mid detail clause requires contractor coordination and sufficient descriptive text.",
            11,
            chunk_type="subsection",
            parent_chunk_id="parent_sec",
            ordinal_in_document=11,
        ),
        _chunk(
            "after",
            "1.2",
            "Following clause",
            "Following subsection text continues requirements after the detail clause.",
            12,
            chunk_type="subsection",
            parent_chunk_id="parent_sec",
            ordinal_in_document=12,
        ),
    ]
    retriever = _seed_retriever(chunks)
    ranked = RankedChunk(
        chunk_id="mid",
        section_number="1.1",
        heading="Detail clause",
        full_text=chunks[1].full_text,
        page_start=11,
        page_end=11,
        ordinal_in_document=11,
        total_score=1.0,
        lexical_score=0.5,
        semantic_score=0.5,
    )
    citations = retriever.expand_with_context([ranked])
    ids = [c.chunk_id for c in citations]
    assert ids[0] == "parent_sec"
    assert "mid" in ids
    assert "after" in ids
    assert len(ids) == len(set(ids))


def test_context_enrichment_boundary_low_ordinal_includes_only_existing_neighbors() -> None:
    chunks = [
        _chunk(
            "first",
            "A",
            "Opening",
            "Opening clause establishes the first obligations with sufficient token length.",
            10,
            ordinal_in_document=0,
        ),
        _chunk(
            "second",
            "B",
            "Next",
            "Next clause provides ordinal neighbor to the right with more requirement language.",
            11,
            ordinal_in_document=1,
        ),
    ]
    retriever = _seed_retriever(chunks)
    ranked = RankedChunk(
        chunk_id="first",
        section_number="A",
        heading="Opening",
        full_text=chunks[0].full_text,
        page_start=10,
        page_end=10,
        ordinal_in_document=0,
        total_score=1.0,
        lexical_score=0.5,
        semantic_score=0.5,
    )
    citations = retriever.expand_with_context([ranked], enrichment=ContextEnrichmentSettings(max_neighbors=2))
    ids = [c.chunk_id for c in citations]
    assert ids[0] == "first"
    assert "second" in ids
    assert len(ids) == len(set(ids))


def test_context_enrichment_dedupes_when_multiple_ranked_share_context() -> None:
    chunks = [
        _chunk("p", "1", "Parent", "Parent text for hierarchy with adequate length for indexing.", 50, chunk_type="section", ordinal_in_document=5),
        _chunk(
            "a",
            "1.1",
            "Clause A",
            "Clause A text about obligations and contractor duties under this agreement.",
            51,
            chunk_type="subsection",
            parent_chunk_id="p",
            ordinal_in_document=6,
        ),
        _chunk(
            "b",
            "1.2",
            "Clause B",
            "Clause B text continues neighboring content for ordinal expansion testing.",
            52,
            chunk_type="subsection",
            parent_chunk_id="p",
            ordinal_in_document=7,
        ),
    ]
    retriever = _seed_retriever(chunks)
    r_a = RankedChunk(
        chunk_id="a",
        section_number="1.1",
        heading="Clause A",
        full_text=chunks[1].full_text,
        page_start=51,
        page_end=51,
        ordinal_in_document=6,
        total_score=2.0,
        lexical_score=1.0,
        semantic_score=1.0,
    )
    r_b = RankedChunk(
        chunk_id="b",
        section_number="1.2",
        heading="Clause B",
        full_text=chunks[2].full_text,
        page_start=52,
        page_end=52,
        ordinal_in_document=7,
        total_score=1.0,
        lexical_score=0.5,
        semantic_score=0.5,
    )
    citations = retriever.expand_with_context([r_a, r_b])
    ids = [c.chunk_id for c in citations]
    assert len(ids) == len(set(ids))
    assert ids.count("p") == 1


def test_context_enrichment_includes_children_for_section_chunks() -> None:
    chunks = [
        _chunk(
            "sec",
            "5",
            "Equipment section",
            "This section summarizes equipment packages and related obligations for the contractor.",
            100,
            chunk_type="section",
            ordinal_in_document=100,
        ),
        _chunk(
            "c1",
            "5.1",
            "Pump package",
            "Pump package subsection includes rated flow and driver details for procurement.",
            101,
            chunk_type="subsection",
            parent_chunk_id="sec",
            ordinal_in_document=101,
        ),
        _chunk(
            "c2",
            "5.2",
            "Motor package",
            "Motor package subsection lists efficiency and enclosure requirements for the plant.",
            102,
            chunk_type="subsection",
            parent_chunk_id="sec",
            ordinal_in_document=102,
        ),
    ]
    retriever = _seed_retriever(chunks)
    ranked = RankedChunk(
        chunk_id="sec",
        section_number="5",
        heading="Equipment section",
        full_text=chunks[0].full_text,
        page_start=100,
        page_end=100,
        ordinal_in_document=100,
        total_score=1.0,
        lexical_score=0.5,
        semantic_score=0.5,
    )
    citations = retriever.expand_with_context([ranked], enrichment=ContextEnrichmentSettings(max_neighbors=0, include_parent=False))
    ids = [c.chunk_id for c in citations]
    assert "c1" in ids
    assert "c2" in ids
    assert ids.index("sec") < ids.index("c1")


def test_context_enrichment_respects_include_parent_false() -> None:
    chunks = [
        _chunk("parent_sec", "1", "Parent", "Parent section text with enough content for the test database.", 20, chunk_type="section", ordinal_in_document=20),
        _chunk(
            "child",
            "1.1",
            "Child",
            "Child subsection text describes specific duties with adequate length here.",
            21,
            chunk_type="subsection",
            parent_chunk_id="parent_sec",
            ordinal_in_document=21,
        ),
    ]
    retriever = _seed_retriever(chunks)
    ranked = RankedChunk(
        chunk_id="child",
        section_number="1.1",
        heading="Child",
        full_text=chunks[1].full_text,
        page_start=21,
        page_end=21,
        ordinal_in_document=21,
        total_score=1.0,
        lexical_score=0.5,
        semantic_score=0.5,
    )
    citations = retriever.expand_with_context(
        [ranked],
        enrichment=ContextEnrichmentSettings(include_parent=False, max_neighbors=1),
    )
    assert all(c.chunk_id != "parent_sec" for c in citations)


def _seed_retriever(chunks: list[ChunkRecord], *, facts: list[ContractFactRow] | None = None) -> HybridRetriever:
    db_path = "file:retrieval_suite?mode=memory&cache=shared"
    store = ContractStore(db_path)
    embedder = HashingEmbedder(dimension=16)
    pages = [PageText(page_num=chunk.page_start, text=chunk.full_text, ocr_used=False) for chunk in chunks]
    extracted_facts = facts
    if extracted_facts is None:
        extracted_facts = [
            ContractFactRow(
                document_id=fact.document_id,
                system=fact.normalized_system,
                system_normalized=fact.normalized_system,
                attribute=fact.normalized_attribute,
                attribute_normalized=fact.normalized_attribute,
                value=fact.raw_value,
                evidence_text=fact.evidence_text,
                source_chunk_id=fact.source_chunk_id,
                page_start=fact.page_start,
                page_end=fact.page_end,
            )
            for fact in extract_contract_facts(chunks)
        ]
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
        facts=extracted_facts,
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
    ordinal_in_document: int | None = None,
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
        ordinal_in_document=page_num if ordinal_in_document is None else ordinal_in_document,
    )
