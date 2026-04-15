from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import ExtractedBlock, PageText
from epc_smart_search.query_planner import build_like_fallback, plan_query
from epc_smart_search.retrieval import HybridRetriever, SearchCoverageCase
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore, build_block_records, pack_vector


def test_direct_section_lookup_wins() -> None:
    db_path = "file:retrieval_contract?mode=memory&cache=shared"
    store = ContractStore(db_path)
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
    )
    retriever = HybridRetriever(store)
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


def test_typo_rescue_recovers_heading_when_primary_terms_are_misspelled() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "termination",
                "12.1",
                "Owner Termination for Convenience",
                "Owner may terminate this Contract for convenience upon written notice.",
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

    ranked = retriever.retrieve("terminat for convenience")

    assert ranked
    assert ranked[0].chunk_id == "termination"


def test_rescue_search_is_skipped_when_primary_relevance_is_already_strong(monkeypatch) -> None:
    db_path = "file:retrieval_rescue_gate?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = _chunk(
        "motors",
        "4.4.3",
        "Electric Motors",
        "Electric motors must meet the design requirements for the project.",
        12,
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
    )
    retriever = HybridRetriever(store)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("rescue search should not run for a strong lexical hit")

    monkeypatch.setattr(store, "search_chunk_feature_rescue_fts", fail_if_called)

    ranked = retriever.retrieve("What does the contract say about electric motors?")

    assert ranked
    assert ranked[0].chunk_id == "motors"


def test_block_fallback_can_recover_table_like_evidence() -> None:
    db_path = "file:retrieval_block_fallback?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = _chunk(
        "equipment_matrix",
        "8.4",
        "Equipment Matrix",
        "Contractor shall provide the listed equipment in the matrix.",
        14,
    )
    page = PageText(
        page_num=14,
        text="Equipment Matrix",
        ocr_used=False,
        blocks=(
            ExtractedBlock(1, "heading", "Equipment Matrix", 1),
            ExtractedBlock(2, "table_row", "HRSG 1  Contractor", 1, ("table_like",)),
        ),
    )
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc",
        page_count=1,
        chunks=[chunk],
        pages=[page],
        features=build_chunk_features([chunk]),
    )
    retriever = HybridRetriever(store)

    ranked = retriever.retrieve("Show me HRSG 1")

    assert ranked
    assert ranked[0].chunk_id == "equipment_matrix"
    assert ranked[0].retrieval_stage == "block_fts"
    assert ranked[0].matched_block_count >= 1


def test_exact_page_hits_can_fall_back_to_block_matches() -> None:
    db_path = "file:retrieval_exact_block?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = _chunk(
        "fuel_summary",
        "A",
        "Appendix A Fuel Gas Summary",
        "Appendix A contains the summary table.",
        205,
        chunk_type="exhibit",
    )
    page = PageText(
        page_num=205,
        text="Appendix A Fuel Gas Summary",
        ocr_used=False,
        blocks=(ExtractedBlock(1, "table_row", "Fuel Gas Summary  45 MMSCFD", 1, ("table_like",)),),
    )
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc",
        page_count=1,
        chunks=[chunk],
        pages=[page],
        features=build_chunk_features([chunk]),
    )
    retriever = HybridRetriever(store)

    hits = retriever.find_exact_page_hits("show me fuel gas summary")

    assert hits
    assert hits[0].page_num == 205


def test_numeric_query_prefers_table_backed_fuel_gas_value() -> None:
    db_path = "file:retrieval_numeric_value?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = _chunk(
        "fuel_summary",
        "A",
        "Appendix A Fuel Gas Summary",
        "Appendix A contains the summary table for the fuel gas system.",
        205,
        chunk_type="exhibit",
    )
    page = PageText(
        page_num=205,
        text=chunk.full_text,
        ocr_used=False,
        blocks=(ExtractedBlock(1, "table_row", "Fuel Gas Summary  45 MMSCFD", 1, ("table_like",)),),
    )
    blocks = build_block_records("doc1", [page], [chunk])
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc",
        page_count=1,
        chunks=[chunk],
        pages=[page],
        features=build_chunk_features([chunk], blocks),
        blocks=blocks,
    )
    retriever = HybridRetriever(store)

    ranked = retriever.retrieve("45 MMSCFD")

    assert ranked
    assert ranked[0].chunk_id == "fuel_summary"
    assert ranked[0].matched_block_count >= 1


def test_numeric_query_finds_pressure_value_late_in_clause() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "fuel_pressure",
                "5.23",
                "Fuel Gas Supply",
                (
                    "The fuel gas system shall be designed for startup, commissioning, and performance testing. "
                    "During those operations the required delivery pressure shall be maintained at 450 psi."
                ),
                270,
            ),
        ]
    )

    ranked = retriever.retrieve("450 psi")

    assert ranked
    assert ranked[0].chunk_id == "fuel_pressure"


def test_numeric_phrase_query_finds_spelled_and_parenthetical_requirement() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "test_results",
                "7.5.3",
                "Air Permit Test Result",
                (
                    "Within twenty-four (24) hours after completion of a test, Contractor shall deliver the results "
                    "with supporting calculations."
                ),
                83,
            ),
        ]
    )

    ranked = retriever.retrieve("twenty-four (24) hours")

    assert ranked
    assert ranked[0].chunk_id == "test_results"


def test_evaluate_coverage_cases_reports_stage_and_match_status() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "motors",
                "4.4.3",
                "Electric Motors",
                "Electric motors must meet the design requirements for the project.",
                12,
            ),
        ]
    )

    results = retriever.evaluate_coverage_cases(
        [SearchCoverageCase(label="motors", query="electric motors", expected_chunk_id="motors")]
    )

    assert len(results) == 1
    assert results[0].found is True
    assert results[0].retrieval_stage in {"exact_lookup", "chunk_fts"}


def test_semantic_reranking_can_promote_paraphrase_match_in_deep_profile() -> None:
    chunks = [
        _chunk(
            "notice",
            "4.4",
            "Owner Cancellation Agreement Notice",
            "Owner shall issue an agreement cancellation notice for administrative filing.",
            24,
        ),
        _chunk(
            "termination",
            "12.1",
            "Owner Termination for Convenience",
            "Owner may terminate this Contract for convenience at any time upon written notice.",
            50,
        ),
    ]
    lexical = _seed_retriever(chunks)
    lexical_ranked = lexical.retrieve("Can the owner cancel the agreement?", profile="deep")
    semantic = _seed_retriever(
        chunks,
        embeddings={
            "notice": [0.0, 1.0, 0.0],
            "termination": [1.0, 0.0, 0.0],
        },
        embedder=_FakeEmbedder(
            {
                "cancel the agreement": [1.0, 0.0, 0.0],
            }
        ),
    )

    semantic_ranked = semantic.retrieve("Can the owner cancel the agreement?", profile="deep")

    assert lexical_ranked
    assert lexical_ranked[0].chunk_id == "notice"
    assert semantic_ranked[0].chunk_id == "termination"
    assert semantic_ranked[0].semantic_score > 0.0


def test_exact_section_lookup_is_not_displaced_by_semantic_similarity() -> None:
    retriever = _seed_retriever(
        [
            _chunk(
                "section_hit",
                "14.2.1",
                "Liquidated Damages",
                "This clause covers liquidated damages.",
                10,
            ),
            _chunk(
                "semantic_distractor",
                "14.2.2",
                "Delay Compensation",
                "This section uses related delay language.",
                11,
            ),
        ],
        embeddings={
            "section_hit": [0.0, 1.0, 0.0],
            "semantic_distractor": [1.0, 0.0, 0.0],
        },
        embedder=_FakeEmbedder(
            {
                "14.2.1": [1.0, 0.0, 0.0],
            }
        ),
    )

    ranked = retriever.retrieve("section 14.2.1", profile="deep")

    assert ranked
    assert ranked[0].chunk_id == "section_hit"


def test_semantic_reranking_falls_back_when_vector_dimensions_do_not_match() -> None:
    chunks = [
        _chunk(
            "notice",
            "4.4",
            "Owner Cancellation Agreement Notice",
            "Owner shall issue an agreement cancellation notice for administrative filing.",
            24,
        ),
        _chunk(
            "termination",
            "12.1",
            "Owner Termination for Convenience",
            "Owner may terminate this Contract for convenience at any time upon written notice.",
            50,
        ),
    ]
    retriever = _seed_retriever(
        chunks,
        embeddings={
            "notice": [0.0, 1.0],
            "termination": [1.0, 0.0],
        },
        model_name="test-semantic",
        dimension=2,
        embedder=_FakeEmbedder(
            {
                "cancel the agreement": [1.0, 0.0, 0.0],
            }
        ),
    )

    ranked = retriever.retrieve("Can the owner cancel the agreement?", profile="deep")

    assert ranked
    assert ranked[0].chunk_id == "notice"
    assert all(chunk.semantic_score == 0.0 for chunk in ranked)


def test_normal_profile_only_uses_semantics_when_lexical_evidence_is_weak() -> None:
    class FakeSemanticReranker:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def rerank(self, document_id: str, query_text: str, candidates) -> bool:
            self.calls.append(query_text)
            return False

    reranker = FakeSemanticReranker()
    retriever = HybridRetriever(
        _seed_store_for_semantic_gate(
            [
                _chunk(
                    "motors",
                    "4.4.3",
                    "Electric Motors",
                    "Electric motors must meet the design requirements for the project.",
                    12,
                ),
                _chunk(
                    "walkdown",
                    "4.4",
                    "Contract Walkdown Checklist",
                    "Company shall complete a contract walkdown checklist before turnover.",
                    24,
                ),
                _chunk(
                    "termination",
                    "12.1",
                    "Owner Termination for Convenience",
                    "Owner may terminate this Contract for convenience at any time upon written notice.",
                    50,
                ),
            ]
        ),
        semantic_reranker=reranker,
    )

    retriever.retrieve("What does the contract say about electric motors?")
    retriever.retrieve("Can a party leave the agreement?")

    assert len(reranker.calls) == 1


def _seed_store_for_semantic_gate(chunks: list[ChunkRecord]) -> ContractStore:
    db_path = "file:retrieval_semantic_gate?mode=memory&cache=shared"
    store = ContractStore(db_path)
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
    )
    return store


def _seed_retriever(
    chunks: list[ChunkRecord],
    *,
    embeddings: dict[str, list[float]] | None = None,
    model_name: str = "test-semantic",
    dimension: int | None = None,
    embedder=None,
) -> HybridRetriever:
    db_path = "file:retrieval_suite?mode=memory&cache=shared"
    store = ContractStore(db_path)
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
        embeddings={chunk_id: pack_vector(vector) for chunk_id, vector in (embeddings or {}).items()},
        model_name=model_name if embeddings else None,
        dimension=dimension or (len(next(iter(embeddings.values()))) if embeddings else None),
    )
    return HybridRetriever(store, embedder=embedder)


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


class _FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]], *, model_name: str = "test-semantic") -> None:
        self.vectors = {key.lower(): value for key, value in vectors.items()}
        self._model_name = model_name
        first = next(iter(vectors.values()))
        self._dimension = len(first)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_available(self) -> bool:
        return True

    def encode(self, texts) -> list[list[float]]:
        encoded: list[list[float]] = []
        for text in texts:
            normalized = str(text).lower()
            vector = next((value for key, value in self.vectors.items() if key in normalized), [0.0] * self._dimension)
            encoded.append(vector)
        return encoded
