from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.indexer import refresh_query_index
from epc_smart_search.ocr_support import ExtractedBlock, PageText
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore, build_block_records, pack_vector


def test_fts_triggers_stay_in_sync() -> None:
    db_path = "file:storage_contract?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="14.2.1",
        heading="Liquidated Damages",
        full_text="Liquidated damages apply when delay occurs.",
        page_start=10,
        page_end=10,
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
        pages=[PageText(page_num=10, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
    )
    feature_rows = store.search_chunk_feature_fts("doc1", 'topic_tags : "liquidated damages" OR heading : "liquidated"', limit=5)
    assert feature_rows
    assert feature_rows[0]["chunk_id"] == "chunk1"

    rescue_rows = store.search_chunk_feature_rescue_fts("doc1", '"liq" OR "iqu" OR "qui"', limit=5)
    assert rescue_rows
    assert rescue_rows[0]["chunk_id"] == "chunk1"
    assert store.get_metadata("search_schema_version") is not None


def test_section_lookup_uses_composite_doc_section_ordinal_index() -> None:
    db_path = "file:storage_contract_plan?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="14.2.1",
        heading="Liquidated Damages",
        full_text="Liquidated damages apply when delay occurs.",
        page_start=10,
        page_end=10,
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
        pages=[PageText(page_num=10, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
    )

    with store._connect() as connection:  # noqa: SLF001
        rows = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT *
            FROM contract_chunks
            WHERE document_id = ? AND section_number = ?
            ORDER BY ordinal_in_document
            """,
            ("doc1", "14.2.1"),
        ).fetchall()

    details = " ".join(str(row[3]) for row in rows)
    assert "idx_contract_chunks_doc_section_ordinal" in details


def test_block_indexes_and_ingest_diagnostics_are_materialized() -> None:
    db_path = "file:storage_contract_blocks?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="8.4",
        heading="Equipment Matrix",
        full_text="Contractor shall provide the listed equipment.",
        page_start=14,
        page_end=14,
        parent_chunk_id=None,
        ordinal_in_document=1,
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

    block_rows = store.search_block_fts("doc1", '"hrsg" OR "contractor"', limit=5)
    assert block_rows
    assert block_rows[0]["chunk_id"] == "chunk1"
    assert store.get_block_count("doc1") == 2
    diagnostics = store.get_ingest_diagnostic_summary("doc1")
    assert diagnostics["table_like_pages"] == 1


def test_chunk_vectors_are_persisted_and_queryable() -> None:
    db_path = "file:storage_contract_vectors?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="8.2",
        heading="Warranty Period",
        full_text="The warranty period begins on substantial completion.",
        page_start=18,
        page_end=18,
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
        pages=[PageText(page_num=18, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
        embeddings={"chunk1": pack_vector([1.0, 0.0, 0.0])},
        model_name="test-semantic",
        dimension=3,
    )

    assert store.get_embedding_count("doc1") == 1
    assert store.get_embedding_metadata("doc1") == ("test-semantic", 3)
    vectors = store.fetch_chunk_vectors("doc1", ["chunk1"])
    assert vectors["chunk1"]["dimension"] == 3
    assert vectors["chunk1"]["vector"] == [1.0, 0.0, 0.0]


def test_refresh_query_index_rebuilds_semantic_vectors(monkeypatch) -> None:
    db_path = "file:storage_refresh_vectors?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="9.1",
        heading="Termination",
        full_text="Owner may terminate this Contract upon written notice.",
        page_start=22,
        page_end=22,
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
        pages=[PageText(page_num=22, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
    )

    monkeypatch.setattr(
        "epc_smart_search.indexer.build_chunk_embeddings",
        lambda chunks, features: ({"chunk1": pack_vector([0.0, 1.0, 0.0])}, "test-semantic", 3),
    )

    refreshed = refresh_query_index(store, "doc1")

    assert refreshed == 1
    assert store.get_embedding_count("doc1") == 1
    assert store.get_embedding_metadata("doc1") == ("test-semantic", 3)


def test_numeric_text_captures_late_body_values_and_table_rows() -> None:
    db_path = "file:storage_numeric_features?mode=memory&cache=shared"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="5.23",
        heading="Fuel Gas System",
        full_text=(
            "The fuel gas system design basis is summarized in Appendix A. "
            "During performance testing the required delivery pressure shall be maintained. "
            "The guaranteed delivery pressure is 450 psi."
        ),
        page_start=14,
        page_end=14,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )
    page = PageText(
        page_num=14,
        text=chunk.full_text,
        ocr_used=False,
        blocks=(
            ExtractedBlock(1, "heading", "Fuel Gas System", 1),
            ExtractedBlock(2, "table_row", "Fuel Gas Summary  45 MMSCFD", 1, ("table_like",)),
        ),
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

    with store._connect() as connection:  # noqa: SLF001
        row = connection.execute(
            """
            SELECT hierarchy_path, numeric_text
            FROM chunk_search_features
            WHERE chunk_id = 'chunk1'
            """
        ).fetchone()

    assert row is not None
    assert "section 5.23 fuel gas system" in str(row["hierarchy_path"]).lower()
    assert "450 psi" in str(row["numeric_text"]).lower()
    assert "45 mmscfd" in str(row["numeric_text"]).lower()
