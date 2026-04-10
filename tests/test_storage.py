from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore


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
