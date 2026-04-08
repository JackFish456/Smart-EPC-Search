from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
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
        embeddings={"chunk1": b"\x00\x00\x00\x00"},
        model_name="test",
        dimension=1,
    )
    rows = store.search_fts("doc1", '"liquidated"', limit=5)
    assert rows
    assert rows[0]["chunk_id"] == "chunk1"
