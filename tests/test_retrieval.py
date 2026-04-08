from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.ocr_support import PageText
from epc_smart_search.retrieval import HashingEmbedder, HybridRetriever
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
        embeddings={"chunk1": pack_vector(embedder.embed(chunk.full_text))},
        model_name=embedder.model_name,
        dimension=embedder.dimension,
    )
    retriever = HybridRetriever(store, embedder)
    ranked = retriever.retrieve("section 14.2.1")
    assert ranked
    assert ranked[0].section_number == "14.2.1"
