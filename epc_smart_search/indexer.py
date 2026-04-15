from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from epc_smart_search.chunking import ChunkRecord, build_document_id, parse_chunks
from epc_smart_search.ocr_support import extract_pages
from epc_smart_search.semantic import LocalEmbedder, build_chunk_semantic_text
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import (
    ContractStore,
    build_block_records,
    build_diagnostic_records,
    pack_vector,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(
    *,
    pdf_path: str | Path,
    db_path: str | Path,
    version_label: str = "v1",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, int | str]:
    path = Path(pdf_path)
    if progress_callback:
        progress_callback("Reading contract pages...")
    pages = extract_pages(str(path))

    document_id = build_document_id(path.name, version_label)
    if progress_callback:
        progress_callback("Parsing contract structure...")
    chunks = parse_chunks(pages, document_id)
    features = build_chunk_features(chunks)
    blocks = build_block_records(document_id, pages, chunks)
    diagnostics = build_diagnostic_records(document_id, pages)
    embeddings, model_name, dimension = build_chunk_embeddings(chunks, features)

    if progress_callback:
        progress_callback("Writing SQLite index...")
    import fitz

    with fitz.open(str(path)) as doc:
        page_count = doc.page_count
    store = ContractStore(db_path)
    store.replace_document(
        document_id=document_id,
        display_name=path.name,
        version_label=version_label,
        file_path=str(path),
        sha256=_sha256(path),
        page_count=page_count,
        chunks=chunks,
        pages=pages,
        features=features,
        blocks=blocks,
        diagnostics=diagnostics,
        embeddings=embeddings,
        model_name=model_name,
        dimension=dimension,
    )
    return {
        "document_id": document_id,
        "page_count": page_count,
        "chunk_count": len(chunks),
        "block_count": len(blocks),
        "embedding_count": len(embeddings or {}),
    }


def build_chunk_embeddings(
    chunks: list[ChunkRecord],
    features,
    *,
    embedder: LocalEmbedder | None = None,
) -> tuple[dict[str, bytes] | None, str | None, int | None]:
    active_embedder = embedder or LocalEmbedder()
    if not active_embedder.is_available():
        return None, None, None
    semantic_texts = [build_chunk_semantic_text(chunk, feature) for chunk, feature in zip(chunks, features)]
    vectors = active_embedder.encode(semantic_texts)
    embeddings = {
        chunk.chunk_id: pack_vector(vector)
        for chunk, vector in zip(chunks, vectors)
        if any(vector)
    }
    return embeddings, active_embedder.model_name, active_embedder.dimension


def refresh_query_index(
    store: ContractStore,
    document_id: str,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> int:
    if progress_callback:
        progress_callback("Refreshing query planner index...")
    rows = store.fetch_document_chunks(document_id)
    chunks = [
        ChunkRecord(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            chunk_type=str(row["chunk_type"]),
            section_number=row["section_number"],
            heading=str(row["heading"]),
            full_text=str(row["full_text"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            parent_chunk_id=row["parent_chunk_id"],
            ordinal_in_document=int(row["ordinal_in_document"]),
        )
        for row in rows
    ]
    features = build_chunk_features(chunks)
    store.replace_search_features(document_id, features)
    if progress_callback:
        progress_callback("Refreshing semantic vectors...")
    embeddings, model_name, dimension = build_chunk_embeddings(chunks, features)
    store.replace_chunk_embeddings(document_id, embeddings, model_name=model_name, dimension=dimension)
    return len(features)
