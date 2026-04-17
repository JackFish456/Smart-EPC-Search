from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import fitz

from epc_smart_search.chunking import ChunkRecord, build_document_id, parse_chunks
from epc_smart_search.fact_extraction import extract_contract_facts
from epc_smart_search.ocr_support import extract_pages
from epc_smart_search.retrieval import HashingEmbedder
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractFactRow, ContractStore, pack_vector


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
    if progress_callback:
        progress_callback("Extracting contract facts...")
    facts = [
        ContractFactRow(
            document_id=fact.document_id,
            system=fact.normalized_system,
            system_normalized=fact.normalized_system,
            attribute=fact.normalized_attribute,
            attribute_normalized=fact.normalized_attribute,
            value=fact.raw_value,
            evidence_text=fact.evidence_text,
            source_chunk_id=fact.source_chunk_id,
            page_start=fact.page,
            page_end=fact.page,
        )
        for fact in extract_contract_facts(chunks)
    ]

    if progress_callback:
        progress_callback("Building local embeddings...")
    embedder = HashingEmbedder()
    embeddings = {
        chunk.chunk_id: pack_vector(embedder.embed(f"{chunk.section_number or ''} {chunk.heading}\n{chunk.full_text}"))
        for chunk in chunks
    }
    features = build_chunk_features(chunks)

    if progress_callback:
        progress_callback("Writing SQLite index...")
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
        facts=facts,
        embeddings=embeddings,
        model_name=embedder.model_name,
        dimension=embedder.dimension,
    )
    return {
        "document_id": document_id,
        "page_count": page_count,
        "chunk_count": len(chunks),
        "fact_count": len(facts),
    }


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
    return len(features)
