from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from epc_smart_search.app_paths import DB_PATH
from epc_smart_search.assistant import validate_contract_store
from epc_smart_search.indexer import build_index
from epc_smart_search.priority_config import PriorityConfig, load_priority_config
from epc_smart_search.preflight import REBUILD_PDF_ENV_VAR, resolve_rebuild_pdf_path
from epc_smart_search.search_features import normalize_text
from epc_smart_search.storage import ContractStore


def rebuild_contract(
    pdf_path: Path,
    out_path: Path,
    *,
    version_label: str = "v1",
    priority_config: PriorityConfig | None = None,
) -> tuple[dict[str, int | str], str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Contract PDF not found: {pdf_path}")
    if out_path.exists():
        raise FileExistsError(f"Output database already exists: {out_path}")
    if out_path.resolve() == Path(DB_PATH).resolve():
        raise ValueError("Refusing to write over the live app database. Choose a fresh output path.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = build_index(pdf_path=pdf_path, db_path=out_path, version_label=version_label, priority_config=priority_config)
    validation = validate_contract_store(ContractStore(out_path))
    if not validation.ready:
        raise RuntimeError(validation.error or "Built contract database did not pass validation.")
    return result, validation.document_id or ""


def build_coverage_report(
    db_path: Path,
    *,
    priority_config: PriorityConfig | None = None,
) -> dict[str, object]:
    store = ContractStore(db_path)
    status = validate_contract_store(store)
    document_id = status.document_id or ""
    diagnostics = store.get_ingest_diagnostics(document_id) if document_id else []
    numeric_summary = {
        "chunks_with_numeric_evidence": store.get_numeric_feature_count(document_id) if document_id else 0,
        "blocks_with_numeric_evidence": store.get_numeric_block_count(document_id) if document_id else 0,
    }
    priority_summary = _build_priority_summary(store, document_id, priority_config)
    return {
        "document_id": document_id,
        "schema_version": store.get_metadata("search_schema_version"),
        "index": {
            "chunk_count": status.chunk_count,
            "block_count": status.block_count,
            "feature_count": status.feature_count,
            "page_text_count": status.page_text_count,
            "diagnostic_count": status.diagnostic_count,
            "embedding_count": status.embedding_count,
        },
        "numeric_coverage": numeric_summary,
        "priority_coverage": priority_summary,
        "semantic": {
            "semantic_model_name": status.semantic_model_name,
            "semantic_dimension": status.semantic_dimension,
            "semantic_index_ready": status.semantic_index_ready,
            "semantic_runtime_available": status.semantic_runtime_available,
            "semantic_ready": status.semantic_ready,
            "semantic_status_reason": status.semantic_status_reason,
        },
        "ingest_summary": store.get_ingest_diagnostic_summary(document_id) if document_id else {},
        "page_diagnostics": [
            {
                "page_num": int(row["page_num"]),
                "meaningful_chars": int(row["meaningful_chars"]),
                "word_count": int(row["word_count"]),
                "block_count": int(row["block_count"]),
                "short_line_count": int(row["short_line_count"]),
                "flags": str(row["flags"]).split(),
            }
            for row in diagnostics
        ],
        "validation": {
            "ready": status.ready,
            "error": status.error,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate a prebuilt EPC contract database.")
    parser.add_argument("--pdf", help=f"Path to the source contract PDF. If omitted, use {REBUILD_PDF_ENV_VAR}.")
    parser.add_argument("--out", required=True, help="Fresh output path for the rebuilt SQLite database.")
    parser.add_argument("--version-label", default="v1", help="Version label used for the generated document ID.")
    parser.add_argument("--priority-config", help="Optional JSON file describing priority sections and numeric focus terms.")
    parser.add_argument("--report-json", help="Optional path for a machine-readable rebuild coverage report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        pdf_path = resolve_rebuild_pdf_path(args.pdf)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out_path = Path(args.out).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve() if args.report_json else None
    try:
        priority_config = load_priority_config(args.priority_config)
        result, document_id = rebuild_contract(
            pdf_path,
            out_path,
            version_label=args.version_label,
            priority_config=priority_config,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(build_coverage_report(out_path, priority_config=priority_config), indent=2),
            encoding="utf-8",
        )

    print(f"Built contract database: {out_path}")
    print(f"Document ID: {document_id}")
    print(f"Pages: {result.get('page_count', '?')}")
    print(f"Chunks: {result.get('chunk_count', '?')}")
    print(f"Blocks: {result.get('block_count', '?')}")
    if report_path is not None:
        print(f"Coverage report: {report_path}")
    return 0


def _build_priority_summary(
    store: ContractStore,
    document_id: str,
    priority_config: PriorityConfig | None,
) -> dict[str, object]:
    rules = list(priority_config.priority_sections) if priority_config is not None else []
    if not document_id or not rules:
        return {
            "configured_sections": 0,
            "matched_sections": [],
            "missing_sections": [],
            "sections_without_numeric_windows": [],
            "sections_without_linked_block_evidence": [],
        }
    matched_sections: list[str] = []
    missing_sections: list[str] = []
    sections_without_numeric_windows: list[str] = []
    sections_without_linked_block_evidence: list[str] = []
    with store._connect() as connection:  # noqa: SLF001
        for rule in rules:
            label = normalize_text(rule.label)
            feature_rows = connection.execute(
                """
                SELECT f.chunk_id, f.numeric_text
                FROM chunk_search_features f
                WHERE f.document_id = ?
                  AND (' ' || f.priority_flags || ' ') LIKE ?
                """,
                (document_id, f"% {label} %"),
            ).fetchall()
            if not feature_rows:
                missing_sections.append(rule.label)
                continue
            matched_sections.append(rule.label)
            chunk_ids = [str(row["chunk_id"]) for row in feature_rows]
            if not any(str(row["numeric_text"] or "").strip() for row in feature_rows):
                sections_without_numeric_windows.append(rule.label)
            placeholders = ", ".join("?" for _ in chunk_ids)
            block_count_row = connection.execute(
                f"""
                SELECT COUNT(*) AS block_count
                FROM contract_blocks
                WHERE document_id = ?
                  AND parent_chunk_id IN ({placeholders})
                """,
                (document_id, *chunk_ids),
            ).fetchone()
            if int(block_count_row["block_count"] or 0) <= 0:
                sections_without_linked_block_evidence.append(rule.label)
    return {
        "configured_sections": len(rules),
        "matched_sections": matched_sections,
        "missing_sections": missing_sections,
        "sections_without_numeric_windows": sections_without_numeric_windows,
        "sections_without_linked_block_evidence": sections_without_linked_block_evidence,
    }


if __name__ == "__main__":
    raise SystemExit(main())
