from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from epc_smart_search.app_paths import DB_PATH
from epc_smart_search.assistant import validate_contract_store
from epc_smart_search.indexer import build_index
from epc_smart_search.preflight import REBUILD_PDF_ENV_VAR, resolve_rebuild_pdf_path
from epc_smart_search.storage import ContractStore


def rebuild_contract(pdf_path: Path, out_path: Path, *, version_label: str = "v1") -> tuple[dict[str, int | str], str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Contract PDF not found: {pdf_path}")
    if out_path.exists():
        raise FileExistsError(f"Output database already exists: {out_path}")
    if out_path.resolve() == Path(DB_PATH).resolve():
        raise ValueError("Refusing to write over the live app database. Choose a fresh output path.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = build_index(pdf_path=pdf_path, db_path=out_path, version_label=version_label)
    validation = validate_contract_store(ContractStore(out_path))
    if not validation.ready:
        raise RuntimeError(validation.error or "Built contract database did not pass validation.")
    return result, validation.document_id or ""


def build_coverage_report(db_path: Path) -> dict[str, object]:
    store = ContractStore(db_path)
    status = validate_contract_store(store)
    document_id = status.document_id or ""
    diagnostics = store.get_ingest_diagnostics(document_id) if document_id else []
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
        "semantic": {
            "semantic_model_name": status.semantic_model_name,
            "semantic_dimension": status.semantic_dimension,
            "semantic_index_ready": status.semantic_index_ready,
            "semantic_runtime_available": status.semantic_runtime_available,
            "semantic_ready": status.semantic_ready,
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
        result, document_id = rebuild_contract(pdf_path, out_path, version_label=args.version_label)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(build_coverage_report(out_path), indent=2), encoding="utf-8")

    print(f"Built contract database: {out_path}")
    print(f"Document ID: {document_id}")
    print(f"Pages: {result.get('page_count', '?')}")
    print(f"Chunks: {result.get('chunk_count', '?')}")
    print(f"Blocks: {result.get('block_count', '?')}")
    if report_path is not None:
        print(f"Coverage report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
