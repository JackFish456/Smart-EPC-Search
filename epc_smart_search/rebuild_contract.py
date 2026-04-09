from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate a prebuilt EPC contract database.")
    parser.add_argument("--pdf", help=f"Path to the source contract PDF. If omitted, use {REBUILD_PDF_ENV_VAR}.")
    parser.add_argument("--out", required=True, help="Fresh output path for the rebuilt SQLite database.")
    parser.add_argument("--version-label", default="v1", help="Version label used for the generated document ID.")
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
    try:
        result, document_id = rebuild_contract(pdf_path, out_path, version_label=args.version_label)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Built contract database: {out_path}")
    print(f"Document ID: {document_id}")
    print(f"Pages: {result.get('page_count', '?')}")
    print(f"Chunks: {result.get('chunk_count', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
