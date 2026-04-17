from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys

from epc_smart_search.app_paths import DB_PATH
from epc_smart_search.assistant import ContractAssistant, validate_contract_store
from epc_smart_search.indexer import build_index
from epc_smart_search.preflight import REBUILD_PDF_ENV_VAR, resolve_rebuild_pdf_path
from epc_smart_search.query_planner import RETRIEVAL_MODE_FACT_LOOKUP, RETRIEVAL_MODE_TOPIC_SUMMARY
from epc_smart_search.storage import ContractStore

SQLITE_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")
DEFAULT_EXACT_QUERY = "What is the configuration of the dew point heaters?"
DEFAULT_SUMMARY_QUERY = "Summarize the closed cooling water system"
DEFAULT_EXPECTED_SYSTEM = "dew point heaters"
DEFAULT_EXPECTED_ATTRIBUTE = "configuration"


@dataclass(slots=True, frozen=True)
class RebuildValidationSummary:
    database_path: Path
    document_id: str
    document_count: int
    page_count: int
    chunk_count: int
    feature_count: int
    fact_count: int
    embedding_count: int
    schema_version: str
    expected_system: str
    expected_attribute: str
    expected_values: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SmokeQuerySummary:
    label: str
    question: str
    retrieval_mode: str
    fact_lookup_attempted: bool
    fact_rows_returned: int
    fallback_reason: str | None
    used_expected_path: bool
    selected_bundle_id: str | None
    answer_text: str


class _NoGemma:
    def ask(self, *args, **kwargs):
        raise RuntimeError("Gemma disabled for rebuild smoke validation.")


def _sqlite_artifact_paths(db_path: Path) -> list[Path]:
    resolved = db_path.expanduser().resolve()
    return [resolved, *(Path(f"{resolved}{suffix}") for suffix in SQLITE_SIDECAR_SUFFIXES)]


def _delete_artifacts(paths: list[Path]) -> list[Path]:
    removed: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            continue
        if resolved.is_dir():
            raise IsADirectoryError(f"Refusing to delete directory during clean reindex: {resolved}")
        resolved.unlink()
        removed.append(resolved)
    return removed


def _count_rows(db_path: Path, sql: str, params: tuple[object, ...] = ()) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def rebuild_contract(
    pdf_path: Path,
    out_path: Path,
    *,
    version_label: str = "v1",
    clean_target: bool = False,
) -> tuple[dict[str, int | str], str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Contract PDF not found: {pdf_path}")
    if out_path.resolve() == Path(DB_PATH).resolve():
        raise ValueError("Refusing to write over the live app database. Choose a fresh output path.")
    if out_path.exists() and not clean_target:
        raise FileExistsError(f"Output database already exists: {out_path}")
    if clean_target:
        _delete_artifacts(_sqlite_artifact_paths(out_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = build_index(pdf_path=pdf_path, db_path=out_path, version_label=version_label)
    validation = validate_contract_store(ContractStore(out_path))
    if not validation.ready:
        raise RuntimeError(validation.error or "Built contract database did not pass validation.")
    return result, validation.document_id or ""


def validate_rebuilt_database(
    db_path: Path,
    *,
    expected_system: str,
    expected_attribute: str,
) -> RebuildValidationSummary:
    resolved_db = db_path.expanduser().resolve()
    if not resolved_db.exists():
        raise FileNotFoundError(f"Rebuilt database was not created: {resolved_db}")
    if resolved_db.stat().st_size <= 0:
        raise RuntimeError(f"Rebuilt database is empty: {resolved_db}")

    store = ContractStore(resolved_db)
    validation = validate_contract_store(store)
    if not validation.ready or not validation.document_id:
        raise RuntimeError(validation.error or "Rebuilt database failed validation.")

    document_id = validation.document_id
    document_count = _count_rows(resolved_db, "SELECT COUNT(*) FROM documents")
    page_count = _count_rows(
        resolved_db,
        "SELECT COUNT(*) FROM contract_pages WHERE document_id = ?",
        (document_id,),
    )
    chunk_count = store.get_chunk_count(document_id)
    feature_count = store.get_feature_count(document_id)
    fact_count = store.get_fact_count(document_id)
    embedding_count = _count_rows(
        resolved_db,
        """
        SELECT COUNT(*)
        FROM chunk_embeddings e
        JOIN contract_chunks c ON c.chunk_id = e.chunk_id
        WHERE c.document_id = ?
        """,
        (document_id,),
    )
    schema_version = store.get_metadata("search_schema_version") or ""
    expected_rows = store.lookup_facts_by_system_attribute(document_id, expected_system, expected_attribute)
    expected_values = tuple(row.value for row in expected_rows if row.value)

    if document_count <= 0:
        raise RuntimeError("Rebuilt database is missing the documents table payload.")
    if page_count <= 0:
        raise RuntimeError("Rebuilt database does not contain extracted pages.")
    if chunk_count <= 0:
        raise RuntimeError("Rebuilt database does not contain contract chunks.")
    if feature_count <= 0:
        raise RuntimeError("Rebuilt database does not contain chunk search features.")
    if feature_count != chunk_count:
        raise RuntimeError("Rebuilt database has mismatched chunk and feature counts.")
    if fact_count <= 0:
        raise RuntimeError("Rebuilt database does not contain structured contract facts.")
    if embedding_count <= 0:
        raise RuntimeError("Rebuilt database does not contain embeddings.")
    if embedding_count != chunk_count:
        raise RuntimeError("Rebuilt database has mismatched chunk and embedding counts.")
    if not expected_values:
        raise RuntimeError(
            "Rebuilt database did not contain the expected fact "
            f"({expected_system} / {expected_attribute})."
        )

    return RebuildValidationSummary(
        database_path=resolved_db,
        document_id=document_id,
        document_count=document_count,
        page_count=page_count,
        chunk_count=chunk_count,
        feature_count=feature_count,
        fact_count=fact_count,
        embedding_count=embedding_count,
        schema_version=schema_version,
        expected_system=expected_system,
        expected_attribute=expected_attribute,
        expected_values=expected_values,
    )


def install_live_database(source_db: Path, live_db: Path) -> list[Path]:
    resolved_source = source_db.expanduser().resolve()
    resolved_live = live_db.expanduser().resolve()
    temp_live = resolved_live.with_suffix(f"{resolved_live.suffix}.tmp")
    removed = _delete_artifacts(_sqlite_artifact_paths(resolved_live) + [temp_live])
    resolved_live.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_source, temp_live)
    temp_live.replace(resolved_live)
    return removed


def run_smoke_queries(
    db_path: Path,
    *,
    exact_query: str,
    summary_query: str,
) -> tuple[SmokeQuerySummary, SmokeQuerySummary]:
    assistant = ContractAssistant(db_path=db_path)
    if not assistant.is_index_ready():
        status = assistant.get_index_status()
        raise RuntimeError(status.error or "Rebuilt database is not queryable through the live pipeline.")

    no_gemma = _NoGemma()
    results: list[SmokeQuerySummary] = []
    for label, question in (("exact", exact_query), ("summary", summary_query)):
        trace = assistant.retriever.retrieve_trace(question)
        answer = assistant.answer_policy.answer(question, None, no_gemma)
        selected_bundle_id = trace.selected_bundle.bundle_id if trace.selected_bundle is not None else None
        if label == "exact":
            used_expected_path = (
                trace.plan.retrieval_mode == RETRIEVAL_MODE_FACT_LOOKUP
                and trace.fact_lookup_attempted
                and trace.fact_rows_returned > 0
                and trace.fact_hit is not None
                and trace.fallback_reason not in {"fact_lookup_miss", "no_fact_rows"}
            )
            if not used_expected_path:
                raise RuntimeError(
                    "Exact smoke query did not return a grounded fact hit. "
                    "Observed "
                    f"retrieval_mode={trace.plan.retrieval_mode}, "
                    f"fact_lookup_attempted={trace.fact_lookup_attempted}, "
                    f"fact_rows_returned={trace.fact_rows_returned}, "
                    f"fallback_reason={trace.fallback_reason}"
                )
        else:
            used_expected_path = (
                trace.plan.retrieval_mode == RETRIEVAL_MODE_TOPIC_SUMMARY
                and not trace.fact_lookup_attempted
                and trace.selected_bundle is not None
            )
            if not used_expected_path:
                raise RuntimeError(
                    "Summary smoke query did not route through the broad retrieval path. "
                    f"Observed retrieval mode: {trace.plan.retrieval_mode}"
                )
        if answer.refused:
            raise RuntimeError(f"Smoke query was refused: {question}")
        results.append(
            SmokeQuerySummary(
                label=label,
                question=question,
                retrieval_mode=trace.plan.retrieval_mode,
                fact_lookup_attempted=trace.fact_lookup_attempted,
                fact_rows_returned=trace.fact_rows_returned,
                fallback_reason=trace.fallback_reason,
                used_expected_path=used_expected_path,
                selected_bundle_id=selected_bundle_id,
                answer_text=answer.text,
            )
        )
    return results[0], results[1]


def run_clean_reindex(
    pdf_path: Path,
    out_path: Path,
    *,
    version_label: str = "v1",
    clean_target: bool = False,
    install_live_db_flag: bool = False,
    expected_system: str = DEFAULT_EXPECTED_SYSTEM,
    expected_attribute: str = DEFAULT_EXPECTED_ATTRIBUTE,
    exact_query: str = DEFAULT_EXACT_QUERY,
    summary_query: str = DEFAULT_SUMMARY_QUERY,
) -> dict[str, object]:
    removed_artifacts: list[Path] = []
    if clean_target:
        removed_artifacts.extend(_delete_artifacts(_sqlite_artifact_paths(out_path)))
    result, document_id = rebuild_contract(
        pdf_path,
        out_path,
        version_label=version_label,
        clean_target=False,
    )
    output_validation = validate_rebuilt_database(
        out_path,
        expected_system=expected_system,
        expected_attribute=expected_attribute,
    )

    active_db_path = out_path.expanduser().resolve()
    live_validation = None
    if install_live_db_flag:
        removed_artifacts.extend(install_live_database(active_db_path, Path(DB_PATH)))
        active_db_path = Path(DB_PATH).expanduser().resolve()
        live_validation = validate_rebuilt_database(
            active_db_path,
            expected_system=expected_system,
            expected_attribute=expected_attribute,
        )

    exact_smoke, summary_smoke = run_smoke_queries(
        active_db_path,
        exact_query=exact_query,
        summary_query=summary_query,
    )
    if clean_target:
        removed_artifacts = [
            *removed_artifacts,
            *[path for path in _sqlite_artifact_paths(out_path) if path.resolve() != active_db_path],
        ]
    return {
        "result": result,
        "document_id": document_id,
        "output_validation": output_validation,
        "live_validation": live_validation,
        "smoke_db_path": active_db_path,
        "removed_artifacts": tuple(dict.fromkeys(removed_artifacts)),
        "smoke_queries": (exact_smoke, summary_smoke),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build, validate, and optionally install a clean EPC contract database.")
    parser.add_argument("--pdf", help=f"Path to the source contract PDF. If omitted, use {REBUILD_PDF_ENV_VAR}.")
    parser.add_argument("--out", required=True, help="Target path for the rebuilt SQLite database.")
    parser.add_argument("--version-label", default="v1", help="Version label used for the generated document ID.")
    parser.add_argument(
        "--clean-target",
        action="store_true",
        help="Delete the target database and SQLite sidecars before rebuilding so the command is repeatable.",
    )
    parser.add_argument(
        "--install-live-db",
        action="store_true",
        help=f"After validation, replace the live runtime database at {Path(DB_PATH).expanduser().resolve()}.",
    )
    parser.add_argument("--expected-system", default=DEFAULT_EXPECTED_SYSTEM, help="System name used for fact validation.")
    parser.add_argument("--expected-attribute", default=DEFAULT_EXPECTED_ATTRIBUTE, help="Attribute used for fact validation.")
    parser.add_argument("--exact-query", default=DEFAULT_EXACT_QUERY, help="Exact-value smoke query to run after rebuild.")
    parser.add_argument("--summary-query", default=DEFAULT_SUMMARY_QUERY, help="Broad summary smoke query to run after rebuild.")
    return parser


def _build_next_command(args: argparse.Namespace, pdf_path: Path, out_path: Path) -> str:
    command = [
        "python",
        "-m",
        "epc_smart_search.rebuild_contract",
        "--pdf",
        str(pdf_path),
        "--out",
        str(out_path),
        "--version-label",
        str(args.version_label),
        "--expected-system",
        str(args.expected_system),
        "--expected-attribute",
        str(args.expected_attribute),
        "--exact-query",
        str(args.exact_query),
        "--summary-query",
        str(args.summary_query),
    ]
    if args.clean_target:
        command.append("--clean-target")
    if args.install_live_db:
        command.append("--install-live-db")
    return subprocess.list2cmdline(command)


def _print_validation_summary(title: str, summary: RebuildValidationSummary) -> None:
    print(title)
    print(f"  Database: {summary.database_path}")
    print(f"  Document ID: {summary.document_id}")
    print(f"  Schema Version: {summary.schema_version}")
    print(
        "  Row Counts: "
        f"documents={summary.document_count}, "
        f"pages={summary.page_count}, "
        f"chunks={summary.chunk_count}, "
        f"features={summary.feature_count}, "
        f"facts={summary.fact_count}, "
        f"embeddings={summary.embedding_count}"
    )
    print(
        "  Expected Fact: "
        f"{summary.expected_system} / {summary.expected_attribute} -> {', '.join(summary.expected_values)}"
    )


def _print_smoke_summary(summary: SmokeQuerySummary) -> None:
    print(f"Smoke Query [{summary.label}]")
    print(f"  Question: {summary.question}")
    print(f"  Retrieval Mode: {summary.retrieval_mode}")
    print(f"  Fact Lookup Attempted: {'yes' if summary.fact_lookup_attempted else 'no'}")
    print(f"  Fact Rows Returned: {summary.fact_rows_returned}")
    print(f"  Fallback Reason: {summary.fallback_reason or 'none'}")
    print(f"  Expected Path Used: {'yes' if summary.used_expected_path else 'no'}")
    print(f"  Selected Bundle: {summary.selected_bundle_id or 'n/a'}")
    print(f"  Answer: {summary.answer_text}")


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
        report = run_clean_reindex(
            pdf_path,
            out_path,
            version_label=args.version_label,
            clean_target=bool(args.clean_target),
            install_live_db_flag=bool(args.install_live_db),
            expected_system=args.expected_system,
            expected_attribute=args.expected_attribute,
            exact_query=args.exact_query,
            summary_query=args.summary_query,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    result = report["result"]
    output_validation = report["output_validation"]
    live_validation = report["live_validation"]
    exact_smoke, summary_smoke = report["smoke_queries"]
    removed_artifacts: tuple[Path, ...] = report["removed_artifacts"]

    print(f"Built contract database: {out_path}")
    print(f"Document ID: {report['document_id']}")
    print(f"Pages: {result.get('page_count', '?')}")
    print(f"Chunks: {result.get('chunk_count', '?')}")
    print(f"Facts: {result.get('fact_count', '?')}")
    print(
        "Fact Extraction: "
        f"chunks_processed={result.get('fact_extraction_chunk_count', '?')}, "
        f"fact_candidates={result.get('fact_candidate_count', '?')}, "
        f"fact_rows_inserted={result.get('fact_rows_inserted', '?')}"
    )
    print(
        "SQLite Writes: "
        f"db_path={result.get('db_path', '?')}, "
        f"chunk_rows_inserted={result.get('chunk_rows_inserted', '?')}, "
        f"feature_rows_inserted={result.get('feature_rows_inserted', '?')}, "
        f"embedding_rows_inserted={result.get('embedding_rows_inserted', '?')}"
    )
    _print_validation_summary("Validation [output]", output_validation)
    if live_validation is not None:
        _print_validation_summary("Validation [live]", live_validation)
    print("Removed Artifacts:")
    if removed_artifacts:
        for path in removed_artifacts:
            print(f"  {path}")
    else:
        print("  none")
    print(f"Smoke Database: {report['smoke_db_path']}")
    _print_smoke_summary(exact_smoke)
    _print_smoke_summary(summary_smoke)
    print("Run Next Time:")
    print(f"  {_build_next_command(args, pdf_path, out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
