from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from epc_smart_search.app_paths import GEMMA_TEST_PYTHON, WORKSPACE_ROOT, find_workspace_sensitive_artifacts

REBUILD_PDF_ENV_VAR = "EPC_CONTRACT_PDF"
PREBUILT_DB_ENV_VAR = "EPC_PREBUILT_DB_PATH"
DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


@dataclass(slots=True, frozen=True)
class PreflightIssue:
    severity: str
    code: str
    message: str


def is_within_workspace(path: Path, workspace_root: Path = WORKSPACE_ROOT) -> bool:
    try:
        path.resolve().relative_to(workspace_root.resolve())
        return True
    except ValueError:
        return False


def validate_external_artifact_path(path_value: str | os.PathLike[str], *, label: str, workspace_root: Path = WORKSPACE_ROOT) -> Path:
    candidate = Path(path_value).expanduser().resolve()
    if is_within_workspace(candidate, workspace_root):
        raise ValueError(f"{label} must live outside the workspace: {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate


def collect_workspace_artifact_warnings(workspace_root: Path = WORKSPACE_ROOT) -> list[PreflightIssue]:
    artifacts = find_workspace_sensitive_artifacts(workspace_root)
    if not artifacts:
        return []
    preview = ", ".join(str(path.relative_to(workspace_root)) for path in artifacts[:4])
    if len(artifacts) > 4:
        preview += ", ..."
    return [
        PreflightIssue(
            severity="warning",
            code="workspace_sensitive_artifacts",
            message=(
                "Sensitive contract-bearing artifacts were detected under the workspace. "
                f"Move them outside the repo when possible. Found: {preview}"
            ),
        )
    ]


def collect_launch_preflight_issues() -> list[PreflightIssue]:
    issues = collect_workspace_artifact_warnings()
    for module_name in ("PySide6", "requests"):
        if importlib.util.find_spec(module_name) is None:
            issues.append(
                PreflightIssue(
                    severity="error",
                    code=f"missing_{module_name.lower()}",
                    message=f"Current app environment is missing required module '{module_name}'.",
                )
            )
    if not GEMMA_TEST_PYTHON.exists():
        issues.append(
            PreflightIssue(
                severity="error",
                code="missing_gemma_python",
                message=(
                    f"Gemma helper Python was not found at {GEMMA_TEST_PYTHON}. "
                    "Set EPC_GEMMA_PYTHON or EPC_GEMMA_TEST_ROOT before launching."
                ),
            )
        )
    elif _python_import_check(GEMMA_TEST_PYTHON, ("flask",)).returncode != 0:
        issues.append(
            PreflightIssue(
                severity="error",
                code="missing_flask_in_gemma_env",
                message=(
                    f"Gemma environment at {GEMMA_TEST_PYTHON} cannot import Flask. "
                    "Install the Gemma service requirements into that environment."
                ),
            )
        )
    return issues


def collect_package_preflight_issues(prebuilt_db_path: str | None) -> list[PreflightIssue]:
    issues = collect_workspace_artifact_warnings()
    if importlib.util.find_spec("PyInstaller") is None:
        issues.append(
            PreflightIssue(
                severity="error",
                code="missing_pyinstaller",
                message="Current packaging environment is missing PyInstaller.",
            )
        )
    raw_prebuilt = (prebuilt_db_path or os.environ.get(PREBUILT_DB_ENV_VAR, "")).strip()
    if not raw_prebuilt:
        issues.append(
            PreflightIssue(
                severity="error",
                code="missing_prebuilt_db",
                message=(
                    f"Provide a prebuilt contract database via package script flags, --prebuilt-db, "
                    f"or {PREBUILT_DB_ENV_VAR}. Keep that file outside the workspace."
                ),
            )
        )
        return issues
    try:
        resolved = validate_external_artifact_path(raw_prebuilt, label="Prebuilt contract database")
    except Exception as exc:
        issues.append(PreflightIssue(severity="error", code="invalid_prebuilt_db", message=str(exc)))
        return issues
    if resolved.suffix.lower() not in DB_SUFFIXES:
        issues.append(
            PreflightIssue(
                severity="error",
                code="invalid_prebuilt_db_suffix",
                message=f"Prebuilt contract database must be a SQLite-style file: {resolved}",
            )
        )
    elif resolved.stat().st_size <= 0:
        issues.append(
            PreflightIssue(
                severity="error",
                code="empty_prebuilt_db",
                message=f"Prebuilt contract database is empty: {resolved}",
            )
        )
    return issues


def resolve_rebuild_pdf_path(raw_path: str | None, *, workspace_root: Path = WORKSPACE_ROOT) -> Path:
    pdf_value = (raw_path or os.environ.get(REBUILD_PDF_ENV_VAR, "")).strip()
    if not pdf_value:
        raise ValueError(f"Provide --pdf or set {REBUILD_PDF_ENV_VAR} to a contract PDF outside the workspace.")
    candidate = validate_external_artifact_path(pdf_value, label="Contract PDF", workspace_root=workspace_root)
    if candidate.suffix.lower() != ".pdf":
        raise ValueError(f"Contract PDF must end in .pdf: {candidate}")
    return candidate


def report_issues(issues: list[PreflightIssue]) -> int:
    exit_code = 0
    for issue in issues:
        print(f"[{issue.severity.upper()}] {issue.message}")
        if issue.severity == "error":
            exit_code = 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EPC Smart Search preflight checks.")
    parser.add_argument("--mode", choices=("launch", "package"), required=True)
    parser.add_argument("--prebuilt-db")
    args = parser.parse_args(argv)
    issues = collect_launch_preflight_issues() if args.mode == "launch" else collect_package_preflight_issues(args.prebuilt_db)
    return report_issues(issues)


def _python_import_check(python_executable: Path, modules: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    imports = "; ".join(f"import {module}" for module in modules)
    return subprocess.run([str(python_executable), "-c", imports], capture_output=True, text=True, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
