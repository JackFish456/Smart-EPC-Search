from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from epc_smart_search.app_paths import WORKSPACE_ROOT, find_workspace_sensitive_artifacts, resolve_gemma_launch_spec, resolve_gemma_test_python
from gemma_runtime import infer_model_mode_from_config

REBUILD_PDF_ENV_VAR = "EPC_CONTRACT_PDF"
PREBUILT_DB_ENV_VAR = "EPC_PREBUILT_DB_PATH"
MODEL_DIR_ENV_VAR = "EPC_SMART_SEARCH_MODEL_DIR"
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
    issues.extend(_collect_local_ai_launch_warnings())
    return issues


def _collect_local_ai_launch_warnings() -> list[PreflightIssue]:
    launch_spec = resolve_gemma_launch_spec()
    if launch_spec.mode in {"disabled", "bundled_service"}:
        return []

    helper_python = launch_spec.service_path or resolve_gemma_test_python()
    if not launch_spec.available:
        return [
            PreflightIssue(
                severity="warning",
                code="local_ai_unavailable",
                message=(
                    (launch_spec.reason or f"Gemma helper Python was not found: {helper_python}")
                    + " Retrieval mode is still available."
                ),
            )
        ]

    if _python_import_check(helper_python, ("flask",)).returncode != 0:
        return [
            PreflightIssue(
                severity="warning",
                code="missing_flask_in_gemma_env",
                message=(
                    f"Gemma environment at {helper_python} cannot import Flask. "
                    "AI mode may not start, but retrieval mode is still available."
                ),
            )
        ]
    return []


def collect_package_preflight_issues(
    prebuilt_db_path: str | None,
    *,
    profile: str = "Lite",
    model_dir: str | None = None,
) -> list[PreflightIssue]:
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
                    f"Provide a prebuilt contract database via -PrebuiltDbPath or {PREBUILT_DB_ENV_VAR}. "
                    "Keep that file outside the workspace."
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
    if profile.strip().lower() == "ai":
        raw_model_dir = (model_dir or os.environ.get(MODEL_DIR_ENV_VAR, "")).strip()
        if not raw_model_dir:
            issues.append(
                PreflightIssue(
                    severity="error",
                    code="missing_model_dir",
                    message=(
                        f"Provide a bundled AI model directory via -ModelDir or {MODEL_DIR_ENV_VAR}. "
                        "Keep that folder outside the workspace."
                    ),
                )
            )
            return issues
        try:
            resolved_model = validate_external_artifact_path(raw_model_dir, label="Bundled AI model directory")
        except Exception as exc:
            issues.append(PreflightIssue(severity="error", code="invalid_model_dir", message=str(exc)))
            return issues
        if not resolved_model.is_dir():
            issues.append(
                PreflightIssue(
                    severity="error",
                    code="model_dir_not_directory",
                    message=f"Bundled AI model directory must be a folder: {resolved_model}",
                )
            )
            return issues
        config_path = resolved_model / "config.json"
        if not config_path.exists():
            issues.append(
                PreflightIssue(
                    severity="error",
                    code="missing_model_config",
                    message=f"Bundled AI model directory is missing config.json: {resolved_model}",
                )
            )
            return issues
        try:
            import json

            model_mode = infer_model_mode_from_config(json.loads(config_path.read_text(encoding="utf-8")))
        except Exception as exc:
            issues.append(
                PreflightIssue(
                    severity="error",
                    code="invalid_model_config",
                    message=f"Could not parse bundled AI model config at {config_path}: {exc}",
                )
            )
            return issues
        if model_mode != "text_only":
            issues.append(
                PreflightIssue(
                    severity="error",
                    code="model_not_text_only",
                    message=f"Bundled AI model must be a text-only Gemma checkpoint: {resolved_model}",
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
    parser.add_argument("--profile", default="Lite")
    parser.add_argument("--model-dir")
    args = parser.parse_args(argv)
    issues = (
        collect_launch_preflight_issues()
        if args.mode == "launch"
        else collect_package_preflight_issues(args.prebuilt_db, profile=args.profile, model_dir=args.model_dir)
    )
    return report_issues(issues)


def _python_import_check(python_executable: Path, modules: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    imports = "; ".join(f"import {module}" for module in modules)
    return subprocess.run([str(python_executable), "-c", imports], capture_output=True, text=True, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
