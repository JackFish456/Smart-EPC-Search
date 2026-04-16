from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_ROOT = WORKSPACE_ROOT.parent
GEMMA_TEST_ROOT = DESKTOP_ROOT / "Gemma Test"
AI_DISABLE_ENV_VAR = "EPC_SMART_SEARCH_DISABLE_AI"
MODEL_DIR_OVERRIDE_ENV_VAR = "EPC_SMART_SEARCH_MODEL_DIR"


@dataclass(slots=True, frozen=True)
class GemmaLaunchSpec:
    mode: str
    service_path: Path | None
    model_dir: Path | None
    available: bool
    reason: str | None = None


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_install_root() -> Path:
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return WORKSPACE_ROOT


def get_resource_root() -> Path:
    if is_frozen_app():
        return Path(getattr(sys, "_MEIPASS", get_install_root())).resolve()
    return WORKSPACE_ROOT


INSTALL_ROOT = get_install_root()
RESOURCE_ROOT = get_resource_root()
ASSETS_DIR = RESOURCE_ROOT / "assets"
BUNDLED_AI_RUNTIME_DIR = INSTALL_ROOT / "ai_runtime"
BUNDLED_GEMMA_SERVICE_PATH = BUNDLED_AI_RUNTIME_DIR / "gemma_service.exe"
BUNDLED_MODEL_DIR = RESOURCE_ROOT / "models" / "gemma"


def resolve_gemma_test_python() -> Path:
    explicit_python = os.environ.get("EPC_GEMMA_PYTHON", "").strip()
    if explicit_python:
        return Path(explicit_python).expanduser()
    explicit_root = os.environ.get("EPC_GEMMA_TEST_ROOT", "").strip()
    if explicit_root:
        return Path(explicit_root).expanduser() / ".venv" / "Scripts" / "python.exe"
    return GEMMA_TEST_ROOT / ".venv" / "Scripts" / "python.exe"


def is_ai_disabled() -> bool:
    raw = os.environ.get(AI_DISABLE_ENV_VAR, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_model_dir_override() -> Path | None:
    raw = os.environ.get(MODEL_DIR_OVERRIDE_ENV_VAR, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _validate_model_dir(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.exists():
        return f"Configured AI model directory was not found: {path}"
    if not path.is_dir():
        return f"Configured AI model directory must be a folder: {path}"
    return None


def resolve_gemma_launch_spec() -> GemmaLaunchSpec:
    helper_python = resolve_gemma_test_python()
    if is_ai_disabled():
        return GemmaLaunchSpec(
            mode="disabled",
            service_path=None,
            model_dir=None,
            available=False,
            reason="AI mode is disabled for this app instance.",
        )

    model_override = resolve_model_dir_override()
    if model_override is not None:
        model_error = _validate_model_dir(model_override)
        if model_error:
            return GemmaLaunchSpec(
                mode="bundled_service" if is_frozen_app() else "external_python",
                service_path=BUNDLED_GEMMA_SERVICE_PATH if is_frozen_app() else helper_python,
                model_dir=model_override,
                available=False,
                reason=model_error,
            )
        if is_frozen_app():
            if BUNDLED_GEMMA_SERVICE_PATH.exists():
                return GemmaLaunchSpec(
                    mode="bundled_service",
                    service_path=BUNDLED_GEMMA_SERVICE_PATH,
                    model_dir=model_override,
                    available=True,
                )
            return GemmaLaunchSpec(
                mode="bundled_service",
                service_path=None,
                model_dir=model_override,
                available=False,
                reason="AI runtime executable is missing from this bundle.",
            )
        if helper_python.exists():
            return GemmaLaunchSpec(
                mode="external_python",
                service_path=helper_python,
                model_dir=model_override,
                available=True,
            )
        return GemmaLaunchSpec(
            mode="external_python",
            service_path=helper_python,
            model_dir=model_override,
            available=False,
            reason="Gemma helper Python was not found for local AI mode.",
        )

    if is_frozen_app():
        if BUNDLED_GEMMA_SERVICE_PATH.exists() and BUNDLED_MODEL_DIR.exists():
            return GemmaLaunchSpec(
                mode="bundled_service",
                service_path=BUNDLED_GEMMA_SERVICE_PATH,
                model_dir=BUNDLED_MODEL_DIR,
                available=True,
            )
        return GemmaLaunchSpec(
            mode="bundled_service",
            service_path=BUNDLED_GEMMA_SERVICE_PATH if BUNDLED_GEMMA_SERVICE_PATH.exists() else None,
            model_dir=BUNDLED_MODEL_DIR if BUNDLED_MODEL_DIR.exists() else None,
            available=False,
            reason="Bundled AI assets are not present in this build.",
        )

    if helper_python.exists():
        return GemmaLaunchSpec(
            mode="external_python",
            service_path=helper_python,
            model_dir=model_override,
            available=True,
        )

    return GemmaLaunchSpec(
        mode="external_python",
        service_path=helper_python,
        model_dir=model_override,
        available=False,
        reason="Gemma helper Python was not found. Retrieval mode is still available.",
    )


def _supports_sqlite(candidate: Path) -> bool:
    probe = candidate / ".sqlite_probe.db"
    try:
        connection = sqlite3.connect(probe)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS probe (value INTEGER)")
            connection.execute("INSERT INTO probe (value) VALUES (1)")
            connection.commit()
        finally:
            connection.close()
        return True
    except (OSError, sqlite3.Error):
        return False
    finally:
        try:
            if probe.exists():
                probe.unlink()
        except OSError:
            pass
        for suffix in ("-journal", "-wal", "-shm"):
            try:
                sidecar = Path(f"{probe}{suffix}")
                if sidecar.exists():
                    sidecar.unlink()
            except OSError:
                pass


def get_app_data_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    candidates: list[Path] = []
    if local_app_data:
        candidates.append(Path(local_app_data) / "EPC Smart Search")
    candidates.append(Path(tempfile.gettempdir()) / "EPC Smart Search")
    candidates.append(Path.home() / ".epc_smart_search")
    candidates.append(WORKSPACE_ROOT / ".epc_smart_search")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if _supports_sqlite(candidate):
            return candidate
    raise OSError("Could not create an application data directory for EPC Smart Search.")


APP_DATA_ROOT = get_app_data_root()
DB_PATH = APP_DATA_ROOT / "contract_store.db"
PRELOADED_DB_PATH = ASSETS_DIR / "contract_store.prebuilt.db"
LOG_PATH = APP_DATA_ROOT / "epc_smart_search.log"
OCR_CACHE_DIR = APP_DATA_ROOT / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
GEMMA_TEST_PYTHON = resolve_gemma_test_python()


def seed_preloaded_db() -> bool:
    if DB_PATH.exists() or not PRELOADED_DB_PATH.exists():
        return False
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRELOADED_DB_PATH, DB_PATH)
    return True


def find_workspace_sensitive_artifacts(workspace_root: Path = WORKSPACE_ROOT) -> list[Path]:
    ignored_dirs = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}
    db_suffixes = {".db", ".sqlite", ".sqlite3"}
    findings: list[Path] = []
    for candidate in workspace_root.rglob("*"):
        if any(part in ignored_dirs for part in candidate.parts):
            continue
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix == ".pdf":
            findings.append(candidate)
            continue
        if suffix in db_suffixes and candidate.stat().st_size > 0:
            findings.append(candidate)
    return sorted(findings)
