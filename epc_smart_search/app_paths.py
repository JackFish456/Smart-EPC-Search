from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_ROOT = WORKSPACE_ROOT.parent
ASSETS_DIR = WORKSPACE_ROOT / "assets"
GEMMA_TEST_ROOT = DESKTOP_ROOT / "Gemma Test"


def _default_gemma_venv_python(gemma_root: Path) -> Path:
    if sys.platform == "win32":
        return gemma_root / ".venv" / "Scripts" / "python.exe"
    return gemma_root / ".venv" / "bin" / "python"


def resolve_gemma_test_python() -> Path:
    explicit_python = os.environ.get("EPC_GEMMA_PYTHON", "").strip()
    if explicit_python:
        return Path(explicit_python).expanduser()
    explicit_root = os.environ.get("EPC_GEMMA_TEST_ROOT", "").strip()
    if explicit_root:
        return _default_gemma_venv_python(Path(explicit_root).expanduser())
    return _default_gemma_venv_python(GEMMA_TEST_ROOT)


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
SEMANTIC_MODEL_PATH = ASSETS_DIR / "semantic_model.json"
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
