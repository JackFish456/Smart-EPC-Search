from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_ROOT = WORKSPACE_ROOT.parent
ASSETS_DIR = WORKSPACE_ROOT / "assets"
CONTRACT_PATH = WORKSPACE_ROOT / "Clean Contract.pdf"
GEMMA_TEST_ROOT = DESKTOP_ROOT / "Gemma Test"
GEMMA_TEST_PYTHON = GEMMA_TEST_ROOT / ".venv" / "Scripts" / "python.exe"


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


def seed_preloaded_db() -> bool:
    if DB_PATH.exists() or not PRELOADED_DB_PATH.exists():
        return False
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRELOADED_DB_PATH, DB_PATH)
    return True
