from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_ROOT = WORKSPACE_ROOT.parent
ASSETS_DIR = WORKSPACE_ROOT / "assets"
CONTRACT_PATH = WORKSPACE_ROOT / "Clean Contract.pdf"
GEMMA_TEST_ROOT = DESKTOP_ROOT / "Gemma Test"
GEMMA_TEST_PYTHON = GEMMA_TEST_ROOT / ".venv" / "Scripts" / "python.exe"


def get_app_data_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    candidates: list[Path] = []
    if local_app_data:
        candidates.append(Path(local_app_data) / "EPC Smart Search")
    candidates.append(Path.home() / ".epc_smart_search")
    candidates.append(WORKSPACE_ROOT / ".epc_smart_search")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    raise OSError("Could not create an application data directory for EPC Smart Search.")


APP_DATA_ROOT = get_app_data_root()
DB_PATH = APP_DATA_ROOT / "contract_store.db"
LOG_PATH = APP_DATA_ROOT / "epc_smart_search.log"
OCR_CACHE_DIR = APP_DATA_ROOT / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
