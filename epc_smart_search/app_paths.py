from __future__ import annotations

import importlib
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
AI_CAPABILITY_DISABLED = "disabled"
AI_CAPABILITY_NO_CUDA = "no_cuda"
AI_CAPABILITY_CUDA_UNDER_4GB = "cuda_under_4gb"
AI_CAPABILITY_CUDA_4GB_TO_7GB = "cuda_4gb_to_7gb"
AI_CAPABILITY_CUDA_8GB_PLUS = "cuda_8gb_plus"
AI_TIER_LITE = "lite"
AI_TIER_MIN = "ai_min"
AI_TIER_HIGH = "ai_high"


@dataclass(slots=True, frozen=True)
class GemmaLaunchSpec:
    mode: str
    service_path: Path | None
    model_dir: Path | None
    available: bool
    tier: str = AI_TIER_LITE
    capability_reason: str = ""
    reason: str | None = None

    @property
    def selected_model_dir(self) -> Path | None:
        return self.model_dir


@dataclass(slots=True, frozen=True)
class HardwareCapability:
    code: str
    vram_gb: float | None
    reason: str


@dataclass(slots=True, frozen=True)
class ModelSelection:
    tier: str
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
BUNDLED_MODEL_DIR_MIN = RESOURCE_ROOT / "models" / "gemma_min"
BUNDLED_MODEL_DIR_HIGH = RESOURCE_ROOT / "models" / "gemma_high"
BUNDLED_MODEL_DIR = BUNDLED_MODEL_DIR_MIN


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


def detect_ai_hardware_capability(torch_module=None) -> HardwareCapability:
    if torch_module is None:
        torch_module = _load_torch_module()
    if torch_module is None:
        return HardwareCapability(
            AI_CAPABILITY_NO_CUDA,
            None,
            "No supported CUDA/NVIDIA GPU is available for AI mode.",
        )
    cuda_module = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda_module, "is_available", None)
    if cuda_module is None or not callable(is_available) or not is_available():
        return HardwareCapability(
            AI_CAPABILITY_NO_CUDA,
            None,
            "No supported CUDA/NVIDIA GPU is available for AI mode.",
        )

    device_count_fn = getattr(cuda_module, "device_count", None)
    try:
        device_count = int(device_count_fn()) if callable(device_count_fn) else 1
    except Exception:
        device_count = 1

    measured_vram_gb: float | None = None
    for device_index in range(max(1, device_count)):
        try:
            properties = cuda_module.get_device_properties(device_index)
        except Exception:
            continue
        total_memory = float(getattr(properties, "total_memory", 0) or 0)
        if total_memory <= 0:
            continue
        device_vram_gb = total_memory / (1024 ** 3)
        measured_vram_gb = max(measured_vram_gb or 0.0, device_vram_gb)

    if measured_vram_gb is None:
        return HardwareCapability(
            AI_CAPABILITY_NO_CUDA,
            None,
            "CUDA GPU memory could not be measured on this machine.",
        )
    if measured_vram_gb < 4.0:
        return HardwareCapability(
            AI_CAPABILITY_CUDA_UNDER_4GB,
            measured_vram_gb,
            f"Detected CUDA GPU memory ({measured_vram_gb:.1f} GB) is below the 4 GB minimum for AI mode.",
        )
    if measured_vram_gb < 8.0:
        return HardwareCapability(
            AI_CAPABILITY_CUDA_4GB_TO_7GB,
            measured_vram_gb,
            f"Detected CUDA GPU memory ({measured_vram_gb:.1f} GB) supports AI-Min.",
        )
    return HardwareCapability(
        AI_CAPABILITY_CUDA_8GB_PLUS,
        measured_vram_gb,
        f"Detected CUDA GPU memory ({measured_vram_gb:.1f} GB) supports AI-High.",
    )


def resolve_model_selection(
    capability: HardwareCapability,
    *,
    disabled: bool = False,
    override_model_dir: Path | None = None,
    override_error: str | None = None,
    model_dir_min: Path | None = None,
    model_dir_high: Path | None = None,
) -> ModelSelection:
    if disabled:
        return ModelSelection(
            AI_TIER_LITE,
            None,
            False,
            "AI mode is disabled for this app instance.",
        )
    if override_model_dir is not None:
        if override_error:
            return ModelSelection(AI_TIER_LITE, override_model_dir, False, override_error)
        if capability.code == AI_CAPABILITY_CUDA_8GB_PLUS:
            return ModelSelection(AI_TIER_HIGH, override_model_dir, True)
        if capability.code == AI_CAPABILITY_CUDA_4GB_TO_7GB:
            return ModelSelection(AI_TIER_MIN, override_model_dir, True)
        return ModelSelection(AI_TIER_LITE, None, False, capability.reason)

    if capability.code == AI_CAPABILITY_CUDA_8GB_PLUS:
        if model_dir_high is not None:
            return ModelSelection(AI_TIER_HIGH, model_dir_high, True)
        if model_dir_min is not None:
            return ModelSelection(AI_TIER_MIN, model_dir_min, True)
        return ModelSelection(AI_TIER_LITE, None, False, "Bundled AI assets are not present in this build.")

    if capability.code == AI_CAPABILITY_CUDA_4GB_TO_7GB:
        if model_dir_min is not None:
            return ModelSelection(AI_TIER_MIN, model_dir_min, True)
        if model_dir_high is not None:
            return ModelSelection(
                AI_TIER_LITE,
                None,
                False,
                "This machine supports AI-Min, but only AI-High assets are present.",
            )
        return ModelSelection(AI_TIER_LITE, None, False, "Bundled AI assets are not present in this build.")

    return ModelSelection(AI_TIER_LITE, None, False, capability.reason)


def resolve_gemma_launch_spec() -> GemmaLaunchSpec:
    if is_ai_disabled():
        disabled_reason = "AI mode is disabled for this app instance."
        return GemmaLaunchSpec(
            mode="disabled",
            service_path=None,
            model_dir=None,
            available=False,
            tier=AI_TIER_LITE,
            capability_reason=disabled_reason,
            reason=disabled_reason,
        )

    capability = detect_ai_hardware_capability()
    frozen_build = is_frozen_app()
    helper_python = resolve_gemma_test_python()
    mode = "bundled_service" if frozen_build else "external_python"
    service_path = _resolve_service_path(mode, helper_python)

    model_override = resolve_model_dir_override()
    model_error = _validate_model_dir(model_override)
    selection = resolve_model_selection(
        capability,
        override_model_dir=model_override,
        override_error=model_error,
        model_dir_min=_existing_model_dir(BUNDLED_MODEL_DIR_MIN),
        model_dir_high=_existing_model_dir(BUNDLED_MODEL_DIR_HIGH),
    )

    if not selection.available:
        return GemmaLaunchSpec(
            mode=mode,
            service_path=service_path,
            model_dir=selection.model_dir,
            available=False,
            tier=selection.tier,
            capability_reason=capability.reason,
            reason=selection.reason,
        )

    service_error = _service_availability_error(mode, service_path)
    if service_error is not None:
        return GemmaLaunchSpec(
            mode=mode,
            service_path=service_path,
            model_dir=selection.model_dir,
            available=False,
            tier=selection.tier,
            capability_reason=capability.reason,
            reason=service_error,
        )

    return GemmaLaunchSpec(
        mode=mode,
        service_path=service_path,
        model_dir=selection.model_dir,
        available=True,
        tier=selection.tier,
        capability_reason=capability.reason,
        reason=None,
    )


def _load_torch_module():
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def _existing_model_dir(path: Path) -> Path | None:
    return path if path.exists() and path.is_dir() else None


def _resolve_service_path(mode: str, helper_python: Path) -> Path | None:
    if mode == "bundled_service":
        return BUNDLED_GEMMA_SERVICE_PATH if BUNDLED_GEMMA_SERVICE_PATH.exists() else None
    return helper_python


def _service_availability_error(mode: str, service_path: Path | None) -> str | None:
    if mode == "bundled_service":
        if service_path is None:
            return "AI runtime executable is missing from this bundle."
        return None
    if service_path is None or not service_path.exists():
        return "Gemma helper Python was not found for local AI mode."
    return None


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
