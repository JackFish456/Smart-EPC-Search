import shutil
import tempfile
import uuid
from pathlib import Path

from epc_smart_search import app_paths
from epc_smart_search.app_paths import HardwareCapability


def test_seed_preloaded_db_copies_snapshot(monkeypatch) -> None:
    base = _test_dir("copies")
    preload = base / "contract_store.prebuilt.db"
    preload.write_bytes(b"seed")
    target = base / "app" / "contract_store.db"

    monkeypatch.setattr(app_paths, "PRELOADED_DB_PATH", preload)
    monkeypatch.setattr(app_paths, "DB_PATH", target)

    copied = app_paths.seed_preloaded_db()

    assert copied is True
    assert target.read_bytes() == b"seed"


def test_seed_preloaded_db_skips_when_target_exists(monkeypatch) -> None:
    base = _test_dir("skips")
    preload = base / "contract_store.prebuilt.db"
    preload.write_bytes(b"seed")
    target = base / "app" / "contract_store.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"existing")

    monkeypatch.setattr(app_paths, "PRELOADED_DB_PATH", preload)
    monkeypatch.setattr(app_paths, "DB_PATH", target)

    copied = app_paths.seed_preloaded_db()

    assert copied is False
    assert target.read_bytes() == b"existing"


def test_get_app_data_root_skips_unwritable_localappdata(monkeypatch) -> None:
    base = _test_dir("fallback")
    local_root = base / "local"
    temp_root = base / "temp"
    home_root = base / "home"
    workspace_root = base / "workspace"

    monkeypatch.setenv("LOCALAPPDATA", str(local_root))
    monkeypatch.setattr(app_paths.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(app_paths, "WORKSPACE_ROOT", workspace_root)
    monkeypatch.setattr(app_paths.Path, "home", staticmethod(lambda: home_root))

    def fake_supports_sqlite(candidate: Path) -> bool:
        return candidate == temp_root / "EPC Smart Search"

    monkeypatch.setattr(app_paths, "_supports_sqlite", fake_supports_sqlite)

    chosen = app_paths.get_app_data_root()

    assert chosen == temp_root / "EPC Smart Search"


def test_resolve_gemma_launch_spec_respects_ai_disable(monkeypatch) -> None:
    monkeypatch.setenv(app_paths.AI_DISABLE_ENV_VAR, "1")

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is False
    assert resolved.mode == "disabled"
    assert resolved.tier == "lite"
    assert "disabled" in (resolved.reason or "").lower()


def test_resolve_gemma_launch_spec_prefers_ai_high_for_frozen_8gb_build(monkeypatch) -> None:
    base = _seed_frozen_ai_assets("frozen_high", include_min=True, include_high=True)
    service_path = base / "ai_runtime" / "gemma_service.exe"
    model_dir_min = base / "models" / "gemma_min"
    model_dir_high = base / "models" / "gemma_high"

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(app_paths.MODEL_DIR_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: True)
    monkeypatch.setattr(app_paths, "BUNDLED_GEMMA_SERVICE_PATH", service_path)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_MIN", model_dir_min)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_HIGH", model_dir_high)
    monkeypatch.setattr(
        app_paths,
        "detect_ai_hardware_capability",
        lambda torch_module=None: HardwareCapability("cuda_8gb_plus", 8.0, "8 GB CUDA GPU detected."),
    )

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is True
    assert resolved.mode == "bundled_service"
    assert resolved.tier == "ai_high"
    assert resolved.service_path == service_path
    assert resolved.selected_model_dir == model_dir_high


def test_resolve_gemma_launch_spec_prefers_ai_min_for_frozen_4gb_build(monkeypatch) -> None:
    base = _seed_frozen_ai_assets("frozen_min", include_min=True, include_high=True)
    service_path = base / "ai_runtime" / "gemma_service.exe"
    model_dir_min = base / "models" / "gemma_min"
    model_dir_high = base / "models" / "gemma_high"

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(app_paths.MODEL_DIR_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: True)
    monkeypatch.setattr(app_paths, "BUNDLED_GEMMA_SERVICE_PATH", service_path)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_MIN", model_dir_min)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_HIGH", model_dir_high)
    monkeypatch.setattr(
        app_paths,
        "detect_ai_hardware_capability",
        lambda torch_module=None: HardwareCapability("cuda_4gb_to_7gb", 4.0, "4 GB CUDA GPU detected."),
    )

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is True
    assert resolved.tier == "ai_min"
    assert resolved.selected_model_dir == model_dir_min


def test_resolve_gemma_launch_spec_falls_back_to_ai_min_when_high_missing_on_8gb_machine(monkeypatch) -> None:
    base = _seed_frozen_ai_assets("frozen_high_missing", include_min=True, include_high=False)
    service_path = base / "ai_runtime" / "gemma_service.exe"
    model_dir_min = base / "models" / "gemma_min"
    model_dir_high = base / "models" / "gemma_high"

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(app_paths.MODEL_DIR_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: True)
    monkeypatch.setattr(app_paths, "BUNDLED_GEMMA_SERVICE_PATH", service_path)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_MIN", model_dir_min)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_HIGH", model_dir_high)
    monkeypatch.setattr(
        app_paths,
        "detect_ai_hardware_capability",
        lambda torch_module=None: HardwareCapability("cuda_8gb_plus", 8.0, "8 GB CUDA GPU detected."),
    )

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is True
    assert resolved.tier == "ai_min"
    assert resolved.selected_model_dir == model_dir_min


def test_resolve_gemma_launch_spec_falls_back_to_lite_when_only_high_assets_exist_on_4gb_machine(monkeypatch) -> None:
    base = _seed_frozen_ai_assets("frozen_only_high", include_min=False, include_high=True)
    service_path = base / "ai_runtime" / "gemma_service.exe"
    model_dir_min = base / "models" / "gemma_min"
    model_dir_high = base / "models" / "gemma_high"

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(app_paths.MODEL_DIR_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: True)
    monkeypatch.setattr(app_paths, "BUNDLED_GEMMA_SERVICE_PATH", service_path)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_MIN", model_dir_min)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_HIGH", model_dir_high)
    monkeypatch.setattr(
        app_paths,
        "detect_ai_hardware_capability",
        lambda torch_module=None: HardwareCapability("cuda_4gb_to_7gb", 4.0, "4 GB CUDA GPU detected."),
    )

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is False
    assert resolved.tier == "lite"
    assert "only ai-high assets" in (resolved.reason or "").lower()
    assert resolved.selected_model_dir is None


def test_resolve_gemma_launch_spec_falls_back_to_lite_when_no_cuda_is_available(monkeypatch) -> None:
    fake_python = _test_dir("gemma_helper_no_cuda") / "python.exe"
    fake_python.write_text("", encoding="utf-8")

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(app_paths.MODEL_DIR_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: False)
    monkeypatch.setattr(app_paths, "resolve_gemma_test_python", lambda: fake_python)
    monkeypatch.setattr(
        app_paths,
        "probe_external_python_runtime",
        lambda helper_python: app_paths.ExternalPythonRuntimeProbe(
            capability=HardwareCapability("no_cuda", None, "No supported CUDA/NVIDIA GPU is available for AI mode."),
            default_model_dir=None,
            model_error="Model path does not exist.",
        ),
    )
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_MIN", _test_dir("unused_min"))
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_HIGH", _test_dir("unused_high"))

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is False
    assert resolved.tier == "lite"
    assert "cuda" in (resolved.reason or "").lower()


def test_resolve_gemma_launch_spec_rejects_missing_model_override(monkeypatch) -> None:
    fake_python = _test_dir("gemma_helper") / "python.exe"
    fake_python.write_text("", encoding="utf-8")

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.setenv(app_paths.MODEL_DIR_OVERRIDE_ENV_VAR, str(fake_python.parent / "missing-model"))
    monkeypatch.setattr(app_paths, "resolve_gemma_test_python", lambda: fake_python)
    monkeypatch.setattr(
        app_paths,
        "probe_external_python_runtime",
        lambda helper_python: app_paths.ExternalPythonRuntimeProbe(
            capability=HardwareCapability("cuda_8gb_plus", 8.0, "8 GB CUDA GPU detected."),
            default_model_dir=fake_python.parent,
            model_error=None,
        ),
    )

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is False
    assert resolved.mode == "external_python"
    assert resolved.tier == "lite"
    assert "not found" in (resolved.reason or "").lower()


def test_resolve_gemma_launch_spec_uses_helper_probe_for_external_python(monkeypatch) -> None:
    fake_python = _test_dir("gemma_helper_external") / "python.exe"
    fake_python.write_text("", encoding="utf-8")
    default_model_dir = _test_dir("gemma_helper_model")
    (default_model_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(app_paths.MODEL_DIR_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: False)
    monkeypatch.setattr(app_paths, "resolve_gemma_test_python", lambda: fake_python)
    monkeypatch.setattr(
        app_paths,
        "probe_external_python_runtime",
        lambda helper_python: app_paths.ExternalPythonRuntimeProbe(
            capability=HardwareCapability("cuda_8gb_plus", 8.0, "8 GB CUDA GPU detected."),
            default_model_dir=default_model_dir,
            model_error=None,
        ),
    )

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is True
    assert resolved.mode == "external_python"
    assert resolved.tier == "ai_high"
    assert resolved.service_path == fake_python
    assert resolved.selected_model_dir == default_model_dir


def test_resolve_gemma_launch_spec_valid_manual_override_bypasses_auto_selection(monkeypatch) -> None:
    base = _seed_frozen_ai_assets("override", include_min=True, include_high=True)
    override_model = _test_dir("override_model")
    service_path = base / "ai_runtime" / "gemma_service.exe"
    model_dir_min = base / "models" / "gemma_min"
    model_dir_high = base / "models" / "gemma_high"
    (override_model / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.setenv(app_paths.MODEL_DIR_OVERRIDE_ENV_VAR, str(override_model))
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: True)
    monkeypatch.setattr(app_paths, "BUNDLED_GEMMA_SERVICE_PATH", service_path)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_MIN", model_dir_min)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR_HIGH", model_dir_high)
    monkeypatch.setattr(
        app_paths,
        "detect_ai_hardware_capability",
        lambda torch_module=None: HardwareCapability("cuda_8gb_plus", 8.0, "8 GB CUDA GPU detected."),
    )

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is True
    assert resolved.tier == "ai_high"
    assert resolved.selected_model_dir == override_model.resolve()


def _test_dir(label: str) -> Path:
    base = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"app_paths_{label}_{uuid.uuid4().hex[:8]}"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _seed_frozen_ai_assets(label: str, *, include_min: bool, include_high: bool) -> Path:
    base = _test_dir(label)
    service_path = base / "ai_runtime" / "gemma_service.exe"
    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_path.write_text("", encoding="utf-8")
    if include_min:
        model_dir_min = base / "models" / "gemma_min"
        model_dir_min.mkdir(parents=True, exist_ok=True)
        (model_dir_min / "config.json").write_text("{}", encoding="utf-8")
    if include_high:
        model_dir_high = base / "models" / "gemma_high"
        model_dir_high.mkdir(parents=True, exist_ok=True)
        (model_dir_high / "config.json").write_text("{}", encoding="utf-8")
    return base
