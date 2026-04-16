import shutil
import tempfile
import uuid
from pathlib import Path

from epc_smart_search import app_paths


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
    assert "disabled" in (resolved.reason or "").lower()


def test_resolve_gemma_launch_spec_prefers_bundled_service_for_frozen_build(monkeypatch) -> None:
    base = _test_dir("frozen_ai")
    service_path = base / "ai_runtime" / "gemma_service.exe"
    model_dir = base / "models" / "gemma"
    service_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    service_path.write_text("", encoding="utf-8")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(app_paths, "is_frozen_app", lambda: True)
    monkeypatch.setattr(app_paths, "BUNDLED_GEMMA_SERVICE_PATH", service_path)
    monkeypatch.setattr(app_paths, "BUNDLED_MODEL_DIR", model_dir)

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is True
    assert resolved.mode == "bundled_service"
    assert resolved.service_path == service_path
    assert resolved.model_dir == model_dir


def test_resolve_gemma_launch_spec_rejects_missing_model_override(monkeypatch) -> None:
    fake_python = _test_dir("gemma_helper") / "python.exe"
    fake_python.write_text("", encoding="utf-8")

    monkeypatch.delenv(app_paths.AI_DISABLE_ENV_VAR, raising=False)
    monkeypatch.setenv(app_paths.MODEL_DIR_OVERRIDE_ENV_VAR, str(fake_python.parent / "missing-model"))
    monkeypatch.setattr(app_paths, "resolve_gemma_test_python", lambda: fake_python)

    resolved = app_paths.resolve_gemma_launch_spec()

    assert resolved.available is False
    assert resolved.mode == "external_python"
    assert "not found" in (resolved.reason or "").lower()


def _test_dir(label: str) -> Path:
    base = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"app_paths_{label}_{uuid.uuid4().hex[:8]}"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base
