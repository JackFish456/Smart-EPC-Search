import shutil
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


def _test_dir(label: str) -> Path:
    base = Path(".tmp_test") / f"app_paths_{label}_{uuid.uuid4().hex[:8]}"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base
