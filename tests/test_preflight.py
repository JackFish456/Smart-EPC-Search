import tempfile
import uuid
from pathlib import Path

import epc_smart_search.preflight as preflight
from epc_smart_search.app_paths import GemmaLaunchSpec


def test_collect_workspace_artifact_warnings_reports_sensitive_files() -> None:
    workspace = _test_dir("workspace")
    (workspace / "contract.pdf").write_bytes(b"%PDF-1.4\n")
    (workspace / "contract_store.prebuilt.db").write_bytes(b"SQLite format 3\x00payload")

    issues = preflight.collect_workspace_artifact_warnings(workspace)

    assert issues
    assert issues[0].severity == "warning"
    assert "contract.pdf" in issues[0].message


def test_resolve_rebuild_pdf_path_rejects_workspace_paths() -> None:
    workspace = _test_dir("workspace_pdf")
    pdf_path = workspace / "contract.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    try:
        preflight.resolve_rebuild_pdf_path(str(pdf_path), workspace_root=workspace)
    except ValueError as exc:
        assert "outside the workspace" in str(exc)
    else:
        raise AssertionError("Expected resolve_rebuild_pdf_path() to reject workspace PDFs")


def test_collect_launch_preflight_issues_warns_when_local_ai_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "collect_workspace_artifact_warnings", lambda workspace_root=preflight.WORKSPACE_ROOT: [])
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        preflight,
        "resolve_gemma_launch_spec",
        lambda: GemmaLaunchSpec(
            mode="external_python",
            service_path=Path(tempfile.gettempdir()) / "missing_gemma_python.exe",
            model_dir=None,
            available=False,
            reason="Gemma helper Python was not found for local AI mode.",
        ),
    )

    issues = preflight.collect_launch_preflight_issues()

    assert any(issue.code == "local_ai_unavailable" and issue.severity == "warning" for issue in issues)


def test_collect_launch_preflight_issues_warns_when_flask_missing_in_gemma_env(monkeypatch) -> None:
    helper_python = _external_db("gemma_helper").with_name("python.exe")
    helper_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(preflight, "collect_workspace_artifact_warnings", lambda workspace_root=preflight.WORKSPACE_ROOT: [])
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        preflight,
        "resolve_gemma_launch_spec",
        lambda: GemmaLaunchSpec(
            mode="external_python",
            service_path=helper_python,
            model_dir=None,
            available=True,
            reason=None,
        ),
    )

    class Result:
        returncode = 1

    monkeypatch.setattr(preflight, "_python_import_check", lambda python_executable, modules: Result())

    issues = preflight.collect_launch_preflight_issues()

    assert any(issue.code == "missing_flask_in_gemma_env" and issue.severity == "warning" for issue in issues)


def test_collect_package_preflight_issues_requires_external_prebuilt_db(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "collect_workspace_artifact_warnings", lambda workspace_root=preflight.WORKSPACE_ROOT: [])
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())

    issues = preflight.collect_package_preflight_issues("")

    assert any(issue.code == "missing_prebuilt_db" for issue in issues)


def test_collect_package_preflight_issues_ai_profile_requires_text_only_model_dir(monkeypatch) -> None:
    workspace = _test_dir("workspace_ai")
    prebuilt_db = _external_db("prebuilt")
    model_dir = _external_model_dir("ai_model")
    (model_dir / "config.json").write_text('{"architectures":["Gemma4ForCausalLM"]}', encoding="utf-8")

    monkeypatch.setattr(preflight, "collect_workspace_artifact_warnings", lambda workspace_root=preflight.WORKSPACE_ROOT: [])
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(preflight, "WORKSPACE_ROOT", workspace)

    issues = preflight.collect_package_preflight_issues(str(prebuilt_db), profile="AI", model_dir=str(model_dir))

    assert issues == []


def test_collect_package_preflight_issues_ai_profile_rejects_multimodal_model_dir(monkeypatch) -> None:
    workspace = _test_dir("workspace_multimodal")
    prebuilt_db = _external_db("prebuilt_multi")
    model_dir = _external_model_dir("ai_model_multimodal")
    (model_dir / "config.json").write_text(
        '{"architectures":["Gemma4ForConditionalGeneration"],"vision_config":{"model_type":"gemma4_vision"}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(preflight, "collect_workspace_artifact_warnings", lambda workspace_root=preflight.WORKSPACE_ROOT: [])
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(preflight, "WORKSPACE_ROOT", workspace)

    issues = preflight.collect_package_preflight_issues(str(prebuilt_db), profile="AI", model_dir=str(model_dir))

    assert any(issue.code == "model_not_text_only" for issue in issues)


def _test_dir(label: str) -> Path:
    base = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"preflight_{label}_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _external_db(label: str) -> Path:
    base = Path(tempfile.gettempdir()) / "epc_smart_search_external" / f"{label}_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    db_path = base / "contract_store.prebuilt.db"
    db_path.write_bytes(b"SQLite format 3\x00payload")
    return db_path


def _external_model_dir(label: str) -> Path:
    base = Path(tempfile.gettempdir()) / "epc_smart_search_external" / f"{label}_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    return base
