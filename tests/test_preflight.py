import tempfile
import uuid
from pathlib import Path

import epc_smart_search.preflight as preflight


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


def test_collect_launch_preflight_issues_reports_missing_gemma_python(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "collect_workspace_artifact_warnings", lambda workspace_root=preflight.WORKSPACE_ROOT: [])
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(preflight, "GEMMA_TEST_PYTHON", Path(tempfile.gettempdir()) / "missing_gemma_python.exe")

    issues = preflight.collect_launch_preflight_issues()

    issue = next(issue for issue in issues if issue.code == "missing_gemma_python")
    assert issue.severity == "warning"
    assert "retrieval mode" in issue.message


def test_collect_package_preflight_issues_requires_external_prebuilt_db(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "collect_workspace_artifact_warnings", lambda workspace_root=preflight.WORKSPACE_ROOT: [])
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())

    issues = preflight.collect_package_preflight_issues("")

    assert any(issue.code == "missing_prebuilt_db" for issue in issues)


def _test_dir(label: str) -> Path:
    base = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"preflight_{label}_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    return base
