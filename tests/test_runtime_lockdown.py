import inspect
import json
import uuid
import html
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import epc_smart_search.assistant as assistant_module
import epc_smart_search.rebuild_contract as rebuild_contract_module
import epc_smart_search.ui.avatar_window as avatar_window_module
import epc_smart_search.ui.chat_dialog as chat_dialog_module
import epc_smart_search_app
from epc_smart_search.assistant import INTERNAL_REBUILD_ERROR
from epc_smart_search.assistant import ContractAssistant
from epc_smart_search.assistant import describe_semantic_status
from epc_smart_search.assistant import IndexValidationResult
from epc_smart_search.assistant import validate_contract_store
from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.config import GREETING
from epc_smart_search.ocr_support import PageText
from epc_smart_search.priority_config import PriorityConfig, PrioritySectionRule
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore, build_block_records, pack_vector
from epc_smart_search.ui.avatar_window import AvatarWindow
from epc_smart_search.ui.chat_dialog import CONTRACT_DATA_UNAVAILABLE_MESSAGE
from epc_smart_search.ui.chat_dialog import CONTRACT_DATA_UNAVAILABLE_STATUS
from epc_smart_search.ui.chat_dialog import ContractChatDialog


def test_validate_contract_store_accepts_seeded_db_without_pdf_dependency() -> None:
    store = _seed_store(_memory_db_uri("valid"))

    status = validate_contract_store(store)

    assert status.ready is True
    assert status.error is None
    assert status.chunk_count == 1
    assert status.feature_count == 1
    assert status.semantic_index_ready is False
    assert status.semantic_runtime_available is False
    assert status.semantic_ready is False
    assert status.semantic_status_reason == "missing_vectors"


def test_validate_contract_store_rejects_wrong_schema_version() -> None:
    store = _seed_store(_memory_db_uri("schema"))
    with store._connect() as connection:  # noqa: SLF001
        connection.execute(
            "UPDATE app_metadata SET value = ? WHERE key = 'search_schema_version'",
            ("999",),
        )
        connection.commit()

    status = validate_contract_store(store)

    assert status.ready is False
    assert "schema" in (status.error or "").lower()


def test_validate_contract_store_rejects_missing_features() -> None:
    store = _seed_store(_memory_db_uri("no_features"), with_features=False)

    status = validate_contract_store(store)

    assert status.ready is False
    assert "missing search features" in (status.error or "").lower()


def test_validate_contract_store_rejects_missing_page_evidence() -> None:
    store = _seed_store(_memory_db_uri("no_pages"))
    with store._connect() as connection:  # noqa: SLF001
        connection.execute("DELETE FROM contract_pages WHERE document_id = ?", ("doc1",))
        connection.commit()

    status = validate_contract_store(store)

    assert status.ready is False
    assert "page evidence" in (status.error or "").lower()


def test_validate_contract_store_rejects_missing_block_coverage() -> None:
    store = _seed_store(_memory_db_uri("no_blocks"))
    with store._connect() as connection:  # noqa: SLF001
        connection.execute("DELETE FROM contract_blocks WHERE document_id = ?", ("doc1",))
        connection.commit()

    status = validate_contract_store(store)

    assert status.ready is False
    assert "block-level search coverage" in (status.error or "").lower()


def test_validate_contract_store_rejects_missing_ingest_diagnostics() -> None:
    store = _seed_store(_memory_db_uri("no_diagnostics"))
    with store._connect() as connection:  # noqa: SLF001
        connection.execute("DELETE FROM page_ingest_diagnostics WHERE document_id = ?", ("doc1",))
        connection.commit()

    status = validate_contract_store(store)

    assert status.ready is False
    assert "ingest diagnostics" in (status.error or "").lower()


def test_validate_contract_store_backfills_legacy_block_coverage(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_contract_store.db"
    store = _seed_store(db_path)
    with store._connect() as connection:  # noqa: SLF001
        connection.execute("DELETE FROM contract_blocks WHERE document_id = ?", ("doc1",))
        connection.execute("DELETE FROM page_ingest_diagnostics WHERE document_id = ?", ("doc1",))
        connection.commit()

    reopened_store = ContractStore(db_path)
    status = validate_contract_store(reopened_store)

    assert reopened_store.get_block_count("doc1") >= 1
    assert reopened_store.get_ingest_diagnostic_count("doc1") == reopened_store.get_page_text_count("doc1")
    assert status.ready is True
    assert status.error is None


def test_contract_assistant_build_index_is_internal_only() -> None:
    assistant = ContractAssistant.__new__(ContractAssistant)

    try:
        assistant.build_index()
    except RuntimeError as exc:
        assert str(exc) == INTERNAL_REBUILD_ERROR
    else:
        raise AssertionError("Expected ContractAssistant.build_index() to be disabled")


def test_chat_dialog_locks_input_when_contract_data_is_unavailable() -> None:
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    class FakeAssistant:
        def is_index_ready(self) -> bool:
            return False

        def get_index_status(self) -> IndexValidationResult:
            return IndexValidationResult(False, "Bundled contract data is missing.", None)

    dialog = ContractChatDialog(FakeAssistant())
    dialog._ensure_index_ready()  # noqa: SLF001

    assert app is not None
    assert dialog._input.isEnabled() is False  # noqa: SLF001
    assert dialog._send_button.isEnabled() is False  # noqa: SLF001
    assert dialog._status.text() == CONTRACT_DATA_UNAVAILABLE_STATUS  # noqa: SLF001
    last_label = dialog._content_layout.itemAt(dialog._content_layout.count() - 1).widget()  # noqa: SLF001
    assert CONTRACT_DATA_UNAVAILABLE_MESSAGE.split(".")[0] in last_label.text()


def test_chat_dialog_clears_history_when_closed() -> None:
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    class FakeAssistant:
        def is_index_ready(self) -> bool:
            return True

        def get_index_status(self) -> IndexValidationResult:
            return IndexValidationResult(True, None, "doc1")

    dialog = ContractChatDialog(FakeAssistant())
    dialog._append_message("user", "Show me the indemnity clause.")  # noqa: SLF001
    dialog._append_message("assistant", "Section 12\n\nContract text: Indemnity applies here.")  # noqa: SLF001

    dialog.close()
    app.processEvents()

    assert dialog._content_layout.count() == 1  # noqa: SLF001
    assert dialog._message_count == 0  # noqa: SLF001
    assert dialog._context_low_locked is False  # noqa: SLF001
    assert dialog._deep_think_button.isChecked() is False  # noqa: SLF001
    first_label = dialog._content_layout.itemAt(0).widget()  # noqa: SLF001
    assert html.escape(GREETING.split(".")[0]) in first_label.text()


def test_describe_semantic_status_reports_lexical_only_mode() -> None:
    status = IndexValidationResult(
        True,
        None,
        "doc1",
        semantic_index_ready=True,
        semantic_runtime_available=False,
        semantic_ready=False,
        semantic_status_reason="missing_runtime_model",
    )

    message = describe_semantic_status(status)

    assert message is not None
    assert "lexical-only mode" in message.lower()


def test_chat_dialog_formats_summary_markdown_like_content() -> None:
    rendered = ContractChatDialog._format_assistant_message(
        "## Overview\n"
        "- **Contractor** shall provide the station.\n"
        "- Pressure must remain at `450 psi`.\n\n"
        "Dependencies:\n"
        "1. Customer provides gas.\n"
        "2. Contractor installs controls."
    )

    assert "Overview" in rendered
    assert "<ul" in rendered
    assert "<ol" in rendered
    assert "font-weight:700;'>Contractor" in rendered
    assert "<code" in rendered
    assert "## Overview" not in rendered
    assert "- **Contractor**" not in rendered


def test_chat_dialog_deep_think_toggle_is_session_local(monkeypatch) -> None:
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    captured: dict[str, object] = {}

    class FakeSignal:
        def connect(self, callback) -> None:
            return None

    class FakeWorker:
        def __init__(
            self,
            assistant,
            question: str,
            request_token: int,
            history=None,
            *,
            deep_think: bool = False,
        ) -> None:
            captured["question"] = question
            captured["request_token"] = request_token
            captured["deep_think"] = deep_think
            self.finished = FakeSignal()
            self.failed = FakeSignal()

        def deleteLater(self) -> None:
            return None

        def start(self) -> None:
            captured["started"] = True

    class FakeAssistant:
        def is_index_ready(self) -> bool:
            return True

        def get_index_status(self) -> IndexValidationResult:
            return IndexValidationResult(True, None, "doc1")

    monkeypatch.setattr(chat_dialog_module, "AskWorker", FakeWorker)

    dialog = ContractChatDialog(FakeAssistant())

    assert dialog._deep_think_button.isChecked() is False  # noqa: SLF001

    dialog._input.setText("What does the contract say about fuel gas supply?")  # noqa: SLF001
    dialog._deep_think_button.setChecked(True)  # noqa: SLF001
    dialog._on_send()  # noqa: SLF001

    assert app is not None
    assert captured["started"] is True
    assert captured["deep_think"] is True
    assert dialog._deep_think_button.isEnabled() is False  # noqa: SLF001
    assert dialog._pending_thinking_label == "Deep Thinking"  # noqa: SLF001

    dialog.close()
    app.processEvents()

    assert dialog._deep_think_button.isChecked() is False  # noqa: SLF001


def test_customer_ui_no_longer_exposes_rebuild_entry_points() -> None:
    assert not hasattr(AvatarWindow, "rebuild_index")
    assert not hasattr(ContractChatDialog, "_rebuild_index")
    assert "Rebuild Contract Index" not in inspect.getsource(avatar_window_module)
    assert "Rebuild Contract Index" not in inspect.getsource(epc_smart_search_app)


def test_show_chat_dialog_restores_and_raises_existing_window() -> None:
    class FakeDialog:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object | None]] = []
            self._minimized = True
            self._state = Qt.WindowState.WindowMinimized

        def isMinimized(self) -> bool:
            return self._minimized

        def windowState(self):
            return self._state

        def setWindowState(self, state) -> None:
            self.calls.append(("setWindowState", state))
            self._state = state
            self._minimized = False

        def show(self) -> None:
            self.calls.append(("show", None))

        def raise_(self) -> None:
            self.calls.append(("raise_", None))

        def activateWindow(self) -> None:
            self.calls.append(("activateWindow", None))

    dialog = FakeDialog()

    epc_smart_search_app._show_chat_dialog(dialog)  # noqa: SLF001

    assert dialog.calls[0][0] == "setWindowState"
    assert dialog.calls[1:] == [("show", None), ("raise_", None), ("activateWindow", None)]


def test_rebuild_cli_creates_and_validates_a_fresh_database(monkeypatch) -> None:
    temp_root = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"runtime_lockdown_{uuid.uuid4().hex[:8]}"
    pdf_path = temp_root / f"contract_{uuid.uuid4().hex[:8]}.pdf"
    out_path = temp_root / f"contract_store_{uuid.uuid4().hex[:8]}.db"
    report_path = temp_root / f"contract_store_{uuid.uuid4().hex[:8]}.report.json"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")
    memory_store = _seed_store(_memory_db_uri("cli"))

    def fake_build_index(*, pdf_path, db_path, version_label="v1", priority_config=None, progress_callback=None):
        Path(db_path).write_bytes(b"stub")
        return {"document_id": "doc1", "page_count": 1, "chunk_count": 1}

    monkeypatch.setattr(rebuild_contract_module, "build_index", fake_build_index)
    monkeypatch.setattr(rebuild_contract_module, "ContractStore", lambda path: memory_store)

    exit_code = rebuild_contract_module.main(
        ["--pdf", str(pdf_path), "--out", str(out_path), "--report-json", str(report_path)]
    )

    assert exit_code == 0
    assert out_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["document_id"] == "doc1"
    assert report["index"]["block_count"] >= 1


def test_build_coverage_report_includes_optional_semantic_status(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "semantic_report.db"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="1",
        heading="General",
        full_text="Contractor shall perform the work.",
        page_start=1,
        page_end=1,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc123",
        page_count=1,
        chunks=[chunk],
        pages=[PageText(page_num=1, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
        embeddings={"chunk1": pack_vector([1.0, 0.0, 0.0])},
        model_name="test-semantic",
        dimension=3,
    )

    class MissingRuntimeEmbedder:
        model_name = None
        dimension = None

        def is_available(self) -> bool:
            return False

    monkeypatch.setattr(assistant_module, "LocalEmbedder", lambda: MissingRuntimeEmbedder())

    report = rebuild_contract_module.build_coverage_report(db_path)

    assert report["index"]["embedding_count"] == 1
    assert report["semantic"]["semantic_index_ready"] is True
    assert report["semantic"]["semantic_runtime_available"] is False
    assert report["semantic"]["semantic_ready"] is False
    assert report["semantic"]["semantic_model_name"] == "test-semantic"
    assert report["semantic"]["semantic_status_reason"] == "missing_runtime_model"


def test_validate_contract_store_reports_runtime_model_mismatch(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "semantic_mismatch.db"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="1",
        heading="General",
        full_text="Contractor shall perform the work.",
        page_start=1,
        page_end=1,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc123",
        page_count=1,
        chunks=[chunk],
        pages=[PageText(page_num=1, text=chunk.full_text, ocr_used=False)],
        features=build_chunk_features([chunk]),
        embeddings={"chunk1": pack_vector([1.0, 0.0, 0.0])},
        model_name="indexed-semantic",
        dimension=3,
    )

    class FakeEmbedder:
        model_name = "runtime-semantic"
        dimension = 3

        def is_available(self) -> bool:
            return True

    monkeypatch.setattr(assistant_module, "LocalEmbedder", lambda: FakeEmbedder())

    status = validate_contract_store(store)

    assert status.ready is True
    assert status.semantic_index_ready is True
    assert status.semantic_runtime_available is False
    assert status.semantic_status_reason == "model_name_mismatch"


def test_build_coverage_report_surfaces_numeric_and_priority_coverage(tmp_path: Path) -> None:
    db_path = tmp_path / "priority_numeric_report.db"
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="5.23",
        heading="Fuel Gas System",
        full_text="The guaranteed delivery pressure shall be maintained at 450 psi.",
        page_start=14,
        page_end=14,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )
    page = PageText(page_num=14, text=chunk.full_text, ocr_used=False)
    blocks = build_block_records("doc1", [page], [chunk])
    priority_config = PriorityConfig(
        priority_sections=(
            PrioritySectionRule(
                label="Fuel Gas System",
                section_numbers=("5.23",),
                heading_terms=("fuel gas",),
                focus_terms=("psi", "pressure"),
            ),
        )
    )
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc123",
        page_count=1,
        chunks=[chunk],
        pages=[page],
        features=build_chunk_features([chunk], blocks, priority_config=priority_config),
        blocks=blocks,
    )

    report = rebuild_contract_module.build_coverage_report(db_path, priority_config=priority_config)

    assert report["numeric_coverage"]["chunks_with_numeric_evidence"] == 1
    assert report["priority_coverage"]["matched_sections"] == ["Fuel Gas System"]
    assert report["priority_coverage"]["missing_sections"] == []


def test_build_coverage_report_flags_missing_priority_sections(tmp_path: Path) -> None:
    db_path = tmp_path / "missing_priority_report.db"
    _seed_store(db_path)
    priority_config = PriorityConfig(
        priority_sections=(
            PrioritySectionRule(
                label="Fuel Gas System",
                section_numbers=("5.23",),
                heading_terms=("fuel gas",),
                focus_terms=("mmscfd",),
            ),
        )
    )

    report = rebuild_contract_module.build_coverage_report(db_path, priority_config=priority_config)

    assert report["priority_coverage"]["missing_sections"] == ["Fuel Gas System"]


def _seed_store(db_path: str | Path, *, with_features: bool = True) -> ContractStore:
    store = ContractStore(db_path)
    chunk = ChunkRecord(
        chunk_id="chunk1",
        document_id="doc1",
        chunk_type="section",
        section_number="1",
        heading="General",
        full_text="Contractor shall perform the work.",
        page_start=1,
        page_end=1,
        parent_chunk_id=None,
        ordinal_in_document=1,
    )
    features = build_chunk_features([chunk]) if with_features else []
    store.replace_document(
        document_id="doc1",
        display_name="Contract.pdf",
        version_label="v1",
        file_path="Contract.pdf",
        sha256="abc123",
        page_count=1,
        chunks=[chunk],
        pages=[PageText(page_num=1, text=chunk.full_text, ocr_used=False)],
        features=features,
    )
    return store


def _memory_db_uri(label: str) -> str:
    return f"file:{label}_{uuid.uuid4().hex[:8]}?mode=memory&cache=shared"
