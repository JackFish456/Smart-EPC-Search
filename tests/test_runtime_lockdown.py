import inspect
import uuid
import html
from pathlib import Path

from PySide6.QtWidgets import QApplication

import epc_smart_search.rebuild_contract as rebuild_contract_module
import epc_smart_search.ui.avatar_window as avatar_window_module
import epc_smart_search_app
from epc_smart_search.assistant import INTERNAL_REBUILD_ERROR
from epc_smart_search.assistant import ContractAssistant
from epc_smart_search.assistant import IndexValidationResult
from epc_smart_search.assistant import validate_contract_store
from epc_smart_search.chunking import ChunkRecord
from epc_smart_search.config import GREETING
from epc_smart_search.ocr_support import PageText
from epc_smart_search.search_features import build_chunk_features
from epc_smart_search.storage import ContractStore
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
    first_label = dialog._content_layout.itemAt(0).widget()  # noqa: SLF001
    assert html.escape(GREETING.split(".")[0]) in first_label.text()


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


def test_customer_ui_no_longer_exposes_rebuild_entry_points() -> None:
    assert not hasattr(AvatarWindow, "rebuild_index")
    assert not hasattr(ContractChatDialog, "_rebuild_index")
    assert "Rebuild Contract Index" not in inspect.getsource(avatar_window_module)
    assert "Rebuild Contract Index" not in inspect.getsource(epc_smart_search_app)


def test_rebuild_cli_creates_and_validates_a_fresh_database(monkeypatch) -> None:
    pdf_path = Path(".tmp_test") / f"contract_{uuid.uuid4().hex[:8]}.pdf"
    out_path = Path(".tmp_test") / f"contract_store_{uuid.uuid4().hex[:8]}.db"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")
    memory_store = _seed_store(_memory_db_uri("cli"))

    def fake_build_index(*, pdf_path, db_path, version_label="v1", progress_callback=None):
        Path(db_path).write_bytes(b"stub")
        return {"document_id": "doc1", "page_count": 1, "chunk_count": 1}

    monkeypatch.setattr(rebuild_contract_module, "build_index", fake_build_index)
    monkeypatch.setattr(rebuild_contract_module, "ContractStore", lambda path: memory_store)

    exit_code = rebuild_contract_module.main(["--pdf", str(pdf_path), "--out", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()


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
        embeddings={"chunk1": b"\x00\x00\x00\x00"},
        model_name="test",
        dimension=1,
    )
    return store


def _memory_db_uri(label: str) -> str:
    return f"file:{label}_{uuid.uuid4().hex[:8]}?mode=memory&cache=shared"
