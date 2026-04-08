from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from epc_smart_search.assistant import AssistantAnswer, ContractAssistant
from epc_smart_search.config import GREETING


def _dialog_style() -> str:
    return """
    QDialog { background-color: #fff4cc; }
    QScrollArea { background-color: #fff4cc; border: 1px solid #000; border-radius: 6px; }
    QLineEdit {
        background-color: #fff;
        color: #000;
        border: 1px solid #000;
        border-radius: 4px;
        padding: 6px;
        font-size: 13px;
    }
    QPushButton {
        background-color: #ffcd23;
        color: #000;
        border: 1px solid #000;
        border-radius: 4px;
        padding: 6px 12px;
        font-weight: bold;
        font-size: 13px;
    }
    QPushButton:hover { background-color: #ffeb99; }
    QLabel { color: #000; }
    """


class IndexWorker(QThread):
    progress = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, assistant: ContractAssistant) -> None:
        super().__init__()
        self._assistant = assistant

    def run(self) -> None:
        try:
            result = self._assistant.build_index(progress_callback=self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class AskWorker(QThread):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, assistant: ContractAssistant, question: str) -> None:
        super().__init__()
        self._assistant = assistant
        self._question = question

    def run(self) -> None:
        try:
            answer = self._assistant.ask(self._question)
            self.finished.emit(answer)
        except Exception as exc:
            self.failed.emit(str(exc))


class ContractChatDialog(QDialog):
    def __init__(self, assistant: ContractAssistant, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._assistant = assistant
        self._index_worker: IndexWorker | None = None
        self._ask_worker: AskWorker | None = None
        self._pending_answer_label: QLabel | None = None

        self.setWindowTitle("Kiewey - EPC Contract Assistant")
        self.setMinimumSize(760, 600)
        self.setStyleSheet(_dialog_style())

        layout = QVBoxLayout(self)
        self._status = QLabel("Checking contract index...")
        layout.addWidget(self._status)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll)

        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask about project scope, obligations, definitions, clauses...")
        self._input.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input)

        self._send_button = QPushButton("Send")
        self._send_button.clicked.connect(self._on_send)
        input_row.addWidget(self._send_button)
        layout.addLayout(input_row)

        self._append_message("assistant", GREETING)
        QTimer.singleShot(0, self._ensure_index_ready)

    def _ensure_index_ready(self) -> None:
        if self._assistant.is_index_ready():
            self._status.setText("Contract index ready.")
            self._set_input_enabled(True)
            return
        self._append_message("assistant", "I’m getting the contract ready for search. This may take a little while on the first run.")
        self._start_index_worker()

    def _rebuild_index(self) -> None:
        self._append_message("assistant", "Rebuilding the contract index now.")
        self._start_index_worker()

    def _start_index_worker(self) -> None:
        if self._index_worker is not None and self._index_worker.isRunning():
            return
        self._set_input_enabled(False)
        self._index_worker = IndexWorker(self._assistant)
        self._index_worker.progress.connect(self._status.setText)
        self._index_worker.finished.connect(self._on_index_finished)
        self._index_worker.failed.connect(self._on_index_failed)
        self._index_worker.start()

    def _on_index_finished(self, result: dict) -> None:
        self._status.setText(
            f"Contract index ready. Pages: {result.get('page_count', '?')} | Chunks: {result.get('chunk_count', '?')}"
        )
        self._append_message(
            "assistant",
            f"Contract index ready. I loaded {result.get('chunk_count', '?')} searchable chunks.",
        )
        self._set_input_enabled(True)

    def _on_index_failed(self, error: str) -> None:
        self._status.setText("Index build failed.")
        self._append_message("assistant", f"I couldn't finish building the contract index.\n\n{error}")
        self._set_input_enabled(True)

    def _on_send(self) -> None:
        question = self._input.text().strip()
        if not question or not self._assistant.is_index_ready():
            return
        self._input.clear()
        self._append_message("user", question)
        self._pending_answer_label = self._append_message("assistant", "...")
        self._set_input_enabled(False)
        self._ask_worker = AskWorker(self._assistant, question)
        self._ask_worker.finished.connect(self._on_answer_ready)
        self._ask_worker.failed.connect(self._on_answer_failed)
        self._ask_worker.start()

    def _on_answer_ready(self, answer: AssistantAnswer) -> None:
        message = answer.text
        if answer.citations:
            citations = "\n".join(self._format_citation_line(citation) for citation in answer.citations)
            message = f"{message}\n\nSources:\n{citations}"
        self._replace_pending_answer(message)
        self._set_input_enabled(True)

    def _on_answer_failed(self, error: str) -> None:
        self._replace_pending_answer(f"I hit an error while answering that question.\n\n{error}")
        self._set_input_enabled(True)

    def _append_message(self, role: str, content: str) -> QLabel:
        label = QLabel(content)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        base_style = (
            "background-color: #ffeb99; margin-left: 70px;"
            if role == "user"
            else "background-color: #fff; margin-right: 70px;"
        )
        label.setStyleSheet(
            base_style
            + " color: #000; border: 1px solid #000; border-radius: 6px; padding: 8px; margin-top: 4px; margin-bottom: 4px;"
        )
        self._content_layout.addWidget(label)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return label

    def _replace_pending_answer(self, content: str) -> None:
        if self._pending_answer_label is not None:
            self._pending_answer_label.setText(content)
            self._pending_answer_label = None
            QTimer.singleShot(0, self._scroll_to_bottom)
            return
        self._append_message("assistant", content)

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_input_enabled(self, enabled: bool) -> None:
        self._input.setEnabled(enabled)
        self._send_button.setEnabled(enabled)

    @staticmethod
    def _format_citation_line(citation) -> str:
        parts = []
        page_label = (
            f"page {citation.page_start}"
            if citation.page_start == citation.page_end
            else f"pages {citation.page_start}-{citation.page_end}"
        )
        parts.append(page_label)
        if citation.section_number:
            parts.append(f"Section {citation.section_number}")
        if citation.heading:
            parts.append(citation.heading)
        if citation.attachment:
            parts.append(citation.attachment)
        return "- " + " | ".join(parts)
