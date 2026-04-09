from __future__ import annotations

from dataclasses import dataclass
import html

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
    QScrollArea#chatScroll {
        background-color: #fff4cc;
        border: 1px solid #000;
        border-radius: 6px;
    }
    QWidget#chatContent {
        background-color: #fff;
        border-radius: 4px;
    }
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
        self._has_announced_index_ready = False
        self._message_cap = 50
        self._message_count = 0
        self._context_low_locked = False
        self._thinking_frame = 0
        self._thinking_timer = QTimer(self)
        self._thinking_timer.setInterval(350)
        self._thinking_timer.timeout.connect(self._advance_thinking_indicator)

        self.setWindowTitle("Kiewey - EPC Contract Assistant")
        self.setMinimumSize(760, 600)
        self.setStyleSheet(_dialog_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self._status = QLabel("Checking contract index...")
        layout.addWidget(self._status)
        self._status.hide()

        self._scroll = QScrollArea()
        self._scroll.setObjectName("chatScroll")
        self._scroll.setWidgetResizable(True)
        self._content = QWidget()
        self._content.setObjectName("chatContent")
        self._scroll.viewport().setStyleSheet("background-color: #fff;")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(16, 12, 16, 12)
        self._content_layout.setSpacing(6)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask Kiewey about the EPC contract...")
        self._input.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input)

        self._send_button = QPushButton("Send")
        self._send_button.clicked.connect(self._on_send)
        input_row.addWidget(self._send_button)
        layout.addLayout(input_row)

        self._append_message("assistant", GREETING, count_toward_cap=False)
        QTimer.singleShot(0, self._ensure_index_ready)

    def _ensure_index_ready(self) -> None:
        if self._assistant.is_index_ready():
            self._clear_status()
            self._announce_index_ready()
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
        self._set_status_message("Building contract index...")
        self._index_worker = IndexWorker(self._assistant)
        self._index_worker.progress.connect(self._set_status_message)
        self._index_worker.finished.connect(self._on_index_finished)
        self._index_worker.failed.connect(self._on_index_failed)
        self._index_worker.start()

    def _on_index_finished(self, result: dict) -> None:
        self._clear_status()
        if self._has_announced_index_ready:
            self._append_message(
                "assistant",
                f"Index rebuild finished. I loaded {result.get('chunk_count', '?')} searchable chunks.",
            )
        else:
            self._announce_index_ready()
        self._set_input_enabled(True)

    def _on_index_failed(self, error: str) -> None:
        self._set_status_message("Index build failed.")
        self._append_message("assistant", f"I couldn't finish building the contract index.\n\n{error}")
        self._set_input_enabled(True)

    def _on_send(self) -> None:
        question = self._input.text().strip()
        if not question or not self._assistant.is_index_ready() or self._context_low_locked:
            return
        self._input.clear()
        self._append_message("user", question)
        self._pending_answer_label = self._append_message("assistant", "", count_toward_cap=False)
        self._start_thinking_indicator()
        self._set_input_enabled(False)
        self._ask_worker = AskWorker(self._assistant, question)
        self._ask_worker.finished.connect(self._on_answer_ready)
        self._ask_worker.failed.connect(self._on_answer_failed)
        self._ask_worker.start()

    def _on_answer_ready(self, answer: AssistantAnswer) -> None:
        self._replace_pending_answer(answer.text)
        if not self._context_low_locked:
            self._set_input_enabled(True)

    def _on_answer_failed(self, error: str) -> None:
        self._replace_pending_answer(f"I hit an error while answering that question.\n\n{error}")
        if not self._context_low_locked:
            self._set_input_enabled(True)

    def _append_message(self, role: str, content: str, *, count_toward_cap: bool = True) -> QLabel:
        label = QLabel()
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
        self._set_message_content(label, role, content)
        self._content_layout.addWidget(label)
        if count_toward_cap:
            self._message_count += 1
        QTimer.singleShot(0, self._scroll_to_bottom)
        return label

    def _replace_pending_answer(self, content: str) -> None:
        if self._pending_answer_label is not None:
            self._stop_thinking_indicator()
            self._set_message_content(self._pending_answer_label, "assistant", content)
            self._pending_answer_label = None
            self._message_count += 1
            self._enforce_message_cap()
            QTimer.singleShot(0, self._scroll_to_bottom)
            return
        self._append_message("assistant", content)
        self._enforce_message_cap()

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_input_enabled(self, enabled: bool) -> None:
        self._input.setEnabled(enabled)
        self._send_button.setEnabled(enabled)

    def _set_status_message(self, message: str) -> None:
        self._status.setText(message)
        self._status.show()

    def _clear_status(self) -> None:
        self._status.clear()
        self._status.hide()

    def _announce_index_ready(self) -> None:
        if self._has_announced_index_ready:
            return
        self._has_announced_index_ready = True

    def _start_thinking_indicator(self) -> None:
        self._thinking_frame = 0
        self._advance_thinking_indicator()
        self._thinking_timer.start()

    def _stop_thinking_indicator(self) -> None:
        self._thinking_timer.stop()

    def _advance_thinking_indicator(self) -> None:
        if self._pending_answer_label is None:
            self._stop_thinking_indicator()
            return
        dot_count = self._thinking_frame % 4
        dots = []
        for index in range(3):
            dot = "." if index < dot_count else "&nbsp;"
            dots.append(f"<span style=\"font-size: 18px; font-weight: 600;\">{dot}</span>")
        self._pending_answer_label.setTextFormat(Qt.TextFormat.RichText)
        self._pending_answer_label.setText(f"<span>Thinking</span>&nbsp;{' '.join(dots)}")
        self._thinking_frame += 1

    def _enforce_message_cap(self) -> None:
        if self._context_low_locked or self._message_count < self._message_cap:
            return
        self._context_low_locked = True
        self._append_message("assistant", "Context Low Please Open Another Chat Window", count_toward_cap=False)
        self._set_input_enabled(False)

    def _set_message_content(self, label: QLabel, role: str, content: str) -> None:
        if role == "assistant":
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setText(self._format_assistant_message(content))
            return
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setText(content)

    @classmethod
    def _format_assistant_message(cls, content: str) -> str:
        stripped = content.strip()
        if not stripped:
            return ""
        paragraphs = [paragraph.strip() for paragraph in stripped.split("\n\n") if paragraph.strip()]
        blocks: list[str] = []
        for paragraph in paragraphs:
            if paragraph.startswith("Section "):
                blocks.append(cls._format_section_block(paragraph))
            elif paragraph.startswith("Page "):
                blocks.append(cls._format_page_block(paragraph))
            elif paragraph.endswith(":"):
                blocks.append(
                    f"<div style='font-size:13px; font-weight:700; margin:2px 0 10px 0;'>{html.escape(paragraph)}</div>"
                )
            else:
                escaped = html.escape(paragraph).replace("\n", "<br>")
                blocks.append(f"<div style='font-size:13px; line-height:1.45; margin:2px 0 10px 0;'>{escaped}</div>")
        return "".join(blocks)

    @classmethod
    def _format_section_block(cls, paragraph: str) -> str:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        heading = html.escape(lines[0]) if lines else ""
        page_line = ""
        body = ""
        for line in lines[1:]:
            if line.startswith("Pages:"):
                page_line = html.escape(line)
            elif line.startswith("Contract text:"):
                body = cls._format_contract_text(line.removeprefix("Contract text:").strip())
            else:
                addition = html.escape(line)
                body = f"{body}<br>{addition}" if body else addition
        page_html = (
            f"<div style='font-size:11px; color:#6b5d00; font-style:italic; margin:1px 0 6px 0;'>{page_line}</div>"
            if page_line
            else ""
        )
        body_html = (
            "<div style='font-size:13px; line-height:1.5;'>"
            "<span style='font-weight:700;'>Contract text:</span> "
            f"{body}</div>"
            if body
            else ""
        )
        return (
            "<div style='margin:4px 0 14px 0; padding-left:10px; border-left:3px solid #ffcd23;'>"
            f"<div style='font-size:14px; font-weight:700; margin:0 0 2px 0;'>{heading}</div>"
            f"{page_html}"
            f"{body_html}"
            "</div>"
        )

    @classmethod
    def _format_page_block(cls, paragraph: str) -> str:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        header = html.escape(lines[0]) if lines else ""
        remainder = " ".join(lines[1:]).strip()
        if not remainder and lines:
            page_label, _, excerpt = lines[0].partition(":")
            header = html.escape(page_label + ":")
            remainder = excerpt.strip()
        excerpt_html = cls._format_contract_text(remainder)
        return (
            "<div style='margin:4px 0 14px 0; padding-left:10px; border-left:3px solid #ffcd23;'>"
            f"<div style='font-size:12px; color:#6b5d00; font-style:italic; margin:0 0 6px 0;'>{header}</div>"
            "<div style='font-size:13px; line-height:1.5;'>"
            f"{excerpt_html}</div>"
            "</div>"
        )

    @staticmethod
    def _format_contract_text(text: str) -> str:
        escaped = html.escape(text).replace("\n", "<br>")
        return f"<span>{escaped}</span>"
