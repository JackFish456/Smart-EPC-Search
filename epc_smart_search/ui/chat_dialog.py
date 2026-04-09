from __future__ import annotations

import html
import re

from PySide6.QtCore import QRectF, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QPainter, QPainterPath
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsEffect,
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


CONTRACT_DATA_UNAVAILABLE_STATUS = "Contract data unavailable. Please contact support."
CONTRACT_DATA_UNAVAILABLE_MESSAGE = (
    "The bundled contract data is unavailable in this build. "
    "Please contact support to reinstall the app or replace the contract package."
)
ANSWER_ERROR_PREFIX = "I hit an error while answering that question."


def _dialog_style() -> str:
    # RoundedRectClipEffect clips the whole scroll subtree (border + viewport + scrollbar)
    # with one antialiased path; viewport stays transparent so only the scroll area fills.
    return """
    QDialog { background-color: #fff4cc; }
    QScrollArea#chatScroll {
        background-color: #ffffff;
        border: 1px solid #000;
        border-radius: 6px;
    }
    QWidget#chatViewport {
        background-color: transparent;
    }
    QWidget#chatContent {
        background-color: transparent;
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
    QPushButton:checked { background-color: #000; color: #fff4cc; }
    QLabel { color: #000; }
    """


class RoundedRectClipEffect(QGraphicsEffect):
    """Antialiased rounded clip for the entire widget subtree (one outline vs binary setMask)."""

    def __init__(self, radius: float = 6.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._radius = radius

    def boundingRectFor(self, source_rect: QRectF) -> QRectF:
        return source_rect

    def draw(self, painter: QPainter) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.sourceBoundingRect()), self._radius, self._radius)
        painter.setClipPath(path)
        self.drawSource(painter)
        painter.restore()


class AskWorker(QThread):
    finished = Signal(int, object)
    failed = Signal(int, str)

    def __init__(
        self,
        assistant: ContractAssistant,
        question: str,
        request_token: int,
        history: list[dict[str, str]] | None = None,
        *,
        deep_think: bool = False,
    ) -> None:
        super().__init__()
        self._assistant = assistant
        self._question = question
        self._request_token = request_token
        self._history = list(history or [])
        self._deep_think = deep_think

    def run(self) -> None:
        try:
            answer = self._assistant.ask(
                self._question,
                history=self._history,
                deep_think=self._deep_think,
            )
            self.finished.emit(self._request_token, answer)
        except Exception as exc:
            self.failed.emit(self._request_token, str(exc))


class ContractChatDialog(QDialog):
    def __init__(self, assistant: ContractAssistant, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._assistant = assistant
        self._ask_worker: AskWorker | None = None
        self._pending_answer_label: QLabel | None = None
        self._pending_question: str | None = None
        self._conversation_history: list[dict[str, str]] = []
        self._message_cap = 50
        self._message_count = 0
        self._context_low_locked = False
        self._request_token = 0
        self._active_request_token: int | None = None
        self._thinking_frame = 0
        self._pending_thinking_label = "Thinking"
        self._thinking_start_timer = QTimer(self)
        self._thinking_start_timer.setSingleShot(True)
        self._thinking_start_timer.setInterval(180)
        self._thinking_start_timer.timeout.connect(self._show_thinking_indicator)
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
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setGraphicsEffect(RoundedRectClipEffect(6.0, self._scroll))
        viewport = self._scroll.viewport()
        viewport.setObjectName("chatViewport")
        viewport.setAutoFillBackground(False)
        self._content = QWidget()
        self._content.setObjectName("chatContent")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(16, 12, 16, 12)
        self._content_layout.setSpacing(6)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask Kiewey about the EPC contract")
        self._input.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input)

        self._deep_think_button = QPushButton("Deep Think")
        self._deep_think_button.setCheckable(True)
        input_row.addWidget(self._deep_think_button)

        self._send_button = QPushButton("Send")
        self._send_button.clicked.connect(self._on_send)
        input_row.addWidget(self._send_button)
        layout.addLayout(input_row)

        self._append_message("assistant", GREETING, count_toward_cap=False)
        QTimer.singleShot(0, self._ensure_index_ready)

    def _ensure_index_ready(self) -> None:
        if self._assistant.is_index_ready():
            self._clear_status()
            self._set_input_enabled(True)
            return
        self._append_message("assistant", CONTRACT_DATA_UNAVAILABLE_MESSAGE, count_toward_cap=False)
        self._set_input_enabled(False)
        self._set_status_message(CONTRACT_DATA_UNAVAILABLE_STATUS)

    def _on_send(self) -> None:
        question = self._input.text().strip()
        if not question or not self._assistant.is_index_ready() or self._context_low_locked:
            return
        deep_think = self._deep_think_button.isChecked()
        self._input.clear()
        self._append_message("user", question)
        self._pending_question = question
        self._pending_answer_label = self._append_message("assistant", "", count_toward_cap=False)
        self._pending_thinking_label = "Deep Thinking" if deep_think else "Thinking"
        self._start_thinking_indicator()
        self._set_input_busy(True)
        self._request_token += 1
        self._active_request_token = self._request_token
        self._ask_worker = AskWorker(
            self._assistant,
            question,
            self._request_token,
            history=self._conversation_history[-8:],
            deep_think=deep_think,
        )
        self._ask_worker.finished.connect(self._on_answer_ready)
        self._ask_worker.failed.connect(self._on_answer_failed)
        self._ask_worker.finished.connect(self._ask_worker.deleteLater)
        self._ask_worker.failed.connect(self._ask_worker.deleteLater)
        self._ask_worker.start()

    def _on_answer_ready(self, request_token: int, answer: AssistantAnswer) -> None:
        self._clear_finished_worker()
        if request_token != self._active_request_token:
            return
        self._active_request_token = None
        if self._pending_question:
            self._conversation_history.append({"role": "user", "content": self._pending_question})
        self._conversation_history.append({"role": "assistant", "content": answer.text})
        self._conversation_history = self._conversation_history[-8:]
        self._pending_question = None
        self._replace_pending_answer(answer.text)
        if not self._context_low_locked:
            self._set_input_busy(False)

    def _on_answer_failed(self, request_token: int, error: str) -> None:
        self._clear_finished_worker()
        if request_token != self._active_request_token:
            return
        self._active_request_token = None
        self._pending_question = None
        self._replace_pending_answer(f"{ANSWER_ERROR_PREFIX}\n\n{error}")
        if not self._context_low_locked:
            self._set_input_busy(False)

    def closeEvent(self, event) -> None:
        self._reset_chat_session()
        super().closeEvent(event)

    def _append_message(
        self,
        role: str,
        content: str,
        *,
        count_toward_cap: bool = True,
    ) -> QLabel:
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
        self._input.setReadOnly(False)
        self._deep_think_button.setEnabled(enabled)
        self._send_button.setEnabled(enabled)
        if enabled:
            self._input.setFocus()

    def _set_input_busy(self, busy: bool) -> None:
        self._input.setEnabled(True)
        self._input.setReadOnly(busy)
        self._deep_think_button.setEnabled(not busy)
        self._send_button.setEnabled(not busy)
        self._input.setFocus()

    def _set_status_message(self, message: str) -> None:
        self._status.setText(message)
        self._status.show()

    def _clear_status(self) -> None:
        self._status.clear()
        self._status.hide()

    def _start_thinking_indicator(self) -> None:
        self._thinking_start_timer.stop()
        self._thinking_timer.stop()
        self._thinking_frame = 0
        self._thinking_start_timer.start()

    def _stop_thinking_indicator(self) -> None:
        self._thinking_start_timer.stop()
        self._thinking_timer.stop()

    def _show_thinking_indicator(self) -> None:
        if self._pending_answer_label is None:
            return
        self._advance_thinking_indicator()
        self._thinking_timer.start()

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
        self._pending_answer_label.setText(f"<span>{self._pending_thinking_label}</span>&nbsp;{' '.join(dots)}")
        self._thinking_frame += 1

    def _enforce_message_cap(self) -> None:
        if self._context_low_locked or self._message_count < self._message_cap:
            return
        self._context_low_locked = True
        self._append_message("assistant", "Context Low Please Open Another Chat Window", count_toward_cap=False)
        self._set_input_enabled(False)

    def _clear_finished_worker(self) -> None:
        worker = self.sender()
        if worker is self._ask_worker:
            self._ask_worker = None

    def _reset_chat_session(self) -> None:
        self._stop_thinking_indicator()
        self._pending_answer_label = None
        self._active_request_token = None
        self._pending_question = None
        self._pending_thinking_label = "Thinking"
        self._conversation_history.clear()
        self._message_count = 0
        self._context_low_locked = False
        self._input.clear()
        self._deep_think_button.setChecked(False)
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._append_message("assistant", GREETING, count_toward_cap=False)
        self._ensure_index_ready()

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
            else:
                blocks.append(cls._format_generic_block(paragraph))
        return "".join(blocks)

    @classmethod
    def _format_generic_block(cls, paragraph: str) -> str:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            return ""
        if cls._is_markdown_header(lines[0]):
            header_html = cls._format_header_block(lines[0].lstrip("#").strip())
            if len(lines) == 1:
                return header_html
            remainder = "\n".join(lines[1:])
            return header_html + cls._format_generic_block(remainder)
        if cls._is_list_block(lines):
            return cls._format_list_block(lines)
        if lines[0].endswith(":"):
            header_html = cls._format_header_block(lines[0].removesuffix(":").strip())
            if len(lines) == 1:
                return header_html
            remainder = lines[1:]
            if cls._is_list_block(remainder):
                return header_html + cls._format_list_block(remainder)
            body = "<br>".join(cls._format_inline_text(line) for line in remainder)
            return header_html + f"<div style='font-size:13px; line-height:1.45; margin:2px 0 10px 0;'>{body}</div>"
        body = "<br>".join(cls._format_inline_text(line) for line in lines)
        return f"<div style='font-size:13px; line-height:1.45; margin:2px 0 10px 0;'>{body}</div>"

    @staticmethod
    def _is_markdown_header(line: str) -> bool:
        return bool(re.match(r"^#{1,3}\s+\S", line))

    @staticmethod
    def _is_list_block(lines: list[str]) -> bool:
        if not lines:
            return False
        unordered = all(re.match(r"^[-*•]\s+\S", line) for line in lines)
        ordered = all(re.match(r"^\d+\.\s+\S", line) for line in lines)
        return unordered or ordered

    @classmethod
    def _format_header_block(cls, text: str) -> str:
        return (
            "<div style='font-size:13px; font-weight:700; margin:2px 0 6px 0;'>"
            f"{cls._format_inline_text(text)}</div>"
        )

    @classmethod
    def _format_list_block(cls, lines: list[str]) -> str:
        ordered = all(re.match(r"^\d+\.\s+\S", line) for line in lines)
        tag = "ol" if ordered else "ul"
        items: list[str] = []
        pattern = r"^\d+\.\s+" if ordered else r"^[-*•]\s+"
        for line in lines:
            item_text = re.sub(pattern, "", line, count=1).strip()
            items.append(f"<li style='margin:0 0 4px 0;'>{cls._format_inline_text(item_text)}</li>")
        return (
            f"<{tag} style='font-size:13px; line-height:1.45; margin:2px 0 12px 18px; padding-left:18px;'>"
            f"{''.join(items)}</{tag}>"
        )

    @staticmethod
    def _format_inline_text(text: str) -> str:
        escaped = html.escape(text)
        escaped = re.sub(
            r"`([^`]+)`",
            lambda match: (
                "<code style='background-color:#fff4cc; border:1px solid #e0cf85; "
                "border-radius:3px; padding:0 3px;'>"
                f"{match.group(1)}</code>"
            ),
            escaped,
        )
        escaped = re.sub(
            r"\*\*([^*]+)\*\*",
            lambda match: f"<span style='font-weight:700;'>{match.group(1)}</span>",
            escaped,
        )
        escaped = re.sub(
            r"(?<!\*)\*([^*]+)\*(?!\*)",
            lambda match: f"<span style='font-style:italic;'>{match.group(1)}</span>",
            escaped,
        )
        return escaped

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
