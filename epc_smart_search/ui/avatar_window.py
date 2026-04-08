from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QCursor, QPixmap
from PySide6.QtWidgets import QLabel, QMenu, QWidget


class AvatarWindow(QWidget):
    open_chat = Signal()
    rebuild_index = Signal()
    exit_requested = Signal()

    def __init__(self, avatar_path: Path) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._drag_offset: QPoint | None = None
        self._label = QLabel(self)
        pixmap = QPixmap(str(avatar_path))
        self._label.setPixmap(
            pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        self._label.adjustSize()
        self.resize(self._label.size())
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_menu(event.globalPosition().toPoint())

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_offset is not None:
                self._drag_offset = None
            self.open_chat.emit()

    def _show_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        open_action = QAction("Open Contract Chat", menu)
        open_action.triggered.connect(self.open_chat.emit)
        rebuild_action = QAction("Rebuild Contract Index", menu)
        rebuild_action.triggered.connect(self.rebuild_index.emit)
        exit_action = QAction("Exit", menu)
        exit_action.triggered.connect(self.exit_requested.emit)
        menu.addAction(open_action)
        menu.addAction(rebuild_action)
        menu.addSeparator()
        menu.addAction(exit_action)
        menu.exec(global_pos)

