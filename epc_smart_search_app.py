from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from epc_smart_search.app_paths import ASSETS_DIR
from epc_smart_search.assistant import ContractAssistant
from epc_smart_search.preflight import collect_workspace_artifact_warnings
from epc_smart_search.ui.avatar_window import AvatarWindow
from epc_smart_search.ui.chat_dialog import ContractChatDialog


def main() -> int:
    for issue in collect_workspace_artifact_warnings():
        print(f"[WARNING] {issue.message}", file=sys.stderr)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    assistant = ContractAssistant()
    chat_dialog = ContractChatDialog(assistant)
    avatar_path = ASSETS_DIR / "kiewey.png"
    avatar = AvatarWindow(avatar_path)
    avatar.open_chat.connect(lambda: _show_chat_dialog(chat_dialog))
    avatar.hide_requested.connect(avatar.hide)
    avatar.exit_requested.connect(app.quit)
    avatar.place_bottom_right()
    avatar.show()
    _show_chat_dialog(chat_dialog)

    tray_icon: QSystemTrayIcon | None = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray_icon = QSystemTrayIcon(QIcon(str(avatar_path)), app)
        tray_icon.setToolTip("Kiewey - EPC Smart Search")
        tray_menu = QMenu()
        open_chat_action = QAction("Open Contract Chat", tray_menu)
        open_chat_action.triggered.connect(lambda: _show_chat_dialog(chat_dialog))
        toggle_avatar_action = QAction(tray_menu)
        toggle_avatar_action.triggered.connect(lambda: _toggle_avatar(avatar))
        exit_action = QAction("Exit", tray_menu)
        exit_action.triggered.connect(app.quit)
        tray_menu.addAction(open_chat_action)
        tray_menu.addAction(toggle_avatar_action)
        tray_menu.addSeparator()
        tray_menu.addAction(exit_action)
        tray_menu.aboutToShow.connect(lambda: _sync_toggle_avatar_action(toggle_avatar_action, avatar))
        _sync_toggle_avatar_action(toggle_avatar_action, avatar)
        tray_icon.setContextMenu(tray_menu)
        tray_icon.activated.connect(lambda reason: _handle_tray_activation(reason, avatar, chat_dialog))
        tray_icon.show()

    exit_code = app.exec()
    if tray_icon is not None:
        tray_icon.hide()
    assistant.gemma.stop()
    return exit_code


def _show_avatar(avatar: AvatarWindow) -> None:
    avatar.place_bottom_right()
    avatar.show()
    avatar.raise_()
    avatar.activateWindow()


def _toggle_avatar(avatar: AvatarWindow) -> None:
    if avatar.isVisible():
        avatar.hide()
        return
    _show_avatar(avatar)


def _sync_toggle_avatar_action(action: QAction, avatar: AvatarWindow) -> None:
    action.setText("Hide Kiewey" if avatar.isVisible() else "Show Kiewey")


def _show_chat_dialog(chat_dialog: ContractChatDialog) -> None:
    if chat_dialog.isMinimized():
        chat_dialog.setWindowState(chat_dialog.windowState() & ~Qt.WindowState.WindowMinimized)
    chat_dialog.show()
    chat_dialog.raise_()
    chat_dialog.activateWindow()


def _handle_tray_activation(reason: QSystemTrayIcon.ActivationReason, avatar: AvatarWindow, chat_dialog: ContractChatDialog) -> None:
    if reason in {QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick}:
        _show_avatar(avatar)
        _show_chat_dialog(chat_dialog)


if __name__ == "__main__":
    raise SystemExit(main())
