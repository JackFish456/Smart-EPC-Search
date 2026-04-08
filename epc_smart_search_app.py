from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from epc_smart_search.app_paths import ASSETS_DIR
from epc_smart_search.assistant import ContractAssistant
from epc_smart_search.ui.avatar_window import AvatarWindow
from epc_smart_search.ui.chat_dialog import ContractChatDialog


def main() -> int:
    app = QApplication(sys.argv)
    assistant = ContractAssistant()
    chat_dialog = ContractChatDialog(assistant)
    avatar = AvatarWindow(ASSETS_DIR / "kiewey.png")
    avatar.open_chat.connect(chat_dialog.show)
    avatar.rebuild_index.connect(chat_dialog._rebuild_index)  # noqa: SLF001
    avatar.exit_requested.connect(app.quit)
    avatar.move(60, 60)
    avatar.show()
    chat_dialog.show()
    exit_code = app.exec()
    assistant.gemma.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
