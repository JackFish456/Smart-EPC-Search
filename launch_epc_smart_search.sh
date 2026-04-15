#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
APP_SCRIPT="$ROOT/epc_smart_search_app.py"

pick_python() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    echo "$ROOT/.venv/bin/python"
    return
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return
  fi
  command -v python3
}

PYTHON="$(pick_python)"
"$PYTHON" -c "import PySide6, requests" 2>/dev/null || {
  echo "Install runtime deps: $PYTHON -m pip install -r requirements-runtime.txt" >&2
  exit 1
}

QT_PATHS="$("$PYTHON" - <<'PY'
from PySide6.QtCore import QLibraryInfo

print(QLibraryInfo.path(QLibraryInfo.PluginsPath))
print(QLibraryInfo.path(QLibraryInfo.PrefixPath))
PY
)"
QT_PLUGIN_PATH_ROOT="$(printf '%s\n' "$QT_PATHS" | sed -n '1p')"
QT_PREFIX_PATH="$(printf '%s\n' "$QT_PATHS" | sed -n '2p')"

if [[ -n "${QT_PLUGIN_PATH_ROOT}" && -d "${QT_PLUGIN_PATH_ROOT}" ]]; then
  export QT_PLUGIN_PATH="${QT_PLUGIN_PATH_ROOT}"
  if [[ -d "${QT_PLUGIN_PATH_ROOT}/platforms" ]]; then
    export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_PLUGIN_PATH_ROOT}/platforms"
  fi
fi

if [[ "${OSTYPE:-}" == darwin* && -n "${QT_PREFIX_PATH}" && -d "${QT_PREFIX_PATH}/lib" ]]; then
  export DYLD_FRAMEWORK_PATH="${QT_PREFIX_PATH}/lib${DYLD_FRAMEWORK_PATH:+:${DYLD_FRAMEWORK_PATH}}"
  export DYLD_LIBRARY_PATH="${QT_PREFIX_PATH}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
fi

"$PYTHON" -m epc_smart_search.preflight --mode launch
exec "$PYTHON" "$APP_SCRIPT"
