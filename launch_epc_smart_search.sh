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

"$PYTHON" -m epc_smart_search.preflight --mode launch
exec "$PYTHON" "$APP_SCRIPT"
