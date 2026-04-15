#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Keep PyInstaller caches inside the repo (avoids permission issues in sandboxes / locked-down home dirs).
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-$ROOT/.pyinstaller}"

PREBUILT_DB_PATH="${EPC_PREBUILT_DB_PATH:-}"
# Default distribution target: Intel Macs (x86_64). Override with EPC_MACOS_TARGET_ARCH or --target-arch.
TARGET_ARCH="${EPC_MACOS_TARGET_ARCH:-x86_64}"

usage() {
  echo "Usage: EPC_PREBUILT_DB_PATH=/path/to/contract_store.prebuilt.db $0" >&2
  echo "   or: $0 --prebuilt-db /path/to/contract_store.prebuilt.db" >&2
  echo "Optional: --target-arch x86_64|arm64|universal2  (default: x86_64 for Intel Macs)" >&2
  echo "Env: EPC_MACOS_TARGET_ARCH, EPC_PREBUILT_DB_PATH" >&2
  echo "The prebuilt database must exist outside this repository directory." >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prebuilt-db)
      [[ $# -ge 2 ]] || usage
      PREBUILT_DB_PATH="$2"
      shift 2
      ;;
    --target-arch)
      [[ $# -ge 2 ]] || usage
      TARGET_ARCH="$2"
      shift 2
      ;;
    -h | --help)
      usage
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "${PREBUILT_DB_PATH}" ]]; then
  usage
fi

case "$TARGET_ARCH" in
  x86_64 | arm64 | universal2) ;;
  *)
    echo "Invalid --target-arch / EPC_MACOS_TARGET_ARCH: $TARGET_ARCH (use x86_64, arm64, or universal2)" >&2
    exit 1
    ;;
esac

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
"$PYTHON" -c "import PyInstaller, PySide6" 2>/dev/null || {
  echo "Install packaging deps: $PYTHON -m pip install -r requirements-dev.txt" >&2
  exit 1
}

PYTHON_MACHINE="$("$PYTHON" -c "import platform; print(platform.machine())")"
if [[ "$TARGET_ARCH" == "x86_64" && "$PYTHON_MACHINE" != "x86_64" ]]; then
  echo "This script defaults to an Intel (x86_64) app for other users." >&2
  echo "Your Python reports machine=${PYTHON_MACHINE} (not x86_64)." >&2
  echo "On Apple Silicon, recreate the venv under Rosetta, then reinstall deps, e.g.:" >&2
  echo "  arch -x86_64 /usr/local/bin/python3.11 -m venv .venv && arch -x86_64 .venv/bin/pip install -r requirements-dev.txt" >&2
  echo "Then run: arch -x86_64 $0 ..." >&2
  echo "Or set EPC_MACOS_TARGET_ARCH=arm64 for Apple Silicon-only builds." >&2
  exit 1
fi
if [[ "$TARGET_ARCH" == "arm64" && "$PYTHON_MACHINE" != "arm64" ]]; then
  echo "arm64 builds need an arm64 Python (this one reports ${PYTHON_MACHINE})." >&2
  exit 1
fi

# PySide6 wheels here target macOS 13+; keep the bundle consistent for recipients.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"

"$PYTHON" -m epc_smart_search.preflight --mode package --prebuilt-db "$PREBUILT_DB_PATH"
STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/epc_smart_search_pkg.XXXXXX")"
cleanup() {
  rm -rf "$STAGE_ROOT"
}
trap cleanup EXIT

STAGE_ASSETS="$STAGE_ROOT/assets"
mkdir -p "$STAGE_ASSETS"
cp "$PREBUILT_DB_PATH" "$STAGE_ASSETS/contract_store.prebuilt.db"

"$PYTHON" -m PyInstaller \
  --noconfirm \
  --windowed \
  --target-architecture "$TARGET_ARCH" \
  --name "EPC Smart Search Lite" \
  --osx-bundle-identifier com.epcsmartsearch.lite \
  --add-data "assets/kiewey.png:assets" \
  --add-data "assets/semantic_model.json:assets" \
  --add-data "$STAGE_ASSETS/contract_store.prebuilt.db:assets" \
  epc_smart_search_app.py

APP_PATH="$ROOT/dist/EPC Smart Search Lite.app"
# Extended attributes (e.g. com.apple.provenance) can break ad-hoc codesign and annoy Gatekeeper on other Macs.
if [[ -d "$APP_PATH" ]]; then
  xattr -cr "$APP_PATH" 2>/dev/null || true
  codesign --force --deep --sign - "$APP_PATH" 2>/dev/null || true
fi
MAIN_EXE="$APP_PATH/Contents/MacOS/EPC Smart Search Lite"
echo "Built: $APP_PATH"
if [[ -f "$MAIN_EXE" ]]; then
  echo "Main executable architectures (lipo): $(lipo -archs "$MAIN_EXE" 2>/dev/null || echo '?')"
  if [[ "$TARGET_ARCH" == "x86_64" ]]; then
    if ! lipo -archs "$MAIN_EXE" 2>/dev/null | grep -q 'x86_64'; then
      echo "ERROR: Expected x86_64 in the main binary for Intel distribution." >&2
      exit 1
    fi
  fi
fi
echo "Recipients: Intel Macs need macOS ${MACOSX_DEPLOYMENT_TARGET}+ and may need to right-click → Open the first time (unsigned app)."
