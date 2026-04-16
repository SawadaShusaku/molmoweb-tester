#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-${PORT:-8010}}"
export MOLMOWEB_GUI_ENDPOINT="${MOLMOWEB_GUI_ENDPOINT:-http://127.0.0.1:8001}"
export MOLMOWEB_GUI_HEADLESS="${MOLMOWEB_GUI_HEADLESS:-false}"

echo "Starting MolmoWeb GUI"
echo "  URL:            http://127.0.0.1:$PORT"
echo "  Model endpoint: $MOLMOWEB_GUI_ENDPOINT"
echo "  Headless:       $MOLMOWEB_GUI_HEADLESS"
echo ""
echo "Open the URL above in your browser and send tasks from the chat box."

if command -v open >/dev/null 2>&1; then
  (sleep 1; open "http://127.0.0.1:$PORT/?lang=ja") >/dev/null 2>&1 &
fi

uv run uvicorn inference.gui_app:app --host 0.0.0.0 --port "$PORT"
