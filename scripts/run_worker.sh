#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# If venv exists, prefer it
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  PY="$(command -v python3)"
fi

# Endpoints
export MOONALT_IPC_IN="${MOONALT_IPC_IN:-ipc:///tmp/capture}"
export MOONALT_BROADTRACK_OUT="${MOONALT_BROADTRACK_OUT:-ipc:///tmp/broadtrack}"
export MOONALT_OUT_MODE="${MOONALT_OUT_MODE:-bind}"   # bind|connect (default bind)

# Protocol expectations
export MOONALT_SCHEMA="${MOONALT_SCHEMA:-golex.vt.input_v1}"
export MOONALT_PIXEL_FORMAT="${MOONALT_PIXEL_FORMAT:-BGR24}"

# YOLO tuning
export MOONALT_YOLO_MODEL="${MOONALT_YOLO_MODEL:-yolov8n-seg.pt}"
export MOONALT_YOLO_IMGSZ="${MOONALT_YOLO_IMGSZ:-640}"
export MOONALT_YOLO_CLASSES="${MOONALT_YOLO_CLASSES:-0,32}"
export MOONALT_YOLO_EVERY_N="${MOONALT_YOLO_EVERY_N:-1}"

echo "[run_worker] IN=$MOONALT_IPC_IN  OUT=$MOONALT_BROADTRACK_OUT  OUT_MODE=$MOONALT_OUT_MODE"
exec "$PY" -m app.worker
