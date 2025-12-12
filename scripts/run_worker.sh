#!/usr/bin/env bash
set -euo pipefail

# Live pipeline defaults:
#   Capture (C++) binds PUSH  -> ipc:///tmp/capture
#   BroadTrack (C++) binds PULL -> ipc:///tmp/broadtrack_in.sock
export MOONALT_IPC_IN="${MOONALT_IPC_IN:-ipc:///tmp/capture}"
export MOONALT_BROADTRACK_OUT="${MOONALT_BROADTRACK_OUT:-ipc:///tmp/broadtrack_in.sock}"

# YOLO tuning
export MOONALT_YOLO_IMGSZ="${MOONALT_YOLO_IMGSZ:-640}"
export MOONALT_YOLO_EVERY_N="${MOONALT_YOLO_EVERY_N:-1}"
export MOONALT_YOLO_CONF="${MOONALT_YOLO_CONF:-0.45}"
export MOONALT_YOLO_IOU="${MOONALT_YOLO_IOU:-0.55}"
export MOONALT_YOLO_MAXDET="${MOONALT_YOLO_MAXDET:-60}"
export MOONALT_YOLO_RETINA="${MOONALT_YOLO_RETINA:-0}"

# Output
export MOONALT_BT_SEND_RAW="${MOONALT_BT_SEND_RAW:-1}"
export MOONALT_LOG_EVERY="${MOONALT_LOG_EVERY:-10}"

# Prefer venv python, fallback to python3
PY_BIN="${PYTHON:-}"
if [[ -z "${PY_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PY_BIN=".venv/bin/python";
  elif command -v python >/dev/null 2>&1; then PY_BIN="python";
  else PY_BIN="python3"; fi
fi

exec "${PY_BIN}" -u -m app.worker
