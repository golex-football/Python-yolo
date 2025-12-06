#!/usr/bin/env bash
set -euo pipefail
export MOONALT_IPC_IN="ipc:///tmp/yolo_input.sock"
export MOONALT_V4L2_DEVICE="${MOONALT_V4L2_DEVICE:-/dev/video0}"
export MOONALT_V4L2_WIDTH="${MOONALT_V4L2_WIDTH:-1920}"
export MOONALT_V4L2_HEIGHT="${MOONALT_V4L2_HEIGHT:-1080}"
export MOONALT_CAPTURE_MAX_FPS="${MOONALT_CAPTURE_MAX_FPS:-30}"
python -u -m app.capture_from_v4l2
