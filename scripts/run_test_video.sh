#!/usr/bin/env bash
set -euo pipefail
export MOONALT_IPC_IN="ipc:///tmp/yolo_input.sock"
: "${MOONALT_TEST_VIDEO:?Set MOONALT_TEST_VIDEO to a readable video path}"
python -u -m app.capture_from_video
