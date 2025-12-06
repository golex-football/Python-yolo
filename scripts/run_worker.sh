#!/usr/bin/env bash
set -euo pipefail
rm -f /tmp/yolo_input.sock /tmp/broadtrack_in.sock
export MOONALT_IPC_IN="ipc:///tmp/yolo_input.sock"
export MOONALT_BROADTRACK_OUT="ipc:///tmp/broadtrack_in.sock"
export MOONALT_YOLO_IMGSZ=${MOONALT_YOLO_IMGSZ:-640}
export MOONALT_YOLO_EVERY_N=${MOONALT_YOLO_EVERY_N:-2}
export MOONALT_YOLO_CONF=${MOONALT_YOLO_CONF:-0.45}
export MOONALT_YOLO_IOU=${MOONALT_YOLO_IOU:-0.55}
export MOONALT_YOLO_MAXDET=${MOONALT_YOLO_MAXDET:-60}
export MOONALT_YOLO_RETINA=${MOONALT_YOLO_RETINA:-0}
export MOONALT_YOLO_HALF=${MOONALT_YOLO_HALF:-1}
export MOONALT_BT_SEND_RAW=${MOONALT_BT_SEND_RAW:-1}
export MOONALT_LOG_EVERY=${MOONALT_LOG_EVERY:-10}
python -u -m app.worker
