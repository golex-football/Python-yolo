#!/usr/bin/env bash
set -euo pipefail
export MOONALT_BROADTRACK_OUT="ipc:///tmp/broadtrack_in.sock"
export MOONALT_VIEWER_MP4="${MOONALT_VIEWER_MP4:-$HOME/Videos/moonalt_out.mp4}"
export MOONALT_VIEWER_FPS="${MOONALT_VIEWER_FPS:-30}"
python -u -m app.viewer
