#!/usr/bin/env bash
set -euo pipefail
SESSION=moonalt_live
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd ~/moonalt-yolo && source .venv/bin/activate && scripts/run_worker.sh"
tmux split-window -v "cd ~/moonalt-yolo && source .venv/bin/activate && scripts/run_viewer.sh"
tmux split-window -h "cd ~/moonalt-yolo && source .venv/bin/activate && scripts/run_live_v4l2.sh"
tmux select-layout tiled
tmux attach -t "$SESSION"
