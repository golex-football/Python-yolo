#!/usr/bin/env bash
set -euo pipefail

PROJ="$HOME/moonalt-yolo"
cd "$PROJ"

# 1) Freeze Python deps (safe even if venv not active)
pip freeze > requirements.lock.txt || true

# 2) Keep YOLO weights with the repo (if cached)
mkdir -p weights
if [ -f "$HOME/.cache/ultralytics/weights/yolov8n-seg.pt" ]; then
  cp -n "$HOME/.cache/ultralytics/weights/yolov8n-seg.pt" weights/
fi

# 3) Save a sample env file (edit later if you want)
cat > .env.sample <<'ENV'
MOONALT_IPC_IN=ipc:///tmp/yolo_input.sock
MOONALT_BROADTRACK_OUT=ipc:///tmp/broadtrack_in.sock
MOONALT_YOLO_IMGSZ=640
MOONALT_YOLO_EVERY_N=2
MOONALT_BT_SEND_RAW=1
ENV

# 4) Create tar snapshot (exclude venv)
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$HOME/backups"
TAR="$HOME/backups/moonalt-yolo-$TS.tgz"
tar --exclude='.venv' -C "$HOME" -czf "$TAR" moonalt-yolo
sha256sum "$TAR" | tee "$TAR.sha256"

# 5) Also create a git bundle (all history) if repo exists
if [ -d .git ]; then
  git bundle create "$HOME/backups/moonalt-yolo-$TS.bundle" --all
fi

# 6) Copy to Windows for extra safety
WIN_DIR="/mnt/c/Users/sina/Backups/moonalt"
mkdir -p "$WIN_DIR"
cp "$TAR" "$TAR.sha256" "$WIN_DIR"/
[ -f "$HOME/backups/moonalt-yolo-$TS.bundle" ] && cp "$HOME/backups/moonalt-yolo-$TS.bundle" "$WIN_DIR"/ || true

echo
echo "Done."
echo "WSL backups:"
ls -lh "$HOME/backups/moonalt-yolo-$TS."{tgz,tgz.sha256} 2>/dev/null || true
[ -f "$HOME/backups/moonalt-yolo-$TS.bundle" ] && ls -lh "$HOME/backups/moonalt-yolo-$TS.bundle" || true
echo
echo "Also copied to: $WIN_DIR"
