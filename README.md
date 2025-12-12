# Python YOLO Worker (Connection-develop compatible)

This repo contains **only** the production YOLO worker used in the live pipeline:

**Capture (C++) → ZMQ/Protobuf → YOLO Worker (Python) → ZMQ/Protobuf → BroadTrack (C++)**

It is intentionally minimal: **no viewer**, **no Python capture**, no demo scripts.

## Default endpoints
- **Input** (from Capture): `ipc:///tmp/capture`  *(Capture binds PUSH, worker connects PULL)*
- **Output** (to BroadTrack): `ipc:///tmp/broadtrack_in.sock` *(BroadTrack binds PULL, worker connects PUSH)*

You can override these using env vars:
- `MOONALT_IPC_IN`
- `MOONALT_BROADTRACK_OUT`

## Install + run
```bash
cd ~/Projects/Python-yolo-connection

python3 -m venv .venv
source .venv/bin/activate

# Some distros don't provide `python` inside venv by default
ln -sf "$(command -v python3)" .venv/bin/python

pip install -U pip
pip install --default-timeout=1000 --retries 30 -r requirements.txt

bash scripts/run_worker.sh
```

## Useful env vars
```bash
# endpoints
export MOONALT_IPC_IN="ipc:///tmp/capture"
export MOONALT_BROADTRACK_OUT="ipc:///tmp/broadtrack_in.sock"

# YOLO tuning
export MOONALT_YOLO_IMGSZ=640
export MOONALT_YOLO_EVERY_N=1
export MOONALT_YOLO_CONF=0.45
export MOONALT_YOLO_IOU=0.55
export MOONALT_YOLO_MAXDET=60
export MOONALT_YOLO_RETINA=0

# output/logging
export MOONALT_BT_SEND_RAW=1
export MOONALT_LOG_EVERY=10
```

## Notes
- If CUDA drivers are not installed (or not working), the worker automatically falls back to **CPU**.
- Input expects `pixel_format == "BGR24"` and checks `len(frame_data) == width*height*3`.

## Troubleshooting
### Worker prints READY but receives nothing
1) Make sure Capture is running and bound the IPC path:
```bash
ls -la /tmp/capture*
```

2) Remove stale IPC files before re-running:
```bash
rm -f /tmp/capture /tmp/capture.sock /tmp/broadtrack_in.sock
```

### "Problem is parsing after recv"
The worker logs:
- received byte length
- protobuf parse errors (if any)
- pixel_format mismatch
- frame_data size mismatch

## Protocol
Authoritative `.proto` files (matching Connection-develop) are in `proto/`.
Runtime Python generated modules are in `services/connection/*_pb2.py`.
