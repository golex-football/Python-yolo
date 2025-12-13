# Python YOLO (Pipeline-ready)

This repo is the YOLO worker for the **Capture → YOLO → BroadTrack** pipeline.

## What was applied (Nazari notes)
- Added a log **immediately after `recv()`** so we confirm data arrival.
- Added a log **right after protobuf parse** to confirm parse correctness.
- Added strict validation (schema, pixel_format, byte-length) and safe drop with logs (no crash).
- ZMQ roles are now:
  - **PULL = connect**
  - **PUSH = bind** (default, configurable)

## ZMQ Endpoints

Defaults:
- **IN**  (Capture → YOLO): `ipc:///tmp/capture`
- **OUT** (YOLO → BroadTrack): `ipc:///tmp/broadtrack`

> Capture-main binds **PUSH** on `ipc:///tmp/capture`, so YOLO must **PULL connect** (correct).

### Important about OUT bind/connect
BroadTrack-main (your uploaded repo) currently **binds PULL** on `ipc:///tmp/broadtrack`.
That configuration requires YOLO to **PUSH connect**.

Because you requested **PUSH bind**, we made the worker default to bind and added:
- `MOONALT_OUT_MODE=bind|connect`

If your BroadTrack side still binds, run YOLO with:
```bash
export MOONALT_OUT_MODE=connect
```

## Proto compatibility (fixed to your uploaded proto files)
This worker uses dynamic `*_pb2.py` generated to match exactly:
- `frame_message.proto` → `InputFrame` fields 1..7
- `yolo_packet.proto` → `YoloPacket` fields 1..8

No field numbers were changed.

## Setup

```bash
cd Python-yolo-main
python3 -m venv .venv
source .venv/bin/activate
ln -sf "$(command -v python3)" .venv/bin/python

pip install -U pip
pip install --default-timeout=1000 --retries 30 -r requirements.txt
```

## Run

```bash
bash scripts/run_worker.sh
```

## GPU Torch install (optional)
When you can install NVIDIA drivers, install GPU wheels inside the venv:

CUDA 12.6:
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

CUDA 11.8:
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## Output format
`YoloPacket.boxes` is a flat float list:
`[x,y,w,h,score,class_id, x,y,w,h,score,class_id, ...]`

Mask is a **union** of all instance masks, resized to raw (w,h), bytes are 0/255.


## BroadTrack-compatible mode
This repo's worker outputs `golex.virtualtracking.model.YoloPacket` (BroadTrack schema) to `MOONALT_BT_OUT` (default `ipc:///tmp/broadtrack_in.sock`). Connection can forward these bytes unchanged to BroadTrack's `ipc:///tmp/broadtrack`.
