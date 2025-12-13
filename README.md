# Python YOLO (Capture → YOLO → BroadTrack) — single-machine IPC

Pipeline (all on the same machine):

Capture (PUSH bind) -> ipc:///tmp/capture -> YOLO (PULL connect) -> ipc:///tmp/broadtrack -> BroadTrack (PULL bind)

## Input messages
The worker can parse (auto-detect):
- CaptureRawFrameV1 (Capture repo)
- CapturePacket (BroadTrack schema)
- InputFrame (legacy)

## Output message (YOLO -> BroadTrack)
The worker sends BroadTrack's protobuf schema:
- golex.virtualtracking.model.YoloPacket
  - raw_frame: BGR24
  - mask_frame: GRAY8
  - box_frame: repeated Box(x,y,width,height,score,class_id)

## ZMQ defaults
Defaults are already correct for a single machine:
- IN:  ipc:///tmp/capture
- OUT: ipc:///tmp/broadtrack
- OUT_MODE: connect  (BroadTrack binds)

You can override with env vars:
- MOONALT_IPC_IN
- MOONALT_BROADTRACK_OUT
- MOONALT_OUT_MODE=connect|bind

## Do I need to build protos?
No. This repo includes the Python protobuf bindings it needs. No `protoc` step is required to run the worker.

## Run
Start in this order:
1) BroadTrack
2) Capture
3) YOLO worker

YOLO (uses your existing venv):
```bash
cd ~/Projects/Python-yolo
source .venv/bin/activate
ln -sf "$(command -v python3)" .venv/bin/python
export CUDA_VISIBLE_DEVICES=""

# single-machine IPC defaults
export MOONALT_IPC_IN="ipc:///tmp/capture"
export MOONALT_BROADTRACK_OUT="ipc:///tmp/broadtrack"
export MOONALT_OUT_MODE="connect"

bash scripts/run_worker.sh
```
