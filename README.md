Here are two ready-to-paste READMEs.

---

# README for the GitHub repo (`Python-yolo`)

```markdown
# Python-yolo — Moonalt YOLO → ZMQ → (BroadTrack/Viewer)

Fast YOLOv8-SEG worker that:
- receives raw frames over **ZeroMQ** (`InputFrame` protobuf, BGR24),
- runs detection/segmentation (players + ball),
- outputs to **ZeroMQ** a `BtPacket` protobuf with:
  - **raw** (optional passthrough BGR24),
  - **mask** (GRAY8 at source size),
  - **boxes** (XYXY in **source** coordinates).

Works on Linux/WSL. GPU (CUDA) recommended; CPU also works.

---

## Layout

```

app/
capture_from_video.py   # test feeder from an MP4
worker.py               # YOLO worker (ZMQ in -> ZMQ out)
viewer.py               # debug viewer (subscribes to worker output)
services/connection/
frame_message_pb2.py    # protobuf: InputFrame
bt_packet_pb2.py        # protobuf: BtPacket
zmq_bind_pull.py        # helper for bind/pull
zmq_bind_push.py        # helper for bind/push
zmq_connect_pull.py     # helper for connect/pull (optional)

````

> If you don’t see these files after cloning, unzip your snapshot and copy them over.

---

## Requirements

- Python **3.10–3.12**
- NVIDIA CUDA (optional, for GPU)
- Packages:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install ultralytics==8.* pyzmq opencv-python-headless protobuf
````

---

## Quick Start (video → worker → viewer)

### Terminal 1 — worker (YOLO)

```bash
cd ~/moonalt-yolo
source .venv/bin/activate

# endpoints
export MOONALT_IPC_IN="ipc:///tmp/yolo_input.sock"
export MOONALT_BROADTRACK_OUT="ipc:///tmp/broadtrack_in.sock"

# safe perf knobs
export MOONALT_YOLO_HALF=0        # keep FP32 (avoids half-precision fuse bugs)
export MOONALT_YOLO_IMGSZ=640
export MOONALT_YOLO_EVERY_N=2     # infer every N frames

python -u -m app.worker
```

### Terminal 2 — capture (test MP4 feeder)

```bash
cd ~/moonalt-yolo
source .venv/bin/activate
export MOONALT_IPC_IN="ipc:///tmp/yolo_input.sock"
export MOONALT_TEST_VIDEO="/mnt/c/Users/sina/Videos/soccer_reencode.mp4"
export MOONALT_CAPTURE_MAX_FPS=30
python -u -m app.capture_from_video
```

### Terminal 3 — viewer (debug only)

```bash
cd ~/moonalt-yolo
source .venv/bin/activate
export MOONALT_BROADTRACK_OUT="ipc:///tmp/broadtrack_in.sock"

# Optionally record an MP4 for inspection:
# export MOONALT_VIEWER_MP4="$HOME/Videos/moonalt_out.mp4"
# export MOONALT_VIEWER_FPS=30

python -u -m app.viewer
```

Stop any process with `Ctrl+C`.

---

## Live Pipeline (no MP4, just data to BroadTrack)

* **Don’t** run the viewer.
* Worker is already publishing `BtPacket` to `MOONALT_BROADTRACK_OUT`.
* Point your BroadTrack bridge/consumer to the same endpoint (e.g. `ipc:///tmp/broadtrack_in.sock` or a `tcp://ip:port` if you want network).

---

## Environment Variables (most used)

| Variable                  | Default                         | Notes                                                    |
| ------------------------- | ------------------------------- | -------------------------------------------------------- |
| `MOONALT_IPC_IN`          | `ipc:///tmp/yolo_input.sock`    | Where capture sends `InputFrame`                         |
| `MOONALT_BROADTRACK_OUT`  | `ipc:///tmp/broadtrack_in.sock` | Where worker pushes `BtPacket`                           |
| `MOONALT_YOLO_IMGSZ`      | `640`                           | YOLO input size (square)                                 |
| `MOONALT_YOLO_EVERY_N`    | `1`                             | Inference every N frames (N>1 reuses last results)       |
| `MOONALT_YOLO_HALF`       | `1`                             | Set to **0** on RTX 40xx/WSL if you see half/dtype error |
| `MOONALT_VIEWER_MP4`      | unset                           | If set, viewer writes MP4 to this path                   |
| `MOONALT_VIEWER_FPS`      | `30`                            | MP4 frame rate                                           |
| `MOONALT_CAPTURE_MAX_FPS` | unset                           | Limit capture push rate                                  |

---

## Message Schemas (summary)

**InputFrame** (from capture → worker)

* `schema: string` (e.g. `"moonalt.input_v1"`)
* `frame_id: int64` (0,1,2,…)
* `timestamp: double` (optional)
* `width, height: uint32`
* `pixel_format: string` (must be `"BGR24"`)
* `frame_data: bytes` (raw `width*height*3`)

**BtPacket** (from worker → BroadTrack/viewer)

* `schema: string` (e.g. `"moonalt.bt_packet_v1"`)
* `frame_id: int64`
* `raw` (optional; passthrough BGR24 at source size)
* `mask` (`GRAY8`, same size as source)
* `boxes` (XYXY, `score`, `class_id`; source coordinates)

---

## Troubleshooting

* **Half/FP16 error**: set `MOONALT_YOLO_HALF=0`.
* **OpenCV can’t open video**: confirm the path exists (`/mnt/c/...`), try a short, ASCII-only path, and ensure the file isn’t locked by another app.
* **Nothing printed by viewer**: ensure the worker is running and both use the same `MOONALT_BROADTRACK_OUT`.
* **WSL interop/browser issues**: `cat <<EOF | sudo tee /etc/wsl.conf
  [interop]
  enabled=true
  appendWindowsPath=true
  EOF

# Then restart WSL (PowerShell): wsl --shutdown

`

---

## License

TBD by repo owner.

````

---

# README for the ZIP snapshot folder  
`Python-yolo-e54e2c40474aaeba4087fdc4fcb6090c735bd91b/README.md`

```markdown
# Python-yolo ZIP Snapshot

This folder is a **working snapshot** of the Moonalt YOLO → ZMQ pipeline
(as tested locally). It’s meant to be unzipped and pushed into the
`golex-football/Python-yolo` repo or run directly.

## What’s inside

- `app/worker.py` — YOLOv8-SEG worker (ZMQ in → ZMQ out)
- `app/capture_from_video.py` — MP4 test feeder
- `app/viewer.py` — debug viewer (optional; can record MP4)
- `services/connection/*.py` — protobuf shims (`InputFrame`, `BtPacket`) and ZMQ helpers

> If you only need live pipeline to BroadTrack, you can skip the viewer.

## Run (three terminals)

### 1) Worker
```bash
cd /path/to/unzipped
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics==8.* pyzmq opencv-python-headless protobuf

export MOONALT_IPC_IN="ipc:///tmp/yolo_input.sock"
export MOONALT_BROADTRACK_OUT="ipc:///tmp/broadtrack_in.sock"

export MOONALT_YOLO_HALF=0
export MOONALT_YOLO_IMGSZ=640
export MOONALT_YOLO_EVERY_N=2

python -u -m app.worker
````

### 2) Capture (test video)

```bash
cd /path/to/unzipped
source .venv/bin/activate
export MOONALT_IPC_IN="ipc:///tmp/yolo_input.sock"
export MOONALT_TEST_VIDEO="/mnt/c/Users/sina/Videos/soccer_reencode.mp4"
export MOONALT_CAPTURE_MAX_FPS=30
python -u -m app.capture_from_video
```

### 3) Viewer (debug only)

```bash
cd /path/to/unzipped
source .venv/bin/activate
export MOONALT_BROADTRACK_OUT="ipc:///tmp/broadtrack_in.sock"

# Optional MP4:
# export MOONALT_VIEWER_MP4="$HOME/Videos/moonalt_out.mp4"
# export MOONALT_VIEWER_FPS=30

python -u -m app.viewer
```

