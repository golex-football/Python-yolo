````markdown
# Python-yolo – Live Soccer YOLO Worker (Frames → Masks/Boxes → ZMQ/Protobuf)

This repo is a **YOLO-based worker** that takes live video frames, runs segmentation/detection, and streams the results over **ZeroMQ + Protobuf** so other systems (e.g. **BroadTrack C++**, Unreal, etc.) can consume them in real time.

It’s designed to sit between:

`Capture (DeckLink / RTSP / file) → YOLO worker (this repo) → BroadTrack / Unreal / tools`

---

## 1. Features

- 🔌 **ZMQ + Protobuf pipeline**
  - Inbound frames: `InputFrame` messages (`raw` BGR frames)
  - Outbound results: `BtPacket` messages (`raw`, `mask`, `boxes[]`)
- 🧠 **YOLOv8 segmentation**
  - Default model: `yolov8n-seg.pt` (small, fast)
  - Runs on GPU via PyTorch
- 🎛️ **Configurable speed / quality**
  - Image size cap (`MOONALT_YOLO_IMGSZ`)
  - Run every Nth frame (`MOONALT_YOLO_EVERY_N`)
  - Capture FPS cap (`MOONALT_CAPTURE_MAX_FPS`)
- 👀 **Debug viewer**
  - 3 OpenCV windows: `raw`, `boxes`, `mask`
  - Display FPS capped by `MOONALT_VIEWER_FPS`
- 🧩 **BroadTrack-ready**
  - Output socket + Protobuf schema is designed so a C++ BroadTrack process can attach as a ZMQ `PULL` client and consume `BtPacket` in real time.

---

## 2. Folder layout

```text
app/
  capture_from_video.py   # reads video, sends InputFrame messages to worker
  worker.py               # YOLO worker: InputFrame -> BtPacket
  viewer.py               # debug viewer: shows raw / boxes / mask

services/
  connection/
    zmq_bind_pull.py      # helper for PULL + bind
    zmq_connect_push.py   # helper for PUSH + connect
    zmq_bind_push.py      # helper for PUSH + bind (outgoing BtPacket)
    zmq_connect_pull.py   # helper for PULL + connect
    frame_message_pb2.py  # generated from frame_message.proto (InputFrame)
    bt_packet_pb2.py      # generated from bt_packet.proto (BtPacket)
  ...

proto/
  frame_message.proto     # defines InputFrame
  bt_packet.proto         # defines BtPacket

scripts/
  run_test_video.sh       # capture from local video file → worker
  run_worker.sh           # launch YOLO worker
  run_viewer.sh           # launch debug viewer

requirements.txt          # Python dependencies (no pinned torch)
limits.env                # optional tuning (FPS, imgsz, etc.)
yolov8n-seg.pt            # default YOLOv8 segmentation model
sitecustomize.py          # minor Python tweaks (safe to ignore)
````

---

## 3. Requirements

* OS: Linux (tested on **Ubuntu 22.04**, including **WSL2**).
* Python: **3.10** recommended.
* GPU: NVIDIA with drivers that support the chosen PyTorch wheels.
* Recommended:

  * For WSL2 GUI: WSLg on Windows 11, or an X server (VcXsrv) on Windows 10.

---

## 4. Installation

### 4.1. Clone / unzip

Copy the repo to your machine, then in a terminal:

```bash
cd /path/to/Python-yolo
```

### 4.2. Create a virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

> Always activate `.venv` before running scripts.

### 4.3. Install PyTorch (GPU, CUDA 11.8)

From the [official PyTorch wheels], for Linux + pip + CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4.4. Install project dependencies

```bash
pip install -r requirements.txt
```

Note:

* `protobuf` is pinned to **3.20.x** to stay compatible with the generated `*_pb2.py` files.
* If you regenerate protos with a newer `protoc` (≥ 3.19) you can update this pin later.

---

## 5. Running the pipeline (test video)

The typical dev setup uses **3 terminals**, all in the repo and with `.venv` active.

### 5.1. Terminal 1 – Capture from test video

Set the path to a test video file (e.g. soccer match) and a max capture FPS:

```bash
cd /path/to/Python-yolo
source .venv/bin/activate

export MOONALT_TEST_VIDEO=/absolute/path/to/soccer_match.mp4
export MOONALT_CAPTURE_MAX_FPS=25   # match the video FPS

bash scripts/run_test_video.sh
```

This script:

* Opens the file via OpenCV.
* Wraps each frame in an `InputFrame` protobuf.
* Sends frames via **ZMQ PUSH** to the worker’s input socket.

You should see logs like:

```text
[capture_video] opening: /.../soccer_match.mp4
[capture_video] sent frame_id=0 (1920x1080) send_fps=25.0
...
```

---

### 5.2. Terminal 2 – YOLO worker

```bash
cd /path/to/Python-yolo
source .venv/bin/activate

# optional tuning (see section 6)
# export MOONALT_YOLO_IMGSZ=640
# export MOONALT_YOLO_EVERY_N=1

bash scripts/run_worker.sh
```

This script:

* Listens for `InputFrame` messages.
* Runs YOLOv8 segmentation on selected frames.
* Builds a `BtPacket` containing:

  * `raw` BGR frame
  * `mask` GRAY8 segmentation mask
  * `boxes[]` detections with coordinates, class id, score
* Sends `BtPacket` via **ZMQ PUSH** to the output endpoint.

You should see logs like:

```text
[worker] READY. Waiting for InputFrame on IPC…
[worker] frame_id=0 size=(1920,1080) mask=(1080,1920) boxes=12 proc_fps=...
...
```

---

### 5.3. Terminal 3 – Debug viewer (3 windows)

```bash
cd /path/to/Python-yolo
source .venv/bin/activate

export MOONALT_VIEWER_FPS=25   # target display FPS
bash scripts/run_viewer.sh
```

The viewer:

* Connects to the worker’s output socket as a ZMQ `PULL`.

* Receives `BtPacket` messages.

* Shows 3 OpenCV windows:

  * `raw`   → original frame
  * `boxes` → frame with green detection rectangles
  * `mask`  → grayscale segmentation mask

* Paces display using `MOONALT_VIEWER_FPS` (doesn’t run faster than this).

Press `q` or `Esc` in any window to exit.

> If windows are empty, make sure capture & worker are running and the ZMQ sockets match (see section 7).

---

## 6. Performance tuning knobs

All tuning is via environment variables (either in your shell or via `limits.env`).

### 6.1. Capture

* `MOONALT_TEST_VIDEO` – path to a local video file.
* `MOONALT_CAPTURE_MAX_FPS` – cap on how fast capture sends frames.

  * If worker can’t keep up, frames are dropped to stay near real time.

### 6.2. YOLO worker

Key envs:

* `MOONALT_YOLO_IMGSZ`

  * Max input size for YOLO (e.g. `640`, `512`).
  * Lower = faster, less detail; higher = slower, better masks.

* `MOONALT_YOLO_EVERY_N`

  * `1` → run YOLO on every frame
  * `2` → YOLO on every 2nd frame (half the YOLO load)
  * Good for balancing FPS vs cost.

Other settings (in `limits.env` or directly in `worker.py`) can control thresholds, classes, etc.

### 6.3. Viewer

* `MOONALT_VIEWER_FPS`

  * Target display FPS (e.g. `25`, `30`).
  * If worker is faster than this, viewer throttles itself.
  * If worker is slower than this, you get whatever FPS the worker can supply.

---

## 7. ZMQ & Protobuf contracts (for C++ / BroadTrack devs)

### 7.1. Ports / endpoints

By default:

* **Capture → Worker**

  * Endpoint (input): `MOONALT_YOLO_IN` env var, default something like `ipc:///tmp/yolo_input.sock` (see code).
  * Pattern:

    * Capture: `PUSH + connect`
    * Worker:  `PULL + bind`

* **Worker → BroadTrack / Viewer**

  * Endpoint (output): `MOONALT_BROADTRACK_OUT`, default: `ipc:///tmp/broadtrack`
  * Pattern:

    * Worker:  `PUSH + bind`
    * Consumer (Viewer or BroadTrack C++): `PULL + connect`

You can override endpoints via env variables, e.g.:

```bash
export MOONALT_BROADTRACK_OUT="tcp://127.0.0.1:5557"
```

### 7.2. InputFrame (capture → worker)

Defined in `proto/frame_message.proto`, generated into `services/connection/frame_message_pb2.py`.

**Conceptual structure:**

```proto
message InputFrame {
  string schema        = 1;   // e.g. "moonalt.input_v1"
  int64  frame_id      = 2;   // 0,1,2,...
  double timestamp     = 3;   // optional, seconds or ms
  uint32 width         = 4;   // frame width
  uint32 height        = 5;   // frame height
  string pixel_format  = 6;   // e.g. "BGR24"
  bytes  frame_data    = 7;   // raw bytes: width * height * 3 for BGR24
}
```

Python capture sends `InputFrame` via ZMQ `PUSH`.

### 7.3. BtPacket (worker → BroadTrack / viewer)

Defined in `proto/bt_packet.proto`, generated into `services/connection/bt_packet_pb2.py`.

**Conceptual structure (simplified):**

```proto
message BtPacket {
  int64 frame_id = 1;

  message RawFrame {
    uint32 width        = 1;
    uint32 height       = 2;
    string pixel_format = 3;   // "BGR24"
    bytes  frame_data   = 4;   // width*height*3 bytes
  }

  message MaskFrame {
    uint32 width        = 1;
    uint32 height       = 2;
    string pixel_format = 3;   // "GRAY8"
    bytes  frame_data   = 4;   // width*height bytes
  }

  message Box {
    int32 x1       = 1;
    int32 y1       = 2;
    int32 x2       = 3;
    int32 y2       = 4;
    int32 cls      = 5;    // class id
    float score    = 6;    // confidence
  }

  RawFrame raw = 2;        // optional
  MaskFrame mask = 3;      // optional
  repeated Box boxes = 4;  // detections
}
```

The worker fills:

* `frame_id` from the incoming `InputFrame`.
* `raw` in BGR24 format.
* `mask` in GRAY8 format (same width/height).
* `boxes` from YOLO predictions.

### 7.4. C++ consumer example (BroadTrack)

A C++ app (e.g. BroadTrack wrapper) should:

1. Generate C++ classes from `bt_packet.proto` using `protoc`.
2. Connect as a ZMQ `PULL` to the output endpoint:

```cpp
zmq::context_t ctx(1);
zmq::socket_t pull(ctx, ZMQ_PULL);
pull.connect("ipc:///tmp/broadtrack");  // or value of MOONALT_BROADTRACK_OUT

while (true) {
    zmq::message_t msg;
    pull.recv(msg, zmq::recv_flags::none);

    moonalt::BtPacket pkt;
    pkt.ParseFromArray(msg.data(), msg.size());

    // pkt.frame_id
    // pkt.raw.width, pkt.raw.height, pkt.raw.pixel_format, pkt.raw.frame_data
    // pkt.mask.width, pkt.mask.height, pkt.mask.frame_data
    // pkt.boxes(i).x1, y1, x2, y2, cls, score
}
```

The worker sets `SNDHWM` on its PUSH socket and sends non-blocking; if the consumer is too slow, frames are dropped to keep the system near real time.

---

## 8. Regenerating protobufs

If you modify `.proto` files or want to regenerate with newer `protoc`:

```bash
cd /path/to/Python-yolo

# Python
protoc -I=proto --python_out=services/connection proto/frame_message.proto
protoc -I=proto --python_out=services/connection proto/bt_packet.proto
```

If you regenerate with `protoc >= 3.19.0`, you **can** upgrade the `protobuf` Python package beyond 3.20.x; otherwise keep the pin.

For C++ consumers (e.g. BroadTrack):

```bash
protoc -I=proto --cpp_out=/path/to/cpp/output proto/bt_packet.proto
```

---

## 9. Notes & future work

* Swap model:

  * Change `yolov8n-seg.pt` to another YOLOv8 model (e.g. `yolov8s-seg.pt`) in `worker.py`.
* Boxes-only mode:

  * You can switch to `yolov8n.pt` (no masks) for extra speed and optionally skip the `mask` field.
* ONNX / TensorRT:

  * The worker is a clean place to plug in an ONNX/TensorRT path instead of the Python YOLO model while keeping the same `InputFrame` → `BtPacket` contract and ZMQ topology.
