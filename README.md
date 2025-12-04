📄 README.md (suggested content)
# Python-yolo – ZMQ YOLO Segmentation Worker

This repo contains a GPU-accelerated YOLOv8 segmentation worker that:

- Receives **raw frames** over ZeroMQ (ZMQ) from a capture module (e.g. DeckLink C++ app).
- Runs a **YOLOv8 segmentation model** (Ultralytics).
- Outputs, for each frame:
  - The **original raw frame** (unchanged, BGR24),
  - A **player mask** (GRAY8, 0/255),
  - A list of **bounding boxes** (players etc.).
- Sends these results over ZMQ to downstream consumers (e.g. **BroadTrack**).

All of this is wrapped in a **Docker image** so it’s easy to run on any machine with a GPU.

## Quick start (Docker only, no Python setup)

1. Install **Docker** (and NVIDIA driver + container toolkit if on Linux).
2. Pull the prebuilt image:

   ```bash
   docker pull sinaabv80/python-yolo:latest


Run the YOLO ZMQ worker:

```bash
docker run --rm -it \
  --gpus all \
  -p 5555:5555 \
  -p 5556:5556 \
  golexfootball/python-yolo:latest
 ```
---

## 1. Requirements

### Hardware

- NVIDIA GPU (e.g. RTX 4060 / 4090 or similar)

### Software

- **Docker Desktop** (Windows) or Docker (Linux)
- **NVIDIA driver** installed on host
- **NVIDIA Container Toolkit** on Linux (for `--gpus all` support)

No Python setup is needed on the host if you use Docker.

---

## 2. Clone the Repo

```bash
git clone https://github.com/golex-football/Python-yolo.git
cd Python-yolo


Repo structure (simplified):

.
├─ app/
│  ├─ yolo_worker_zmq.py      # main worker
│  ├─ debug_downstream_zmq.py # optional debug consumer
│  └─ capture_obs_zmq.py      # legacy OBS capture, not used in Docker flow
├─ docs/
│  └─ INSTRUCTIONS.md         # internal notes
├─ Dockerfile                 # builds GPU-enabled container
├─ requirements.txt           # Python deps
└─ README.md                  # you are here

3. Build the Docker Image

From the repo root:

docker build -t python-yolo:latest .


This will:

Pull nvidia/cuda:12.1.0-runtime-ubuntu22.04

Install Python + Torch (CUDA 12.1 build) + Ultralytics + OpenCV + pyzmq

Copy the project into /workspace in the image

Configure the container to run app/yolo_worker_zmq.py by default

4. Run the Worker (Docker)

Run the worker with GPU access:

docker run --rm -it \
  --gpus all \
  -p 5555:5555 \
  -p 5556:5556 \
  python-yolo:latest


You should see logs like:

[WORKER] MODEL_NAME = yolov8s-seg.pt, INFER_WIDTH = 960
[WORKER] mode = zmq
[WORKER] Torch CUDA available: True
[WORKER] Using CUDA device 0
[WORKER][ZMQ] ZMQ PULL connected to tcp://0.0.0.0:5555
[WORKER] ZMQ PUSH bound to tcp://0.0.0.0:5556


At this point the worker is:

Listening on tcp://0.0.0.0:5555 (mapped to host localhost:5555) for raw frames

Sending results to tcp://0.0.0.0:5556 (mapped to host localhost:5556)

5. ZMQ Contracts (for C++ developers)

There are two ZMQ directions:

Capture → YOLO worker

YOLO worker → BroadTrack / downstream

All communication uses PUSH / PULL sockets (fast, lossy, “UDP-like” behavior).

5.1 Capture Side → Worker (input)

Worker socket: PULL

Endpoint (inside container): tcp://0.0.0.0:5555

From host / C++ side: tcp://127.0.0.1:5555

Message type: 2-part multipart

Socket setup (C++ pseudo):

zmq::context_t ctx(1);
zmq::socket_t sock(ctx, ZMQ_PUSH);
sock.connect("tcp://127.0.0.1:5555");


Per frame, send:

Part 0 – JSON meta (UTF-8 string):

{
  "schema":       "moonalt.raw_frame_v1",
  "frame_id":     1234,
  "timestamp_ms": 1733345678123,
  "width":        1920,
  "height":       1080,
  "pixel_format": "BGR24"
}


Part 1 – Raw frame bytes:

Layout: BGR24, row-major

Size: width * height * 3 bytes

C++ example (meta + frame):

nlohmann::json meta;
meta["schema"]       = "moonalt.raw_frame_v1";
meta["frame_id"]     = frame_id;     // monotonic int
meta["timestamp_ms"] = unix_ms;      // optional but recommended
meta["width"]        = width;
meta["height"]       = height;
meta["pixel_format"] = "BGR24";

std::string meta_str = meta.dump();

zmq::message_t meta_msg(meta_str.size());
memcpy(meta_msg.data(), meta_str.data(), meta_str.size());

zmq::message_t frame_msg(frame_bgr.size()); // vector<uint8_t> frame_bgr;
memcpy(frame_msg.data(), frame_bgr.data(), frame_bgr.size());

// Send: [meta, frame]
sock.send(meta_msg, zmq::send_flags::sndmore);
sock.send(frame_msg, zmq::send_flags::none);


Important:

frame_bgr.size() must equal width * height * 3

No resizing in C++ unless you update width/height accordingly

frame_id should be strictly increasing (0,1,2,3,…) – it’s preserved in output

5.2 Worker → BroadTrack / Downstream (output)

Worker socket: PUSH

Endpoint (inside container): tcp://0.0.0.0:5556

From host / C++ side: tcp://127.0.0.1:5556

Message type: 3-part multipart

Socket setup (C++ pseudo):

zmq::context_t ctx(1);
zmq::socket_t sock(ctx, ZMQ_PULL);
sock.connect("tcp://127.0.0.1:5556");


Per frame, receive 3 parts:

Part 0 – JSON meta (UTF-8 string)

Example structure:

{
  "schema":       "moonalt.yolo_output_v1",
  "frame_id":     1234,
  "width":        1920,
  "height":       1080,
  "pixel_format": "BGR24",
  "mask_format":  "GRAY8",
  "bboxes": [
    [x1, y1, x2, y2],
    [x1, y1, x2, y2]
  ]
}


Part 1 – Raw frame bytes

Same as input: BGR24

Size: width * height * 3 bytes

This is the original frame, unchanged.

Part 2 – Mask bytes

GRAY8 mask (0 = background, 255 = player/foreground)

Size: width * height bytes

C++ example receive path:

std::vector<zmq::message_t> parts;
zmq::recv_multipart(sock, std::back_inserter(parts));

if (parts.size() != 3) {
    // handle error
}

std::string meta_str(static_cast<char*>(parts[0].data()), parts[0].size());
nlohmann::json meta = nlohmann::json::parse(meta_str);

int frame_id = meta["frame_id"];
int width    = meta["width"];
int height   = meta["height"];

// Part 1: raw BGR frame
uint8_t* frame_data = static_cast<uint8_t*>(parts[1].data());
// Wrap into cv::Mat (no copy):
cv::Mat frame_bgr(height, width, CV_8UC3, frame_data);

// Part 2: mask GRAY8
uint8_t* mask_data = static_cast<uint8_t*>(parts[2].data());
cv::Mat mask_gray(height, width, CV_8UC1, mask_data);

// Convert bboxes JSON -> vector<cv::Rect>
std::vector<cv::Rect> boxes;
for (auto& bb : meta["bboxes"]) {
    int x1 = bb[0];
    int y1 = bb[1];
    int x2 = bb[2];
    int y2 = bb[3];
    boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
}

// Now you have:
//   - frame_bgr : original frame
//   - mask_gray : segmentation mask
//   - boxes     : bounding boxes
// You can feed these into BroadTrack or other systems.

6. Changing ZMQ Endpoints (Optional)

By default, the Docker image uses:

ENV ZMQ_IN_ENDPOINT=tcp://0.0.0.0:5555
ENV ZMQ_OUT_ENDPOINT=tcp://0.0.0.0:5556


You can override these when running the container:

docker run --rm -it --gpus all \
  -e ZMQ_IN_ENDPOINT=tcp://0.0.0.0:7000 \
  -e ZMQ_OUT_ENDPOINT=tcp://0.0.0.0:7001 \
  -p 7000:7000 -p 7001:7001 \
  python-yolo:latest


Then your C++ apps should use tcp://127.0.0.1:7000 (input) and tcp://127.0.0.1:7001 (output).

7. Dev Notes

The worker is optimized for throughput over reliability:

Uses PUSH/PULL sockets.

Frames can be dropped if the consumer is too slow.

frame_id propagates from capture → YOLO → output so you can:

Detect dropped frames,

Reuse previous data for missing frames on the Unreal side if needed.

Model name and resolution are configured inside app/yolo_worker_zmq.py.

If you’re integrating from C++ and need small adjustments to the schema or ports, keep the same general contract (multipart, JSON + raw buffers) to avoid breaking the Python side.


---

## What you should do now

1. Open `C:\Users\sina\Desktop\moonalt-soccer-pipeline\README.md`.
2. Replace its content with the block above.
3. Then in PowerShell:

   ```powershell
   cd C:\Users\sina\Desktop\moonalt-soccer-pipeline
   git add README.md
   git commit -m "Add README with Docker + ZMQ usage"
   git push
