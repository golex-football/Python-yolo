

Python YOLO – Connection-Based Worker

This repository contains a minimal, production-ready YOLO worker designed to sit between Capture and BroadTrack using ZMQ + Protobuf, as defined in the Connection-develop module.

It only includes what is required for the live pipeline and intentionally removes all viewers, test captures, and demo scripts.


---

🎯 Purpose

Capture (C++ / Connection)
   └── ZMQ + Protobuf (InputFrame)
        ↓
   YOLO Worker (this repo)
        ↓
   ZMQ + Protobuf (YOLO output)
        └── BroadTrack (C++)

Real-time, frame-by-frame processing

One input message → one YOLO inference → one output message

No buffering, no global smoothing, no offline passes

CPU-first (GPU ready when drivers are available)



---

📁 Repository Structure

Python-yolo-connection/
├── app/
│   └── worker.py                 # Main entry point (ONLY runtime logic)
│
├── services/
│   └── connection/
│       ├── __init__.py
│       ├── frame_message_pb2.py  # Input protobuf (from Capture)
│       ├── yolo_packet_pb2.py    # Output protobuf (to BroadTrack)
│       ├── zmq_connect_pull.py   # ZMQ PULL wrapper (connect)
│       └── zmq_connect_push.py   # ZMQ PUSH wrapper (connect)
│
├── proto/                        # Reference only (not used at runtime)
│   ├── frame_message.proto
│   └── yolo_packet.proto
│
├── scripts/
│   └── run_worker.sh             # Production run script
│
├── requirements.txt
├── yolov8n-seg.pt                # YOLOv8 segmentation model
└── README.md


---

❌ What Was Removed (Intentionally)

These files do not exist in this repo by design:

Viewers (viewer.py, viewer_gui.py)

Python capture sources (capture_from_video.py, capture_from_v4l2.py)

Test / demo scripts

tmux helpers, debug runners, local tools


➡️ All capture and visualization is handled outside this repo.


---

🔌 Communication Details

ZMQ Pattern

Capture → YOLO

Pattern: PUSH → PULL

YOLO side: PULL + connect


YOLO → BroadTrack

Pattern: PUSH

YOLO side: PUSH + connect



Default Endpoints

IN  = ipc:///tmp/capture
OUT = ipc:///tmp/broadtrack_in.sock

> IPC requires Capture, YOLO, and BroadTrack to run on the same machine.




---

📦 Input Protobuf (from Capture)

Message: InputFrame

Required fields:

schema → must match expected schema (e.g. golex.vt.input_v1)

width

height

pixel_format → must be BGR24

frame_data → raw bytes


Validation performed by worker:

len(frame_data) == width * height * 3

If this fails, the frame is dropped with a log, not crashed.


---

📤 Output Protobuf (to BroadTrack)

One output message per input frame

Encoded using yolo_packet_pb2

Structure matches Connection-develop expectations


> If BroadTrack parsing fails, check that yolo_packet.proto is byte-for-byte identical to the C++ side.




---

🚀 Setup & Run

1) Create and activate virtual environment

cd Python-yolo-connection

python3 -m venv .venv
source .venv/bin/activate

# Ensure `python` exists (scripts expect it)
ln -sf "$(command -v python3)" .venv/bin/python

2) Install dependencies

pip install -U pip
pip install --default-timeout=1000 --retries 30 -r requirements.txt

3) Run the worker

bash scripts/run_worker.sh

Expected log:

[worker] READY
[worker] recv bytes: ...
[worker] parsed frame: schema=... w=... h=... pix=BGR24 bytes=...


---

🧪 Debugging & Logging

The worker logs every critical stage:

After recv() → confirms data arrival

After ParseFromString() → confirms protobuf validity

Frame size validation → catches width/height mismatches

Pixel format validation → avoids garbage frames


If YOLO receives data but does nothing:

The issue is parsing, not ZMQ

Compare proto files and pixel format first



---

🧠 CPU / GPU Behavior

GPU not required

Automatically runs on CPU if:


torch.cuda.is_available() == False

GPU can be enabled later without code changes.


---

✅ Guarantees

No hidden entry points

No unused files

No viewer-only logic

One clear main: app/worker.py

Fully aligned with Connection-develop



---
