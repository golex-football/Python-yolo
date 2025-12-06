[README.md](https://github.com/user-attachments/files/23997810/README.md)

# ✅ **README.md (Final — For Your ZIP Exactly As It Is)**


---

# # 🔷 YOLO Mask & Boxes Worker (Capture → YOLO → BroadTrack)

This project implements a **real-time YOLOv8 segmentation & detection worker**.
It is designed to operate in a processing pipeline:

1. **Capture (C++):** Sends raw frames via ZeroMQ + Protobuf
2. **YOLO Worker (Python):** Performs segmentation + bounding box detection
3. **BroadTrack / Any Consumer:** Receives YOLO output via ZeroMQ + Protobuf

The system is intended to run fully inside **WSL2 Ubuntu**.

---

# # 📁 Project Structure (matches your ZIP)

```
Python-yolo/
│
├── app/
│   ├── worker.py                # Main YOLO worker
│   ├── viewer.py                # Visual debug output
│   ├── capture_from_video.py    # Test input using an MP4
│
├── services/
│   └── connection/
│       ├── frame_message.proto      # Protobuf message used by C++ Capture
│       ├── frame_message_pb2.py     # Auto-generated protobuf Python file
│       ├── bt_packet_pb2.py         # Output message for BroadTrack
│       ├── zmq_bind_pull.py         # ZMQ PULL implementation (bind)
│       ├── zmq_bind_push.py         # ZMQ PUSH implementation (connect)
│       ├── zmq_connect_pull.py      # Additional ZMQ helper
│
├── weights/
│   └── yolov8n-seg.pt               # YOLOv8 segmentation model
│
├── requirements.txt
└── README.md
```

Everything in this README assumes the project looks exactly like above.

---

# # 📌 1. Requirements (WSL2 Ubuntu)

Install system dependencies:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip protobuf-compiler python3-grpc-tools
```

---

# # 📌 2. Python Setup

```bash
cd ~/Python-yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# # 📌 3. Rebuilding Protobuf (if needed)

You only need this if `frame_message.proto` is modified.

```bash
cd ~/Python-yolo/services/connection
python3 -m grpc_tools.protoc -I. --python_out=. frame_message.proto
```

This regenerates:

```
frame_message_pb2.py
```

---

# # 📌 4. IPC Endpoints (Pipeline Standard)

All parts of the pipeline **must use exactly these two addresses**:

```
MOONALT_IPC_IN          = ipc:///tmp/moonalt_yolo_in
MOONALT_BROADTRACK_OUT  = ipc:///tmp/moonalt_bt_in
```

### Used by:

| Component             | Address                  | Mode           |
| --------------------- | ------------------------ | -------------- |
| Capture (C++)         | `MOONALT_IPC_IN`         | PUSH → connect |
| YOLO Worker           | `MOONALT_IPC_IN`         | PULL → bind    |
| YOLO Worker           | `MOONALT_BROADTRACK_OUT` | PUSH → connect |
| BroadTrack / Consumer | `MOONALT_BROADTRACK_OUT` | PULL → bind    |

These must match across ALL apps.

---

# # 📌 5. Running the YOLO Worker

### Terminal 1:

```bash
cd ~/Python-yolo
source .venv/bin/activate

export MOONALT_IPC_IN="ipc:///tmp/moonalt_yolo_in"
export MOONALT_BROADTRACK_OUT="ipc:///tmp/moonalt_bt_in"

python -u app/worker.py
```

The worker now:

* receives frames from Capture
* runs YOLO segmentation & detection
* sends results to BroadTrack (or any consumer)

---

# # 📌 6. Running the Capture (C++)

### Terminal 2:

```bash
export MOONALT_IPC_IN="ipc:///tmp/moonalt_yolo_in"
./capture_binary
```

The Capture must send this protobuf message:

```proto
message InputFrame {
  string schema        = 1;
  int64  frame_id      = 2;
  double timestamp     = 3;
  uint32 width         = 4;
  uint32 height        = 5;
  string pixel_format  = 6; // e.g. "BGR24"
  bytes  frame_data    = 7;
}
```

This matches exactly the Python pb2 in your ZIP.

---

# # 📌 7. Running BroadTrack / Any Consumer

### Terminal 3:

```bash
export MOONALT_BROADTRACK_OUT="ipc:///tmp/moonalt_bt_in"
```

Example minimal Python receiver:

```python
from services.connection.zmq_bind_pull import ZMQBindPull
from services.connection.bt_packet_pb2 import BtPacket
import zmq, os

ctx = zmq.Context.instance()
endpoint = os.environ["MOONALT_BROADTRACK_OUT"]

pull = ZMQBindPull(ctx, endpoint)

while True:
    data = pull.recv_bytes()
    pkt = BtPacket()
    pkt.ParseFromString(data)
    print("Frame:", pkt.frame_id)
```

---

# # 📌 8. Test Without Capture (Local MP4)

```bash
cd ~/Python-yolo
source .venv/bin/activate
python -u app/capture_from_video.py --video yourfile.mp4
```

This simulates Capture and feeds frames to the worker.

---

# # 📌 9. Test Without BroadTrack (Viewer)

```bash
cd ~/Python-yolo
source .venv/bin/activate

export MOONALT_BROADTRACK_OUT="ipc:///tmp/moonalt_bt_in"
python -u app/viewer.py
```

Viewer visualizes worker output.

---

# # 📌 10. Notes

* The worker reads/writes protobuf messages exactly matching the Capture and BroadTrack expectations
* ZeroMQ implementation follows bind/connect best-practices
* The system is fully compatible with WSL2 real-time pipelines

---

# 🔷 End of README
