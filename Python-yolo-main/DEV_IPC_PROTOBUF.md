# Dev notes: IPC + Protobuf refactor (WSL/Linux)

## 1) Create WSL env
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pyzmq protobuf grpcio-tools ultralytics opencv-python
```

## 2) Compile protobuf
```bash
bash scripts/compile_proto.sh
```

This generates: `services/proto/messages_pb2.py`

## 3) Run the pipeline (IPC sockets)
- Producer (OBS): sends raw frames to `ipc:///tmp/moonalt_raw`
```bash
python app/capture_obs_zmq.py --mode obs --camera-index 0
```

- Worker: pulls raw frames, runs YOLO, pushes output to `ipc:///tmp/moonalt_out`
```bash
python app/yolo_worker_zmq.py --mode zmq
```

- Debug consumer: pulls worker output and visualizes
```bash
python app/debug_downstream_zmq.py
```

## Notes
- We kept the multipart structure:
  - Raw producer -> worker: `[RawFrameMetaV1, frame_bytes]`
  - Worker -> downstream: `[YoloOutputV1, frame_bytes, mask_bytes]`
- Endpoints changed from TCP to IPC under `/tmp`. Ensure the same machine processes.
- If you need TCP again, change endpoints in the files or via env vars and use the same ZmqPush/ZmqPull wrappers.
