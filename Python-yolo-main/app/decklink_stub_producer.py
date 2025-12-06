"""
This file is ONLY a reference implementation.

Your C++ DeckLink application must replicate EXACTLY this ZMQ contract:

PUSH → tcp://127.0.0.1:5555

Multipart message (2 parts):
  part[0] = JSON meta:
      {
        "schema": "moonalt.raw_frame_v1",
        "frame_id": <int>,
        "timestamp_ms": <int>,
        "width": <int>,
        "height": <int>,
        "pixel_format": "BGR24"
      }

  part[1] = raw BGR24 frame bytes
"""

import cv2
import zmq
import time
import json

ZMQ_ENDPOINT = "tcp://127.0.0.1:5555"

def main():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.connect(ZMQ_ENDPOINT)

    cap = cv2.VideoCapture(0)  # webcam for testing only

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]

        meta = {
            "schema": "moonalt.raw_frame_v1",
            "frame_id": frame_id,
            "timestamp_ms": int(time.time() * 1000),
            "width": w,
            "height": h,
            "pixel_format": "BGR24"
        }

        meta_bytes = json.dumps(meta).encode()
        frame_bytes = frame.tobytes()

        socket.send_multipart([meta_bytes, frame_bytes])

        frame_id += 1

if __name__ == "__main__":
    main()
