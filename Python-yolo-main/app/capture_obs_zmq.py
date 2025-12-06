"""
capture_obs_zmq.py

Raw frame producer / bridge for the Moonalt soccer pipeline.

Two modes:

  1) OBS mode  (default)
     - Reads frames from an OBS Virtual Camera using OpenCV.
     - Sends raw frames over ZMQ as `moonalt.raw_frame_v1` to the YOLO worker.

  2) DeckLink mode
     - Acts as a bridge between a C++ DeckLink capture app and the YOLO worker.
     - Receives frames from DeckLink app over ZMQ (`decklink.raw_frame_v1`).
     - Forwards them (without modifying the pixel data) as `moonalt.raw_frame_v1`.

This way you can switch between OBS virtual cam and the real DeckLink capture
feed by just changing the --mode flag (or an env var).
"""

import argparse
import json
import os
import time
from typing import Any, Dict

import cv2
import numpy as np
import zmq


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Default camera index for OBS virtual camera.
DEFAULT_CAMERA_INDEX = 1

# Outgoing endpoint toward YOLO worker.
# yolo_worker_zmq.py uses:
#   in_socket = ctx.socket(zmq.PULL)
#   in_socket.connect(ZMQ_IN_ENDPOINT)
# so here we BIND a PUSH socket on the same endpoint.
ZMQ_RAW_OUT = os.environ.get("ZMQ_RAW_OUT", "tcp://127.0.0.1:5555")

# Endpoint where we listen for DeckLink frames (when in --mode decklink).
# C++ DeckLink app should:
#   socket = ctx.socket(ZMQ.PUSH);
#   socket.connect("tcp://127.0.0.1:6000");
ZMQ_DECKLINK_IN = os.environ.get("ZMQ_DECKLINK_IN", "tcp://127.0.0.1:6000")

# Show local preview window for debugging.
SHOW_PREVIEW = True

# Simple FPS throttling for OBS mode (0 = no throttle).
OBS_TARGET_FPS = 60.0


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def build_raw_frame_meta(
    frame_id: int,
    width: int,
    height: int,
    timestamp_ms: int,
    source: str,
    pixel_format: str = "BGR24",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build the canonical meta JSON for moonalt.raw_frame_v1 messages.
    This is what yolo_worker_zmq.py expects.
    """
    meta: Dict[str, Any] = {
        "schema": "moonalt.raw_frame_v1",
        "frame_id": int(frame_id),
        "timestamp_ms": int(timestamp_ms),
        "width": int(width),
        "height": int(height),
        "pixel_format": pixel_format,
        "source": source,  # "obs" or "decklink"
    }
    if extra:
        # keep all capture-specific metadata here so nothing is lost
        meta["source_meta"] = extra
    return meta


# ---------------------------------------------------------------------------
# OBS MODE
# ---------------------------------------------------------------------------

def run_obs_mode(camera_index: int) -> None:
    print(f"[CAPTURE][OBS] Using camera index {camera_index}")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[CAPTURE][OBS] ERROR: Could not open OBS virtual camera.")
        return

    # ZMQ: PUSH -> bind, worker will PULL -> connect
    ctx = zmq.Context.instance()
    out_sock = ctx.socket(zmq.PUSH)
    out_sock.bind(ZMQ_RAW_OUT)
    print(f"[CAPTURE][OBS] ZMQ PUSH bound on {ZMQ_RAW_OUT}")

    frame_id = 0
    prev_log = time.time()
    sent_this_sec = 0

    if OBS_TARGET_FPS > 0:
        min_interval = 1.0 / OBS_TARGET_FPS
    else:
        min_interval = 0.0
    last_send_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CAPTURE][OBS] WARNING: Failed to read frame from camera.")
            time.sleep(0.01)
            continue

        now = time.time()
        if min_interval > 0 and (now - last_send_time) < min_interval:
            # simple FPS throttle
            time.sleep(max(0.0, min_interval - (now - last_send_time)))
            now = time.time()

        h, w = frame.shape[:2]

        # Build meta & send
        meta = build_raw_frame_meta(
            frame_id=frame_id,
            width=w,
            height=h,
            timestamp_ms=now_ms(),
            source="obs",
        )
        try:
            out_sock.send_multipart(
                [json.dumps(meta).encode("utf-8"), frame.tobytes()],
                copy=False,
            )
        except Exception as e:
            print(f"[CAPTURE][OBS] ERROR sending frame: {e}")
            break

        frame_id += 1
        sent_this_sec += 1
        last_send_time = now

        # Preview
        if SHOW_PREVIEW:
            cv2.imshow("OBS raw", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("[CAPTURE][OBS] ESC pressed, stopping.")
                break

        # Log FPS
        if now - prev_log >= 1.0:
            print(f"[CAPTURE][OBS] FPS ~ {sent_this_sec} | last frame_id {frame_id - 1}")
            prev_log = now
            sent_this_sec = 0

    cap.release()
    cv2.destroyAllWindows()
    print("[CAPTURE][OBS] Stopped.")


# ---------------------------------------------------------------------------
# DECKLINK MODE
# ---------------------------------------------------------------------------

def bytes_to_bgr_frame(
    buf: bytes,
    width: int,
    height: int,
) -> np.ndarray | None:
    """
    Interpret a raw BGR24 buffer as an OpenCV frame for preview only.
    """
    expected = width * height * 3
    if len(buf) != expected:
        print(
            f"[CAPTURE][DECKLINK] WARNING: buffer size {len(buf)} "
            f"!= width*height*3 ({expected})"
        )
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    return arr.reshape((height, width, 3))


def run_decklink_mode() -> None:
    """
    Bridge between C++ DeckLink app and YOLO worker.

    Incoming ZMQ (from C++ DeckLink app):
      - Endpoint: ZMQ_DECKLINK_IN (default tcp://127.0.0.1:6000)
      - Socket: PUSH (C++) --> PULL (Python, bound here)
      - Multipart [2]:
          Part 0: UTF-8 JSON meta:
              {
                "schema": "decklink.raw_frame_v1",
                "frame_id": <int>,                 # optional
                "timestamp_ms": <int>,             # optional
                "width": <int>,
                "height": <int>,
                "pixel_format": "BGR24",
                "device_index": <int>,             # optional
                "device_name": "<string>",         # optional
                "display_mode_code": <int>,        # optional (BMDDisplayMode enum)
                "display_mode_name": "<string>"    # optional
                ... any other fields are OK
              }
          Part 1: raw frame bytes, exactly width * height * 3, BGR24

    Outgoing ZMQ (to YOLO worker) is the same contract that OBS uses:
      - schema: "moonalt.raw_frame_v1"
      - PULL (worker) <-- PUSH (here)
    """
    ctx = zmq.Context.instance()

    # PULL from DeckLink producer (C++ app will connect with PUSH).
    in_sock = ctx.socket(zmq.PULL)
    in_sock.bind(ZMQ_DECKLINK_IN)
    print(f"[CAPTURE][DECKLINK] ZMQ PULL bound on {ZMQ_DECKLINK_IN}")

    # PUSH to YOLO worker.
    out_sock = ctx.socket(zmq.PUSH)
    out_sock.bind(ZMQ_RAW_OUT)
    print(f"[CAPTURE][DECKLINK] ZMQ PUSH bound on {ZMQ_RAW_OUT}")

    local_frame_id = 0
    prev_log = time.time()
    forwarded_this_sec = 0

    while True:
        try:
            parts = in_sock.recv_multipart()
        except Exception as e:
            print(f"[CAPTURE][DECKLINK] ERROR receiving ZMQ message: {e}")
            break

        if len(parts) < 2:
            print("[CAPTURE][DECKLINK] WARNING: received message with < 2 parts")
            continue

        try:
            meta_in = json.loads(parts[0].decode("utf-8"))
        except Exception as e:
            print(f"[CAPTURE][DECKLINK] WARNING: bad JSON meta: {e}")
            continue

        frame_bytes = parts[1]

        # Validate / pull basic fields
        width = int(meta_in.get("width", 0))
        height = int(meta_in.get("height", 0))
        if width <= 0 or height <= 0:
            print("[CAPTURE][DECKLINK] WARNING: invalid width/height in meta, skipping frame")
            continue

        # Decide frame_id: use DeckLink-provided one if present, otherwise our local counter.
        if "frame_id" in meta_in:
            frame_id = int(meta_in["frame_id"])
        else:
            frame_id = local_frame_id
            local_frame_id += 1

        # Timestamp: prefer DeckLink-provided, else now().
        ts = int(meta_in.get("timestamp_ms", now_ms()))

        pixel_format = meta_in.get("pixel_format", "BGR24")

        # Build canonical meta for the YOLO worker, but preserve all original fields in "source_meta".
        meta_out = build_raw_frame_meta(
            frame_id=frame_id,
            width=width,
            height=height,
            timestamp_ms=ts,
            source="decklink",
            pixel_format=pixel_format,
            extra=meta_in,
        )

        try:
            out_sock.send_multipart(
                [json.dumps(meta_out).encode("utf-8"), frame_bytes],
                copy=False,
            )
        except Exception as e:
            print(f"[CAPTURE][DECKLINK] ERROR forwarding frame: {e}")
            break

        forwarded_this_sec += 1

        # Preview only for debugging (uses a copy of the buffer as a NumPy array).
        if SHOW_PREVIEW:
            frame = bytes_to_bgr_frame(frame_bytes, width, height)
            if frame is not None:
                cv2.imshow("DeckLink raw (preview)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[CAPTURE][DECKLINK] ESC pressed, stopping.")
                    break

        now = time.time()
        if now - prev_log >= 1.0:
            print(
                f"[CAPTURE][DECKLINK] FPS ~ {forwarded_this_sec} | "
                f"last frame_id {frame_id}"
            )
            prev_log = now
            forwarded_this_sec = 0

    cv2.destroyAllWindows()
    print("[CAPTURE][DECKLINK] Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OBS/DeckLink raw frame producer/bridge for Moonalt soccer pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["obs", "decklink"],
        default=os.environ.get("CAPTURE_MODE", "obs"),
        help="Input source: 'obs' (OpenCV camera) or 'decklink' (ZMQ from C++ DeckLink app)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="Camera index to use in OBS mode (default 1)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable local OpenCV preview window",
    )
    return parser.parse_args()


def main() -> None:
    global SHOW_PREVIEW

    args = parse_args()
    if args.no_preview:
        SHOW_PREVIEW = False

    print(
        f"[CAPTURE] Starting in mode='{args.mode}' | "
        f"ZMQ_RAW_OUT={ZMQ_RAW_OUT} | ZMQ_DECKLINK_IN={ZMQ_DECKLINK_IN}"
    )

    if args.mode == "obs":
        run_obs_mode(args.camera_index)
    else:
        run_decklink_mode()


if __name__ == "__main__":
    main()
