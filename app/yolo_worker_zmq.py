# ============================================================================
# YOLO WORKER + CAPTURE (OBS or ZMQ input)
#
# Modes:
#   --mode obs
#       - Capture frames directly from an OBS virtual camera (OpenCV).
#       - No ZMQ input; frames come from the camera.
#
#   --mode zmq  (default)
#       - Receive raw frames over ZMQ (moonalt.raw_frame_v1).
#
# In BOTH modes, the worker:
#   - Runs YOLOv8-seg on the frame (resized to INFER_WIDTH).
#   - Produces:
#       * Binary mask (players + ball) in original resolution.
#       * Player/ball bounding boxes in original resolution.
#       * Unmodified raw frame bytes.
#   - Sends them out over ZMQ to BroadTrack/Unreal:
#
# ZMQ OUTPUT CONTRACT (TO BROADTRACK / UNREAL / DEBUG)
#
# Socket type: PUSH (here) ---> PULL (downstream)
# Endpoint   : ZMQ_OUT_ENDPOINT (default "tcp://127.0.0.1:5556")
#
# Multipart message (3 parts):
#   Part 0: UTF-8 JSON:
#       {
#         "schema": "moonalt.yolo_output_v1",
#         "frame_id": <int>,
#         "width": <int>,
#         "height": <int>,
#         "pixel_format": "BGR24",
#         "mask_format": "GRAY8",
#         "bboxes": [
#             [x1, y1, x2, y2],
#             ...
#         ]
#       }
#
#   Part 1: Raw frame bytes (EXACTLY the input frame)
#           - BGR24, width * height * 3 bytes
#
#   Part 2: Mask bytes:
#           - width * height bytes
#           - uint8, 0 or 255
#
# ZMQ INPUT CONTRACT (MODE = "zmq"):
#
# Socket type: PULL (here) <--- PUSH (capture/DeckLink producer)
# Endpoint   : ZMQ_IN_ENDPOINT (default "tcp://127.0.0.1:5555")
#
# Multipart message (2 parts):
#   Part 0: UTF-8 JSON:
#       {
#         "schema": "moonalt.raw_frame_v1",
#         "frame_id": <int>,
#         "timestamp_ms": <int>,
#         "width": <int>,
#         "height": <int>,
#         "pixel_format": "BGR24"
#         ... (can have extra fields, ignored here)
#       }
#
#   Part 1: Raw frame bytes (BGR24).
#
# In --mode obs, we do NOT use this; we just create frame_id in Python.
#
# "UDP-style" behavior on ZMQ:
#   - tcp:// endpoints with:
#       * RCVHWM = 1 on input (drop backlog, always process latest)
#       * SNDHWM = 1 + NOBLOCK on output (drop frames if consumer is slow)
# ============================================================================

import argparse
import json
import os
import time

import cv2
import numpy as np
import zmq
from ultralytics import YOLO

# Try to import torch to detect GPU
try:
    import torch
except ImportError:
    torch = None


# ------------- CONFIG -------------

MODEL_NAME  = "yolov8s-seg.pt"   # seg model, auto-download on first run
INFER_WIDTH = 960                # resize width for YOLO (faster when smaller)
CONF_THRES  = 0.5
USE_HALF    = True               # FP16 on GPU if possible
SHOW_VIS    = True               # set False for max speed

PERSON_CLASS_ID       = 0
SPORTS_BALL_CLASS_ID  = 32

# ZMQ endpoints
ZMQ_IN_ENDPOINT  = os.environ.get("ZMQ_IN_ENDPOINT",  "tcp://127.0.0.1:5555")
ZMQ_OUT_ENDPOINT = os.environ.get("ZMQ_OUT_ENDPOINT", "tcp://127.0.0.1:5556")

# OBS camera index (for --mode obs)
DEFAULT_CAMERA_INDEX = 1


# ----------------------------------
# Helper functions
# ----------------------------------

def build_binary_mask(results, frame_shape):
    """
    Build a single-channel binary mask (uint8, 0 or 255) where:
    - 255 (white) = all persons + sports balls
    -   0 (black) = everything else
    """
    h, w = frame_shape[:2]

    if results.masks is None:
        return np.zeros((h, w), dtype=np.uint8)

    masks = results.masks.data.cpu().numpy()         # [N, mask_h, mask_w]
    classes = results.boxes.cls.cpu().numpy().astype(int)

    combined = np.zeros((h, w), dtype=np.uint8)

    for i, m in enumerate(masks):
        cls_id = classes[i]

        # keep only players + ball
        if cls_id not in (PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID):
            continue

        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m_bin = (m_resized > 0.5).astype(np.uint8) * 255
        combined = np.maximum(combined, m_bin)

    # smooth & clean a bit (you can simplify this for more speed)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined


def extract_bboxes_person_and_ball(results, infer_shape, orig_shape):
    """
    Extract bounding boxes for 'person' + 'sports ball' detections from YOLO results.
    Returned coords are in ORIGINAL resolution.

    - results: YOLO result for the RESIZED frame (frame_resized)
    - infer_shape: shape of frame_resized (h, w, 3)
    - orig_shape: shape of the original raw frame (h, w, 3)
    """
    infer_h, infer_w = infer_shape[:2]
    orig_h, orig_w = orig_shape[:2]

    if results.boxes is None or len(results.boxes) == 0:
        return []

    sx = orig_w / float(infer_w)
    sy = orig_h / float(infer_h)

    boxes_xyxy = results.boxes.xyxy.cpu().numpy()   # [N, 4]
    classes = results.boxes.cls.cpu().numpy().astype(int)

    bboxes = []

    for box, cls_id in zip(boxes_xyxy, classes):
        if cls_id not in (PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID):
            continue

        x1, y1, x2, y2 = box

        x1 *= sx
        y1 *= sy
        x2 *= sx
        y2 *= sy

        x1 = int(max(0, min(orig_w - 1, x1)))
        y1 = int(max(0, min(orig_h - 1, y1)))
        x2 = int(max(0, min(orig_w - 1, x2)))
        y2 = int(max(0, min(orig_h - 1, y2)))

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def process_and_send_frame(
    frame_bgr_orig,
    frame_bytes_raw,
    frame_id,
    model,
    device_arg,
    out_socket,
    cuda_available,
    dropped_out_counter,
):
    """
    Run YOLO on a single frame and send result over ZMQ.
    Returns (infer_time_ms, num_bboxes, dropped_out_counter).
    """

    # -------- YOLO INFERENCE --------
    orig_h, orig_w = frame_bgr_orig.shape[:2]
    scale = INFER_WIDTH / float(orig_w)
    new_w = INFER_WIDTH
    new_h = int(orig_h * scale)
    frame_resized = cv2.resize(frame_bgr_orig, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    half_flag = bool(cuda_available and USE_HALF)

    t0 = time.time()
    results_list = model.predict(
        frame_resized,
        conf=CONF_THRES,
        imgsz=INFER_WIDTH,
        device=device_arg,
        verbose=False,
        classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
        half=half_flag,
    )
    infer_time = (time.time() - t0) * 1000.0  # ms

    results = results_list[0]

    # Build mask at inference size, then upscale to full resolution
    mask_resized = build_binary_mask(results, frame_resized.shape)
    mask = cv2.resize(mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Extract bboxes in original resolution
    bboxes = extract_bboxes_person_and_ball(
        results,
        frame_resized.shape,
        frame_bgr_orig.shape,
    )

    # -------- BUILD OUTPUT META --------
    meta_out = {
        "schema":       "moonalt.yolo_output_v1",
        "frame_id":     int(frame_id),
        "width":        int(orig_w),
        "height":       int(orig_h),
        "pixel_format": "BGR24",
        "mask_format":  "GRAY8",
        "bboxes":       bboxes,
    }
    meta_out_bytes = json.dumps(meta_out).encode("utf-8")
    mask_bytes     = mask.tobytes()

    # -------- SEND TO BROADTRACK / DOWNSTREAM --------
    # 3-part message: meta, raw frame (unchanged), mask
    try:
        out_socket.send_multipart(
            [meta_out_bytes, frame_bytes_raw, mask_bytes],
            flags=zmq.NOBLOCK,
        )
    except zmq.Again:
        dropped_out_counter += 1
    except Exception as e:
        print(f"[WORKER] ERROR sending OUT over ZMQ: {e}")
        raise

    # -------- VISUALIZATION (dev only) --------
    if SHOW_VIS:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        vis_orig = frame_bgr_orig.copy()
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(vis_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)

        vis_orig_small = cv2.resize(vis_orig, (960, 540))
        vis_mask_small = cv2.resize(mask_bgr,  (960, 540))
        stacked = np.hstack((vis_orig_small, vis_mask_small))
        cv2.imshow("WORKER: Raw+BBoxes (left) | Mask (right)", stacked)

    return infer_time, len(bboxes), dropped_out_counter
# ----------------------------------
# Run modes
# ----------------------------------

def run_from_zmq(model, device_arg, cuda_available):
    """
    Original behavior: PULL raw frames from ZMQ, run YOLO, PUSH output.
    """
    ctx = zmq.Context()

    # PULL raw frames from capture
    in_socket = ctx.socket(zmq.PULL)
    in_socket.setsockopt(zmq.RCVHWM, 1)
    in_socket.setsockopt(zmq.LINGER, 0)
    in_socket.connect(ZMQ_IN_ENDPOINT)
    print(f"[WORKER][ZMQ] ZMQ PULL connected to {ZMQ_IN_ENDPOINT}")

    # PUSH processed output to BroadTrack/Unreal/etc.
    out_socket = ctx.socket(zmq.PUSH)
    out_socket.setsockopt(zmq.SNDHWM, 1)
    out_socket.setsockopt(zmq.LINGER, 0)
    out_socket.bind(ZMQ_OUT_ENDPOINT)
    print(f"[WORKER] ZMQ PUSH bound to {ZMQ_OUT_ENDPOINT}")

    prev_time   = time.time()
    frame_count = 0
    dropped_out = 0

    try:
        while True:
            # -------- RECEIVE RAW FRAME (drain to latest) --------
            try:
                parts = in_socket.recv_multipart()
                # Drain any backlog, keep latest
                while True:
                    try:
                        newer = in_socket.recv_multipart(flags=zmq.NOBLOCK)
                        parts = newer
                    except zmq.Again:
                        break
            except Exception as e:
                print(f"[WORKER] ERROR receiving from ZMQ: {e}")
                break

            if len(parts) != 2:
                print(f"[WORKER] WARNING: expected 2 parts, got {len(parts)}")
                continue

            meta_bytes, frame_bytes_raw = parts
            meta_in = json.loads(meta_bytes.decode("utf-8"))

            frame_id = meta_in.get("frame_id", -1)
            w        = meta_in["width"]
            h        = meta_in["height"]

            # Reconstruct raw BGR frame from bytes for YOLO / vis.
            # NOTE: frame_bytes_raw itself will be passed on untouched.
            frame_bgr_orig = np.frombuffer(frame_bytes_raw, dtype=np.uint8).reshape(h, w, 3)

            infer_time, num_boxes, dropped_out = process_and_send_frame(
                frame_bgr_orig,
                frame_bytes_raw,
                frame_id,
                model,
                device_arg,
                out_socket,
                cuda_available,
                dropped_out,
            )

            # -------- STATS --------
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(
                    f"[WORKER][ZMQ] FPS: {fps:.1f} | Inference: {infer_time:.1f} ms | "
                    f"BBoxes: {num_boxes} | frame_id: {frame_id} | dropped_out: {dropped_out}"
                )
                prev_time   = now
                frame_count = 0
                dropped_out = 0

            if SHOW_VIS and cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        if SHOW_VIS:
            cv2.destroyAllWindows()
        try:
            in_socket.close(0)
            out_socket.close(0)
            ctx.term()
        except Exception:
            pass

        print("[WORKER][ZMQ] Stopped.")


def run_from_obs(model, device_arg, cuda_available, camera_index: int):
    """
    New behavior: open OBS virtual camera directly, no ZMQ input.
    Still sends YOLO output over ZMQ to BroadTrack/Unreal.
    """
    print(f"[WORKER][OBS] Using camera index {camera_index}")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[WORKER][OBS] ERROR: Could not open OBS virtual camera.")
        return

    ctx = zmq.Context()
    out_socket = ctx.socket(zmq.PUSH)
    out_socket.setsockopt(zmq.SNDHWM, 1)
    out_socket.setsockopt(zmq.LINGER, 0)
    out_socket.bind(ZMQ_OUT_ENDPOINT)
    print(f"[WORKER][OBS] ZMQ PUSH bound to {ZMQ_OUT_ENDPOINT}")

    frame_id    = 0
    prev_time   = time.time()
    frame_count = 0
    dropped_out = 0

    try:
        while True:
            ret, frame_bgr_orig = cap.read()
            if not ret:
                print("[WORKER][OBS] WARNING: Failed to read frame from camera.")
                time.sleep(0.01)
                continue

            h, w = frame_bgr_orig.shape[:2]
            frame_bytes_raw = frame_bgr_orig.tobytes()

            infer_time, num_boxes, dropped_out = process_and_send_frame(
                frame_bgr_orig,
                frame_bytes_raw,
                frame_id,
                model,
                device_arg,
                out_socket,
                cuda_available,
                dropped_out,
            )

            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(
                    f"[WORKER][OBS] FPS: {fps:.1f} | Inference: {infer_time:.1f} ms | "
                    f"BBoxes: {num_boxes} | frame_id: {frame_id} | dropped_out: {dropped_out}"
                )
                prev_time   = now
                frame_count = 0
                dropped_out = 0

            frame_id += 1

            if SHOW_VIS and cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        if SHOW_VIS:
            cv2.destroyAllWindows()
        try:
            out_socket.close(0)
            ctx.term()
        except Exception:
            pass

        print("[WORKER][OBS] Stopped.")
# ----------------------------------
# CLI + main
# ----------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Moonalt YOLO worker with OBS or ZMQ input"
    )
    parser.add_argument(
        "--mode",
        choices=["zmq", "obs"],
        default=os.environ.get("WORKER_MODE", "zmq"),
        help="Input mode: 'zmq' (PULL moonalt.raw_frame_v1) or 'obs' (OpenCV camera)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="Camera index for OBS mode (default 1)",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable OpenCV visualization windows for maximum speed",
    )
    return parser.parse_args()


def main():
    global SHOW_VIS

    args = parse_args()
    if args.no_vis:
        SHOW_VIS = False

    print(f"[WORKER] MODEL_NAME = {MODEL_NAME}, INFER_WIDTH = {INFER_WIDTH}")
    print(f"[WORKER] mode = {args.mode}")

    # ------------ YOLO MODEL LOAD ------------
    model = YOLO(MODEL_NAME)

    if torch is not None:
        cuda_available = torch.cuda.is_available()
        print("[WORKER] Torch CUDA available:", cuda_available)
        try:
            print("[WORKER] Model first param device:", next(model.model.parameters()).device)
        except Exception as e:
            print("[WORKER] Could not inspect model device:", e)
    else:
        cuda_available = False
        print("[WORKER] Torch not imported")

    if cuda_available:
        device_arg = 0
        print("[WORKER] Using CUDA device 0")
    else:
        device_arg = "cpu"
        print("[WORKER] Using CPU")

    # ------------ Run in selected mode ------------
    if args.mode == "obs":
        run_from_obs(model, device_arg, cuda_available, camera_index=args.camera_index)
    else:
        run_from_zmq(model, device_arg, cuda_available)


if __name__ == "__main__":
    main()




#python app\yolo_worker_zmq.py --mode obs --camera-index 1

#python app\yolo_worker_zmq.py --mode zmq
