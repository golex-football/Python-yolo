# ============================================================================
# YOLO WORKER (OBS or IPC input) -> Protobuf over ZMQ IPC output
#
# Modes:
#   --mode zmq  (default)
#       - Receives InputFrame protobuf messages via IPC (ZMQ PULL helper).
#   --mode obs
#       - Captures frames from a camera (e.g., OBS Virtual Camera) and
#         directly builds OutputFrame protobuf messages.
#
# In both modes, this worker:
#   - Runs YOLOv8-seg on the frame (optionally FP16 on CUDA).
#   - Produces:
#       * Binary GRAY8 mask (players+ball) in original resolution.
#       * Bounding boxes (xyxy) in original resolution.
#   - Sends a single OutputFrame protobuf message via IPC (ZMQ PUSH helper).
#
# IMPORTANT:
#   - IPC endpoints are provided by helpers in services/connection and bind to:
#       IN  : ipc:///tmp/yolo_input.sock   (worker binds in helper; producers CONNECT)
#       OUT : ipc:///tmp/yolo_output.sock  (worker binds in helper; consumers CONNECT)
#   - IPC does not work on Windows host. Run in WSL/Linux.
# ============================================================================

import argparse
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from services.connection.zmq_pull import ZMQPull
from services.connection.zmq_push import ZMQPush
from services.connection.frame_message_pb2 import OutputFrame

# ----------------------------------
# Config
# ----------------------------------
# Inference resolution: frames are resized to width=INFER_WIDTH keeping aspect ratio
INFER_WIDTH: int = int(os.environ.get("INFER_WIDTH", 960))

# Run FP16 if CUDA is available and this flag is True
USE_HALF: bool = os.environ.get("USE_HALF", "1") not in ("0", "false", "False")

# Visualize windows (for development)
SHOW_VIS: bool = os.environ.get("SHOW_VIS", "0") in ("1", "true", "True")

# YOLO classes (Ultralytics common: 0=person, 32=sports ball for COCO models)
PERSON_CLASS_ID: int = int(os.environ.get("PERSON_CLASS_ID", "0"))
SPORTS_BALL_CLASS_ID: int = int(os.environ.get("SPORTS_BALL_CLASS_ID", "32"))


# ----------------------------------
# Helpers for boxes and masks
# ----------------------------------
def extract_bboxes(results, frame_shape: Tuple[int, int]) -> List[List[int]]:
    """
    Extract xyxy bounding boxes in original resolution as integers.
    """
    h, w = frame_shape[:2]
    if results.boxes is None or results.boxes.xyxy is None:
        return []
    boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # (N, 4)
    classes = None
    try:
        classes = results.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        pass

    bboxes: List[List[int]] = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        if classes is not None:
            cls_id = int(classes[i])
            # keep only players + ball
            if cls_id not in (PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID):
                continue
        x1i = max(0, min(int(round(x1)), w - 1))
        y1i = max(0, min(int(round(y1)), h - 1))
        x2i = max(0, min(int(round(x2)), w - 1))
        y2i = max(0, min(int(round(y2)), h - 1))
        if x2i > x1i and y2i > y1i:
            bboxes.append([x1i, y1i, x2i, y2i])
    return bboxes


def build_binary_mask(results, frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Combine instance masks for PERSON + SPORTS_BALL into a single GRAY8 mask.
    """
    h, w = frame_shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    # No masks available
    if results.masks is None or getattr(results.masks, "data", None) is None:
        return combined

    masks = results.masks.data.cpu().numpy()         # [N, mask_h, mask_w], values 0..1
    classes = None
    try:
        classes = results.boxes.cls.cpu().numpy().astype(int)
    except Exception:
        pass

    for i, m in enumerate(masks):
        if classes is not None:
            cls_id = int(classes[i])
            if cls_id not in (PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID):
                continue
        # Resize mask to original resolution
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m_bin = (m_resized > 0.5).astype(np.uint8) * 255
        combined = np.maximum(combined, m_bin)

    # Optional smoothing / cleanup
    if np.any(combined):
        combined = cv2.GaussianBlur(combined, (5, 5), 0)
        _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined


# ----------------------------------
# Core per-frame processing
# ----------------------------------
def process_and_build_output(
    frame_bgr_orig: np.ndarray,
    frame_bytes_raw: bytes,
    frame_id: int,
    model: YOLO,
    device_arg: str,
    cuda_available: bool,
) -> Tuple[float, int, OutputFrame]:
    """
    Run YOLO on a single frame and build a Protobuf OutputFrame (no I/O here).

    Returns:
        (infer_time_ms: float, num_bboxes: int, out_msg: OutputFrame)
    """
    # -------- YOLO INFERENCE --------
    orig_h, orig_w = frame_bgr_orig.shape[:2]
    scale = INFER_WIDTH / float(orig_w)
    new_w = INFER_WIDTH
    new_h = max(1, int(round(orig_h * scale)))
    frame_resized = cv2.resize(frame_bgr_orig, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    half_flag = bool(cuda_available and USE_HALF)

    t0 = time.time()
    # ultralytics forward; no gradient
    results = model.predict(
        frame_resized,
        imgsz=(new_h, new_w),
        verbose=False,
        half=half_flag,
    )[0]
    infer_time_ms = (time.time() - t0) * 1000.0

    # -------- POST --------
    bboxes = extract_bboxes(results, (orig_h, orig_w))
    mask = build_binary_mask(results, (orig_h, orig_w))

    # -------- BUILD OUTPUT PROTO --------
    out_msg = OutputFrame()
    out_msg.schema       = "moonalt.yolo_output_v1"
    out_msg.frame_id     = int(frame_id)
    out_msg.width        = int(orig_w)
    out_msg.height       = int(orig_h)
    out_msg.pixel_format = "BGR24"
    out_msg.mask_format  = "GRAY8"
    out_msg.frame_data   = frame_bytes_raw           # unchanged input
    out_msg.mask_data    = mask.tobytes()

    for (x1, y1, x2, y2) in bboxes:
        bb = out_msg.boxes.add()
        bb.x1, bb.y1, bb.x2, bb.y2 = float(x1), float(y1), float(x2), float(y2)

    # -------- VIS (dev only) --------
    if SHOW_VIS:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis_orig = frame_bgr_orig.copy()
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(vis_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        vis_orig_small = cv2.resize(vis_orig, (960, 540))
        vis_mask_small = cv2.resize(mask_bgr,  (960, 540))
        stacked = np.hstack((vis_orig_small, vis_mask_small))
        cv2.imshow("WORKER: Raw+BBoxes (left) | Mask (right)", stacked)

    return infer_time_ms, len(bboxes), out_msg


# ----------------------------------
# Run modes
# ----------------------------------
def run_from_zmq(model: YOLO, device_arg: str, cuda_available: bool) -> None:
    """
    Receive InputFrame protobuf via IPC, run YOLO, send OutputFrame protobuf via IPC.
    """
    pull = ZMQPull(os.environ.get("MOONALT_IPC_IN", "/tmp/yolo_input.sock"))
    push = ZMQPush(os.environ.get("MOONALT_IPC_OUT", "/tmp/yolo_output.sock"))
    print("[WORKER][IPC] PULL bound at /tmp/yolo_input.sock | PUSH bound at /tmp/yolo_output.sock")

    prev_time   = time.time()
    frame_count = 0

    try:
        while True:
            in_msg = pull.recv_frame()  # protobuf InputFrame
            frame_id = in_msg.frame_id
            w        = in_msg.width
            h        = in_msg.height

            frame_bytes_raw = in_msg.frame_data
            frame_bgr_orig  = np.frombuffer(frame_bytes_raw, dtype=np.uint8).reshape(h, w, 3)

            infer_ms, num_boxes, out_msg = process_and_build_output(
                frame_bgr_orig,
                frame_bytes_raw,
                frame_id,
                model,
                device_arg,
                cuda_available,
            )
            push.send_frame(out_msg)

            # stats
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"[WORKER][IPC] FPS: {fps:.1f} | Inference: {infer_ms:.1f} ms | BBoxes: {num_boxes} | frame_id: {frame_id}")
                prev_time   = now
                frame_count = 0

            if SHOW_VIS and cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if SHOW_VIS:
            cv2.destroyAllWindows()
        print("[WORKER][IPC] Stopped.")


def run_from_obs(model: YOLO, device_arg: str, cuda_available: bool, camera_index: int) -> None:
    """
    Capture frames from a camera (e.g., OBS Virtual Camera), process and send OutputFrame via IPC.
    """
    print(f"[WORKER][OBS] Using camera index {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[WORKER][OBS] ERROR: Could not open camera.")
        return

    push = ZMQPush(os.environ.get("MOONALT_IPC_OUT", "/tmp/yolo_output.sock"))
    print("[WORKER][OBS->IPC] PUSH bound at /tmp/yolo_output.sock")

    frame_id    = 0
    prev_time   = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame_bgr_orig = cap.read()
            if not ret:
                print("[WORKER][OBS] WARNING: Failed to read frame from camera.")
                time.sleep(0.01)
                continue

            h, w = frame_bgr_orig.shape[:2]
            frame_bytes_raw = frame_bgr_orig.tobytes()

            infer_ms, num_boxes, out_msg = process_and_build_output(
                frame_bgr_orig,
                frame_bytes_raw,
                frame_id,
                model,
                device_arg,
                cuda_available,
            )
            push.send_frame(out_msg)

            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"[WORKER][OBS] FPS: {fps:.1f} | Inference: {infer_ms:.1f} ms | BBoxes: {num_boxes} | frame_id: {frame_id}")
                prev_time   = now
                frame_count = 0

            frame_id += 1

            if SHOW_VIS and cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        if SHOW_VIS:
            cv2.destroyAllWindows()
        print("[WORKER][OBS] Stopped.")


# ----------------------------------
# CLI + main
# ----------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moonalt YOLO worker (IPC protobuf)")
    parser.add_argument("--mode", choices=["zmq", "obs"], default="zmq",
                        help="Input source: zmq (IPC InputFrame) or obs (camera).")
    parser.add_argument("--model", type=str, default=os.environ.get("YOLO_MODEL", "yolov8n-seg.pt"),
                        help="Ultralytics YOLOv8-seg model path.")
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", ""),
                        help='Torch device, e.g., "cuda:0" or "cpu" (empty = auto).')
    parser.add_argument("--camera-index", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")),
                        help="Camera index for --mode obs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load YOLO
    model = YOLO(args.model)
    # (Ultralytics handles device internally; args.device maintained for compatibility)
    cuda_available = any((getattr(model.model, 'device', None) or [])) or (os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"))

    if args.mode == "obs":
        run_from_obs(model, args.device, cuda_available, camera_index=args.camera_index)
    else:
        run_from_zmq(model, args.device, cuda_available)


if __name__ == "__main__":
    main()
