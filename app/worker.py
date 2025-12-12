# app/worker.py — Production YOLO worker for Connection-develop
# Capture (C++) -> ZMQ/Protobuf(InputFrame) -> YOLO -> ZMQ/Protobuf(YoloPacket) -> BroadTrack (C++)

import os
import time
import zmq
import numpy as np
import cv2
import torch

from services.connection.zmq_connect_pull import ZMQConnectPull
from services.connection.zmq_connect_push import ZMQConnectPush
from services.connection.frame_message_pb2 import InputFrame
from services.connection.yolo_packet_pb2 import YoloPacket


def _env_int(k: str, d: int) -> int:
    v = os.environ.get(k)
    try:
        return int(v) if v is not None else d
    except Exception:
        return d


def _env_float(k: str, d: float) -> float:
    v = os.environ.get(k)
    try:
        return float(v) if v is not None else d
    except Exception:
        return d


def _env_list_int(k: str, d: list[int]) -> list[int]:
    v = os.environ.get(k)
    if not v:
        return d
    try:
        return [int(x) for x in v.replace(",", " ").split()]
    except Exception:
        return d


def _norm_ep(envkey: str, default: str) -> str:
    v = os.environ.get(envkey, default)
    if v.startswith("ipc://") or v.startswith("tcp://"):
        return v
    if v.startswith("/"):
        return "ipc://" + v
    return v


def _to_bgr(frame_bytes: bytes, w: int, h: int) -> np.ndarray:
    # writable copy
    return np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w, 3).copy()


def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1)); y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1)); y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


# ---------------- config ----------------
IMGSZ_CAP = _env_int("MOONALT_YOLO_IMGSZ", 640)
CONF_T = _env_float("MOONALT_YOLO_CONF", 0.45)
IOU_T = _env_float("MOONALT_YOLO_IOU", 0.55)
MAXDET = _env_int("MOONALT_YOLO_MAXDET", 60)
RETINA = bool(_env_int("MOONALT_YOLO_RETINA", 0))
EVERY_N = max(1, _env_int("MOONALT_YOLO_EVERY_N", 1))
CLASSES = _env_list_int("MOONALT_YOLO_CLASSES", [0, 32])  # person=0, sports ball=32
SEND_RAW = bool(_env_int("MOONALT_BT_SEND_RAW", 1))
LOG_EVERY = max(1, _env_int("MOONALT_LOG_EVERY", 10))

IN_EP = _norm_ep("MOONALT_IPC_IN", "ipc:///tmp/capture")
OUT_EP = _norm_ep("MOONALT_BROADTRACK_OUT", "ipc:///tmp/broadtrack_in.sock")


# ---------------- YOLO (always FP32) ----------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
try:
    from ultralytics import YOLO

    model = YOLO("yolov8n-seg.pt")
    HAVE_YOLO = True
    print("YOLOv8n-seg (FP32) — will fuse on first predict")
    print(f"[worker] YOLO ready. device={DEVICE} imgsz_cap={IMGSZ_CAP} classes={CLASSES}")
except Exception as e:
    HAVE_YOLO = False
    model = None
    print(f"[worker][WARN] YOLO load failed -> empty outputs. ({e})")


def _infer(frame_bgr: np.ndarray):
    """Returns: boxes_xyxy, scores, classes, mask_gray(HxW uint8)."""
    h, w = frame_bgr.shape[:2]
    if not HAVE_YOLO or model is None:
        return [], [], [], np.zeros((h, w), dtype=np.uint8)

    res = model.predict(
        source=frame_bgr,
        imgsz=IMGSZ_CAP,
        conf=CONF_T,
        iou=IOU_T,
        max_det=MAXDET,
        classes=CLASSES,
        retina_masks=RETINA,
        device=DEVICE,
        half=False,  # force FP32
        verbose=False,
    )[0]

    boxes_xyxy, scores, classes = [], [], []
    if res.boxes is not None and len(res.boxes) > 0:
        b = res.boxes
        xyxy = b.xyxy.cpu().numpy()
        scrs = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, scrs, cls):
            x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(float(s))
            classes.append(int(c))

    mask_gray = np.zeros((h, w), dtype=np.uint8)
    m = getattr(res, "masks", None)
    if m is not None and m.data is not None and len(m.data) > 0:
        try:
            union = (m.data > 0.5).any(dim=0).cpu().numpy().astype(np.uint8) * 255
            mh, mw = union.shape[:2]
            if (mw, mh) != (w, h):
                union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_gray = union
        except Exception:
            pass

    return boxes_xyxy, scores, classes, mask_gray


def run_worker():
    ctx = zmq.Context.instance()

    # Per your team decision:
    #   Capture binds PUSH  -> worker connects PULL
    #   BroadTrack binds PULL -> worker connects PUSH
    pull = ZMQConnectPull(ctx, IN_EP)
    push = ZMQConnectPush(ctx, OUT_EP)

    print(f"[worker] READY. IN(PULL connect)={IN_EP}  OUT(PUSH connect)={OUT_EP}")

    ema_fps = None
    frame_counter = 0
    last_boxes = last_scores = last_classes = None
    last_mask = None

    try:
        while True:
            raw = pull.recv_bytes()
            if (frame_counter % LOG_EVERY) == 0:
                print("[worker] recv bytes:", len(raw))

            msg = InputFrame()
            try:
                msg.ParseFromString(raw)
            except Exception as e:
                print("[worker][ERR] protobuf ParseFromString failed:", e)
                frame_counter += 1
                continue

            w = int(msg.width)
            h = int(msg.height)

            # Connection-develop expects string pixel_format.
            if getattr(msg, "pixel_format", "") != "BGR24":
                if (frame_counter % LOG_EVERY) == 0:
                    print("[worker][ERR] unsupported pixel_format:", getattr(msg, "pixel_format", ""))
                frame_counter += 1
                continue

            expected = w * h * 3
            got = len(msg.frame_data)
            if got != expected:
                print(f"[worker][ERR] frame_data size mismatch: got={got} expected={expected} (w={w} h={h})")
                frame_counter += 1
                continue

            frame_bgr = _to_bgr(msg.frame_data, w, h)

            do_infer = (frame_counter % EVERY_N) == 0
            if do_infer:
                t0 = time.time()
                boxes_xyxy, scores, classes, mask_gray = _infer(frame_bgr)
                dt = max(1e-3, time.time() - t0)
                inst = 1.0 / dt
                ema_fps = inst if ema_fps is None else (0.9 * ema_fps + 0.1 * inst)
                last_boxes, last_scores, last_classes, last_mask = boxes_xyxy, scores, classes, mask_gray
            else:
                boxes_xyxy, scores, classes, mask_gray = (
                    last_boxes or [],
                    last_scores or [],
                    last_classes or [],
                    last_mask if last_mask is not None else np.zeros((h, w), dtype=np.uint8),
                )

            pkt = YoloPacket()
            pkt.schema = "golex.vt.yolo_v1"
            pkt.frame_id = int(msg.frame_id)

            if SEND_RAW:
                pkt.raw_width = w
                pkt.raw_height = h
                pkt.raw_pixel_format = "BGR24"
                pkt.raw_frame = msg.frame_data

            pkt.mask_width = w
            pkt.mask_height = h
            pkt.mask_frame = mask_gray.tobytes()

            # layout: [x, y, w, h, score, class_id, ...]
            for (x1, y1, x2, y2), s, c in zip(boxes_xyxy, scores, classes):
                pkt.boxes.append(float(x1))
                pkt.boxes.append(float(y1))
                pkt.boxes.append(float(x2 - x1))
                pkt.boxes.append(float(y2 - y1))
                pkt.boxes.append(float(s))
                pkt.boxes.append(float(c))

            push.send(pkt.SerializeToString())

            if (frame_counter % LOG_EVERY) == 0:
                pf = 0.0 if ema_fps is None else ema_fps
                print(f"[worker] frame_id={msg.frame_id} size=({w},{h}) boxes={len(boxes_xyxy)} proc_fps={pf:.1f}")

            frame_counter += 1

    except KeyboardInterrupt:
        print("[worker] Ctrl+C -> exit")
    finally:
        try:
            pull.close()
            push.close()
            ctx.term()
        except Exception:
            pass


if __name__ == "__main__":
    run_worker()
