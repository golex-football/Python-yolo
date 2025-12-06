# app/worker.py  — FP32, simple, stable
import os, time, zmq, numpy as np, cv2, torch
from services.connection.zmq_bind_pull import ZMQBindPull
from services.connection.zmq_bind_push import ZMQBindPush
from services.connection.frame_message_pb2 import InputFrame
from services.connection.bt_packet_pb2 import BtPacket

# ---------------- env helpers ----------------
def _env_int(k, d): 
    v = os.environ.get(k); 
    try: return int(v) if v is not None else d
    except: return d

def _env_float(k, d):
    v = os.environ.get(k); 
    try: return float(v) if v is not None else d
    except: return d

def _env_list_int(k, d):
    v = os.environ.get(k)
    if not v: return d
    try: return [int(x) for x in v.replace(",", " ").split()]
    except: return d

def _norm_ipc(envkey, default):
    v = os.environ.get(envkey, default)
    if v.startswith("ipc://") or v.startswith("tcp://"): return v
    if v.startswith("/"): return "ipc://" + v
    return v

def _to_bgr(frame_bytes, w, h):
    # return writable copy
    return np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w, 3).copy()

def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1)); y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1)); y2 = max(0, min(int(y2), h - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

# ---------------- config ----------------
IMGSZ_CAP   = _env_int("MOONALT_YOLO_IMGSZ", 640)
CONF_T      = _env_float("MOONALT_YOLO_CONF", 0.45)
IOU_T       = _env_float("MOONALT_YOLO_IOU", 0.55)
MAXDET      = _env_int("MOONALT_YOLO_MAXDET", 60)
RETINA      = bool(_env_int("MOONALT_YOLO_RETINA", 0))
EVERY_N     = _env_int("MOONALT_YOLO_EVERY_N", 1)
CLASSES     = _env_list_int("MOONALT_YOLO_CLASSES", [0, 32])  # person=0, sports ball=32
SEND_RAW    = bool(_env_int("MOONALT_BT_SEND_RAW", 1))
LOG_EVERY   = _env_int("MOONALT_LOG_EVERY", 10)

IN_EP  = _norm_ipc("MOONALT_IPC_IN", "ipc:///tmp/yolo_input.sock")
OUT_EP = _norm_ipc("MOONALT_BROADTRACK_OUT", "ipc:///tmp/broadtrack_in.sock")

# ---------------- YOLO (always FP32) ----------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n-seg.pt")
    print("YOLOv8n-seg (FP32) — will fuse on first predict")
    print(f"[worker] YOLO ready. device={device} imgsz_cap={IMGSZ_CAP} classes={CLASSES}")
    HAVE_YOLO = True
except Exception as e:
    print(f"[worker] YOLO load failed -> CPU empty outputs. ({e})")
    model = None
    HAVE_YOLO = False

# ---------------- inference ----------------
def _infer(frame_bgr: np.ndarray):
    """
    Returns:
      boxes_xyxy, scores, classes, mask_gray(HxW uint8)
    """
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
        device=device,
        half=False,              # <- force FP32, avoids dtype mismatches
        verbose=False,
    )[0]

    boxes_xyxy, scores, classes = [], [], []
    if res.boxes is not None and len(res.boxes) > 0:
        b = res.boxes
        xyxy = b.xyxy.cpu().numpy()
        scrs = b.conf.cpu().numpy()
        cls  = b.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, scrs, cls):
            x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(float(s))
            classes.append(int(c))

    mask_gray = np.zeros((h, w), dtype=np.uint8)
    m = getattr(res, "masks", None)
    if m is not None and m.data is not None and len(m.data) > 0:
        try:
            union = (m.data > 0.5).any(dim=0).cpu().numpy().astype(np.uint8) * 255  # mh x mw
            mh, mw = union.shape[:2]
            if (mw, mh) != (w, h):
                union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_gray = union
        except Exception:
            pass

    return boxes_xyxy, scores, classes, mask_gray

# ---------------- main loop ----------------
def run_worker():
    ctx = zmq.Context.instance()
    pull = ZMQBindPull(ctx, IN_EP)
    push = ZMQBindPush(ctx, OUT_EP)
    print("[worker] READY. Waiting for InputFrame on IPC…")

    ema_fps = None
    frame_counter = 0
    last_boxes = last_scores = last_classes = None
    last_mask = None

    try:
        while True:
            raw = pull.recv_bytes()
            msg = InputFrame()
            msg.ParseFromString(raw)

            w = int(msg.width); h = int(msg.height)
            if msg.pixel_format != "BGR24":
                continue

            frame_bgr = _to_bgr(msg.frame_data, w, h)

            do_infer = (EVERY_N <= 1) or (frame_counter % EVERY_N == 0)
            if do_infer:
                t0 = time.time()
                boxes_xyxy, scores, classes, mask_gray = _infer(frame_bgr)
                dt = max(1e-3, time.time() - t0)                   # clamp to 1ms
                inst = 1.0 / dt
                ema_fps = inst if ema_fps is None else (0.9 * ema_fps + 0.1 * inst)
                last_boxes, last_scores, last_classes = boxes_xyxy, scores, classes
                last_mask = mask_gray
            else:
                boxes_xyxy, scores, classes = last_boxes, last_scores, last_classes
                mask_gray = last_mask
                if boxes_xyxy is None: boxes_xyxy, scores, classes = [], [], []
                if mask_gray is None: mask_gray = np.zeros((h, w), dtype=np.uint8)

            pkt = BtPacket()
            pkt.schema = "moonalt.bt_packet_v1"
            pkt.frame_id = int(msg.frame_id)

            if SEND_RAW:
                pkt.raw.schema = "moonalt.raw_v1"
                pkt.raw.frame_id = pkt.frame_id
                pkt.raw.width = w; pkt.raw.height = h
                pkt.raw.pixel_format = "BGR24"
                pkt.raw.frame_data = msg.frame_data   # unchanged pass-through

            pkt.mask.schema = "moonalt.mask_v1"
            pkt.mask.frame_id = pkt.frame_id
            pkt.mask.width = w; pkt.mask.height = h
            pkt.mask.pixel_format = "GRAY8"
            pkt.mask.frame_data = mask_gray.tobytes()

            pkt.boxes.schema = "moonalt.boxes_v1"
            pkt.boxes.frame_id = pkt.frame_id
            pkt.boxes.width = w; pkt.boxes.height = h
            for (x1, y1, x2, y2), s, c in zip(boxes_xyxy, scores, classes):
                b = pkt.boxes.boxes.add()
                b.x1, b.y1, b.x2, b.y2 = int(x1), int(y1), int(x2), int(y2)
                b.score = float(s); b.class_id = int(c)

            push.send(pkt.SerializeToString())

            if (frame_counter % max(1, LOG_EVERY)) == 0:
                pf = 0.0 if ema_fps is None else ema_fps
                print(f"[worker] frame_id={msg.frame_id} size=({w},{h}) "
                      f"mask=({h},{w}) boxes={len(pkt.boxes.boxes)} proc_fps={pf:.1f}")

            frame_counter += 1

    except KeyboardInterrupt:
        print("[worker] Ctrl+C -> exit")
    finally:
        try:
            pull.close(); push.close(); ctx.term()
        except Exception:
            pass

if __name__ == "__main__":
    run_worker()
