# app/worker.py — real-time YOLO worker for Connection-develop pipeline
import os
import time
import traceback

import zmq
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from services.connection.zmq_connect_pull import ZMQPull
from services.connection.zmq_connect_push import ZMQPush
from services.connection.frame_message_pb2 import InputFrame
from services.connection.bt_model_pb2 import YoloPacket, RawFrame, MaskFrame, BoxFrame, Box, PF_BGR24, PF_GRAY8, CapturePacket
from services.connection.capture_rawframe_v1_pb2 import CaptureRawFrameV1 as CaptureRawFrameV1


def _env(k: str, d: str) -> str:
    v = os.environ.get(k)
    return d if v is None or v == "" else v

def _env_int(k: str, d: int) -> int:
    try:
        return int(_env(k, str(d)))
    except Exception:
        return d

def _env_float(k: str, d: float) -> float:
    try:
        return float(_env(k, str(d)))
    except Exception:
        return d

def _parse_int_list(s: str, default: list[int]) -> list[int]:
    try:
        parts = s.replace(",", " ").split()
        return [int(x) for x in parts] if parts else default
    except Exception:
        return default

def _norm_ep(v: str) -> str:
    # accept ipc:///tmp/x, tcp://..., or /tmp/x -> ipc:///tmp/x
    if v.startswith(("ipc://", "tcp://")):
        return v
    if v.startswith("/"):
        return "ipc://" + v
    return v

def _pick_device() -> str:
    # if driver not installed, cuda will be unavailable; we auto-fallback to CPU
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_union_mask(result, out_h: int, out_w: int) -> np.ndarray:
    """Return uint8 mask in (H,W) with values 0/255."""
    if getattr(result, "masks", None) is None:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    data = result.masks.data  # torch tensor: (N, mh, mw)
    if data is None or len(data) == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    m = (data > 0.5).any(dim=0)  # (mh, mw) bool
    m = m.detach().cpu().numpy().astype(np.uint8) * 255

    # resize to raw frame size if needed
    if m.shape[0] != out_h or m.shape[1] != out_w:
        m = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return m


def main():
    # ---------------- endpoints ----------------
    IN_EP  = _norm_ep(_env("MOONALT_IPC_IN", "ipc:///tmp/capture"))                 # Capture PUSH bind -> YOLO PULL connect
    OUT_EP = _norm_ep(_env("MOONALT_BROADTRACK_OUT", "ipc:///tmp/broadtrack"))     # YOLO PUSH bind (default) -> BroadTrack PULL connect

    OUT_MODE = _env("MOONALT_OUT_MODE", "connect").lower().strip()  # bind|connect (default connect (BroadTrack binds))

    # ---------------- io / perf knobs ----------------
    SCHEMA_EXPECT = _env("MOONALT_SCHEMA", "golex.vt.input_v1")
    PIX_EXPECT    = _env("MOONALT_PIXEL_FORMAT", "BGR24")

    MODEL_PATH = _env("MOONALT_YOLO_MODEL", "yolov8n-seg.pt")
    IMGSZ      = _env_int("MOONALT_YOLO_IMGSZ", 640)
    CLASSES    = _parse_int_list(_env("MOONALT_YOLO_CLASSES", "0,32"), [0, 32])
    EVERY_N    = _env_int("MOONALT_YOLO_EVERY_N", 1)

    device = _pick_device()

    print(f"[worker] starting. device={device} model={MODEL_PATH} imgsz={IMGSZ} classes={CLASSES} every_n={EVERY_N}", flush=True)

    model = YOLO(MODEL_PATH)
    print("[worker] YOLO ready.", flush=True)

    ctx = zmq.Context.instance()
    pull = ZMQPull(ctx, IN_EP)
    push = ZMQPush(ctx, OUT_EP, mode=OUT_MODE)

    print(f"[worker] READY. IN(connect PULL)={IN_EP}  OUT({OUT_MODE} PUSH)={OUT_EP}", flush=True)

    n = 0
    while True:
        try:
            raw = pull.recv()
            print(f"[worker] recv bytes={len(raw)}", flush=True)  # Nazari: log right after receive            # Parse input frame (CaptureRawFrameV1 | BroadTrack CapturePacket | legacy InputFrame)
            frame_id = None
            ts_ns = time.time_ns()
            raw_bgr_bytes = None
            w = h = None
            msg = None  # only used for legacy InputFrame schema checks

            # 1) Capture-main RawFrame (sent directly on ipc:///tmp/capture in your Capture-main repo)
            caprf = CaptureRawFrameV1()
            try:
                caprf.ParseFromString(raw)
            except Exception:
                caprf = None

            if caprf is not None and getattr(caprf, "frame_data", None) is not None and len(caprf.frame_data) > 0 and caprf.width > 0 and caprf.height > 0:
                frame_id = int(caprf.frame_id)
                w = int(caprf.width)
                h = int(caprf.height)
                raw_bgr_bytes = bytes(caprf.frame_data)
                # Capture-main uses timestamp millis (see DateTime::currentTimestampMillis)
                if getattr(caprf, "timestamp", 0):
                    ts_ns = int(caprf.timestamp) * 1_000_000
                print(f"[worker] parsed CaptureRawFrameV1: frame_id={frame_id} {w}x{h} bytes={len(raw_bgr_bytes)}", flush=True)

            else:
                # 2) BroadTrack CapturePacket (if a future Capture/Connection sends it)
                cap = CapturePacket()
                try:
                    cap.ParseFromString(raw)
                except Exception:
                    cap = None

                if cap is not None and cap.raw_frame is not None and getattr(cap.raw_frame, "frame_data", None) is not None and len(cap.raw_frame.frame_data) > 0 and cap.raw_frame.width > 0 and cap.raw_frame.height > 0:
                    frame_id = int(cap.frame_id)
                    w = int(cap.raw_frame.width)
                    h = int(cap.raw_frame.height)
                    raw_bgr_bytes = bytes(cap.raw_frame.frame_data)
                    if getattr(cap.raw_frame, "timestamp", 0):
                        ts_ns = int(cap.raw_frame.timestamp)
                    print(f"[worker] parsed CapturePacket: frame_id={frame_id} {w}x{h} bytes={len(raw_bgr_bytes)}", flush=True)

                else:
                    # 3) Legacy InputFrame (string-based schema)
                    msg = InputFrame()
                    try:
                        msg.ParseFromString(raw)
                    except Exception as e:
                        print(f"[worker][ERR] ParseFromString failed for all known input messages: {e!r}", flush=True)
                        continue

                    frame_id = int(msg.frame_id)
                    # InputFrame timestamp is double seconds in some builds; if present, convert to ns
                    try:
                        if getattr(msg, "timestamp", 0):
                            ts_ns = int(float(msg.timestamp) * 1e9)
                    except Exception:
                        pass

                    w = int(msg.width)
                    h = int(msg.height)
                    raw_bgr_bytes = bytes(msg.frame_data)
                    print(f"[worker] parsed InputFrame: frame_id={frame_id} {w}x{h} bytes={len(raw_bgr_bytes)} schema={getattr(msg,'schema','')} pf={getattr(msg,'pixel_format','')}", flush=True)

            # schema/pixel checks (drop bad frames but keep worker alive)

            if msg is not None:
                if SCHEMA_EXPECT and msg.schema != SCHEMA_EXPECT:
                    print(f"[worker][ERR] schema mismatch: got={msg.schema} expect={SCHEMA_EXPECT}", flush=True)
                    continue
                if msg.pixel_format != PIX_EXPECT:
                    print(f"[worker][ERR] pixel_format unsupported: got={msg.pixel_format} expect={PIX_EXPECT}", flush=True)
                    continue

            expected = w * h * 3
            if len(raw_bgr_bytes) != expected:
                print(f"[worker][ERR] frame_data size mismatch: got={len(raw_bgr_bytes)} expected={expected}", flush=True)
                continue

            frame = np.frombuffer(raw_bgr_bytes, dtype=np.uint8).reshape((h, w, 3))

            n += 1
            if EVERY_N > 1 and (n % EVERY_N) != 0:
                continue

            t0 = time.time()
            results = model.predict(source=frame, imgsz=IMGSZ, device=device, classes=CLASSES, verbose=False)
            dt_ms = (time.time() - t0) * 1000.0
            r0 = results[0]

            # mask (union)
            mask = _make_union_mask(r0, h, w)

            # boxes -> flattened float list [x,y,w,h,score,class_id, ...]
            boxes_f = []
            if getattr(r0, "boxes", None) is not None and r0.boxes is not None and len(r0.boxes) > 0:
                xywh = r0.boxes.xywh.detach().cpu().numpy()
                conf = r0.boxes.conf.detach().cpu().numpy()
                cls  = r0.boxes.cls.detach().cpu().numpy()
                for (x, y, bw, bh), sc, ci in zip(xywh, conf, cls):
                    boxes_f.extend([float(x), float(y), float(bw), float(bh), float(sc), float(ci)])

            out = YoloPacket()
            out.frame_id = int(frame_id)

            # raw frame (BroadTrack schema: nested RawFrame)
            out.raw_frame.timestamp = int(ts_ns)
            out.raw_frame.width = w
            out.raw_frame.height = h
            out.raw_frame.pixel_format = PF_BGR24
            out.raw_frame.frame_data = raw_bgr_bytes  # BGR24 bytes

            # mask frame (GRAY8)
            out.mask_frame.timestamp = int(ts_ns)
            out.mask_frame.width = w
            out.mask_frame.height = h
            out.mask_frame.pixel_format = PF_GRAY8
            out.mask_frame.frame_data = mask.tobytes()

            # boxes (uint32 xywh in pixels)
            out.box_frame.width = w
            out.box_frame.height = h
            for (x, y, bw, bh), sc, ci in zip(xywh, conf, cls):
                b = out.box_frame.boxes.add()
                # clamp + convert to uint32
                xi = int(max(0, min(w - 1, round(float(x - bw/2)))))
                yi = int(max(0, min(h - 1, round(float(y - bh/2)))))
                wi = int(max(0, min(w, round(float(bw)))))
                hi = int(max(0, min(h, round(float(bh)))))
                b.x = xi
                b.y = yi
                b.width = wi
                b.height = hi
                b.score = float(sc)
                b.class_id = int(ci)

            payload = out.SerializeToString()
            push.send(payload)

            print(f"[worker] sent bytes={len(payload)} boxes={len(boxes_f)//6} dt={dt_ms:.2f}ms", flush=True)

        except KeyboardInterrupt:
            print("[worker] Ctrl+C -> exit", flush=True)
            break
        except Exception:
            print("[worker][ERR] loop exception:\n" + traceback.format_exc(), flush=True)
            continue

    try:
        pull.close()
    except Exception:
        pass
    try:
        push.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()