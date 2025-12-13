# app/worker.py — real-time YOLO worker (BroadTrack-compatible output)
#
# Input:  golex.virtualtracking.InputFrame (from services/connection/frame_message_pb2.py)
# Output: golex.virtualtracking.model.YoloPacket (runtime-defined; matches BroadTrack src/model/protobuf)
#
# Designed to send to Connection (as pass-through) which forwards bytes unchanged to BroadTrack.

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

from services.broadtrack.proto_runtime import YoloPacket as BT_YoloPacket
from services.broadtrack.proto_runtime import PF_BGR24, PF_GRAY8

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()

def _norm_ep(ep: str) -> str:
    # allow users to pass /tmp/x.sock and convert to ipc:///tmp/x.sock
    ep = ep.strip()
    if ep.startswith("ipc://") or ep.startswith("tcp://"):
        return ep
    if ep.startswith("/"):
        return "ipc://" + ep
    return ep

def _decode_bgr24(width: int, height: int, b: bytes) -> np.ndarray:
    # expects width*height*3 bytes
    arr = np.frombuffer(b, dtype=np.uint8)
    expected = int(width) * int(height) * 3
    if arr.size != expected:
        raise ValueError(f"BGR24 size mismatch: got={arr.size} expected={expected}")
    img = arr.reshape((int(height), int(width), 3))
    return img

def _encode_mask_gray8(mask: np.ndarray, w: int, h: int) -> bytes:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        # try squeeze
        mask = np.squeeze(mask)
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.tobytes()

def _yolo_seg(model, img_bgr: np.ndarray):
    # returns: mask (H,W) uint8 in {0,255}, and boxes list
    # boxes: (x,y,w,h,score,class_id)
    # Ultralytics expects BGR in numpy; it converts internally.
    res = model.predict(source=img_bgr, verbose=False)
    r0 = res[0]

    # mask
    mask = None
    if getattr(r0, "masks", None) is not None and r0.masks is not None:
        # masks.data: (N,H,W) float {0,1}
        m = r0.masks.data
        if torch.is_tensor(m):
            m = m.detach().cpu()
        m = m.numpy()
        if m.ndim == 3 and m.shape[0] > 0:
            m = m[0]  # take first mask
        m = (m > 0.5).astype(np.uint8) * 255
        mask = m
    else:
        mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)

    # boxes
    boxes_out = []
    if getattr(r0, "boxes", None) is not None and r0.boxes is not None and len(r0.boxes) > 0:
        bxyxy = r0.boxes.xyxy
        conf = r0.boxes.conf
        cls  = r0.boxes.cls
        if torch.is_tensor(bxyxy): bxyxy = bxyxy.detach().cpu().numpy()
        if torch.is_tensor(conf): conf = conf.detach().cpu().numpy()
        if torch.is_tensor(cls):  cls  = cls.detach().cpu().numpy()

        for i in range(bxyxy.shape[0]):
            x1, y1, x2, y2 = bxyxy[i].tolist()
            score = float(conf[i]) if conf is not None else 0.0
            class_id = int(cls[i]) if cls is not None else 0
            x = int(max(0, round(x1)))
            y = int(max(0, round(y1)))
            w = int(max(0, round(x2 - x1)))
            h = int(max(0, round(y2 - y1)))
            boxes_out.append((x, y, w, h, score, class_id))

    return mask, boxes_out

def build_bt_packet(frame_id: int, raw_bgr: bytes, w: int, h: int, mask_gray8: bytes, boxes):
    pkt = BT_YoloPacket()
    pkt.frame_id = int(frame_id)

    pkt.raw_frame.width = int(w)
    pkt.raw_frame.height = int(h)
    pkt.raw_frame.pixel_format = PF_BGR24
    pkt.raw_frame.frame_data = raw_bgr

    pkt.mask_frame.width = int(w)
    pkt.mask_frame.height = int(h)
    pkt.mask_frame.pixel_format = PF_GRAY8
    pkt.mask_frame.frame_data = mask_gray8

    pkt.box_frame.width = int(w)
    pkt.box_frame.height = int(h)
    for (x, y, bw, bh, score, class_id) in boxes:
        b = pkt.box_frame.boxes.add()
        b.x = int(x); b.y = int(y); b.width = int(bw); b.height = int(bh)
        b.score = float(score); b.class_id = int(class_id)

    return pkt.SerializeToString()

def main():
    # Capture -> YOLO
    in_ep = _norm_ep(_env("MOONALT_IPC_IN", "ipc:///tmp/capture"))  # capture PUSH bind -> YOLO PULL connect

    # YOLO -> Connection (pass-through) -> BroadTrack
    # Connection should forward these bytes unchanged to BroadTrack's PULL bind endpoint (usually ipc:///tmp/broadtrack).
    out_ep = _norm_ep(_env("MOONALT_BT_OUT", "ipc:///tmp/broadtrack_in.sock"))

    # model
    weights = _env("MOONALT_YOLO_WEIGHTS", "yolov8n-seg.pt")
    device  = _env("MOONALT_DEVICE", "cpu")
    print(f"[worker] IN={in_ep} OUT={out_ep} weights={weights} device={device}", flush=True)

    model = YOLO(weights)
    try:
        if device.lower() != "cpu":
            model.to(device)
    except Exception:
        # ultralytics manages device internally; don't fail hard
        pass

    pull = ZMQPull(in_ep)
    push = ZMQPush(out_ep)

    last_log = 0.0
    while True:
        try:
            raw = pull.recv()
            msg = InputFrame()
            msg.ParseFromString(raw)

            w = int(msg.width); h = int(msg.height)
            img_bgr = _decode_bgr24(w, h, msg.frame_data)

            mask, boxes = _yolo_seg(model, img_bgr)
            mask_bytes = _encode_mask_gray8(mask, w, h)

            bt_bytes = build_bt_packet(msg.frame_id, msg.frame_data, w, h, mask_bytes, boxes)
            push.send(bt_bytes)

            now = time.time()
            if now - last_log > 1.0:
                last_log = now
                print(f"[worker] frame_id={msg.frame_id} sent_bytes={len(bt_bytes)} boxes={len(boxes)}", flush=True)

        except KeyboardInterrupt:
            print("[worker] stopped", flush=True)
            break
        except Exception as e:
            print(f"[worker][ERR] {e!r}", flush=True)
            traceback.print_exc()
            time.sleep(0.1)

    try: pull.close()
    except Exception: pass
    try: push.close()
    except Exception: pass

if __name__ == "__main__":
    main()
