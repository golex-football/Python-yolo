import os
import time
import zmq
import cv2
import numpy as np

from services.connection.zmq_connect_pull import ZMQConnectPull
from services.connection.bt_packet_pb2 import BtPacket

def _norm_ipc(envkey, default):
    v = os.environ.get(envkey, default)
    if v.startswith("ipc://") or v.startswith("tcp://"): return v
    if v.startswith("/"): return "ipc://" + v
    return v

def _expand(p):
    return os.path.expanduser(p) if p else p

def main():
    ctx = zmq.Context.instance()
    endpt = _norm_ipc("MOONALT_BROADTRACK_OUT", "ipc:///tmp/broadtrack")
    pull = ZMQConnectPull(ctx, endpt)

    # target display FPS (e.g. 25 or 30)
    fps = float(os.environ.get("MOONALT_VIEWER_FPS", "30"))
    target_dt = 1.0 / max(1e-3, fps)
    last_ts = time.time()

    # 3 windows: raw, boxes, mask
    cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
    cv2.namedWindow("boxes", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    print(f"[viewer] CONNECTED -> {endpt}")
    print(f"[viewer] Display FPS target -> {fps:.1f}")

    try:
        while True:
            data = pull.recv()
            pkt = BtPacket()
            pkt.ParseFromString(data)

            raw_bgr = None
            mask_gray = None

            # raw frame
            if pkt.HasField("raw") and pkt.raw.pixel_format == "BGR24":
                w, h = int(pkt.raw.width), int(pkt.raw.height)
                raw_bgr = np.frombuffer(
                    pkt.raw.frame_data, dtype=np.uint8
                ).reshape(h, w, 3).copy()

            # mask frame
            if pkt.HasField("mask") and pkt.mask.pixel_format == "GRAY8":
                mw, mh = int(pkt.mask.width), int(pkt.mask.height)
                mask_gray = np.frombuffer(
                    pkt.mask.frame_data, dtype=np.uint8
                ).reshape(mh, mw).copy()

            # if we only have mask, fake a raw from it
            if raw_bgr is None and mask_gray is not None:
                raw_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)

            if raw_bgr is None:
                # nothing usable in this packet
                continue

            # boxes view
            boxes_view = raw_bgr.copy()
            for b in pkt.boxes.boxes:
                cv2.rectangle(
                    boxes_view,
                    (b.x1, b.y1),
                    (b.x2, b.y2),
                    (0, 255, 0),
                    2,
                )

            # ensure we always have some mask to show
            if mask_gray is None:
                mask_gray = np.zeros(raw_bgr.shape[:2], dtype=np.uint8)

            # show 3 windows
            cv2.imshow("raw", raw_bgr)
            cv2.imshow("boxes", boxes_view)
            cv2.imshow("mask", mask_gray)

            # try to keep a steady FPS
            now = time.time()
            elapsed = now - last_ts
            if elapsed < target_dt:
                wait_ms = int((target_dt - elapsed) * 1000)
            else:
                wait_ms = 1

            key = cv2.waitKey(max(1, wait_ms)) & 0xFF
            last_ts = now

            if key == 27 or key == ord("q"):
                break

            print(f"[viewer] frame_id={pkt.frame_id} boxes={len(pkt.boxes.boxes)}")

    except KeyboardInterrupt:
        print("[viewer] Ctrl+C -> exit")
    finally:
        pull.close()
        ctx.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
