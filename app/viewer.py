import os
import time
import cv2
import numpy as np
import zmq

from services.connection.zmq_bind_pull import ZMQBindPull
from services.connection.yolo_packet_pb2 import YoloPacket


def _norm_ipc(envkey, default):
    v = os.environ.get(envkey, default)
    if v.startswith("ipc://") or v.startswith("tcp://"):
        return v
    if v.startswith("/"):
        return "ipc://" + v
    return v


def _unpack_boxes(flat: list[float]):
    """Input layout: [x, y, w, h, score, class_id, ...]"""
    out = []
    if not flat:
        return out
    n = len(flat) // 6
    for i in range(n):
        x, y, w, h, s, c = flat[i * 6 : i * 6 + 6]
        out.append((float(x), float(y), float(w), float(h), float(s), int(c)))
    return out


def main():
    ctx = zmq.Context.instance()
    endpt = _norm_ipc("MOONALT_BROADTRACK_OUT", "ipc:///tmp/broadtrack_in.sock")
    pull = ZMQBindPull(ctx, endpt)

    fps = float(os.environ.get("MOONALT_VIEWER_FPS", "30"))
    target_dt = 1.0 / max(1e-3, fps)
    last_ts = time.time()

    cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
    cv2.namedWindow("boxes", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    print(f"[viewer] BIND(PULL) -> {endpt}")
    print(f"[viewer] Display FPS target -> {fps:.1f}")

    try:
        while True:
            data = pull.recv_bytes()
            pkt = YoloPacket()
            pkt.ParseFromString(data)

            raw_bgr = None
            if getattr(pkt, "raw_pixel_format", "") == "BGR24" and pkt.raw_frame:
                w, h = int(pkt.raw_width), int(pkt.raw_height)
                if w > 0 and h > 0:
                    raw_bgr = (
                        np.frombuffer(pkt.raw_frame, dtype=np.uint8)
                        .reshape(h, w, 3)
                        .copy()
                    )

            mask_gray = None
            if pkt.mask_frame and pkt.mask_width > 0 and pkt.mask_height > 0:
                mw, mh = int(pkt.mask_width), int(pkt.mask_height)
                mask_gray = (
                    np.frombuffer(pkt.mask_frame, dtype=np.uint8)
                    .reshape(mh, mw)
                    .copy()
                )

            if raw_bgr is None and mask_gray is not None:
                raw_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)

            if raw_bgr is None:
                continue

            boxes_view = raw_bgr.copy()
            for (x, y, bw, bh, score, class_id) in _unpack_boxes(list(pkt.boxes)):
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + bw), int(y + bh)
                cv2.rectangle(boxes_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    boxes_view,
                    f"{class_id}:{score:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            if mask_gray is None:
                mask_gray = np.zeros(raw_bgr.shape[:2], dtype=np.uint8)

            cv2.imshow("raw", raw_bgr)
            cv2.imshow("boxes", boxes_view)
            cv2.imshow("mask", mask_gray)

            now = time.time()
            elapsed = now - last_ts
            wait_ms = int(max(1, (target_dt - elapsed) * 1000)) if elapsed < target_dt else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            last_ts = now

            if key == 27 or key == ord("q"):
                break

            print(f"[viewer] frame_id={pkt.frame_id} boxes={len(pkt.boxes) // 6}")

    except KeyboardInterrupt:
        print("[viewer] Ctrl+C -> exit")
    finally:
        pull.close()
        ctx.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
