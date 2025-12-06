import zmq
import json
import numpy as np
import cv2

ZMQ_IN_ENDPOINT = "tcp://127.0.0.1:5556"  # must match worker's OUT endpoint

def main():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(ZMQ_IN_ENDPOINT)
    print(f"[DOWNSTREAM] ZMQ PULL connected to {ZMQ_IN_ENDPOINT}")

    while True:
        try:
            parts = socket.recv_multipart()
        except Exception as e:
            print(f"[DOWNSTREAM] ERROR receiving from ZMQ: {e}")
            break

        if len(parts) != 3:
            print(f"[DOWNSTREAM] WARNING: expected 3 parts, got {len(parts)}")
            continue

        meta_bytes, frame_bytes, mask_bytes = parts
        meta = json.loads(meta_bytes.decode("utf-8"))

        frame_id     = meta.get("frame_id", -1)
        w            = meta["width"]
        h            = meta["height"]
        pixel_format = meta.get("pixel_format", "BGR24")
        bboxes       = meta.get("bboxes", [])

        # reconstruct raw frame (BGR)
        if pixel_format != "BGR24":
            print(f"[DOWNSTREAM] Unsupported pixel_format={pixel_format}")
            continue

        frame_bgr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w, 3)

        # reconstruct mask
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(h, w)

        # visualize raw + overlay
        vis_raw = frame_bgr.copy()
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(vis_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = frame_bgr.copy()
        overlay[mask == 255] = (0, 255, 255)  # cyan on players

        vis_raw_small = cv2.resize(vis_raw,  (960, 540))
        overlay_small = cv2.resize(overlay, (960, 540))
        combined = np.hstack((vis_raw_small, overlay_small))

        cv2.imshow("DOWNSTREAM: Raw+BBoxes (left) | Raw+Mask overlay (right)", combined)
        print(f"[DOWNSTREAM] frame_id={frame_id}, bboxes={len(bboxes)}")

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    print("[DOWNSTREAM] Stopped.")

if __name__ == "__main__":
    main()
