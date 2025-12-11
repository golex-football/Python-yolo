import os, zmq, cv2, numpy as np
from services.connection.yolo_packet_pb2 import YoloPacket

OUT_EP = os.environ.get("MOONALT_BROADTRACK_OUT", "ipc:///tmp/broadtrack_in.sock")

def main():
    # Prefer xcb on WSLg
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PULL)
    endpoint = OUT_EP if OUT_EP.startswith("ipc://") else f"ipc://{OUT_EP}"
    # Worker connects PUSH -> viewer binds PULL
    s.bind(endpoint)
    print(f"[viewer_gui] BIND(PULL) -> {endpoint}")

    while True:
        data = s.recv()
        pkt = YoloPacket()
        pkt.ParseFromString(data)

        w, h = int(pkt.raw_width), int(pkt.raw_height)
        if w <= 0 or h <= 0 or pkt.raw_pixel_format != "BGR24":
            continue

        frame = np.frombuffer(pkt.raw_frame, dtype=np.uint8).reshape((h, w, 3))
        mw, mh = int(pkt.mask_width), int(pkt.mask_height)
        mask = np.frombuffer(pkt.mask_frame, dtype=np.uint8).reshape((mh, mw)) if mw > 0 and mh > 0 else np.zeros((h, w), dtype=np.uint8)

        vis = frame.copy()
        boxes = list(pkt.boxes)
        for i in range(len(boxes) // 6):
            x, y, bw, bh, score, class_id = boxes[i*6:(i+1)*6]
            p1 = (int(x), int(y))
            p2 = (int(x + bw), int(y + bh))
            cv2.rectangle(vis, p1, p2, (0,255,0), 2)
            cv2.putText(vis, f"{int(class_id)}:{score:.2f}", (p1[0], max(0, p1[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        both = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow("YoloPacket (raw | mask)", both)
        if cv2.waitKey(1) == 27:  # ESC
            break

if __name__ == "__main__":
    main()
