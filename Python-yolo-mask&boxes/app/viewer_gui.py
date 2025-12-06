import os, zmq, cv2, numpy as np
from services.connection.bt_packet_pb2 import BtPacket

OUT_EP = os.environ.get("MOONALT_BROADTRACK_OUT", "/tmp/broadtrack_in.sock")

def main():
    # Prefer xcb on WSLg
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PULL)
    endpoint = OUT_EP if OUT_EP.startswith("ipc://") else f"ipc://{OUT_EP}"
    s.connect(endpoint)
    print(f"[viewer_gui] CONNECTED -> {endpoint}")

    while True:
        data = s.recv()
        pkt = BtPacket()
        pkt.ParseFromString(data)

        w, h = pkt.width, pkt.height
        frame = np.frombuffer(pkt.frame_data, dtype=np.uint8).reshape((h, w, 3))
        mask  = np.frombuffer(pkt.mask_data,  dtype=np.uint8).reshape((h, w))

        vis = frame.copy()
        for b in pkt.boxes:
            p1 = (int(b.x1), int(b.y1))
            p2 = (int(b.x2), int(b.y2))
            cv2.rectangle(vis, p1, p2, (0,255,0), 2)

        both = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow("BtPacket (raw | mask)", both)
        if cv2.waitKey(1) == 27:  # ESC
            break

if __name__ == "__main__":
    main()
