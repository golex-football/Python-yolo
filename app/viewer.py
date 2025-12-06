import os
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
    endpt = _norm_ipc("MOONALT_BROADTRACK_OUT", "ipc:///tmp/broadtrack_in.sock")
    pull = ZMQConnectPull(ctx, endpt)

    out_path = _expand(os.environ.get("MOONALT_VIEWER_MP4", ""))  # e.g. "~/Videos/moonalt_out.mp4"
    fps = float(os.environ.get("MOONALT_VIEWER_FPS", "30"))
    writer = None
    w = h = 0

    print(f"[viewer] CONNECTED -> {endpt}")
    if out_path:
        print(f"[viewer] Target MP4  -> {out_path} (will open on first frame)")

    try:
        while True:
            data = pull.recv()
            pkt = BtPacket()
            pkt.ParseFromString(data)

            frame = None
            if pkt.HasField("raw") and pkt.raw.pixel_format == "BGR24":
                w, h = int(pkt.raw.width), int(pkt.raw.height)
                frame = np.frombuffer(pkt.raw.frame_data, dtype=np.uint8).reshape(h, w, 3).copy()
            elif pkt.HasField("mask"):
                w, h = int(pkt.mask.width), int(pkt.mask.height)
                m = np.frombuffer(pkt.mask.frame_data, dtype=np.uint8).reshape(h, w)
                frame = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

            if frame is None:
                # nothing usable in this packet
                continue

            # draw boxes
            for b in pkt.boxes.boxes:
                cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)

            # open writer lazily when we know frame size
            if out_path and writer is None:
                os.makedirs(os.path.dirname(_expand(out_path)), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(_expand(out_path), fourcc, fps, (w, h))
                if writer.isOpened():
                    print(f"[viewer] Writing MP4 -> {_expand(out_path)} at {fps:.1f} fps")
                else:
                    print("[viewer] WARN: could not open writer, disabling MP4 output")
                    writer = None

            if writer:
                writer.write(frame)

            print(f"[viewer] frame_id={pkt.frame_id} boxes={len(pkt.boxes.boxes)}")

    except KeyboardInterrupt:
        print("[viewer] Ctrl+C -> exit")
    finally:
        if writer:
            writer.release()
        pull.close()
        ctx.term()

if __name__ == "__main__":
    main()
