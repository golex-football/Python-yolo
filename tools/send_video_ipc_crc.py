import os, time, math, argparse, cv2, zmq, zlib
from services.connection.frame_message_pb2 import InputFrame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--endpoint", default=os.environ.get("MOONALT_IPC_IN","ipc:///tmp/yolo_input.sock"))
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUSH); s.connect(args.endpoint)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit(f"Cannot open: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not src_fps or src_fps < 1e-3: src_fps = 25.0
    send_fps = args.fps if args.fps > 0 else src_fps
    delay = 1.0 / max(1e-6, send_fps)

    print(f"[PROD] {args.video} → {args.endpoint} @ {send_fps:.2f} FPS for {args.seconds}s")
    t0 = time.time(); frame_id = 0
    while True:
        if args.seconds > 0 and (time.time()-t0) >= args.seconds:
            print("[PROD] done.")
            break
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        h, w = frame.shape[:2]
        raw = frame.tobytes()
        crc = zlib.crc32(raw) & 0xffffffff

        m = InputFrame(schema="moonalt.yolo_input_v1",
                       frame_id=frame_id, timestamp=time.time(),
                       width=w, height=h, pixel_format="BGR24")
        m.frame_data = raw
        # piggyback CRC via the reserved/optional string "debug" field if you have it;
        # if you don't, we embed as schema suffix (still visible in logs).
        m.schema = f"moonalt.yolo_input_v1|crc32={crc}"
        s.send(m.SerializeToString())

        if frame_id % 15 == 0:
            print(f"[PROD] frame {frame_id} w={w} h={h} crc32={crc}")
        frame_id += 1
        time.sleep(delay)

if __name__ == "__main__":
    main()
