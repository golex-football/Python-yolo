import os, time, math, argparse, cv2, zmq
from services.connection.frame_message_pb2 import InputFrame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to video file")
    ap.add_argument("--endpoint", default=os.environ.get("MOONALT_IPC_IN","ipc:///tmp/yolo_input.sock"),
                    help="endpoint to CONNECT (worker binds PULL)")
    ap.add_argument("--fps", type=float, default=0.0, help="0 = use file FPS")
    ap.add_argument("--seconds", type=float, default=0.0, help="run N seconds then exit (0 = infinite)")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUSH)  # producer PUSH -> worker PULL
    s.connect(args.endpoint)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or math.isnan(src_fps) or src_fps < 1e-3:
        src_fps = 25.0
    send_fps = args.fps if args.fps > 0 else src_fps
    delay = 1.0 / max(1e-6, send_fps)

    print(f"[SEND] {args.video} → {args.endpoint} at {send_fps:.2f} FPS")
    frame_id = 0
    t0 = time.time()
    try:
        while True:
            if args.seconds > 0 and (time.time() - t0) >= args.seconds:
                print(f"[SEND] reached {args.seconds}s, exiting.")
                break

            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                continue

            h, w = frame.shape[:2]
            m = InputFrame(schema="moonalt.yolo_input_v1",
                           frame_id=frame_id, timestamp=time.time(),
                           width=w, height=h, pixel_format="BGR24")
            m.frame_data = frame.tobytes()
            s.send(m.SerializeToString())
            frame_id += 1
            time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

if __name__ == "__main__":
    main()
