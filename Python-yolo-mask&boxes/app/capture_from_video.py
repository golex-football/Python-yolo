import os
import time
import cv2
import zmq
import numpy as np

from services.connection.frame_message_pb2 import InputFrame

def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _norm_ipc(envkey: str, default: str) -> str:
    v = os.environ.get(envkey, default)
    if v.startswith("ipc://") or v.startswith("tcp://"):
        return v
    if v.startswith("/"):
        return "ipc://" + v
    return v

def main():
    # ----- config -----
    endpt = _norm_ipc("MOONALT_IPC_IN", "ipc:///tmp/yolo_input.sock")
    path = os.environ.get("MOONALT_TEST_VIDEO")
    if not path or not os.path.exists(path):
        raise RuntimeError(f"Set MOONALT_TEST_VIDEO to a readable file. Got: {path}")

    log_every = _env_int("MOONALT_LOG_EVERY", 30)
    max_fps = float(os.environ.get("MOONALT_CAPTURE_MAX_FPS", "30"))
    min_dt = 1.0 / max_fps if max_fps > 0 else 0.0
    last_send_t = 0.0

    # ----- zmq push -----
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUSH)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.SNDHWM, 2)
    s.setsockopt(zmq.IMMEDIATE, 1)
    s.connect(endpt)

    print(f"[capture_video] CONNECTED -> {endpt}")
    print(f"[capture_video] opening: {path}")

    # ----- video -----
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frame_id = 0
    log_frames = 0
    log_t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # rewind for testing loops
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            h, w = frame.shape[:2]
            msg = InputFrame()
            msg.schema = "moonalt.input_v1"
            msg.frame_id = int(frame_id)
            msg.width = int(w)
            msg.height = int(h)
            msg.pixel_format = "BGR24"
            msg.frame_data = frame.tobytes()  # raw, unmodified

            # simple fps cap (drop if worker can't keep up)
            now = time.time()
            if min_dt > 0 and last_send_t > 0:
                to_sleep = min_dt - (now - last_send_t)
                if to_sleep > 0:
                    time.sleep(to_sleep)
            last_send_t = time.time()

            try:
                s.send(msg.SerializeToString(), flags=zmq.NOBLOCK)
            except zmq.Again:
                # worker queue full -> drop this frame
                pass

            frame_id += 1
            log_frames += 1
            if log_frames % max(1, log_every) == 0:
                dt = max(1e-6, time.time() - log_t0)
                fps = log_frames / dt
                print(f"[capture_video] sent frame_id={frame_id} ({w}x{h}) send_fps={fps:.1f}")
                log_frames = 0
                log_t0 = time.time()

    except KeyboardInterrupt:
        print("[capture_video] Ctrl+C -> exit")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            s.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass

if __name__ == "__main__":
    main()
