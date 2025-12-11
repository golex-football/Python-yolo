import os
import time
import cv2
import zmq
import numpy as np

from services.connection.zmq_bind_push import ZMQBindPush
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
    endpt = _norm_ipc("MOONALT_IPC_IN", "ipc:///tmp/capture")
    video_path = os.environ.get("MOONALT_TEST_VIDEO")
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError(f"Set MOONALT_TEST_VIDEO to a readable file. Got: {video_path}")

    log_every = _env_int("MOONALT_LOG_EVERY", 30)
    max_fps = float(os.environ.get("MOONALT_CAPTURE_MAX_FPS", "30"))
    min_dt = 1.0 / max_fps if max_fps > 0 else 0.0
    last_send_t = 0.0

    # ----- zmq push -----
    # Capture binds PUSH, worker connects PULL.
    ctx = zmq.Context.instance()
    # If this is ipc://, make sure stale socket file won't break bind()
    if endpt.startswith("ipc://"):
        ipc_path = endpt[len("ipc://"):]
        try:
            if ipc_path and os.path.exists(ipc_path):
                os.remove(ipc_path)
        except Exception:
            pass

    push = ZMQBindPush(ctx, endpt)
    print(f"[capture_video] BIND -> {endpt}")
    print(f"[capture_video] opening: {video_path}")

    # ----- video -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

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
            # Fill BOTH string and enum fields (enum is preferred)
            msg.schema = "golex.vt.input_v1"
            msg.schema_id = 1  # SCHEMA_GOLEX_VT_INPUT_V1
            msg.frame_id = int(frame_id)
            msg.timestamp = float(time.time())
            msg.width = int(w)
            msg.height = int(h)
            msg.pixel_format = "BGR24"
            msg.pixel_format_id = 1  # PIXEL_FORMAT_BGR24
            msg.frame_data = frame.tobytes()  # raw, unmodified

            # simple fps cap (drop if worker can't keep up)
            now = time.time()
            if min_dt > 0 and last_send_t > 0:
                to_sleep = min_dt - (now - last_send_t)
                if to_sleep > 0:
                    time.sleep(to_sleep)
            last_send_t = time.time()

            try:
                push.send(msg.SerializeToString())
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
            push.close()
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass

if __name__ == "__main__":
    main()
