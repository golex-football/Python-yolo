import os, time
import numpy as np
import zmq
from services.connection.frame_message_pb2 import InputFrame

IN_EP = os.environ.get("MOONALT_IPC_IN", "ipc:///tmp/capture")

def main():
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUSH)
    endpoint = IN_EP if IN_EP.startswith(("ipc://", "tcp://")) else f"ipc://{IN_EP}"
    # Capture binds PUSH, worker connects PULL
    if endpoint.startswith("ipc://"):
        ipc_path = endpoint[len("ipc://"):]
        try:
            if ipc_path and os.path.exists(ipc_path):
                os.remove(ipc_path)
        except Exception:
            pass
    s.bind(endpoint)
    print(f"[capture_test] BIND -> {endpoint}")

    w, h = 960, 540
    frame_id = 0

    while True:
        # Build a moving gradient frame (BGR)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = 64
        # uint8-safe demo gradient
        shift = frame_id & 0xFF  # same as % 256
        col = (np.arange(w, dtype=np.uint16) + shift) % 256
        row = (np.arange(h, dtype=np.uint16) + shift) % 256
        img[..., 1] = col.astype(np.uint8)[None, :]
        img[..., 2] = row.astype(np.uint8)[:, None]

        msg = InputFrame(
            schema="golex.vt.input_v1",
            frame_id=frame_id,
            timestamp=time.time(),
            width=w,
            height=h,
            pixel_format="BGR24",
            schema_id=1,
            pixel_format_id=1,
        )
        msg.frame_data = img.tobytes()
        s.send(msg.SerializeToString())

        if frame_id % 25 == 0:
            print(f"[capture_test] sent frame_id={frame_id}")

        frame_id += 1
        time.sleep(0.02)  # ~50 fps

if __name__ == "__main__":
    main()
