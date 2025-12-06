import os, time, zmq, cv2
from services.connection.zmq_bind_push import ZMQBindPush
from services.connection.frame_message_pb2 import InputFrame

def _norm_ipc(k,d):
    v=os.environ.get(k,d)
    if v.startswith(("ipc://","tcp://")): return v
    if v.startswith("/"): return "ipc://"+v
    return v
def _env_int(k,d):
    v=os.environ.get(k); 
    try: return int(v) if v is not None else d
    except: return d

def main():
    ctx=zmq.Context.instance()
    endpt=_norm_ipc("MOONALT_IPC_IN","ipc:///tmp/yolo_input.sock")
    push=ZMQBindPush(ctx, endpt)  # capture BINDs PUSH, worker CONNECTs PULL? (we BIND to keep symmetry with your current setup)

    dev=os.environ.get("MOONALT_V4L2_DEVICE","/dev/video0")
    W=_env_int("MOONALT_V4L2_WIDTH",1920)
    H=_env_int("MOONALT_V4L2_HEIGHT",1080)
    FPS=_env_int("MOONALT_CAPTURE_MAX_FPS",30)

    cap=cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open V4L2 device: {dev}")

    print(f"[capture_v4l2] BIND -> {endpt}  device={dev} {W}x{H}@{FPS}")
    frame_id=0; t0=time.time(); sent=0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok: 
                time.sleep(0.005); 
                continue

            msg=InputFrame()
            msg.schema="moonalt.input_v1"
            msg.frame_id=frame_id
            msg.timestamp=0.0
            msg.width=W; msg.height=H
            msg.pixel_format="BGR24"
            msg.frame_data=frame_bgr.tobytes()

            push.send_nowait(msg.SerializeToString())

            frame_id+=1; sent+=1
            if sent%30==0:
                elapsed=max(1e-6, time.time()-t0)
                print(f"[capture_v4l2] sent frame_id={frame_id} send_fps={sent/elapsed:.1f}")
                t0=time.time(); sent=0
    except KeyboardInterrupt:
        print("[capture_v4l2] Ctrl+C -> exit")
    finally:
        cap.release()
        push.close(); ctx.term()

if __name__=="__main__":
    main()
