import os, cv2, zmq, numpy as np, time
from pathlib import Path
from services.connection.frame_message_pb2 import OutputFrame

HEADLESS = os.environ.get("HEADLESS","0") == "1"
DUMP = os.environ.get("DUMP","1") == "1" if HEADLESS else os.environ.get("DUMP","0") == "1"
OUTDIR = Path(os.environ.get("OUTDIR","_out"))
if DUMP: OUTDIR.mkdir(parents=True, exist_ok=True)

def to_img(buf, w, h):
    if not buf or len(buf) != w*h*3: return None
    return np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))

def to_mask(buf, w, h):
    if not buf or len(buf) != w*h: return None
    return np.frombuffer(buf, dtype=np.uint8).reshape((h, w))

def draw_boxes(img, boxes):
    if img is None: return
    H, W = img.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
        if max(x1, y1, x2, y2) <= 1.5:
            x1, y1, x2, y2 = x1*W, y1*H, x2*W, y2*H
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0,255,0), 2)
        label = f"{getattr(b,'class_id',-1)}:{getattr(b,'score',0.0):.2f}"
        cv2.putText(img, label, (p1[0], max(0,p1[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

def main():
    OUT = os.environ.get("MOONALT_IPC_OUT","ipc:///tmp/yolo_output.sock")
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PULL); s.connect(OUT)
    print(f"[RECV] connected to {OUT}")
    if HEADLESS:
        print("[RECV] HEADLESS=1: no GUI, ", "DUMP=on → saving to", OUTDIR if DUMP else "DUMP=off")

    last_save = 0.0
    while True:
        blob = s.recv()
        m = OutputFrame(); m.ParseFromString(blob)
        w, h = m.width, m.height
        raw = to_img(getattr(m, "frame_data", b""), w, h)
        vis = to_img(getattr(m, "vis_frame_data", b""), w, h)
        msk = to_mask(getattr(m, "mask_data", b""), w, h)

        disp = raw.copy() if raw is not None else (vis.copy() if vis is not None else np.zeros((h,w,3), np.uint8))
        draw_boxes(disp, m.boxes)
        if not HEADLESS:
            cv2.imshow("raw+boxes", disp)
            cv2.imshow("mask", cv2.applyColorMap(msk, cv2.COLORMAP_JET) if msk is not None else np.zeros((h,w,3),np.uint8))
            cv2.imshow("vis", vis if vis is not None else disp)
            if cv2.waitKey(1) & 0xFF == 'q': break

        # throttle disk writes to ~5Hz
        now = time.time()
        if DUMP and now - last_save > 0.2:
            cv2.imwrite(str(OUTDIR / f"raw_boxes_{m.frame_id:06d}.jpg"), disp)
            if msk is not None:
                cv2.imwrite(str(OUTDIR / f"mask_{m.frame_id:06d}.png"), msk)
            if vis is not None:
                cv2.imwrite(str(OUTDIR / f"vis_{m.frame_id:06d}.jpg"), vis)
            last_save = now

if __name__ == "__main__":
    main()
