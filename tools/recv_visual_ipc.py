import os, cv2, zmq, numpy as np, argparse, time
from pathlib import Path
from services.connection.frame_message_pb2 import OutputFrame

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
        # normalized -> pixels
        if max(x1, y1, x2, y2) <= 1.5:
            x1, y1, x2, y2 = x1*W, y1*H, x2*W, y2*H
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0,255,0), 2)
        label = f"{getattr(b,'class_id',-1)}:{getattr(b,'score',0.0):.2f}"
        cv2.putText(img, label, (p1[0], max(0,p1[1]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

def synth_mask_from_boxes(w, h, boxes):
    mask = np.zeros((h, w), np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
        if max(x1, y1, x2, y2) <= 1.5:
            x1, y1, x2, y2 = x1*w, y1*h, x2*w, y2*h
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("MOONALT_IPC_OUT","ipc:///tmp/yolo_output.sock"))
    ap.add_argument("--save", action="store_true", help="record MP4 files")
    ap.add_argument("--outdir", default="_vid", help="output folder for MP4s")
    ap.add_argument("--fps", type=float, default=25.0, help="recording FPS")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PULL)
    s.connect(args.endpoint)
    print(f"[RECV] connected to {args.endpoint}")
    print("[RECV] windows: raw+boxes, mask(colorized) — press 'q' to quit")

    writer_raw = writer_mask = None
    outdir = Path(args.outdir)
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[REC] saving to: {outdir} (FPS={args.fps})")

    last_print = 0.0
    while True:
        blob = s.recv()
        m = OutputFrame(); m.ParseFromString(blob)
        w, h = m.width, m.height

        raw = to_img(getattr(m, "frame_data", b""), w, h)
        vis = to_img(getattr(m, "vis_frame_data", b""), w, h)
        mask = to_mask(getattr(m, "mask_data", b""), w, h)

        # build frames for display
        disp_raw = raw.copy() if raw is not None else (vis.copy() if vis is not None else np.zeros((h,w,3), np.uint8))
        draw_boxes(disp_raw, m.boxes)

        if mask is None:
            # synthesize from boxes so you always SEE a mask
            mask = synth_mask_from_boxes(w, h, m.boxes)

        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        # show
        cv2.imshow("raw+boxes", disp_raw)
        cv2.imshow("mask", mask_color)

        # record (lazy init once size known)
        if args.save:
            if writer_raw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_raw  = cv2.VideoWriter(str(outdir/"raw_boxes.mp4"), fourcc, args.fps, (w,h))
                writer_mask = cv2.VideoWriter(str(outdir/"mask.mp4"), fourcc, args.fps, (w,h))
            writer_raw.write(disp_raw)
            writer_mask.write(mask_color)

        # status print @ ~2Hz
        now = time.time()
        if now - last_print > 0.5:
            print(f"[RECV] frame_id={m.frame_id}  size={w}x{h}  boxes={len(m.boxes)}")
            last_print = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if writer_raw: writer_raw.release()
    if writer_mask: writer_mask.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
