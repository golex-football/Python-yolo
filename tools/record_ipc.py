import os, cv2, zmq, numpy as np, time, argparse
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
        if max(x1, y1, x2, y2) <= 1.5:  # normalized coords -> px
            x1, y1, x2, y2 = x1*W, y1*H, x2*W, y2*H
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0,255,0), 2)

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
    ap.add_argument("--outdir", default="_vid")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    raw_path  = outdir / "raw_boxes.mp4"
    mask_path = outdir / "mask.mp4"
    print(f"[RECORD] saving to: {raw_path} and {mask_path} @ {args.fps} FPS")

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PULL); s.connect(args.endpoint)
    print(f"[RECV] connected to {args.endpoint}")

    writer_raw = writer_mask = None
    frames = 0; t0 = time.time()
    try:
        while True:
            blob = s.recv()
            m = OutputFrame(); m.ParseFromString(blob)
            w, h = m.width, m.height

            raw = to_img(getattr(m, "frame_data", b""), w, h)
            vis = to_img(getattr(m, "vis_frame_data", b""), w, h)
            mask = to_mask(getattr(m, "mask_data", b""), w, h)

            # build display frames
            disp = raw.copy() if raw is not None else (vis.copy() if vis is not None else np.zeros((h,w,3), np.uint8))
            draw_boxes(disp, m.boxes)
            if mask is None:
                mask = synth_mask_from_boxes(w, h, m.boxes)
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            # lazy init writers
            if writer_raw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_raw  = cv2.VideoWriter(str(raw_path),  fourcc, args.fps, (w,h))
                writer_mask = cv2.VideoWriter(str(mask_path), fourcc, args.fps, (w,h))

            writer_raw.write(disp)
            writer_mask.write(mask_color)
            frames += 1

            # status ~2Hz
            if time.time() - t0 > 0.5:
                print(f"[RECORD] wrote {frames} frames (last frame_id={m.frame_id}, boxes={len(m.boxes)})")
                frames = 0; t0 = time.time()
    except KeyboardInterrupt:
        print("[RECORD] Ctrl-C, closing...")
    finally:
        if writer_raw: writer_raw.release()
        if writer_mask: writer_mask.release()

if __name__ == "__main__":
    main()
