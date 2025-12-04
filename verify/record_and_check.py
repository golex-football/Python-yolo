import os, cv2, zmq, numpy as np, time, re, zlib, json, argparse
from pathlib import Path
from services.connection.frame_message_pb2 import OutputFrame

CRC_RE = re.compile(r"crc32=(\d+)")

def to_img(buf, w, h):
    if not buf or len(buf) != w*h*3: return None
    return np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))

def to_mask(buf, w, h):
    if not buf or len(buf) != w*h: return None
    return np.frombuffer(buf, dtype=np.uint8).reshape((h, w))

def boxes_are_normalized(boxes):
    return all(max(b.x1,b.y1,b.x2,b.y2) <= 1.5 for b in boxes)

def boxes_in_bounds_px(boxes, w, h):
    for b in boxes:
        if not (0 <= b.x1 <= w and 0 <= b.x2 <= w and 0 <= b.y1 <= h and 0 <= b.y2 <= h):
            return False
        if b.x2 <= b.x1 or b.y2 <= b.y1: return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("MOONALT_IPC_OUT","ipc:///tmp/yolo_output.sock"))
    ap.add_argument("--outdir", default="_vid")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--bt_dir", default="_bt/human-bboxes", help="BroadTrack bbox export dir")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    bt_dir = Path(args.bt_dir); bt_dir.mkdir(parents=True, exist_ok=True)

    raw_path  = outdir / "raw_boxes.mp4"
    mask_path = outdir / "mask.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_raw = writer_mask = None

    print(f"[CHECK] saving raw+boxes → {raw_path}")
    print(f"[CHECK] saving mask      → {mask_path}")

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PULL); s.connect(args.endpoint)
    print(f"[RECV] connected to {args.endpoint}")

    t0 = time.time(); frames = 0
    stats = {"crc_match":0, "crc_total":0, "norm_boxes":0, "abs_boxes_ok":0, "w":0, "h":0}

    try:
        while True:
            if args.seconds > 0 and (time.time()-t0) >= args.seconds:
                print("[CHECK] time elapsed, stopping.")
                break

            blob = s.recv()
            m = OutputFrame(); m.ParseFromString(blob)
            w, h = m.width, m.height
            stats["w"], stats["h"] = w, h

            # reconstruct raw & mask
            raw  = to_img(getattr(m,"frame_data",b""), w, h)
            vis  = to_img(getattr(m,"vis_frame_data",b""), w, h)
            mask = to_mask(getattr(m,"mask_data",b""), w, h)
            if raw is None and vis is not None:
                raw = vis.copy()
            if mask is None:
                # fallback: build binary mask from boxes
                mask = np.zeros((h,w), np.uint8)
                for b in m.boxes:
                    x1,y1,x2,y2 = b.x1,b.y1,b.x2,b.y2
                    if max(x1,y1,x2,y2) <= 1.5: x1,y1,x2,y2 = x1*w,y1*h,x2*w,y2*h
                    cv2.rectangle(mask,(int(x1),int(y1)),(int(x2),int(y2)),255,-1)

            # init writers
            if writer_raw is None:
                writer_raw  = cv2.VideoWriter(str(raw_path), fourcc, args.fps, (w,h))
                writer_mask = cv2.VideoWriter(str(mask_path),fourcc, args.fps, (w,h))

            # draw boxes on a copy (expect pixel-space)
            disp = raw.copy() if raw is not None else np.zeros((h,w,3), np.uint8)
            if boxes_are_normalized(m.boxes):
                stats["norm_boxes"] += 1
                # scale normalized -> px for drawing
                for b in m.boxes:
                    x1, y1, x2, y2 = int(b.x1*w), int(b.y1*h), int(b.x2*w), int(b.y2*h)
                    cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)
            else:
                if boxes_in_bounds_px(m.boxes, w, h): stats["abs_boxes_ok"] += 1
                for b in m.boxes:
                    cv2.rectangle(disp,(int(b.x1),int(b.y1)),(int(b.x2),int(b.y2)),(0,255,0),2)

            writer_raw.write(disp)
            writer_mask.write(cv2.applyColorMap(mask, cv2.COLORMAP_JET))

            # CRC check: producer stuffed crc32 in input schema suffix; worker should copy schema to output.meta/schema
            # We re-hash the *raw bytes we received* and compare.
            src_crc = None
            for field in [getattr(m,"schema",""), getattr(m,"meta","")]:
                if not field: continue
                mt = CRC_RE.search(field)
                if mt: src_crc = int(mt.group(1)); break
            if src_crc is not None and raw is not None:
                crc = (zlib.crc32(raw.tobytes()) & 0xffffffff)
                stats["crc_total"] += 1
                if crc == src_crc: stats["crc_match"] += 1

            frames += 1
    except KeyboardInterrupt:
        print("[CHECK] Ctrl-C")
    finally:
        if writer_raw: writer_raw.release()
        if writer_mask: writer_mask.release()

    # quick invariants
    print("\n==== SUMMARY ====")
    print(json.dumps(stats, indent=2))
    # sanity on mask: binary-ish?
    if mask is not None:
        uniq = np.unique(mask)
        print(f"[MASK] unique values (sample): {uniq[:10]}  (expect mostly {0,255})")
    print("[CHECK] BroadTrack export written as .txt xyxy per frame under _bt/human-bboxes/")
