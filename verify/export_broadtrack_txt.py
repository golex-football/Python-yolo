import os, zmq, argparse
from pathlib import Path
from services.connection.frame_message_pb2 import OutputFrame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("MOONALT_IPC_OUT","ipc:///tmp/yolo_output.sock"))
    ap.add_argument("--outdir", default="_bt/human-bboxes")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PULL); s.connect(args.endpoint)
    print(f"[BT] connected to {args.endpoint}, writing to {outdir}")

    try:
        while True:
            blob = s.recv()
            m = OutputFrame(); m.ParseFromString(blob)
            w, h = m.width, m.height
            # write one file per frame_id: "<frame_id>.txt", lines: x1 y1 x2 y2 score
            p = outdir / f"{m.frame_id:06d}.txt"
            with p.open("w") as f:
                for b in m.boxes:
                    x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
                    # normalize -> pixel if needed
                    if max(x1,y1,x2,y2) <= 1.5:
                        x1, y1, x2, y2 = x1*w, y1*h, x2*w, y2*h
                    f.write(f"{int(x1)} {int(y1)} {int(x2)} {int(y2)} {getattr(b,'score',1.0):.3f}\n")
            if m.frame_id % 30 == 0:
                print(f"[BT] wrote {p.name} (boxes={len(m.boxes)})")
    except KeyboardInterrupt:
        print("[BT] Ctrl-C, done.")

if __name__ == "__main__":
    main()
