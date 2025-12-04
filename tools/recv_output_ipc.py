import os, zmq
from services.connection.frame_message_pb2 import OutputFrame
OUT = os.environ.get("MOONALT_IPC_OUT", "ipc:///tmp/yolo_output.sock")
ctx = zmq.Context.instance(); s = ctx.socket(zmq.PULL); s.connect(OUT)
print(f"[RECV] connected to {OUT}")
while True:
    blob = s.recv()
    m = OutputFrame(); m.ParseFromString(blob)
    print(f"[RECV] frame_id={m.frame_id} {m.width}x{m.height} boxes={len(m.boxes)}")
