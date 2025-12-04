import os, zmq, time
from services.connection.frame_message_pb2 import InputFrame, OutputFrame

IN  = os.environ.get("MOONALT_IPC_IN","ipc:///tmp/yolo_input.sock")
OUT = os.environ.get("MOONALT_IPC_OUT","ipc:///tmp/yolo_output.sock")
print(f"[SMOKE] expecting IPC endpoints:\n  IN : {IN}\n  OUT: {OUT}")
assert IN.startswith("ipc://") and OUT.startswith("ipc://"), "Not using IPC!"

# quick REQ/REP ping on the OUT address won't work (PUSH/PULL), so we just bind dummies and check create/close.
ctx = zmq.Context.instance()
pull = ctx.socket(zmq.PULL); pull.setsockopt(zmq.RCVTIMEO, 100); pull.connect(OUT)
push = ctx.socket(zmq.PUSH); push.setsockopt(zmq.SNDTIMEO, 100); push.connect(IN)
print("[SMOKE] ZMQ sockets created and connected (PUSH->IN, PULL<-OUT).")
print("[SMOKE] Protobuf classes present:", bool(InputFrame), bool(OutputFrame))
pull.close(); push.close()
print("[SMOKE] OK.")
