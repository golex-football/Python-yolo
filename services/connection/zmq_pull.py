import os
import zmq
from .frame_message_pb2 import InputFrame

DEFAULT_IN_SOCK = os.environ.get("MOONALT_IPC_IN", "/tmp/yolo_input.sock")

class ZMQPull:
    """
    Worker-side PULL socket over IPC. Worker BINDs; producers CONNECT.
    """
    def __init__(self, socket_path: str = DEFAULT_IN_SOCK):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PULL)

        # Clean stale .sock file if exists (Linux/WSL)
        if socket_path.startswith("/"):
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass

        self.socket.bind(f"ipc://{socket_path}")

    def recv_frame(self) -> InputFrame:
        data = self.socket.recv()   # single protobuf blob
        msg = InputFrame()
        msg.ParseFromString(data)
        return msg
