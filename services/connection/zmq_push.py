import os
import zmq
from .frame_message_pb2 import OutputFrame

DEFAULT_OUT_SOCK = os.environ.get("MOONALT_IPC_OUT", "/tmp/yolo_output.sock")

class ZMQPush:
    """
    Worker-side PUSH socket over IPC. Worker BINDs; consumers CONNECT.
    """
    def __init__(self, socket_path: str = DEFAULT_OUT_SOCK):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUSH)

        # Clean stale .sock file if exists (Linux/WSL)
        if socket_path.startswith("/"):
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass

        self.socket.bind(f"ipc://{socket_path}")

    def send_frame(self, frame_msg: OutputFrame) -> None:
        self.socket.send(frame_msg.SerializeToString())
