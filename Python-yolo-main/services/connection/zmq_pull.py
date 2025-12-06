import zmq

# Default IPC endpoints (Linux/WSL-safe sockets)
DEFAULT_RAW_IN  = "ipc:///tmp/moonalt_raw"   # producer binds PUSH, worker connects PULL
DEFAULT_OUT     = "ipc:///tmp/moonalt_out"   # worker binds PUSH, consumers connect PULL

class ZmqPull:
    """ZMQ PULL wrapper for receiving raw frames (multipart: [meta_pb, frame_bytes])."""
    def __init__(self, endpoint: str = DEFAULT_RAW_IN):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVHWM, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(endpoint)
        self.endpoint = endpoint

    def recv(self) -> tuple[bytes, bytes]:
        parts = self.sock.recv_multipart()
        if len(parts) != 2:
            raise ValueError(f"Expected 2 parts (meta, frame_bytes), got {len(parts)} parts from {self.endpoint}")
        return parts[0], parts[1]
