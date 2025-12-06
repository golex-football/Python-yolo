import zmq

# Default IPC endpoints (Linux/WSL-safe sockets)
DEFAULT_OUT     = "ipc:///tmp/moonalt_out"   # worker binds PUSH, consumers connect PULL

class ZmqPush:
    """ZMQ PUSH wrapper for sending worker output (multipart: [meta_pb, frame_bytes, mask_bytes])."""
    def __init__(self, endpoint: str = DEFAULT_OUT, bind: bool = True):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        if bind:
            self.sock.bind(endpoint)
        else:
            self.sock.connect(endpoint)
        self.endpoint = endpoint

    def send(self, meta_pb: bytes, frame_bytes: bytes, mask_bytes: bytes):
        # Non-blocking: drop if downstream is slow
        try:
            self.sock.send_multipart([meta_pb, frame_bytes, mask_bytes], flags=zmq.NOBLOCK)
        except zmq.Again:
            # downstream slow: drop
            pass
