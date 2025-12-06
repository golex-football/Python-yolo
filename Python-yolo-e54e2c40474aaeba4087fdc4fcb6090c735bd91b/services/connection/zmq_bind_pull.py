import zmq

class ZMQBindPull:
    def __init__(self, ctx, endpoint: str):
        self.ctx = ctx
        self.socket = ctx.socket(zmq.PULL)
        # Block forever waiting for frames (no timeout -> no zmq.Again)
        self.socket.setsockopt(zmq.RCVTIMEO, -1)
        self.socket.bind(endpoint)

    def recv_bytes(self) -> bytes:
        return self.socket.recv()

    def close(self):
        try:
            self.socket.close(0)
        except Exception:
            pass
