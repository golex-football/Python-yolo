import zmq

class ZMQBindPull:
    def __init__(self, ctx, endpoint: str):
        s = ctx.socket(zmq.PULL)
        s.setsockopt(zmq.RCVTIMEO, -1)   # block until message arrives
        s.bind(endpoint)
        self.socket = s

    def recv_bytes(self) -> bytes:
        return self.socket.recv()

    def close(self):
        try:
            self.socket.close(0)
        except Exception:
            pass
