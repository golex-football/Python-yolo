import zmq

class ZMQPull:
    """Simple PULL socket that CONNECTs to endpoint."""

    def __init__(self, endpoint: str, rcvhwm: int = 4, linger_ms: int = 0):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVHWM, int(rcvhwm))
        self.sock.setsockopt(zmq.LINGER, int(linger_ms))
        self.sock.connect(endpoint)

    def recv(self) -> bytes:
        return self.sock.recv()

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
