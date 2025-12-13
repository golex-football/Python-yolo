import zmq

class ZMQPush:
    """Simple PUSH socket that CONNECTs to endpoint."""

    def __init__(self, endpoint: str, sndhwm: int = 4, linger_ms: int = 0):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, int(sndhwm))
        self.sock.setsockopt(zmq.LINGER, int(linger_ms))
        self.sock.connect(endpoint)

    def send(self, data: bytes):
        self.sock.send(data)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
