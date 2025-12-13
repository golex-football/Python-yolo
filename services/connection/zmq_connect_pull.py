import zmq

class ZMQPull:
    """PULL socket that CONNECTs to the endpoint."""

    def __init__(self, ctx: zmq.Context, endpoint: str, rcvhwm: int = 4):
        self.sock = ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVHWM, int(rcvhwm))
        self.sock.connect(endpoint)

    def recv(self) -> bytes:
        return self.sock.recv()

    def close(self):
        self.sock.close(0)
