import zmq


class ZMQConnectPush:
    """PUSH socket that CONNECTs to an endpoint.

    Use this when the receiver (PULL) is the one doing bind().
    """

    def __init__(self, ctx, endpoint: str, *, sndhwm: int = 100):
        s = ctx.socket(zmq.PUSH)
        s.setsockopt(zmq.SNDHWM, int(sndhwm))
        s.setsockopt(zmq.SNDTIMEO, 0)  # non-blocking send
        s.connect(endpoint)
        self.socket = s

    def send(self, data: bytes) -> bool:
        try:
            self.socket.send(data, flags=zmq.DONTWAIT)
            return True
        except zmq.Again:
            return False

    def close(self):
        try:
            self.socket.close(0)
        except Exception:
            pass
