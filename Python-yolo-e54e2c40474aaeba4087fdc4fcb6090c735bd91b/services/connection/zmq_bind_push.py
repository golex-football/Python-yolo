import zmq

class ZMQBindPush:
    def __init__(self, ctx, endpoint: str):
        self.ctx = ctx
        s = ctx.socket(zmq.PUSH)
        # Keep memory bounded and avoid blocking forever if no consumer:
        s.setsockopt(zmq.SNDHWM, 100)   # queue up to 100 messages
        s.setsockopt(zmq.SNDTIMEO, 0)   # non-blocking send
        s.bind(endpoint)
        self.socket = s

    def send(self, data: bytes) -> bool:
        try:
            self.socket.send(data, flags=zmq.DONTWAIT)
            return True
        except zmq.Again:
            return False  # downstream not keeping up; drop frame

    def close(self):
        try:
            self.socket.close(0)
        except Exception:
            pass
