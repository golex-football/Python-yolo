import zmq

class ZMQConnectPull:
    def __init__(self, ctx, endpoint: str):
        self.ctx = ctx
        self.socket = ctx.socket(zmq.PULL)
        self.socket.connect(endpoint)

    def recv(self) -> bytes:
        return self.socket.recv()

    def recv_bytes(self) -> bytes:
        return self.socket.recv()

    def close(self):
        try:
            self.socket.close(0)
        except Exception:
            pass
