import os
import zmq

def _ipc_path(endpoint: str) -> str | None:
    if endpoint.startswith("ipc://"):
        return endpoint[len("ipc://"):]
    return None

class ZMQPush:
    """ZMQ PUSH socket with explicit bind/connect mode.

    mode:
      - 'bind'    -> this socket binds; peer connects
      - 'connect' -> this socket connects; peer binds
    """

    def __init__(self, ctx: zmq.Context, endpoint: str, mode: str = "connect", sndhwm: int = 4, linger_ms: int = 0):
        self.endpoint = endpoint
        self.mode = (mode or "connect").strip().lower()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, int(sndhwm))
        self.sock.setsockopt(zmq.LINGER, int(linger_ms))

        if self.mode == "bind":
            path = _ipc_path(endpoint)
            if path:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            self.sock.bind(endpoint)
        else:
            self.mode = "connect"
            self.sock.connect(endpoint)

    def send(self, data: bytes):
        self.sock.send(data)

    def close(self):
        try:
            self.sock.close(0)
        finally:
            if self.mode == "bind":
                path = _ipc_path(self.endpoint)
                if path:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
