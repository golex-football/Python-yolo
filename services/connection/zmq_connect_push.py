import os
import zmq

def _ipc_path(endpoint: str) -> str | None:
    if endpoint.startswith("ipc://"):
        return endpoint[len("ipc://"):]
    return None

class ZMQPush:
    """PUSH socket. Default is bind (as requested).

    mode:
      - 'bind'    -> this socket owns the endpoint (other side connects)
      - 'connect' -> other side owns the endpoint (this connects)
    """

    def __init__(self, ctx: zmq.Context, endpoint: str, *, mode: str = "bind", sndhwm: int = 4):
        self.endpoint = endpoint
        self.mode = (mode or "bind").strip().lower()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, int(sndhwm))

        if self.mode == "bind":
            # remove stale ipc file (common cause of bind failure)
            path = _ipc_path(endpoint)
            if path:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            self.sock.bind(endpoint)
        elif self.mode == "connect":
            self.sock.connect(endpoint)
        else:
            raise ValueError(f"Invalid ZMQPush mode: {self.mode!r}")

    def send(self, data: bytes) -> None:
        # blocking send is okay here; if you want drop-on-backpressure, switch to DONTWAIT.
        self.sock.send(data)

    def close(self):
        try:
            self.sock.close(0)
        finally:
            # cleanup ipc file if we bound it
            if self.mode == "bind":
                path = _ipc_path(self.endpoint)
                if path:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
