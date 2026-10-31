"""Client/server split, like RECAST3D's adapter -> SliceRecon -> visualiser pipeline.

    adapter  (minirecast push ...)  --projections-->  server (minirecast serve)
    viewer   (minirecast view ...)  <--slices------>  server

The server holds the projections and does all reconstruction, so it can run on
the machine next to the detector (or a compute node) while the viewer runs on
your laptop. Connections use Python's standard library
(multiprocessing.connection): messages are authenticated with a shared key
(HMAC), and carry a JSON header + raw float arrays - no pickle, so a peer cannot
make the other side execute code.

For a remote server, keep it bound to 127.0.0.1 and use an SSH tunnel:
    ssh -L 5558:localhost:5558 user@gpu-host     # then: --server localhost:5558
"""
import json
import os
import threading
import time
from multiprocessing.connection import Client, Listener

import numpy as np

from .geometry import ConeVecGeometry, ParallelGeometry
from .recon import SliceReconstructor, slice_points, plane_size_for_box

DEFAULT_PORT = 5558


def authkey():
    return os.environ.get("MINIRECAST_KEY", "minirecast-demo").encode()


def parse_address(text):
    host, _, port = text.rpartition(":")
    return (host or "127.0.0.1", int(port or DEFAULT_PORT))


# ------------------------------------------------------------------ messages
def send(conn, header, array=None):
    header = dict(header)
    if array is not None:
        array = np.ascontiguousarray(array)
        header["_array"] = {"dtype": array.dtype.str, "shape": list(array.shape)}
    conn.send_bytes(json.dumps(header).encode())
    if array is not None:
        conn.send_bytes(array.tobytes())


def recv(conn):
    header = json.loads(conn.recv_bytes().decode())
    array = None
    spec = header.pop("_array", None)
    if spec is not None:
        dtype = np.dtype(spec["dtype"])
        if dtype.kind not in "fiu":                      # numbers only
            raise ValueError(f"refusing array dtype {dtype}")
        array = np.frombuffer(conn.recv_bytes(), dtype=dtype).reshape(spec["shape"])
    return header, array


def geometry_message(geom):
    """(header, array) describing a geometry."""
    if isinstance(geom, ParallelGeometry):
        return ({"cmd": "geometry", "kind": "parallel", "rows": geom.rows, "cols": geom.cols,
                 "pixel_u": geom.pixel_u, "pixel_v": geom.pixel_v,
                 "box": [float(b) for b in geom.volume_box()]},
                np.asarray(geom.angles, dtype=np.float64))
    return ({"cmd": "geometry", "kind": "cone", "rows": geom.rows, "cols": geom.cols,
             "angular_range": float(geom.angular_range),
             "box": [float(b) for b in geom.volume_box()]},
            np.asarray(geom.vectors, dtype=np.float64))


def geometry_from_message(h, arr):
    if h["kind"] == "parallel":
        return ParallelGeometry(np.asarray(arr, float), int(h["rows"]), int(h["cols"]),
                                float(h["pixel_u"]), float(h["pixel_v"]),
                                box=np.asarray(h["box"], float))
    return ConeVecGeometry(np.asarray(arr, float).reshape(-1, 12), int(h["rows"]),
                           int(h["cols"]), float(h["angular_range"]),
                           box=np.asarray(h["box"], float))


# ------------------------------------------------------------------ server
class Server:
    def __init__(self, host="127.0.0.1", port=DEFAULT_PORT, window="shepp-logan",
                 backend="auto", gaussian=0.0, tilt=0.0, verbose=True):
        self.address = (host, port)
        self.verbose = verbose
        self.opts = dict(window=window, backend=backend, gaussian=gaussian, tilt=tilt)
        self.recon = None
        self.lock = threading.Lock()

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def serve_forever(self):
        with Listener(self.address, authkey=authkey()) as listener:
            self.log(f"[server] listening on {self.address[0]}:{self.address[1]} "
                  f"(waiting for an adapter and viewers)")
            while True:
                try:
                    conn = listener.accept()
                except Exception as e:                   # bad key, port scan, ...
                    self.log(f"[server] rejected connection: {e}")
                    continue
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        peer = "client"
        try:
            while True:
                h, arr = recv(conn)
                cmd = h.get("cmd")
                if cmd == "hello":
                    peer = h.get("role", "client")
                    self.log(f"[server] {peer} connected")
                    send(conn, {"ok": True})
                elif cmd == "geometry":
                    geom = geometry_from_message(h, arr)
                    with self.lock:
                        self.recon = SliceReconstructor(None, geom, verbose=False, **self.opts)
                    self.log(f"[server] new scan: {geom.n_angles} projections "
                          f"{geom.rows}x{geom.cols} ({h['kind']} beam), "
                          f"backend={self.recon.backend}")
                    send(conn, {"ok": True})
                elif cmd == "projections":
                    n = self._recon().add_projections(h["indices"], arr)
                    if self.recon.backend == "numba" and n == len(h["indices"]):
                        self.recon.warmup()
                    send(conn, {"ok": True, "n": n})
                elif cmd == "end":
                    self.log(f"[server] adapter finished ({self._recon().n} projections)")
                    send(conn, {"ok": True})
                elif cmd == "info":
                    r = self.recon
                    send(conn, {"ready": r is not None, **(r.info() if r else {})})
                elif cmd == "slice":
                    r = self._recon()
                    img, _ = r.reconstruct_plane(h["origin"], h["normal"], n=int(h["n"]),
                                                 zoom=float(h.get("zoom", 1.0)),
                                                 center=h.get("center"))
                    send(conn, {"version": r.version}, img.astype(np.float32))
                elif cmd == "set_cor":
                    self._recon().set_cor_shift(float(h["px"]))
                    send(conn, {"ok": True})
                elif cmd == "auto_cor":
                    best, _ = self._recon().estimate_cor(verbose=False)
                    send(conn, {"best": best})
                elif cmd == "preview":
                    vol, origin, h_ = self._recon().preview_volume(int(h.get("maxdim", 64)))
                    send(conn, {"origin": [float(o) for o in origin], "spacing": float(h_)}, vol)
                else:
                    send(conn, {"error": f"unknown command {cmd!r}"})
        except (EOFError, ConnectionError, OSError):
            self.log(f"[server] {peer} disconnected")
        except Exception as e:  # keep serving other clients
            self.log(f"[server] error from {peer}: {e!r}")
            try:
                send(conn, {"error": repr(e)})
            except Exception:
                pass
        finally:
            conn.close()

    def _recon(self):
        if self.recon is None:
            raise RuntimeError("no scan yet - start an adapter (minirecast push ...)")
        return self.recon


# ------------------------------------------------------------------ clients
class _Conn:
    def __init__(self, address, role):
        self.conn = Client(address, authkey=authkey())
        self.lock = threading.Lock()
        self.call({"cmd": "hello", "role": role})

    def call(self, header, array=None):
        with self.lock:
            send(self.conn, header, array)
            h, arr = recv(self.conn)
        if "error" in h:
            raise RuntimeError(h["error"])
        return h, arr


class RemoteReconstructor:
    """Viewer-side proxy with the same interface the viewers use locally."""

    def __init__(self, address, wait=True):
        self.address = address
        self._c = _Conn(address, "viewer")
        self._preview_conn = None
        info = self.info()
        while wait and not info.get("ready"):
            print("[viewer] waiting for the server to receive a scan ...")
            time.sleep(1.0)
            info = self.info()
        self.box = np.asarray(info["box"], float)
        self.kind = info["kind"]
        self.native_pixel = float(info["native_pixel"])
        self.backend = "remote:" + info.get("backend", "?")
        self.n_total = info["n_total"]

    # state -----------------------------------------------------------
    def info(self):
        return self._c.call({"cmd": "info"})[0]

    @property
    def n(self):
        return self.info()["n"]

    @property
    def cor_shift(self):
        return self.info()["cor_shift"]

    def plane_size(self, normal):
        return plane_size_for_box(self.box, normal)

    def inside_box(self, pts):
        return np.all(np.abs(pts) <= self.box, axis=-1)

    # reconstruction ----------------------------------------------------
    def reconstruct_plane(self, origin, normal, n=256, size=None, zoom=1.0, center=None):
        pts, _ = slice_points(self.box, origin, normal, n, size, zoom, center)
        h = {"cmd": "slice", "origin": [float(v) for v in origin],
             "normal": [float(v) for v in normal], "n": int(n), "zoom": float(zoom),
             "center": None if center is None else [float(v) for v in center]}
        _, img = self._c.call(h)
        return np.asarray(img, dtype=np.float64), pts

    def preview_volume(self, maxdim=64):
        if self._preview_conn is None:                  # own connection: no UI blocking
            self._preview_conn = _Conn(self.address, "viewer-preview")
        h, vol = self._preview_conn.call({"cmd": "preview", "maxdim": int(maxdim)})
        return vol, np.asarray(h["origin"]), float(h["spacing"])

    def set_cor_shift(self, px):
        self._c.call({"cmd": "set_cor", "px": float(px)})

    def estimate_cor(self, verbose=True):
        best = self._c.call({"cmd": "auto_cor"})[0]["best"]
        if verbose:
            print(f"[viewer] server COR search: best shift {best:+.2f} px")
        return best, None

    def warmup(self):
        pass


class RemoteSink:
    """Adapter-side: send projections to the server (blocks until accepted)."""

    def __init__(self, address, geom):
        self._c = _Conn(address, "adapter")
        self._c.call(*geometry_message(geom))

    def add_projections(self, indices, block):
        h, _ = self._c.call({"cmd": "projections",
                             "indices": [int(i) for i in np.asarray(indices).ravel()]},
                            np.asarray(block, dtype=np.float32))
        return h.get("n")

    def end(self):
        self._c.call({"cmd": "end"})
