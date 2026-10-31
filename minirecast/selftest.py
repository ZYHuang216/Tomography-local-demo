"""Quick correctness/speed check on this machine (analytic phantom, writes no files)."""
import socket
import threading
import time

import numpy as np

from .phantom import EllipsoidPhantom, project_parallel, project_cone
from .geometry import ParallelGeometry, circular_cone_vectors
from . import recon as R
from . import preprocess
from .stream import Feeder, arrival_order

PH = EllipsoidPhantom(scale=50.0)


def _score(img, pts, radius=45.0):
    t = PH.evaluate(pts)
    m = np.linalg.norm(pts, axis=-1) < radius
    return np.median(np.abs(img[m] - t[m])), np.corrcoef(img[m], t[m])[0, 1]


def _report(name, ok, msg):
    print(f"{name:34s} {msg}  {'OK' if ok else 'FAIL'}")
    return ok


def _cases():
    return [
        ("parallel", ParallelGeometry(np.arange(180) * np.pi / 180, 96, 128, 0.9, 1.15),
         project_parallel),
        ("cone", circular_cone_vectors(180, 300.0, 600.0, 128, 160, 1.5, det_shift_u=2.0),
         project_cone),
    ]


def test_accuracy_and_backends():
    ok = True
    for name, geom, proj_fn in _cases():
        proj = proj_fn(PH, geom)
        backends = ["numpy"] + (["numba"] if R.HAVE_NUMBA else [])
        imgs = {}
        for be in backends:
            rec = R.SliceReconstructor(proj, geom, backend=be, verbose=False)
            rec.warmup()
            t0 = time.perf_counter()
            img, pts = rec.reconstruct_plane((3.0, -2.0, 5.0), (1.0, 0.5, 0.8), n=128, size=110)
            dt = time.perf_counter() - t0
            med, corr = _score(img, pts)
            imgs[be] = img
            ok &= _report(f"{name} {be} oblique slice",
                          med < 0.01 and corr > 0.85,
                          f"{dt * 1000:6.1f} ms  median|err| {med:.4f} corr {corr:.3f}")
        if "numba" in imgs:
            diff = np.max(np.abs(imgs["numba"] - imgs["numpy"]))
            ok &= _report(f"{name} numba == numpy", diff < 1e-3, f"max diff {diff:.1e}")
    return ok


def test_streaming():
    """Projections streamed in groups (any order) == batch reconstruction."""
    ok = True
    for name, geom, proj_fn in _cases():
        proj = proj_fn(PH, geom)
        batch = R.SliceReconstructor(proj, geom, backend="numpy", verbose=False)
        live = R.SliceReconstructor(None, geom, backend="numpy", verbose=False)
        Feeder(live, proj, order="interleaved", group=17, rate=0, verbose=False).run()
        a, _ = batch.reconstruct_plane((0, 0, 3.0), (0.2, 0.1, 1.0), n=64)
        b, _ = live.reconstruct_plane((0, 0, 3.0), (0.2, 0.1, 1.0), n=64)
        diff = np.max(np.abs(a - b))
        ok &= _report(f"{name} stream == batch", diff < 1e-4 and live.n == geom.n_angles,
                      f"{live.n} projections, max diff {diff:.1e}")
    order = arrival_order(10, "interleaved")
    ok &= _report("interleaved order", sorted(order) == list(range(10)), str([int(i) for i in order]))
    return ok


def test_cor_shift():
    """Data with a shifted rotation centre is fixed by set_cor_shift / estimate_cor."""
    ok = True
    true_shift = 3.0
    for name in ("parallel", "cone"):
        if name == "parallel":
            geom = ParallelGeometry(np.arange(180) * np.pi / 180, 32, 160, 0.9, 0.9)
            proj = project_parallel(PH, geom)
            # detector moved by +3 px: content appears 3 px further along the columns
            proj = np.roll(proj, int(true_shift), axis=2)
        else:
            moved = circular_cone_vectors(180, 300.0, 600.0, 32, 160, 1.5,
                                          det_shift_u=-true_shift)
            proj = project_cone(PH, moved)
            geom = circular_cone_vectors(180, 300.0, 600.0, 32, 160, 1.5)
        rec = R.SliceReconstructor(proj, geom, backend="auto", verbose=False)
        img0, pts = rec.reconstruct_plane((0, 0, 0), (0, 0, 1), n=96, size=110)
        best, _ = rec.estimate_cor(np.arange(-5.0, 5.01, 0.5), n=96, verbose=False)
        img1, _ = rec.reconstruct_plane((0, 0, 0), (0, 0, 1), n=96, size=110)
        e0, e1 = _score(img0, pts)[0], _score(img1, pts)[0]
        ok &= _report(f"{name} COR search", abs(best - true_shift) <= 0.5 and e1 < e0,
                      f"found {best:+.1f} px (true {true_shift:+.1f}), "
                      f"median|err| {e0:.4f} -> {e1:.4f}")
    return ok


def test_preprocess():
    ok = True
    rng = np.random.default_rng(1)
    block = rng.normal(1.0, 0.3, (40, 3, 50)).astype(np.float32)
    blur = preprocess.gaussian_blur(block, 2.0)
    ok &= _report("gaussian keeps mean, cuts noise",
                  abs(blur.mean() - block.mean()) < 0.01 and blur.std() < 0.5 * block.std(),
                  f"std {block.std():.3f} -> {blur.std():.3f}")
    img = np.zeros((41, 1, 41), np.float32)
    img[20, 0, 30] = 1.0                                   # point right of centre
    rot = preprocess.rotate_projections(img, 90.0)
    r, c = np.unravel_index(np.argmax(rot[:, 0, :]), (41, 41))
    ok &= _report("tilt rotation", (r, c) in ((10, 20), (30, 20)), f"peak moved to {(int(r), int(c))}")
    return ok


def test_hdf5_loader():
    """DXchange reader on an in-memory fake file (h5py not needed for the test)."""
    from .hdf5 import load_dxchange
    geom = ParallelGeometry(np.arange(90) * np.pi / 90, 40, 64, 1.7, 1.7)
    p = project_parallel(PH, geom) * 0.01
    flat, dark = 5000.0, 100.0
    data = (dark + (flat - dark) * np.exp(-p)).transpose(1, 0, 2)    # [A, R, C]
    files = {"exchange/data": data.astype(np.uint16),
             "exchange/data_white": np.full((2, 40, 64), flat, np.uint16),
             "exchange/data_dark": np.full((2, 40, 64), dark, np.uint16),
             "exchange/theta": np.rad2deg(geom.angles),
             "measurement/instrument/detector/pixel_size": np.array([1700.0])}   # um

    class F(dict):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class FakeH5:
        @staticmethod
        def File(path, mode):
            return F(files)

    proj, g = load_dxchange("fake.h5", sample=1, h5=FakeH5, verbose=False)
    rec = R.SliceReconstructor(proj / 0.01, g, backend="auto", verbose=False)
    img, pts = rec.reconstruct_plane((0, 0, 0), (0, 0, 1), n=64, size=100)
    med, corr = _score(img, pts)
    return _report("hdf5 (Data Exchange) loader", corr > 0.85 and abs(g.pixel_u - 1.7) < 1e-9,
                   f"pixel {g.pixel_u} mm, corr {corr:.3f}")


def test_network():
    """adapter -> server -> viewer on localhost gives the same slice as local."""
    from . import net
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = net.Server("127.0.0.1", port, backend="numpy", verbose=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.3)
    geom = circular_cone_vectors(60, 300.0, 600.0, 48, 64, 3.5)
    proj = project_cone(PH, geom)
    sink = net.RemoteSink(("127.0.0.1", port), geom)
    Feeder(sink, proj, order="interleaved", group=16, rate=0, verbose=False).run()
    sink.end()
    viewer = net.RemoteReconstructor(("127.0.0.1", port))
    img_r, _ = viewer.reconstruct_plane((0, 0, 1.0), (0.3, 0.0, 1.0), n=48, zoom=1.5,
                                        center=(2.0, 0, 0))
    local = R.SliceReconstructor(proj, geom, backend="numpy", verbose=False)
    img_l, _ = local.reconstruct_plane((0, 0, 1.0), (0.3, 0.0, 1.0), n=48, zoom=1.5,
                                       center=(2.0, 0, 0))
    viewer.set_cor_shift(1.0)
    vol, _, _ = viewer.preview_volume(12)
    diff = np.max(np.abs(img_r - img_l))
    return _report("network adapter/server/viewer",
                   diff < 1e-4 and viewer.info()["n"] == 60 and vol.ndim == 3,
                   f"max diff {diff:.1e}, preview {vol.shape}")


def test_volume_and_model():
    """3D volume == point reconstruction; Otsu threshold separates the phantom."""
    from . import model as M
    geom = circular_cone_vectors(60, 300.0, 600.0, 48, 64, 3.5)
    rec = R.SliceReconstructor(project_cone(PH, geom), geom, backend="numpy", verbose=False)
    vol, origin, h = rec.reconstruct_volume(voxel=5.0)
    i, j, k = (np.array(vol.shape) * [0.3, 0.6, 0.5]).astype(int)
    p = origin + h * np.array([i, j, k])
    same = abs(vol[i, j, k] - rec.reconstruct_points(p[None])[0]) < 1e-5
    iso = M.otsu_threshold(vol)
    return _report("3D volume + Otsu threshold", same and 0.05 < iso < 0.9,
                   f"volume {vol.shape}, iso {iso:.3f}")


def run():
    ok = True
    for t in (test_accuracy_and_backends, test_streaming, test_cor_shift, test_preprocess,
              test_hdf5_loader, test_network, test_volume_and_model):
        try:
            ok &= t()
        except Exception as e:
            ok = False
            print(f"{t.__name__:34s} ERROR {e!r}  FAIL")
    if not R.HAVE_NUMBA:
        print("numba not installed: using the slower numpy backend")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1
