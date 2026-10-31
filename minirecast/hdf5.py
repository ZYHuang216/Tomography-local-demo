"""Parallel-beam (synchrotron) data in Data Exchange HDF5 format.

This is the format of APS / TomoBank (https://tomobank.readthedocs.io) and of
many other beamlines (written by e.g. dxchange / tomopy tools):

    /exchange/data        projections   [angles, rows, cols]
    /exchange/data_white  flat fields   [n, rows, cols]     (optional)
    /exchange/data_dark   dark fields   [n, rows, cols]     (optional)
    /exchange/theta       angles in degrees                 (optional)

Needs h5py (installed by setup.sh).
"""
import numpy as np

from .geometry import ParallelGeometry

PIXEL_SIZE_KEYS = (
    "measurement/instrument/detection_system/objective/resolution",   # um, APS 2-BM/32-ID
    "measurement/instrument/detector/pixel_size",
    "measurement/instrument/detector/pixel_size_x",
)


def _bin(a, sample):
    """Average sample x sample blocks of the last two axes."""
    if sample <= 1:
        return a.astype(np.float32)
    *lead, h, w = a.shape
    h2, w2 = (h // sample) * sample, (w // sample) * sample
    a = a[..., :h2, :w2].reshape(*lead, h2 // sample, sample, w2 // sample, sample)
    return a.mean(axis=(-1, -3)).astype(np.float32)


def _read_pixel_size(f):
    for key in PIXEL_SIZE_KEYS:
        if key in f:
            v = float(np.ravel(f[key][()])[0])
            if v > 0:
                return v / 1000.0 if v > 0.05 else v      # um -> mm when it looks like um
    return None


def load_dxchange(path, sample=2, skip=1, rows=None, pixel_size=None, h5=None, verbose=True):
    """Return (projections [rows, angles, cols] as -log(I/I0), ParallelGeometry).

    rows       : 'start:stop' detector-row range to load (saves memory/time)
    pixel_size : detector pixel in mm (default: from the file, else 1 = pixel units)
    """
    if h5 is None:
        try:
            import h5py
        except ImportError as e:
            raise SystemExit("h5py is not installed - run ./setup.sh again") from e
        h5 = h5py
    r0, r1 = None, None
    if rows:
        a, _, b = str(rows).partition(":")
        r0, r1 = int(a) if a else None, int(b) if b else None
    with h5.File(path, "r") as f:
        data = f["exchange/data"]
        n_all = data.shape[0]
        rsl = slice(r0, r1)
        proj = _bin(np.asarray(data[::skip, rsl, :]), sample)              # [A, R, C]
        flat = (_bin(np.asarray(f["exchange/data_white"][:, rsl, :]), sample).mean(0)
                if "exchange/data_white" in f else None)
        dark = (_bin(np.asarray(f["exchange/data_dark"][:, rsl, :]), sample).mean(0)
                if "exchange/data_dark" in f else None)
        if "exchange/theta" in f:
            theta = np.asarray(f["exchange/theta"][()], dtype=float)[::skip]
        else:
            theta = np.linspace(0.0, 180.0, n_all, endpoint=False)[::skip]
        px = pixel_size or _read_pixel_size(f) or 1.0
    if dark is None:
        dark = np.zeros(proj.shape[1:], np.float32)
    if flat is None:
        flat = np.full(proj.shape[1:], float(np.percentile(proj, 99.9)), np.float32)
    trans = (proj - dark) / np.maximum(flat - dark, 1e-6)
    proj = -np.log(np.clip(trans, 1e-3, None)).astype(np.float32)
    proj = np.ascontiguousarray(np.transpose(proj, (1, 0, 2)))           # [R, A, C]
    theta = theta[: proj.shape[1]]
    span = np.ptp(theta) * len(theta) / max(len(theta) - 1, 1)
    if not (170 < span < 190 or 350 < span < 370):
        print(f"warning: angles cover {span:.0f} deg; FBP assumes 180 or 360")
    pixel = px * sample
    geom = ParallelGeometry(np.deg2rad(theta), proj.shape[0], proj.shape[2], pixel, pixel)
    if verbose:
        unit = "mm" if px != 1.0 else "px (no pixel size in file; use --pixel-size)"
        print(f"[hdf5] {path}: {proj.shape[1]}/{n_all} projections, "
              f"{proj.shape[0]}x{proj.shape[2]} after {sample}x{sample} binning, "
              f"pixel {pixel:g} {unit}, angles {theta[0]:.1f}..{theta[-1]:.1f} deg")
    return proj, geom
