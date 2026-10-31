"""Load a FleX-ray (CWI) cone-beam scan, e.g. the rat skull used in the RECAST3D docs
(Zenodo doi:10.5281/zenodo.1164088).

Self-contained re-implementation of what CWI's flexDATA does for this scanner
(and what RECAST3D's examples/adapters/flexdata_adapter.py relies on):
  * parse 'scan settings.txt'  (SOD, SDD, voxel size, motor positions, ROI)
  * apply a scanner correction profile (motor offsets measured by CWI)
  * build ASTRA cone_vec vectors: source, detector centre, u, v per projection
  * read the TIFF stack, bin it, flat/dark-correct and take -log
The geometry was checked against flexDATA's get_vectors() for the skull scan.
"""
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .geometry import ConeVecGeometry

# Correction profiles copied from flexDATA (flexdata/correct.py). The skull was
# scanned in 2017; flexDATA's own skull example uses 'cwi-flexray-2019-04-24'.
PROFILES = {
    "none": {},
    "cwi-flexray-2019-04-24": {"det_tan": 24.0, "src_ort": -7.0, "axs_tan": -0.5},
    "cwi-flexray-2020-03-26": {"det_tan": 24.300, "src_ort": -6.086, "axs_tan": -0.524},
    "cwi-flexray-2022-10-28": {"det_tan": 24.0485, "src_ort": -5.7730, "axs_tan": -0.5010},
    "cwi-flexray-2024-07-25": {"det_tan": 24.0042, "src_ort": -6.2879, "axs_tan": -0.5143},
}


# --------------------------------------------------------------------------
# Scan settings
# --------------------------------------------------------------------------
def parse_scan_settings(path):
    """Return a dict with geometry parameters (mm) from 'scan settings.txt'."""
    fname = os.path.join(path, "scan settings.txt")
    rec, raw = {}, {}
    with open(fname, "r", errors="replace") as f:
        for line in f:
            key, sep, val = line.partition(":")
            if not sep:
                continue
            key = key.strip().lower()
            raw[key] = val.strip()
            m = re.match(r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", val)
            rec[key] = float(m.group(1)) if m else val.strip()

    def need(k):
        if k not in rec:
            raise KeyError(f"'{k}' missing in {fname}")
        return rec[k]

    sod, sdd = need("sod"), need("sdd")
    p = dict(
        src2obj=sod,
        det2obj=sdd - sod,
        img_pixel=need("voxel size") / 1000.0,          # um -> mm (for the stored binning)
        src_ort=rec.get("ver_tube", 0.0),
        src_tan=rec.get("tra_tube", 0.0),
        det_ort=rec.get("ver_det", 0.0),
        det_tan=rec.get("tra_det", 0.0),
        axs_tan=rec.get("tra_obj", 0.0),
        theta_min=rec.get("start angle", 0.0),
        theta_max=rec.get("last angle", 360.0),
        roi=[int(float(v)) for v in raw.get("roi (ltrb)", "0,0,1943,1535").split(",")],
        name=rec.get("sample name", ""),
    )
    p["det_pixel"] = p["img_pixel"] * sdd / sod             # stored (binned) detector pixel
    # ROI is given in 1x1 pixels; centre of the full detector is (971, 767), pitch 74.8 um
    roi = p["roi"]
    p["det_tan"] += ((roi[0] + roi[2]) // 2 - 971) * 0.0748
    p["det_ort"] += ((roi[1] + roi[3]) // 2 - 767) * 0.0748
    return p


def cone_vectors(p, thetas_deg, pixel):
    """ASTRA cone_vec vectors for a circular FleX-ray orbit (no axis tilts).

    Matches flexDATA: rotation about +z, source starts at -y, volume centred
    vertically between source and detector height."""
    z0 = (p["det_ort"] * p["src2obj"] + p["src_ort"] * p["det2obj"]) / (p["src2obj"] + p["det2obj"])
    th = np.deg2rad(np.asarray(thetas_deg, dtype=float))
    c, s = np.cos(th), np.sin(th)

    def rot(x, y):  # rotate (x, y) about z by every theta
        return np.stack([c * x - s * y, s * x + c * y], axis=1)

    V = np.zeros((len(th), 12))
    V[:, 0:2] = rot(p["src_tan"] - p["axs_tan"], -p["src2obj"])
    V[:, 2] = p["src_ort"] - z0
    V[:, 3:5] = rot(p["det_tan"] - p["axs_tan"], p["det2obj"])
    V[:, 5] = p["det_ort"] - z0
    V[:, 6:8] = rot(pixel, 0.0)
    V[:, 11] = pixel
    return V


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
def _imread(fname):
    try:
        import tifffile
        return tifffile.imread(fname)
    except ImportError:
        from PIL import Image
        return np.array(Image.open(fname))


def _read_binned(fname, sample):
    im = _imread(fname).astype(np.float32)
    if sample > 1:
        h, w = (im.shape[0] // sample) * sample, (im.shape[1] // sample) * sample
        im = im[:h, :w].reshape(h // sample, sample, w // sample, sample).mean(axis=(1, 3))
    return np.flipud(im)          # flexDATA convention: detector rows bottom-up


def _files(path, prefix):
    return sorted(f for f in glob.glob(os.path.join(path, prefix + "*"))
                  if f.lower().endswith((".tif", ".tiff")))


def object_box(proj, V, full_box, margin=1.15):
    """Tight half-size box (mm) around the object, from its shadow on the detector."""
    # mean attenuation per detector column / row over all angles: robust to noise
    col_prof = proj.mean(axis=(0, 1))
    row_prof = proj.mean(axis=(1, 2))
    cols = np.nonzero(col_prof > 0.03 * col_prof.max())[0]
    rows = np.nonzero(row_prof > 0.03 * row_prof.max())[0]
    if cols.size == 0 or rows.size == 0:
        return full_box
    s, d, u, v = V[0, 0:3], V[0, 3:6], V[0, 6:9], V[0, 9:12]
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    mag = abs((d - s) @ n) / abs(-s @ n)
    # where the rotation axis / mid-plane project on the detector (pixel units)
    t = ((d - s) @ n) / ((-s) @ n)
    hit = s + t * (-s) - d
    c0 = hit @ u / (u @ u) + (proj.shape[2] - 1) / 2
    r0 = hit @ v / (v @ v) + (proj.shape[0] - 1) / 2
    pu, pv = np.linalg.norm(u), np.linalg.norm(v)
    rxy = max(abs(cols[0] - c0), abs(cols[-1] - c0)) * pu / mag
    rz = max(abs(rows[0] - r0), abs(rows[-1] - r0)) * pv / mag
    box = np.array([rxy, rxy, rz]) * margin + 1.0
    return np.minimum(box, full_box)


def _shadow_range(profile, frac=0.1, pad=16):
    idx = np.nonzero(profile > frac * profile.max())[0]
    if idx.size == 0:
        return 0, len(profile)
    return max(0, idx[0] - pad), min(len(profile), idx[-1] + 1 + pad)


def load_flexray(path, sample=4, skip=2, profile="cwi-flexray-2019-04-24",
                 threads=8, auto_box=True, crop=True, verbose=True):
    """Return (projections [rows, angles, cols] as -log(I/I0), ConeVecGeometry).

    Memory-friendly: projections are corrected one by one straight into a single
    float32 array. With crop=True the detector is first cropped to the object's
    shadow (found from 24 sample projections), which typically halves the data
    and makes --sample 1 (full resolution) feasible on a laptop."""
    path = os.path.expanduser(path)
    p = parse_scan_settings(path)
    for k, v in PROFILES[profile].items():
        p[k] += v

    scans = _files(path, "scan_")
    if not scans:
        raise FileNotFoundError(f"no scan_*.tif files in {path}")
    n_total = len(scans)
    scans = scans[::skip]
    dark = np.mean([_read_binned(f, sample) for f in _files(path, "di")], axis=0)
    flat = np.mean([_read_binned(f, sample) for f in _files(path, "io")], axis=0)
    gain = 1.0 / np.maximum(flat - dark, 1.0)
    R_full, C_full = dark.shape

    def corrected(fname):
        im = (_read_binned(fname, sample) - dark) * gain
        np.clip(im, 1e-3, None, out=im)
        np.log(im, out=im)
        np.negative(im, out=im)
        return im

    r0, r1, c0, c1 = 0, R_full, 0, C_full
    if crop:
        sample_idx = np.linspace(0, len(scans) - 1, min(24, len(scans))).astype(int)
        with ThreadPoolExecutor(threads) as ex:
            some = np.stack(list(ex.map(corrected, [scans[i] for i in sample_idx])))
        # strongest attenuation seen in each detector row / column over the sampled
        # angles (99th percentile: ignores single noisy pixels) - keeps thin edges
        # such as the zygomatic arches that a mean profile would cut off
        r0, r1 = _shadow_range(np.percentile(some, 99, axis=2).max(axis=0))
        c0, c1 = _shadow_range(np.percentile(some, 99, axis=1).max(axis=0))
        del some

    proj = np.empty((r1 - r0, len(scans), c1 - c0), dtype=np.float32)

    def load_one(i):
        proj[:, i, :] = corrected(scans[i])[r0:r1, c0:c1]

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(load_one, range(len(scans))))

    # projections are equally spaced over [theta_min, theta_max), last one not repeated
    step = (p["theta_max"] - p["theta_min"]) / n_total
    thetas = p["theta_min"] + np.arange(0, n_total, skip)[: proj.shape[1]] * step
    pixel = p["det_pixel"] * sample
    V = cone_vectors(p, thetas, pixel)
    # cropping moves the centre of the (smaller) detector
    dc = (c0 + c1 - 1) / 2.0 - (C_full - 1) / 2.0
    dr = (r0 + r1 - 1) / 2.0 - (R_full - 1) / 2.0
    V[:, 3:6] += dc * V[:, 6:9] + dr * V[:, 9:12]

    rows, cols = proj.shape[0], proj.shape[2]
    vox = p["img_pixel"] * sample
    box = np.array([C_full * vox / 2, C_full * vox / 2, R_full * vox / 2])
    if auto_box:
        box = object_box(proj, V, box)
    rng = np.deg2rad(p["theta_max"] - p["theta_min"])
    geom = ConeVecGeometry(V, rows, cols, angular_range=rng, box=box)
    if verbose:
        mb = proj.nbytes / 1e6
        print(f"[flexray] {p['name'] or os.path.basename(path)}: {proj.shape[1]}/{n_total} "
              f"projections, {R_full}x{C_full} px after {sample}x{sample} binning, "
              f"cropped to {rows}x{cols} ({mb:.0f} MB)")
        print(f"[flexray] SOD={p['src2obj']:.1f} mm  SDD={p['src2obj'] + p['det2obj']:.1f} mm  "
              f"voxel={vox * 1000:.0f} um  profile={profile}")
        print(f"[flexray] view box (half sizes) = {np.round(box, 1)} mm")
    return proj, geom
