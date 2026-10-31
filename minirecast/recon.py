"""On-demand slice reconstruction (CPU), in the spirit of RECAST3D/SliceRecon.

Projections are filtered as they arrive (FBP for parallel beam, FDK for cone
beam). Any arbitrarily oriented slice is then reconstructed by back-projecting
only onto that slice's pixels, using all projections received so far. That is
what makes live (streaming) reconstruction and interactive slicing possible.

Two back-projection backends are available:
  * "numba" - parallel JIT kernels (fast, used when numba is installed)
  * "numpy" - vectorised reference implementation (slower, no extra deps)
"""
import threading
import time

import numpy as np

from .geometry import ParallelGeometry, ConeVecGeometry
from . import preprocess

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is optional
    HAVE_NUMBA = False

# numba's default threading layer must not run two parallel kernels at once
# (e.g. a slice request while the preview volume is computed) -> serialise them.
_KERNEL_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def ramp_response(n, tau, window="shepp-logan"):
    """Frequency response of the discrete Ram-Lak filter (Kak & Slaney),
    zero-padded to avoid wrap-around, with an optional apodisation window."""
    N = int(2 ** np.ceil(np.log2(2 * n)))
    idx = np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])
    h = np.zeros(N)
    h[0] = 1.0 / (4.0 * tau * tau)
    odd = idx % 2 == 1
    h[odd] = -1.0 / (np.pi * idx[odd] * tau) ** 2
    H = np.real(np.fft.fft(h)) * tau
    f = np.fft.fftfreq(N)              # cycles / sample, in [-0.5, 0.5)
    if window in (None, "ram-lak", "none"):
        pass
    elif window == "shepp-logan":
        H *= np.sinc(f)
    elif window == "hann":
        H *= 0.5 * (1.0 + np.cos(2.0 * np.pi * f))
    elif window == "cosine":
        H *= np.cos(np.pi * f)
    else:
        raise ValueError(f"unknown filter window {window!r}")
    return H, N


def filter_rows(data, tau, window="shepp-logan"):
    """Ramp-filter along the last axis."""
    n = data.shape[-1]
    H, N = ramp_response(n, tau, window)
    F = np.fft.rfft(data, n=N, axis=-1)
    F *= H[: N // 2 + 1]
    return np.fft.irfft(F, n=N, axis=-1)[..., :n].astype(np.float32)


# --------------------------------------------------------------------------
# Per-angle constants for cone_vec geometries
# --------------------------------------------------------------------------
def cone_constants(vectors):
    """For each angle: src, det, u, v, u/|u|^2, v/|v|^2, unit normal (src->det), SDD, SOD."""
    V = np.asarray(vectors, dtype=np.float64)
    s, d, u, v = V[:, 0:3], V[:, 3:6], V[:, 6:9], V[:, 9:12]
    n = np.cross(u, v)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    sign = np.sign(np.einsum('ij,ij->i', d - s, n))
    n *= sign[:, None]                                   # point from source to detector
    sdd = np.einsum('ij,ij->i', d - s, n)                # source-detector-plane distance
    sod = np.einsum('ij,ij->i', -s, n)                   # source-to-axis (origin) distance
    uu = u / np.einsum('ij,ij->i', u, u)[:, None]
    vv = v / np.einsum('ij,ij->i', v, v)[:, None]
    return dict(s=s, d=d, u=u, v=v, uu=uu, vv=vv, n=n, sdd=sdd, sod=sod)


def fdk_filter_block(block, K, window="shepp-logan"):
    """FDK cosine pre-weighting + ramp filtering for a block of projections.
    block: [rows, k, cols]; K: cone constants of those k projections.
    Returns [k, rows, cols]."""
    R, A, C = block.shape
    cc = np.arange(C) - (C - 1) / 2.0
    rr = np.arange(R) - (R - 1) / 2.0
    out = np.empty((A, R, C), dtype=np.float32)
    for k in range(A):
        s, d, u, v, n = K['s'][k], K['d'][k], K['u'][k], K['v'][k], K['n'][k]
        sdd, sod = K['sdd'][k], K['sod'][k]
        p0 = s + sdd * n                                    # principal point
        off = d - p0
        un, vn = np.linalg.norm(u), np.linalg.norm(v)
        a = (off @ u) / un + cc * un                         # mm along u (cols)
        b = (off @ v) / vn + rr * vn                         # mm along v (rows)
        w = sdd / np.sqrt(sdd ** 2 + a[None, :] ** 2 + b[:, None] ** 2)
        tau_iso = un * sod / sdd                             # pixel pitch at the axis
        out[k] = filter_rows(block[:, k, :] * w, tau_iso, window)
    return out


# --------------------------------------------------------------------------
# Back-projection kernels
# --------------------------------------------------------------------------
def _bilinear(img, r, c):
    """Bilinear sample of a 2D image at float (r, c); zero outside."""
    R, C = img.shape
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    fr = r - r0
    fc = c - c0
    out = np.zeros(r.shape, dtype=np.float64)
    for dr, wr in ((0, 1.0 - fr), (1, fr)):
        for dc, wc in ((0, 1.0 - fc), (1, fc)):
            rr = r0 + dr
            cc = c0 + dc
            ok = (rr >= 0) & (rr < R) & (cc >= 0) & (cc < C)
            vals = np.zeros(r.shape)
            vals[ok] = img[rr[ok], cc[ok]]
            out += wr * wc * vals
    return out


def bp_parallel_numpy(q, cos_t, sin_t, pu, pv, pts, scale, col_shift):
    A, R, C = q.shape
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    row = z / pv + (R - 1) / 2.0
    acc = np.zeros(len(pts))
    for k in range(A):
        col = (-x * sin_t[k] + y * cos_t[k]) / pu + (C - 1) / 2.0 + col_shift
        acc += _bilinear(q[k], row, col)
    return acc * scale


def bp_cone_numpy(q, S, D, UU, VV, N, SDD, SOD, pts, scale):
    A, R, C = q.shape
    acc = np.zeros(len(pts))
    for k in range(A):
        rel = pts - S[k]
        L = rel @ N[k]
        t = SDD[k] / L
        X = S[k] + t[:, None] * rel - D[k]
        col = X @ UU[k] + (C - 1) / 2.0
        row = X @ VV[k] + (R - 1) / 2.0
        w = (SOD[k] / L) ** 2
        acc += w * _bilinear(q[k], row, col)
    return acc * scale


if HAVE_NUMBA:
    @njit(cache=True, fastmath=True, inline='always')
    def _bil_nb(img, k, r, c, R, C):
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))
        fr = r - r0
        fc = c - c0
        v = 0.0
        if 0 <= r0 < R:
            if 0 <= c0 < C:
                v += (1.0 - fr) * (1.0 - fc) * img[k, r0, c0]
            if 0 <= c0 + 1 < C:
                v += (1.0 - fr) * fc * img[k, r0, c0 + 1]
        if 0 <= r0 + 1 < R:
            if 0 <= c0 < C:
                v += fr * (1.0 - fc) * img[k, r0 + 1, c0]
            if 0 <= c0 + 1 < C:
                v += fr * fc * img[k, r0 + 1, c0 + 1]
        return v

    @njit(parallel=True, cache=True, fastmath=True)
    def bp_parallel_numba(q, cos_t, sin_t, pu, pv, pts, scale, col_shift):
        A, R, C = q.shape
        M = pts.shape[0]
        out = np.zeros(M)
        cr = (R - 1) / 2.0
        cc = (C - 1) / 2.0 + col_shift
        for i in prange(M):
            x = pts[i, 0]
            y = pts[i, 1]
            row = pts[i, 2] / pv + cr
            acc = 0.0
            for k in range(A):
                col = (-x * sin_t[k] + y * cos_t[k]) / pu + cc
                acc += _bil_nb(q, k, row, col, R, C)
            out[i] = acc * scale
        return out

    @njit(parallel=True, cache=True, fastmath=True)
    def bp_cone_numba(q, S, D, UU, VV, N, SDD, SOD, pts, scale):
        A, R, C = q.shape
        M = pts.shape[0]
        out = np.zeros(M)
        cr = (R - 1) / 2.0
        cc = (C - 1) / 2.0
        for i in prange(M):
            px = pts[i, 0]
            py = pts[i, 1]
            pz = pts[i, 2]
            acc = 0.0
            for k in range(A):
                rx = px - S[k, 0]
                ry = py - S[k, 1]
                rz = pz - S[k, 2]
                L = rx * N[k, 0] + ry * N[k, 1] + rz * N[k, 2]
                t = SDD[k] / L
                X0 = S[k, 0] + t * rx - D[k, 0]
                X1 = S[k, 1] + t * ry - D[k, 1]
                X2 = S[k, 2] + t * rz - D[k, 2]
                col = X0 * UU[k, 0] + X1 * UU[k, 1] + X2 * UU[k, 2] + cc
                row = X0 * VV[k, 0] + X1 * VV[k, 1] + X2 * VV[k, 2] + cr
                w = SOD[k] / L
                acc += w * w * _bil_nb(q, k, row, col, R, C)
            out[i] = acc * scale
        return out


# --------------------------------------------------------------------------
# Plane helpers (shared by local and remote viewers)
# --------------------------------------------------------------------------
def plane_basis(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    if abs(n @ z) < 0.9:
        e1 = np.cross(z, n)
    else:
        e1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return n, e1, e2


def plane_points(origin, normal, size, n):
    """n x n grid of points (mm) on a square plane of side `size`."""
    _, e1, e2 = plane_basis(normal)
    t = (np.arange(n) - (n - 1) / 2.0) * (size / n)
    a, b = np.meshgrid(t, t, indexing='xy')
    pts = np.asarray(origin, float) + a[..., None] * e1 + b[..., None] * e2
    return pts  # (n, n, 3); rows follow e2, columns follow e1


def plane_size_for_box(box, normal):
    """Side length that covers the volume box's cross-section in this plane."""
    _, e1, e2 = plane_basis(normal)
    return 2.0 * float(max(np.abs(e1) @ box, np.abs(e2) @ box))


def slice_points(box, origin, normal, n=256, size=None, zoom=1.0, center=None):
    """Points of an n x n slice through the plane (origin, normal), centred on the
    point of the plane closest to `center` (default: box centre), `zoom` times
    smaller than the box cross-section. Returns (pts (n,n,3), mask inside box)."""
    n_hat = np.asarray(normal, float) / np.linalg.norm(normal)
    o = np.asarray(origin, float)
    c = np.zeros(3) if center is None else np.asarray(center, float)
    o = c - ((c - o) @ n_hat) * n_hat
    if size is None:
        size = plane_size_for_box(box, normal) / max(float(zoom), 1e-6)
    pts = plane_points(o, normal, size, n)
    mask = np.all(np.abs(pts) <= box, axis=-1)
    return pts, mask


def planes_intersection(normals, origins, fallback=(0.0, 0.0, 0.0)):
    """Point where (up to) three planes cross, least-squares; fallback if degenerate."""
    N = np.asarray(normals, float)
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    b = np.einsum('ij,ij->i', N, np.asarray(origins, float))
    if np.linalg.matrix_rank(N, tol=1e-3) < 3:
        return np.asarray(fallback, float)
    return np.linalg.lstsq(N, b, rcond=None)[0]


def volume_grid(box, maxdim):
    """Regular grid covering the box with at most `maxdim` voxels along the longest side."""
    h = 2.0 * float(np.max(box)) / maxdim
    dims = np.maximum(np.ceil(2.0 * np.asarray(box) / h).astype(int), 2)
    origin = -np.asarray(box) + h / 2.0
    return dims, origin, h


# --------------------------------------------------------------------------
# Public reconstructor (also the local "service" used by the viewer)
# --------------------------------------------------------------------------
class SliceReconstructor:
    """Filter projections as they arrive, then reconstruct any slice on demand.

    proj : optional [rows, angles, cols] array with ALL projections (batch mode).
           Omit it and call add_projections() to stream data in (live mode).
    gaussian : sigma (detector pixels) of a Gaussian low-pass on each projection.
    tilt : detector tilt correction in degrees (rotates each projection).
    """

    def __init__(self, proj=None, geom=None, window="shepp-logan", backend="auto",
                 gaussian=0.0, tilt=0.0, verbose=True):
        if geom is None:
            raise ValueError("geometry is required")
        if backend == "auto":
            backend = "numba" if HAVE_NUMBA else "numpy"
        if backend == "numba" and not HAVE_NUMBA:
            raise RuntimeError("numba backend requested but numba is not installed")
        self.backend = backend
        self.geom = geom
        self.window = window
        self.gaussian = float(gaussian)
        self.tilt = float(tilt)
        self.verbose = verbose
        self.box = np.asarray(geom.volume_box(), dtype=float)
        self.n_total = geom.n_angles
        self.cor_shift = 0.0
        self.version = 0
        self._lock = threading.RLock()
        self._order = np.zeros(self.n_total, dtype=np.int64)
        self._received = np.zeros(self.n_total, dtype=bool)
        self.n = 0
        self._q = np.zeros((self.n_total, geom.rows, geom.cols), dtype=np.float32)

        if isinstance(geom, ParallelGeometry):
            self.kind = "parallel"
            self._cos_all = np.cos(geom.angles)
            self._sin_all = np.sin(geom.angles)
            self._active = dict(cos=self._cos_all[:0], sin=self._sin_all[:0])
        elif isinstance(geom, ConeVecGeometry):
            self.kind = "cone"
            self._K_all = cone_constants(geom.vectors)
            self._active = {k: v[:0] for k, v in self._K_all.items()}
            if geom.angular_range < 2 * np.pi - 1e-3 and verbose:
                print("note: short scan (<360 deg) - no Parker weights, expect some artefacts")
        else:
            raise TypeError(type(geom))

        if proj is not None:
            t0 = time.perf_counter()
            for i in range(0, self.n_total, 64):        # in chunks: low peak memory
                self.add_projections(np.arange(i, min(i + 64, self.n_total)),
                                     proj[:, i:i + 64, :])
            if verbose:
                print(f"[minirecast] filtered {self.n} projections ({geom.rows}x{geom.cols}) "
                      f"in {time.perf_counter() - t0:.2f}s, backend={self.backend}")

    # ----------------------------------------------------------- ingest
    def add_projections(self, indices, block):
        """Add projections (geometry indices `indices`, data [rows, k, cols])."""
        indices = np.asarray(indices, dtype=np.int64).ravel()
        block = np.asarray(block, dtype=np.float32)
        if block.shape[1] != len(indices):
            raise ValueError("block has %d projections but %d indices"
                             % (block.shape[1], len(indices)))
        keep = ~self._received[indices]
        indices, block = indices[keep], block[:, keep, :]
        if len(indices) == 0:
            return self.n
        block = preprocess.apply(block, gaussian=self.gaussian, tilt=self.tilt)
        if self.kind == "parallel":
            filt = np.transpose(filter_rows(block, self.geom.pixel_u, self.window), (1, 0, 2))
        else:
            K = {k: v[indices] for k, v in self._K_all.items()}
            filt = fdk_filter_block(block, K, self.window)
        with self._lock:
            n0, n1 = self.n, self.n + len(indices)
            self._q[n0:n1] = filt
            self._order[n0:n1] = indices
            self._received[indices] = True
            order = self._order[:n1]
            if self.kind == "parallel":
                active = dict(cos=self._cos_all[order], sin=self._sin_all[order])
            else:
                active = {k: np.ascontiguousarray(v[order]) for k, v in self._K_all.items()}
            self._active = active
            self.n = n1
            self.version += 1
        return self.n

    # ----------------------------------------------------------- info
    @property
    def native_pixel(self):
        """Detector pixel size at the rotation axis (mm): the finest real detail.
        Zooming to smaller slice pixels than this only interpolates."""
        if self.kind == "parallel":
            return float(self.geom.pixel_u)
        K = self._K_all
        return float(np.linalg.norm(K['u'][0]) * K['sod'][0] / K['sdd'][0])

    def info(self):
        return dict(kind=self.kind, n=int(self.n), n_total=int(self.n_total),
                    version=int(self.version), cor_shift=float(self.cor_shift),
                    box=[float(b) for b in self.box], native_pixel=self.native_pixel,
                    backend=self.backend)

    def plane_size(self, normal):
        return plane_size_for_box(self.box, normal)

    def inside_box(self, pts):
        return np.all(np.abs(pts) <= self.box, axis=-1)

    # ----------------------------------------------------------- reconstruction
    def _snapshot(self):
        with self._lock:
            return self.n, self._q[:self.n], self._active, self.cor_shift

    def reconstruct_points(self, pts, cor_shift=None):
        pts = np.ascontiguousarray(np.asarray(pts, dtype=np.float64).reshape(-1, 3))
        n, q, act, shift = self._snapshot()
        if cor_shift is not None:                 # trial value (COR search), not stored
            shift = float(cor_shift)
        if n == 0 or len(pts) == 0:
            return np.zeros(len(pts))
        scale = np.pi / n            # uniform angles over 180 (parallel) or 360 deg
        with _KERNEL_LOCK:
            if self.kind == "parallel":
                pu, pv = self.geom.pixel_u, self.geom.pixel_v
                fn = bp_parallel_numba if self.backend == "numba" else bp_parallel_numpy
                return fn(q, act['cos'], act['sin'], pu, pv, pts, scale, float(shift))
            # shift > 0: rotation axis appears `shift` columns right of where the
            # geometry says (same meaning as for parallel beam) = detector moved left
            D = act['d'] - shift * act['u'] if shift else act['d']
            D = np.ascontiguousarray(D)
            fn = bp_cone_numba if self.backend == "numba" else bp_cone_numpy
            return fn(q, act['s'], D, act['uu'], act['vv'], act['n'], act['sdd'],
                      act['sod'], pts, scale)

    def reconstruct_plane(self, origin, normal, n=256, size=None, zoom=1.0, center=None,
                          cor_shift=None):
        """Reconstruct an n x n slice of the plane through `origin` with the given normal.

        Zoom works like a microscope objective: the slice still has n x n pixels,
        but covers a `zoom` times smaller area around `center` (projected onto the
        plane), so every zoom step reconstructs real extra detail.
        Returns (image, points); pixels outside the volume box are 0."""
        pts, mask = slice_points(self.box, origin, normal, n, size, zoom, center)
        flat = pts.reshape(-1, 3)
        img = np.zeros(len(flat))
        m = mask.ravel()
        if m.any():
            img[m] = self.reconstruct_points(flat[m], cor_shift)
        return img.reshape(n, n), pts

    def reconstruct_volume(self, voxel=None, maxdim=None, chunk=262144, progress=None):
        """Full 3D reconstruction of the volume box (FDK/FBP on a regular grid).

        voxel  : voxel size in mm (default: the detector pixel at the axis)
        maxdim : alternatively, number of voxels along the longest side
        Computed in z-slabs; `progress(fraction)` is called after each slab.
        Returns (volume [nx, ny, nz] float32, origin (mm), voxel size (mm))."""
        if maxdim is not None:
            dims, origin, h = volume_grid(self.box, maxdim)
        else:
            h = float(voxel or self.native_pixel)
            dims = np.maximum(np.ceil(2.0 * self.box / h).astype(int), 2)
            origin = -self.box + h / 2.0
        nx, ny, nz = (int(d) for d in dims)
        x = origin[0] + h * np.arange(nx)
        y = origin[1] + h * np.arange(ny)
        z = origin[2] + h * np.arange(nz)
        X, Y = np.meshgrid(x, y, indexing="ij")
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        vol = np.empty((nx, ny, nz), dtype=np.float32)
        per_slab = max(1, chunk // (nx * ny))
        for k0 in range(0, nz, per_slab):       # slabs: slice requests can interleave
            ks = np.arange(k0, min(nz, k0 + per_slab))
            pts = np.concatenate([np.column_stack([xy, np.full(len(xy), z[k])]) for k in ks])
            vals = self.reconstruct_points(pts)
            vol[:, :, ks] = vals.reshape(len(ks), nx, ny).transpose(1, 2, 0)
            if progress is not None:
                progress(ks[-1] / max(nz - 1, 1))
        return vol, np.asarray(origin, float), h

    def preview_volume(self, maxdim=64, chunk=65536):
        """Low-resolution reconstruction of the whole box (like SliceRecon's
        --preview-size). Returns (volume [nx, ny, nz] float32, origin, spacing)."""
        return self.reconstruct_volume(maxdim=maxdim, chunk=chunk)

    def warmup(self):
        """Trigger numba compilation (falls back to numpy if it fails)."""
        if self.n == 0:
            return
        try:
            self.reconstruct_points(np.zeros((4, 3)))
        except Exception as e:  # pragma: no cover
            if self.backend != "numba":
                raise
            print(f"[minirecast] numba failed ({e!r}); using numpy backend")
            self.backend = "numpy"

    # ----------------------------------------------------------- rotation centre
    def set_cor_shift(self, px):
        """Shift the rotation centre by `px` detector pixels: the axis is taken to
        project `px` columns to the right of the nominal position. Back-projection
        only, no re-filtering needed, so it can be changed live."""
        with self._lock:
            self.cor_shift = float(px)
            self.version += 1

    def estimate_cor(self, shifts=None, n=192, z=0.0, verbose=True):
        """Pick the COR shift whose central axial slice has the lowest histogram
        entropy (the same criterion as tomopy.find_center): a wrong centre smears
        every edge into double arcs, which spreads the grey-value histogram.
        Works for parallel and cone beam. Default search: +-3 px in 0.25 px steps
        (calibrated scanners); pass e.g. np.arange(-30, 30.1, 1) for raw beamline data."""
        if shifts is None:
            shifts = np.arange(-3.0, 3.01, 0.25)
        scores = []
        for sft in shifts:
            img, _ = self.reconstruct_plane((0.0, 0.0, z), (0.0, 0.0, 1.0), n=n,
                                            cor_shift=float(sft))
            hist, _ = np.histogram(img[img != 0], bins=128)
            prob = hist[hist > 0] / max(hist.sum(), 1)
            scores.append(float(np.sum(prob * np.log(prob))))      # = -entropy
        best = float(shifts[int(np.argmax(scores))])
        self.set_cor_shift(best)
        if verbose:
            print("[minirecast] COR search: " + " ".join(
                f"{s:+.2f}:{v:.3g}" for s, v in zip(shifts, scores)))
            print(f"[minirecast] best COR shift {best:+.2f} px")
        return best, np.array(scores)
