"""Analytic 3D Shepp-Logan phantom.

Each ellipsoid has an exact line integral, so the simulated projections are
free of discretisation error.  We use them both for the demo data and to check
the reconstruction against the true phantom.
"""
import numpy as np

# Modified 3D Shepp-Logan (as in phantom3d.m / Toft), rotation about z only.
# columns: value, a, b, c, x0, y0, z0, phi[deg]
MODIFIED_SHEPP_LOGAN = np.array([
    [ 1.0, .6900, .920, .810,  0.00,  0.0000,  0.00,   0],
    [-0.8, .6624, .874, .780,  0.00, -0.0184,  0.00,   0],
    [-0.2, .1100, .310, .220,  0.22,  0.0000,  0.00, -18],
    [-0.2, .1600, .410, .280, -0.22,  0.0000,  0.00,  18],
    [ 0.1, .2100, .250, .410,  0.00,  0.3500, -0.15,   0],
    [ 0.1, .0460, .046, .050,  0.00,  0.1000,  0.25,   0],
    [ 0.1, .0460, .046, .050,  0.00, -0.1000,  0.25,   0],
    [ 0.1, .0460, .023, .050, -0.08, -0.6050,  0.00,   0],
    [ 0.1, .0230, .023, .020,  0.00, -0.6060,  0.00,   0],
    [ 0.1, .0230, .046, .020,  0.06, -0.6050,  0.00,   0],
])


def _ellipsoid_frames(table, scale):
    """Return (values, centers, inverse transforms M) with M @ (x - c) -> unit sphere."""
    vals, centers, mats = [], [], []
    for v, a, b, c, x0, y0, z0, phi in table:
        p = np.deg2rad(phi)
        R = np.array([[np.cos(p), -np.sin(p), 0.0],
                      [np.sin(p),  np.cos(p), 0.0],
                      [0.0, 0.0, 1.0]])
        L_inv = np.diag([1.0 / (a * scale), 1.0 / (b * scale), 1.0 / (c * scale)])
        vals.append(v)
        centers.append(np.array([x0, y0, z0]) * scale)
        mats.append(L_inv @ R.T)
    return np.array(vals), np.array(centers), np.array(mats)


class EllipsoidPhantom:
    """Sum of constant-density ellipsoids, coordinates in mm."""

    def __init__(self, scale=1.0, table=MODIFIED_SHEPP_LOGAN):
        self.scale = float(scale)
        self.vals, self.centers, self.mats = _ellipsoid_frames(table, self.scale)

    def evaluate(self, points):
        """Phantom value at points (..., 3)."""
        pts = np.asarray(points, dtype=np.float64)
        out = np.zeros(pts.shape[:-1])
        for v, c, M in zip(self.vals, self.centers, self.mats):
            y = (pts - c) @ M.T
            out += v * (np.einsum('...i,...i->...', y, y) <= 1.0)
        return out

    def line_integrals(self, starts, dirs):
        """Exact integrals along lines start + t*dir (dir unit vectors), shape (..., 3)."""
        starts = np.asarray(starts, dtype=np.float64)
        dirs = np.asarray(dirs, dtype=np.float64)
        out = np.zeros(np.broadcast_shapes(starts.shape, dirs.shape)[:-1])
        for v, c, M in zip(self.vals, self.centers, self.mats):
            p = (starts - c) @ M.T
            q = dirs @ M.T
            qq = np.einsum('...i,...i->...', q, q)
            pq = np.einsum('...i,...i->...', p, q)
            pp = np.einsum('...i,...i->...', p, p)
            disc = pq * pq - qq * (pp - 1.0)
            chord = 2.0 * np.sqrt(np.clip(disc, 0.0, None)) / qq
            out += v * chord
        return out


def project_parallel(phantom, geom):
    """Analytic parallel-beam projections, returned as [rows, angles, cols]."""
    u, v = geom.detector_coords()
    data = np.empty((geom.rows, len(geom.angles), geom.cols), dtype=np.float32)
    for k, th in enumerate(geom.angles):
        d = np.array([np.cos(th), np.sin(th), 0.0])
        eu = np.array([-np.sin(th), np.cos(th), 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        starts = u[None, :, None] * eu + v[:, None, None] * ez - 10.0 * phantom.scale * d
        data[:, k, :] = phantom.line_integrals(starts, d)
    return data


def project_cone(phantom, geom):
    """Analytic cone-beam projections for a cone_vec geometry, [rows, angles, cols]."""
    R, C = geom.rows, geom.cols
    cc = np.arange(C) - (C - 1) / 2.0
    rr = np.arange(R) - (R - 1) / 2.0
    data = np.empty((R, geom.n_angles, C), dtype=np.float32)
    for k, vec in enumerate(geom.vectors):
        s, d, u, v = vec[0:3], vec[3:6], vec[6:9], vec[9:12]
        pix = d + cc[None, :, None] * u + rr[:, None, None] * v
        dirs = pix - s
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        data[:, k, :] = phantom.line_integrals(s[None, None, :], dirs)
    return data
