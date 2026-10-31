"""Scan geometries.

Projection data are always stored ASTRA-style as [rows, angles, cols]. Cone-beam
geometries use ASTRA 'cone_vec' vectors (source, detector centre, u, v in mm),
the same format RECAST3D/SliceRecon receive from the FleX-ray adapter.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ParallelGeometry:
    angles: np.ndarray          # radians
    rows: int
    cols: int
    pixel_u: float = 1.0        # mm
    pixel_v: float = 1.0        # mm
    box: np.ndarray = None      # half-sizes (x, y, z) in mm; default: full field of view

    def detector_coords(self):
        u = (np.arange(self.cols) - (self.cols - 1) / 2.0) * self.pixel_u
        v = (np.arange(self.rows) - (self.rows - 1) / 2.0) * self.pixel_v
        return u, v

    @property
    def n_angles(self):
        return len(self.angles)

    def volume_box(self):
        """Half-sizes (x, y, z) of the reconstructable box in mm."""
        if self.box is not None:
            return np.asarray(self.box, dtype=float)
        h = self.cols * self.pixel_u / 2.0
        return np.array([h, h, self.rows * self.pixel_v / 2.0])


@dataclass
class ConeVecGeometry:
    vectors: np.ndarray         # (n_angles, 12): src(3), det(3), u(3), v(3)
    rows: int
    cols: int
    angular_range: float = 2 * np.pi   # radians covered by the scan
    box: np.ndarray = None      # half-sizes (x, y, z) in mm

    @property
    def n_angles(self):
        return self.vectors.shape[0]

    def volume_box(self):
        if self.box is not None:
            return np.asarray(self.box, dtype=float)
        # field of view at the rotation axis
        s, d, u, v = np.split(self.vectors[0], 4)
        mag = np.linalg.norm(d - s) / np.linalg.norm(s)
        hu = self.cols * np.linalg.norm(u) / mag / 2.0
        hv = self.rows * np.linalg.norm(v) / mag / 2.0
        return np.array([hu, hu, hv])


def circular_cone_vectors(n_angles, sod, sdd, rows, cols, pixel, angular_range=2 * np.pi,
                          det_shift_u=0.0):
    """Ideal circular cone-beam orbit around the z axis (for the synthetic test)."""
    angles = np.arange(n_angles) * angular_range / n_angles
    vecs = np.zeros((n_angles, 12))
    for k, a in enumerate(angles):
        ray = np.array([np.cos(a), np.sin(a), 0.0])        # source -> detector
        tan = np.array([-np.sin(a), np.cos(a), 0.0])
        vecs[k, 0:3] = -sod * ray
        vecs[k, 3:6] = (sdd - sod) * ray + det_shift_u * pixel * tan
        vecs[k, 6:9] = pixel * tan
        vecs[k, 9:12] = np.array([0.0, 0.0, pixel])
    return ConeVecGeometry(vecs, rows, cols, angular_range)
