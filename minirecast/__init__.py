"""minirecast - a tiny CPU-only, RECAST3D-style on-demand slice reconstructor.

Runs on macOS (Apple Silicon) without CUDA. See README.md.
"""
from .geometry import ParallelGeometry, ConeVecGeometry, circular_cone_vectors
from .recon import SliceReconstructor, HAVE_NUMBA

__version__ = "0.1.0"
