"""Projection pre-processing applied on arrival (like SliceRecon's --gaussian / --tilt).

Pure NumPy, so it runs everywhere. Blocks are [rows, k, cols].
"""
import numpy as np


def gaussian_blur(block, sigma):
    """Separable Gaussian low-pass (sigma in detector pixels) on each projection.
    Reduces noise before filtering, at the cost of resolution."""
    if sigma <= 0:
        return block
    out = block.astype(np.float32, copy=True)
    pad = int(np.ceil(3 * sigma))
    for axis in (0, 2):                                    # detector rows, then columns
        n = out.shape[axis]
        widths = [(0, 0)] * 3
        widths[axis] = (pad, pad)
        padded = np.pad(out, widths, mode="edge")
        N = padded.shape[axis]
        f = np.fft.rfftfreq(N)
        H = np.exp(-2.0 * (np.pi * f * sigma) ** 2)        # FT of a unit-area Gaussian
        shape = [1, 1, 1]
        shape[axis] = len(H)
        F = np.fft.rfft(padded, axis=axis) * H.reshape(shape)
        res = np.fft.irfft(F, n=N, axis=axis)
        out = np.take(res, np.arange(pad, pad + n), axis=axis).astype(np.float32)
    return out


def rotate_projections(block, degrees):
    """Rotate every projection by `degrees` about the detector centre (bilinear).
    Use it to correct a detector that is tilted relative to the rotation axis:
    pass the opposite of the measured tilt."""
    if abs(degrees) < 1e-9:
        return block
    R, A, C = block.shape
    t = np.deg2rad(degrees)
    rr, cc = np.meshgrid(np.arange(R) - (R - 1) / 2.0, np.arange(C) - (C - 1) / 2.0,
                         indexing="ij")
    # source position for every output pixel (inverse rotation)
    rs = np.cos(t) * rr + np.sin(t) * cc + (R - 1) / 2.0
    cs = -np.sin(t) * rr + np.cos(t) * cc + (C - 1) / 2.0
    r0 = np.floor(rs).astype(int)
    c0 = np.floor(cs).astype(int)
    fr = (rs - r0)[..., None]
    fc = (cs - c0)[..., None]
    out = np.zeros((R, C, A), dtype=np.float32)
    for dr, wr in ((0, 1 - fr), (1, fr)):
        for dc, wc in ((0, 1 - fc), (1, fc)):
            ri, ci = r0 + dr, c0 + dc
            ok = (ri >= 0) & (ri < R) & (ci >= 0) & (ci < C)
            vals = np.zeros((R, C, A), dtype=np.float32)
            vals[ok] = block[ri[ok], :, ci[ok]]
            out += (wr * wc * vals).astype(np.float32)
    return np.ascontiguousarray(np.transpose(out, (0, 2, 1)))


def apply(block, gaussian=0.0, tilt=0.0):
    if tilt:
        block = rotate_projections(block, -tilt)
    if gaussian:
        block = gaussian_blur(block, gaussian)
    return block
