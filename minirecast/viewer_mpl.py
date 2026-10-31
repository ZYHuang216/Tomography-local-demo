"""Matplotlib fallback viewer and a RECAST3D-like 3D snapshot renderer."""
import numpy as np


def _clim(imgs):
    vals = np.concatenate([i.ravel() for i in imgs])
    vals = vals[vals != 0]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [1.0, 99.7])
    return max(lo, 0.0) if hi > 0 else lo, hi


def snapshot_3d(recon, filename, res=200, origin=(0.0, 0.0, 0.0), cmap="bone", title=None,
                zoom=1.0):
    """Render three orthogonal on-demand slices in a 3D box (like RECAST3D) to a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    box = recon.box
    planes = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
    slices = [recon.reconstruct_plane(origin, n, n=res, zoom=zoom, center=origin) for n in planes]
    lo, hi = _clim([s[0] for s in slices])
    norm = colors.Normalize(lo, hi)
    cmap_ = matplotlib.colormaps[cmap]

    fig = plt.figure(figsize=(7, 7), facecolor="#1b1b1b")
    ax = fig.add_subplot(projection="3d", facecolor="#1b1b1b")
    for img, pts in slices:
        inside = recon.inside_box(pts)
        rgba = cmap_(norm(img))
        rgba[~inside, 3] = 0.0
        # clip plane to the volume box for display
        P = np.clip(pts, -box, box)
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], facecolors=rgba, rstride=1, cstride=1,
                        shade=False, linewidth=0, antialiased=False)
    # box outline
    bx, by, bz = box
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            ax.plot([-bx, bx], [s1 * by] * 2, [s2 * bz] * 2, color="#bbbbbb", lw=0.8)
            ax.plot([s1 * bx] * 2, [-by, by], [s2 * bz] * 2, color="#bbbbbb", lw=0.8)
            ax.plot([s1 * bx] * 2, [s2 * by] * 2, [-bz, bz], color="#bbbbbb", lw=0.8)
    ax.set_box_aspect(box)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-55)
    if title:
        ax.set_title(title, color="#dddddd")
    import os
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    fig.savefig(filename, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return filename


def run_viewer_mpl(recon, res=192, cmap="bone"):
    """Three orthogonal slices + one oblique slice with sliders (2D fallback)."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    box = recon.box
    fig, axs = plt.subplots(2, 2, figsize=(9, 9.5))
    plt.subplots_adjust(bottom=0.24)
    names = ["axial (z)", "sagittal (x)", "coronal (y)", "oblique"]
    ims = []
    for ax, name in zip(axs.ravel(), names):
        ims.append(ax.imshow(np.zeros((res, res)), cmap=cmap, origin="lower"))
        ax.set_title(name)
        ax.set_axis_off()

    sl = {}
    specs = [("x", -box[0], box[0]), ("y", -box[1], box[1]), ("z", -box[2], box[2]),
             ("tilt", 0.0, 90.0), ("azimuth", 0.0, 180.0), ("zoom", 1.0, 16.0)]
    for j, (name, lo, hi) in enumerate(specs):
        axs_ = fig.add_axes([0.15, 0.17 - j * 0.026, 0.7, 0.018])
        init = (lo + hi) / 2 if name in ("x", "y", "z") else (1.0 if name == "zoom" else 30.0)
        sl[name] = Slider(axs_, name, lo, hi, valinit=init)

    def update(_=None):
        x, y, z = sl["x"].val, sl["y"].val, sl["z"].val
        t, a = np.deg2rad(sl["tilt"].val), np.deg2rad(sl["azimuth"].val)
        oblique = (np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t))
        reqs = [((0, 0, z), (0, 0, 1)), ((x, 0, 0), (1, 0, 0)), ((0, y, 0), (0, 1, 0)),
                ((x, y, z), oblique)]
        zoom, c = sl["zoom"].val, (x, y, z)    # zoom around the planes' crossing point
        imgs = [recon.reconstruct_plane(o, n, n=res, zoom=zoom, center=c)[0] for o, n in reqs]
        fig.suptitle(f"zoom {zoom:.1f}x  ~{recon.plane_size((0, 0, 1)) / zoom / res * 1000:.0f} um/px "
                     f"(axial)  centre ({x:+.1f}, {y:+.1f}, {z:+.1f}) mm")
        lo, hi = _clim(imgs)
        for im, img in zip(ims, imgs):
            im.set_data(img)
            im.set_clim(lo, hi)
        fig.canvas.draw_idle()

    for s in sl.values():
        s.on_changed(update)
    update()
    plt.show()


def save_slices(recon, outdir, res=512, origin=(0.0, 0.0, 0.0), cmap="gray", zoom=1.0):
    """Reconstruct the three central orthogonal slices and save them as PNG + .npy."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    named = {"axial": (0, 0, 1), "sagittal": (1, 0, 0), "coronal": (0, 1, 0)}
    imgs = {k: recon.reconstruct_plane(origin, n, n=res, zoom=zoom, center=origin)[0]
            for k, n in named.items()}
    lo, hi = _clim(list(imgs.values()))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="black")
    for ax, (k, img) in zip(axs, imgs.items()):
        size = recon.plane_size(named[k]) / zoom
        np.save(os.path.join(outdir, f"{k}.npy"), img.astype(np.float32))
        plt.imsave(os.path.join(outdir, f"{k}.png"), img, cmap=cmap, vmin=lo, vmax=hi,
                   origin="lower")
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, origin="lower",
                  extent=[-size / 2, size / 2, -size / 2, size / 2])   # mm from centre
        ax.set_title(f"{k}  ({res}x{res}, {size / res * 1000:.0f} um/px, zoom {zoom:g}x)", color="w")
        ax.tick_params(colors="w")
        ax.set_xlabel("mm from zoom centre", color="w")
    finest = min(recon.plane_size(n) for n in named.values()) / zoom / res
    if finest < 0.7 * recon.native_pixel:
        print(f"[minirecast] note: {finest * 1000:.0f} um/px is finer than the data "
              f"({recon.native_pixel * 1000:.0f} um): zooming further only interpolates")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "orthogonal_slices.png"), dpi=90, facecolor="black")
    plt.close(fig)
    return outdir
