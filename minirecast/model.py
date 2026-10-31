"""3D model: full-volume reconstruction -> bone surface mesh (+ export / 3D viewer).

    volume  (FDK on a voxel grid)  ->  threshold (Otsu, automatic)
            ->  marching cubes surface  ->  keep main pieces  ->  smooth
            ->  STL / PLY / OBJ for 3D viewers or 3D printing; TIFF stack for Fiji

Surface extraction uses VTK (pyvista, installed by setup.sh); scikit-image is
used as a fallback if pyvista is missing.
"""
import os
import struct
import time

import numpy as np


# ----------------------------------------------------------------- threshold
def otsu_threshold(vol, bins=512):
    """Otsu's threshold on the non-zero voxels (separates air from bone)."""
    v = vol[vol != 0]
    lo, hi = np.percentile(v, [0.1, 99.95])
    hist, edges = np.histogram(np.clip(v, lo, hi), bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


# ----------------------------------------------------------------- surface
def image_data(vol, origin, spacing):
    import pyvista as pv
    grid = pv.ImageData(dimensions=vol.shape, spacing=(spacing,) * 3, origin=tuple(origin))
    grid["density"] = vol.ravel(order="F")
    return grid


def extract_surface(vol, origin, spacing, iso, smooth=15, min_piece=0.002, decimate=0.0):
    """Iso-surface of the volume. Returns a pyvista.PolyData if pyvista is available,
    else a dict(points, faces) from scikit-image.

    smooth    : Taubin smoothing iterations (keeps volume; 0 = off)
    min_piece : drop disconnected pieces smaller than this fraction of the surface
                (noise specks); 0 keeps everything
    decimate  : fraction of triangles to remove (0..0.95) for smaller files
    """
    try:
        import pyvista as pv  # noqa: F401
    except ImportError:
        return _extract_surface_skimage(vol, origin, spacing, iso)
    grid = image_data(vol, origin, spacing)
    surf = grid.contour([iso], scalars="density", method="flying_edges")
    if surf.n_points == 0:
        raise ValueError(f"no surface at iso={iso:g}; try a lower --iso")
    if min_piece > 0:
        conn = surf.connectivity("all")
        # RegionId exists as point AND cell data; use the cell (triangle) labels
        ids = np.asarray(conn.cell_data["RegionId"]).astype(np.int64)
        counts = np.bincount(ids)
        keep = np.nonzero(counts >= min_piece * len(ids))[0]
        surf = conn.extract_cells(np.isin(ids, keep)).extract_surface()
    surf = surf.triangulate()
    if decimate > 0:
        surf = surf.decimate(min(float(decimate), 0.95))
    if smooth > 0:
        surf = surf.smooth_taubin(n_iter=int(smooth), pass_band=0.05)
    return surf.compute_normals(auto_orient_normals=True)


def _extract_surface_skimage(vol, origin, spacing, iso):
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(vol, level=iso, spacing=(spacing,) * 3)
    return {"points": verts + np.asarray(origin), "faces": faces}


def mesh_arrays(mesh):
    """(points [n,3], triangles [m,3]) from a pyvista mesh or the skimage dict."""
    if isinstance(mesh, dict):
        return mesh["points"], mesh["faces"]
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return np.asarray(mesh.points), faces


def write_stl(path, points, faces):
    """Binary STL (units: mm)."""
    tri = points[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    data = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data["n"], data["v"] = n, tri
    with open(path, "wb") as f:
        f.write(b"minirecast surface (mm)".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        f.write(data.tobytes())


def save_mesh(mesh, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not isinstance(mesh, dict):
        mesh.save(path)                              # .stl .ply .obj .vtp ...
    elif path.lower().endswith(".stl"):
        write_stl(path, *mesh_arrays(mesh))
    else:
        raise SystemExit("without pyvista only .stl export is available")
    return path


def save_volume(vol, origin, spacing, path):
    """.tif (Fiji/ImageJ stack, z first) | .npz | .vti (ParaView)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        stack = np.ascontiguousarray(vol.transpose(2, 1, 0))           # [z, y, x]
        tifffile.imwrite(path, stack, imagej=True,
                         resolution=(1.0 / spacing, 1.0 / spacing),
                         metadata={"spacing": spacing, "unit": "mm", "axes": "ZYX"})
    elif ext == ".npz":
        np.savez_compressed(path, volume=vol, origin=origin, spacing=spacing)
    elif ext in (".vti", ".vtk"):
        image_data(vol, origin, spacing).save(path)
    else:
        raise SystemExit(f"unknown volume format {ext} (use .tif, .npz or .vti)")
    return path


# ----------------------------------------------------------------- pictures
def snapshot(mesh, path, title=None, elev=15, azim=-60):
    """PNG of the surface. Uses pyvista off-screen if possible, else matplotlib."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not isinstance(mesh, dict):
        try:
            import pyvista as pv
            pl = pv.Plotter(off_screen=True, window_size=(1000, 1100))
            pl.set_background("#1b1b1b")
            pl.add_mesh(mesh, color="#efe6d2", smooth_shading=True, specular=0.3)
            pl.camera_position = "iso"
            pl.camera.elevation, pl.camera.azimuth = elev, azim
            if title:
                pl.add_text(title, font_size=10, color="#dddddd")
            pl.screenshot(path)
            pl.close()
            return path
        except Exception:
            pass
    return _snapshot_mpl(*mesh_arrays(mesh), path, title, elev, azim)


def _snapshot_mpl(points, faces, path, title, elev, azim, max_faces=250000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(faces) > max_faces:
        faces = faces[np.random.default_rng(0).choice(len(faces), max_faces, replace=False)]
    tri = points[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    shade = 0.25 + 0.75 * np.abs(n @ light)
    base = np.array([0.94, 0.90, 0.82])
    fig = plt.figure(figsize=(8, 9), facecolor="#1b1b1b")
    ax = fig.add_subplot(projection="3d", facecolor="#1b1b1b")
    ax.add_collection3d(Poly3DCollection(tri, facecolors=shade[:, None] * base,
                                         edgecolor="none", linewidths=0))
    lo, hi = points.min(0), points.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, color="#dddddd")
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ----------------------------------------------------------------- 3D viewer
def view_model(vol, origin, spacing, iso, smooth=15, min_piece=0.002, title="3D model"):
    """Interactive surface viewer.

    Slider : iso threshold (re-extracts the surface)
    Keys   : l = cut the model open with a plane (drag it), again = off
             o = add/remove volume rendering      space = reset camera
             s/w = surface/wireframe (VTK)        q = quit
    """
    import pyvista as pv

    state = {"iso": iso, "clip": False, "vol": False, "plane": ((1.0, 0, 0), (0, 0, 0))}
    grid = image_data(vol, origin, spacing)
    pl = pv.Plotter(window_size=(1200, 1000), title=title)
    pl.set_background("#1b1b1b")
    pl.add_axes()

    def surface():
        surf = extract_surface(vol, origin, spacing, state["iso"], smooth, min_piece)
        if state["clip"]:
            n, o = state["plane"]
            surf = surf.clip(normal=n, origin=o)
        return surf

    def redraw():
        t = time.perf_counter()
        mesh = surface()
        pl.add_mesh(mesh, color="#efe6d2", smooth_shading=True, specular=0.3,
                    name="surface", reset_camera=False)
        pl.add_text(f"iso {state['iso']:.3g}   {mesh.n_cells:,} triangles"
                    f"{'   (cut open)' if state['clip'] else ''}",
                    position="lower_left", font_size=9, color="#e0c060", name="info")
        print(f"[model] iso {state['iso']:.4g}: {mesh.n_cells:,} triangles "
              f"({time.perf_counter() - t:.1f}s)")

    def on_iso(value):
        state["iso"] = float(value)
        redraw()

    def on_plane(normal, origin_):
        state["plane"] = (tuple(normal), tuple(origin_))
        redraw()

    def toggle_clip():
        state["clip"] = not state["clip"]
        if state["clip"]:
            pl.add_plane_widget(on_plane, normal=state["plane"][0], origin=(0, 0, 0),
                                bounds=grid.bounds, interaction_event="end",
                                test_callback=False)
        else:
            pl.clear_plane_widgets()
        redraw()

    def toggle_volume():
        state["vol"] = not state["vol"]
        if state["vol"]:
            hi = float(np.percentile(vol, 99.9))
            pl.add_volume(grid, scalars="density", cmap="bone", clim=(state["iso"] * 0.5, hi),
                          opacity="sigmoid", show_scalar_bar=False, name="volume",
                          reset_camera=False)
        else:
            pl.remove_actor("volume")
        pl.render()

    def reset():
        pl.camera_position = "iso"
        pl.reset_camera()
        pl.render()

    lo, hi = np.percentile(vol[vol != 0], [50, 99.9])
    redraw()
    pl.add_slider_widget(on_iso, [float(lo), float(hi)], value=state["iso"],
                         title="iso threshold (bone surface)", pointa=(0.55, 0.92),
                         pointb=(0.97, 0.92), style="modern", interaction_event="end",
                         title_height=0.02, fmt="%.3g")
    pl.add_key_event("l", toggle_clip)
    pl.add_key_event("o", toggle_volume)
    pl.add_key_event("space", reset)
    pl.add_text("drag: rotate   scroll: zoom   slider: threshold\n"
                "l: cut open with a plane   o: volume rendering   space: reset",
                position="upper_left", font_size=9, color="#cccccc")
    pl.camera_position = "iso"
    pl.reset_camera()
    pl.show()
