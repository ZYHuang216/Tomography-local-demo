"""RECAST3D-style 3D viewer.

Left : 3D scene - three slicing planes in the volume box, optional low-resolution
       preview volume, axes, rotation-centre slider, contrast sliders + histogram.
Right: the three slices as flat 2D images (always visible).

Works with a local SliceReconstructor or a RemoteReconstructor (minirecast view
--server ...). While projections are still arriving (live scan), the slices are
re-reconstructed whenever new data comes in.

Mouse
  sphere handle: move slice | arrow shaft: push along normal | arrow tip: tilt
  left-drag on empty space: rotate camera | scroll: camera zoom
Keys
  + / =  zoom in at the planes' crossing (re-reconstructs finer pixels)
  -      zoom out          0  reset zoom
  k      auto-contrast     m  next colour map       g  axes on/off
  o      preview volume on/off                      x  auto rotation centre
  n      look straight at slice 1, 2, 3 in turn     space  reset camera
  h      hide/show slice handles                    q  quit
"""
import threading
import time

import numpy as np

from .recon import planes_intersection, plane_basis

ZOOM_STEP = 1.5
MAX_ZOOM = 32.0
SLICE_COLORS = ("#ff6b6b", "#5fd35f", "#5aa9ff")     # slice 1/2/3 handles + panel labels
CMAPS = ("bone", "gray", "viridis", "magma", "inferno", "cividis")


def _slice_name(normal):
    n = np.abs(np.asarray(normal, float))
    n /= np.linalg.norm(n)
    for axis, name in ((2, "axial"), (0, "sagittal"), (1, "coronal")):
        if n[axis] > 0.999:
            return name
    tilt = np.degrees(np.arccos(np.clip(n.max(), -1, 1)))
    return f"oblique ({tilt:.0f} deg off axis)"


def run_viewer(recon, title="minirecast", res=256, preview_res=96, cmap="bone", panel=True,
               preview=False, preview_dim=None, cor_range=10.0, poll_ms=250):
    import pyvista as pv

    box = np.asarray(recon.box, float)
    bounds = (-box[0], box[0], -box[1], box[1], -box[2], box[2])
    fast = "numba" in str(getattr(recon, "backend", "")) or "remote" in str(
        getattr(recon, "backend", ""))
    stream_res = min(res, 2 * preview_res)
    preview_dim = preview_dim or (96 if fast else 48)

    if panel:   # big 3D view on the left, three 2D slice views stacked on the right
        pl = pv.Plotter(shape="1|3", splitting_position=0.68, window_size=(1600, 980),
                        title=title, border=False)
    else:
        pl = pv.Plotter(window_size=(1150, 980), title=title)
    # pyvista binds +/- to "thicker/thinner lines" by default; we use them for zoom
    try:
        for key in ("plus", "minus"):
            pl.iren._key_press_event_callbacks.pop(key, None)
    except AttributeError:
        pass

    MAIN = 0
    pl.subplot(MAIN)
    pl.set_background("#1b1b1b")
    pl.add_mesh(pv.Box(bounds=bounds).outline(), color="#bbbbbb", line_width=1.5)

    # (normal, handle position). Handles are spread out so they don't overlap.
    bx, by, bz = 0.6 * box
    init = [((0.0, 0.0, 1.0), (-bx, -by, 0.0)),
            ((1.0, 0.0, 0.0), (0.0, -by, bz)),
            ((0.0, 1.0, 0.0), (bx, 0.0, -bz))]
    planes = [[np.array(n, float), np.array(o, float)] for n, o in init]
    info = recon.info()
    state = dict(clim=(0.0, 1.0), manual_clim=False, zoom=1.0, center=np.zeros(3),
                 cmap=cmap, version=info["version"], n=info["n"], n_total=info["n_total"],
                 cor=float(info["cor_shift"]), preview=preview, preview_job=None,
                 preview_result=None, preview_version=-1, preview_time=0.0,
                 cor_job=None, cor_result=None, rate=0.0, last_n=(info["n"], time.time()),
                 cam_slice=-1, axes=True)

    # ------------------------------------------------------------ slices
    def make_grid(img, pts):
        g = pv.StructuredGrid(pts[..., 0], pts[..., 1], pts[..., 2])
        g["slice"] = img.ravel(order="F")
        return g

    def update_grid(g, img, pts):
        # resolution may change between drag-preview and full-res updates
        g.dimensions = (pts.shape[0], pts.shape[1], 1)
        g.points = np.column_stack([pts[..., i].ravel(order="F") for i in range(3)])
        g["slice"] = img.ravel(order="F")
        g.Modified()

    def slice_size(i):
        return recon.plane_size(planes[i][0]) / state["zoom"]

    def slice_of(i, n):
        normal, origin = planes[i]
        return recon.reconstruct_plane(origin, normal, n=n, zoom=state["zoom"],
                                       center=state["center"])

    def make_image(img, size):
        n = img.shape[0]
        im = pv.ImageData(dimensions=(n, n, 1), spacing=(size / n, size / n, 1.0),
                          origin=(-size / 2 + size / (2 * n), -size / 2 + size / (2 * n), 0.0))
        im["slice"] = img.ravel()          # rows follow the plane's 2nd axis, columns the 1st
        return im

    def update_image(im, img, size):
        n = img.shape[0]
        im.dimensions = (n, n, 1)
        im.spacing = (size / n, size / n, 1.0)
        im.origin = (-size / 2 + size / (2 * n), -size / 2 + size / (2 * n), 0.0)
        im["slice"] = img.ravel()
        im.Modified()

    grids, images = [], []
    t0 = time.perf_counter()
    for i in range(3):
        img, pts = slice_of(i, res)
        grids.append(make_grid(img, pts))
        images.append(make_image(img, slice_size(i)))
    print(f"[viewer] first 3 slices in {time.perf_counter() - t0:.2f}s")

    def slice_values():
        vals = np.concatenate([g["slice"] for g in grids])
        return vals[vals != 0]

    def auto_clim():
        vals = slice_values()
        if vals.size == 0:
            return (0.0, 1.0)
        lo, hi = np.percentile(vals, [1.0, 99.7])
        lo = max(lo, 0.0) if hi > 0 else lo
        return (float(lo), float(hi if hi > lo else lo + 1e-6))

    state["clim"] = auto_clim()
    pl.subplot(MAIN)
    slice_actors = [pl.add_mesh(g, scalars="slice", cmap=cmap, clim=state["clim"],
                                show_scalar_bar=False, lighting=False) for g in grids]
    panel_actors = []

    def panel_label(i):
        n, _ = planes[i]
        px = slice_size(i) / res * 1000.0
        return f"slice {i + 1}: {_slice_name(n)}   {slice_size(i):.1f} mm, ~{px:.0f} um/px"

    def refresh_panel_labels():
        if not panel:
            return
        for i in range(3):
            pl.subplot(i + 1)
            pl.add_text(panel_label(i), position="upper_left", font_size=8,
                        color=SLICE_COLORS[i], name=f"label{i}")
        pl.subplot(MAIN)

    def fit_panel(i):
        if panel:
            pl.renderers[i + 1].reset_camera()

    if panel:
        for i in range(3):
            pl.subplot(i + 1)
            pl.set_background("#111111")
            panel_actors.append(pl.add_mesh(images[i], scalars="slice", cmap=cmap,
                                            clim=state["clim"], show_scalar_bar=False,
                                            lighting=False))
            pl.view_xy()
            pl.enable_parallel_projection()
            pl.reset_camera()
        refresh_panel_labels()
        pl.subplot(MAIN)

    def recompute(i, n):
        img, pts = slice_of(i, n)
        update_grid(grids[i], img, pts)
        if panel:
            update_image(images[i], img, slice_size(i))

    def recompute_all(n):
        for i in range(3):
            recompute(i, n)

    # ------------------------------------------------------------ contrast + histogram
    pl.subplot(MAIN)
    chart = pv.Chart2D(size=(0.31, 0.19), loc=(0.67, 0.135), x_label="", y_label="")
    try:
        chart.background_color = (0.08, 0.08, 0.08, 0.85)
        chart.x_axis.label_visible = False
        chart.y_axis.label_visible = False
        chart.y_axis.tick_labels_visible = False
        chart.grid = False
    except Exception:
        pass
    hist_bars = chart.bar([0.0, 1.0], [0.0, 0.0], color="#8a8a8a")
    lo_line = chart.line([0, 0], [0, 1], color="#5aa9ff", width=2.0)
    hi_line = chart.line([1, 1], [0, 1], color="#ff6b6b", width=2.0)
    pl.add_chart(chart)

    def update_histogram():
        vals = slice_values()
        if vals.size < 10:
            return
        a, b = np.percentile(vals, [0.1, 99.9])
        span = max(b - a, 1e-6)
        counts, edges = np.histogram(vals, bins=60, range=(a - 0.1 * span, b + 0.1 * span))
        centers = 0.5 * (edges[1:] + edges[:-1])
        counts = np.log1p(counts)
        hist_bars.update(centers, counts)
        top = float(counts.max()) if counts.size else 1.0
        lo, hi = state["clim"]
        lo_line.update([lo, lo], [0, top])
        hi_line.update([hi, hi], [0, top])
        state["hist_range"] = (float(edges[0]), float(edges[-1]))

    def apply_clim():
        for a in slice_actors + panel_actors:
            a.mapper.scalar_range = state["clim"]
        if "preview_actor" in state:
            try:
                state["preview_actor"].mapper.scalar_range = state["clim"]
            except Exception:
                pass
        update_histogram()

    def set_slider(widget, lo, hi, value):
        rep = widget.GetRepresentation()
        rep.SetMinimumValue(lo)
        rep.SetMaximumValue(hi)
        rep.SetValue(value)

    def sync_contrast_sliders():
        lo_r, hi_r = state.get("hist_range", (state["clim"][0], state["clim"][1]))
        lo_r, hi_r = min(lo_r, state["clim"][0]), max(hi_r, state["clim"][1])
        set_slider(min_slider, lo_r, hi_r, state["clim"][0])
        set_slider(max_slider, lo_r, hi_r, state["clim"][1])

    def on_min(value):
        lo, hi = state["clim"]
        state["clim"] = (float(min(value, hi - 1e-9)), hi)
        state["manual_clim"] = True
        apply_clim()

    def on_max(value):
        lo, hi = state["clim"]
        state["clim"] = (lo, float(max(value, lo + 1e-9)))
        state["manual_clim"] = True
        apply_clim()

    slider_style = dict(interaction_event="always", style="modern", title_height=0.018,
                        slider_width=0.02, tube_width=0.006, fmt="%.3g")
    max_slider = pl.add_slider_widget(on_max, [state["clim"][0], state["clim"][1] * 1.5 + 1e-6],
                                      value=state["clim"][1], title="contrast max",
                                      pointa=(0.68, 0.085), pointb=(0.97, 0.085),
                                      color="#ff6b6b", **slider_style)
    min_slider = pl.add_slider_widget(on_min, [state["clim"][0], state["clim"][1] * 1.5 + 1e-6],
                                      value=state["clim"][0], title="contrast min",
                                      pointa=(0.68, 0.03), pointb=(0.97, 0.03),
                                      color="#5aa9ff", **slider_style)
    update_histogram()
    sync_contrast_sliders()

    def contrast_auto():
        state["clim"] = auto_clim()
        state["manual_clim"] = False
        apply_clim()
        sync_contrast_sliders()
        pl.render()

    # ------------------------------------------------------------ rotation centre
    def on_cor(value):
        state["cor"] = float(value)
        recon.set_cor_shift(float(value))       # version changes -> timer refreshes slices
        print(f"[viewer] rotation centre shift {value:+.2f} px")

    cor_slider = pl.add_slider_widget(
        on_cor, [-cor_range, cor_range], value=state["cor"],
        title="rotation centre shift [detector px]  (x: auto)",
        pointa=(0.56, 0.93), pointb=(0.97, 0.93), color="#e0c060",
        interaction_event="end", style="modern", title_height=0.018, fmt="%+.2f",
        slider_width=0.02, tube_width=0.006)

    def auto_cor():
        if state["cor_job"] is not None and state["cor_job"].is_alive():
            return
        print("[viewer] searching rotation centre ...")

        def job():
            best, _ = recon.estimate_cor(verbose=False)
            state["cor_result"] = best
        state["cor_job"] = threading.Thread(target=job, daemon=True)
        state["cor_job"].start()

    # ------------------------------------------------------------ preview volume
    def start_preview_job():
        def job(version):
            try:
                vol, origin, h = recon.preview_volume(preview_dim)
                state["preview_result"] = (vol, origin, h, version)
            except Exception as e:  # pragma: no cover
                print(f"[viewer] preview failed: {e!r}")
        state["preview_time"] = time.time()
        state["preview_job"] = threading.Thread(target=job, args=(state["version"],),
                                                daemon=True)
        state["preview_job"].start()

    def show_preview_result():
        vol, origin, h, version = state["preview_result"]
        state["preview_result"] = None
        state["preview_version"] = version
        if not state["preview"]:
            return
        img = pv.ImageData(dimensions=vol.shape, spacing=(h, h, h), origin=tuple(origin))
        img["v"] = vol.ravel(order="F")
        pl.subplot(MAIN)
        state["preview_actor"] = pl.add_volume(
            img, scalars="v", cmap=state["cmap"], clim=state["clim"],
            opacity=[0.0, 0.0, 0.02, 0.06, 0.12, 0.2], show_scalar_bar=False,
            name="preview", reset_camera=False)

    def toggle_preview():
        state["preview"] = not state["preview"]
        if state["preview"]:
            state["preview_version"] = -1                  # (re)compute now
            print(f"[viewer] preview volume on ({preview_dim} voxels on the long side)")
        else:
            pl.remove_actor("preview")
            state.pop("preview_actor", None)
            pl.render()

    # ------------------------------------------------------------ widgets
    pl.subplot(MAIN)
    widgets = []
    for i, (normal, origin) in enumerate(init):
        def on_move(normal, origin, i=i):
            planes[i] = [np.array(normal, float), np.array(origin, float)]
            recompute(i, preview_res)

        w = pl.add_plane_widget(on_move, normal=normal, origin=origin, bounds=bounds,
                                normal_rotation=True, interaction_event="always",
                                test_callback=False, outline_translation=False, implicit=True,
                                color=SLICE_COLORS[i])
        # make VTK's own plane almost invisible (still pickable) and hide its outline:
        # the reconstructed slice itself is what you see
        try:
            w.GetPlaneProperty().SetOpacity(0.03)
            w.GetSelectedPlaneProperty().SetOpacity(0.12)
            w.GetOutlineProperty().SetOpacity(0.0)
            w.GetSelectedOutlineProperty().SetOpacity(0.0)
        except AttributeError:
            pass

        def on_release(obj, event, i=i):
            planes[i] = [np.array(obj.GetNormal(), float), np.array(obj.GetOrigin(), float)]
            t = time.perf_counter()
            recompute(i, res)
            fit_panel(i)                 # tilting changes the slice's size
            refresh_panel_labels()
            pl.render()
            print(f"[viewer] slice {i + 1}: {res}x{res} in {time.perf_counter() - t:.3f}s")

        w.AddObserver("EndInteractionEvent", on_release)
        widgets.append(w)

    # ------------------------------------------------------------ zoom
    def zoom_text():
        px = recon.plane_size(planes[0][0]) / state["zoom"] / res * 1000.0
        c = state["center"]
        native = recon.native_pixel * 1000.0
        warn = "  (finer than data)" if px < 0.7 * native else ""
        return (f"zoom {state['zoom']:.2f}x  ~{px:.0f} um/px (data {native:.0f} um){warn}   "
                f"centre ({c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f}) mm")

    def status_text():
        n, nt = state["n"], state["n_total"]
        live = (f"LIVE  {state['rate']:.0f} proj/s" if n < nt else "complete")
        return (f"projections {n}/{nt} ({100.0 * n / max(nt, 1):.0f}%)  {live}   "
                f"COR {state['cor']:+.2f} px   backend {recon.backend}\n" + zoom_text())

    def show_status():
        pl.subplot(MAIN)
        pl.add_text(status_text(), position="lower_left", font_size=9, color="#e0c060",
                    name="status")

    def set_zoom(new_zoom, center=None):
        old = state["zoom"]
        state["zoom"] = float(np.clip(new_zoom, 1.0, MAX_ZOOM))
        if center is not None:
            state["center"] = np.asarray(center, float)
        if state["zoom"] == 1.0:
            state["center"] = np.zeros(3)
        t = time.perf_counter()
        for i in range(3):
            recompute(i, res)
            fit_panel(i)
        refresh_panel_labels()
        # camera follows: look at the zoom centre and magnify by the same factor
        pl.subplot(MAIN)
        cam = pl.camera
        if state["zoom"] == 1.0:
            pl.reset_camera()
        else:
            shift = state["center"] - np.array(cam.focal_point)
            cam.focal_point = tuple(state["center"])
            cam.position = tuple(np.array(cam.position) + shift)
            cam.zoom(state["zoom"] / old)
        show_status()
        pl.render()
        print(f"[viewer] {zoom_text()}  (3 slices in {time.perf_counter() - t:.2f}s)")

    def zoom_in():
        # zoom where the three planes cross (move them onto the feature first)
        c = planes_intersection([p[0] for p in planes], [p[1] for p in planes],
                                fallback=state["center"])
        c = np.clip(c, -box, box)
        set_zoom(state["zoom"] * ZOOM_STEP, center=c)

    # ------------------------------------------------------------ camera / axes / colours
    def look_at_next_slice():
        state["cam_slice"] = (state["cam_slice"] + 1) % 3
        i = state["cam_slice"]
        normal, origin = planes[i]
        n_hat, _, e2 = plane_basis(normal)
        c = np.asarray(state["center"], float)
        c = c - ((c - origin) @ n_hat) * n_hat              # slice centre
        pl.subplot(MAIN)
        cam = pl.camera
        dist = np.linalg.norm(np.array(cam.position) - np.array(cam.focal_point))
        cam.focal_point = tuple(c)
        cam.position = tuple(c + dist * n_hat)
        cam.up = tuple(e2)
        pl.render()
        print(f"[viewer] camera: looking at slice {i + 1} ({_slice_name(normal)})")

    def reset_camera():
        pl.subplot(MAIN)
        pl.camera_position = "iso"
        pl.reset_camera()
        if state["zoom"] > 1.0:
            pl.camera.focal_point = tuple(state["center"])
            pl.camera.zoom(state["zoom"])
        pl.render()

    def toggle_axes():
        pl.subplot(MAIN)
        state["axes"] = not state["axes"]
        (pl.renderer.show_axes if state["axes"] else pl.renderer.hide_axes)()
        pl.render()

    def next_cmap():
        state["cmap"] = CMAPS[(CMAPS.index(state["cmap"]) + 1) % len(CMAPS)
                              if state["cmap"] in CMAPS else 0]
        for a in slice_actors + panel_actors:
            a.mapper.lookup_table.cmap = state["cmap"]
            a.mapper.scalar_range = state["clim"]
        if state["preview"]:
            state["preview_version"] = -1                  # rebuild with the new map
        print(f"[viewer] colour map: {state['cmap']}")
        pl.render()

    def toggle_widgets():
        for w in widgets:
            w.SetEnabled(0 if w.GetEnabled() else 1)
        pl.render()

    # ------------------------------------------------------------ live updates (timer)
    def tick(step=None):
        try:
            inf = recon.info()
        except Exception as e:                              # server gone, ...
            if step and step % 20 == 0:
                print(f"[viewer] no update: {e!r}")
            return
        changed = False
        now = time.time()
        if inf["n"] != state["last_n"][0]:
            dn, dt = inf["n"] - state["last_n"][0], now - state["last_n"][1]
            state["rate"] = dn / max(dt, 1e-3)
            state["last_n"] = (inf["n"], now)
        state["n"], state["n_total"] = inf["n"], inf["n_total"]
        if inf["version"] != state["version"]:
            state["version"] = inf["version"]
            state["cor"] = float(inf["cor_shift"])
            recompute_all(res if inf["n"] >= inf["n_total"] else stream_res)
            if not state["manual_clim"]:
                state["clim"] = auto_clim()
                apply_clim()
                sync_contrast_sliders()
            else:
                update_histogram()
            changed = True
        if state["cor_result"] is not None:
            best, state["cor_result"] = state["cor_result"], None
            cor_slider.GetRepresentation().SetValue(best)
            state["cor"] = best
            print(f"[viewer] auto rotation centre: {best:+.2f} px")
            changed = True
        if state["preview"]:
            if state["preview_result"] is not None:
                show_preview_result()
                changed = True
            busy = state["preview_job"] is not None and state["preview_job"].is_alive()
            if (not busy and state["preview_version"] != state["version"]
                    and now - state["preview_time"] > 2.0 and state["n"] > 0):
                start_preview_job()
        if changed or step == 1:
            show_status()
            pl.render()

    # ------------------------------------------------------------ keys + start
    for key in ("plus", "equal", "KP_Add", "+", "="):
        pl.add_key_event(key, zoom_in)
    for key in ("minus", "KP_Subtract", "-", "underscore"):
        pl.add_key_event(key, lambda: set_zoom(state["zoom"] / ZOOM_STEP))
    for key in ("0", "KP_0"):
        pl.add_key_event(key, lambda: set_zoom(1.0))
    pl.add_key_event("k", contrast_auto)
    pl.add_key_event("m", next_cmap)
    pl.add_key_event("g", toggle_axes)
    pl.add_key_event("o", toggle_preview)
    pl.add_key_event("x", auto_cor)
    pl.add_key_event("n", look_at_next_slice)
    pl.add_key_event("space", reset_camera)
    pl.add_key_event("h", toggle_widgets)

    pl.subplot(MAIN)
    pl.add_axes()
    pl.add_text("sphere: move  arrow shaft: push  arrow tip: tilt\n"
                "+/- 0: zoom   k m: contrast/colours   o: preview   x: auto COR\n"
                "n: view slice   space: reset view   g: axes   h: handles",
                position="upper_left", font_size=8, color="#cccccc")
    show_status()
    pl.camera_position = "iso"
    pl.add_timer_event(max_steps=10 ** 9, duration=int(poll_ms), callback=tick)
    pl.show()
