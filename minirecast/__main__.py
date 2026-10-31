"""minirecast - RECAST3D-style on-demand slice reconstruction on the CPU.

Local (one process):
    python -m minirecast flexray skull_image_downloaded/skull [--stream]
    python -m minirecast hdf5 tomo_00001.h5 [--stream]
    python -m minirecast phantom [--geometry cone] [--stream]

Distributed (like RECAST3D: adapter -> reconstruction server -> viewer):
    python -m minirecast serve  [--port 5558]
    python -m minirecast view   --server localhost:5558
    python -m minirecast push   --server localhost:5558 flexray skull_image_downloaded/skull

3D model (whole volume -> bone surface):
    python -m minirecast model flexray skull_image_downloaded/skull --sample 2 \
           --save-mesh results/skull.stl --save-volume results/skull.tif

    python -m minirecast selftest
"""
import argparse
import sys
import time

import numpy as np


# ---------------------------------------------------------------- data sources
def add_phantom_args(sp):
    sp.add_argument("--geometry", default="parallel", choices=["parallel", "cone"])
    sp.add_argument("--angles", type=int, default=360)
    sp.add_argument("--det", type=int, default=256, help="detector size (pixels)")
    sp.add_argument("--photons", type=float, default=0, help="add Poisson noise (e.g. 2000)")


def add_flexray_args(sp):
    sp.add_argument("path", help="folder with scan_*.tif and 'scan settings.txt'")
    sp.add_argument("--sample", type=int, default=4, help="detector binning")
    sp.add_argument("--skip", type=int, default=2, help="use every N-th projection")
    sp.add_argument("--profile", default="cwi-flexray-2019-04-24",
                    help="FleX-ray geometry correction profile ('none' to disable)")
    sp.add_argument("--full-box", action="store_true",
                    help="whole field of view instead of a box around the object")
    sp.add_argument("--no-crop", action="store_true",
                    help="keep the whole detector (default: crop to the object's shadow)")


def add_hdf5_args(sp):
    sp.add_argument("path", help="Data Exchange HDF5 file (/exchange/data, data_white, ...)")
    sp.add_argument("--sample", type=int, default=2, help="detector binning")
    sp.add_argument("--skip", type=int, default=1, help="use every N-th projection")
    sp.add_argument("--rows", default=None, metavar="START:STOP",
                    help="detector rows to load (before binning), e.g. 500:800")
    sp.add_argument("--pixel-size", type=float, default=None,
                    help="detector pixel in mm (default: from the file)")


def load_source(args):
    """Return (projections [rows, angles, cols], geometry, title)."""
    if args.source == "phantom":
        from .phantom import EllipsoidPhantom, project_parallel, project_cone
        from .geometry import ParallelGeometry, circular_cone_vectors
        ph = EllipsoidPhantom(scale=50.0)
        t0 = time.perf_counter()
        if args.geometry == "parallel":
            geom = ParallelGeometry(np.arange(args.angles) * np.pi / args.angles,
                                    rows=args.det, cols=args.det, pixel_u=110.0 / args.det,
                                    pixel_v=110.0 / args.det)
            proj = project_parallel(ph, geom)
        else:
            geom = circular_cone_vectors(args.angles, sod=300.0, sdd=600.0, rows=args.det,
                                         cols=args.det, pixel=240.0 / args.det)
            proj = project_cone(ph, geom)
        if args.photons:
            rng = np.random.default_rng(0)
            counts = rng.poisson(args.photons * np.exp(-proj * 0.02))
            proj = (-np.log(np.maximum(counts, 1) / args.photons) / 0.02).astype(np.float32)
        print(f"[phantom] {args.angles} {args.geometry}-beam projections "
              f"({args.det}x{args.det}) in {time.perf_counter() - t0:.1f}s")
        return proj, geom, f"phantom ({args.geometry} beam)"
    if args.source == "flexray":
        from .flexray import load_flexray
        proj, geom = load_flexray(args.path, sample=args.sample, skip=args.skip,
                                  profile=args.profile, auto_box=not args.full_box,
                                  crop=not args.no_crop)
        return proj, geom, f"FleX-ray: {args.path}"
    if args.source == "hdf5":
        from .hdf5 import load_dxchange
        proj, geom = load_dxchange(args.path, sample=args.sample, skip=args.skip,
                                   rows=args.rows, pixel_size=args.pixel_size)
        return proj, geom, f"HDF5: {args.path}"
    raise ValueError(args.source)


# ---------------------------------------------------------------- option groups
def add_recon_args(sp):
    g = sp.add_argument_group("reconstruction")
    g.add_argument("--filter", default="shepp-logan",
                   choices=["ram-lak", "shepp-logan", "hann", "cosine"])
    g.add_argument("--backend", default="auto", choices=["auto", "numba", "numpy"])
    g.add_argument("--gaussian", type=float, default=0.0, metavar="SIGMA",
                   help="Gaussian low-pass on each projection (detector pixels)")
    g.add_argument("--tilt", type=float, default=0.0, metavar="DEG",
                   help="correct a detector tilted by DEG degrees")


def add_stream_args(sp, default_stream=False):
    g = sp.add_argument_group("live acquisition (simulated)")
    if not default_stream:
        g.add_argument("--stream", action="store_true",
                       help="feed projections in groups as during a scan")
    g.add_argument("--rate", type=float, default=150.0, help="projections per second")
    g.add_argument("--group", type=int, default=20, help="projections per group")
    g.add_argument("--order", default="sequential", choices=["sequential", "interleaved"],
                   help="acquisition order (interleaved = sharp image early)")


def add_view_args(sp):
    g = sp.add_argument_group("viewing / output")
    g.add_argument("--viewer", default="pyvista", choices=["pyvista", "mpl", "none"])
    g.add_argument("--res", type=int, default=256, help="slice resolution (pixels)")
    g.add_argument("--preview-res", type=int, default=None,
                   help="resolution while dragging (default: 96 numba / 48 numpy)")
    g.add_argument("--no-panel", action="store_true", help="hide the 2D slice panel")
    g.add_argument("--preview", action="store_true", help="start with the 3D preview volume on")
    g.add_argument("--preview-dim", type=int, default=None,
                   help="preview volume size on the long side (default 96 numba / 48 numpy)")
    g.add_argument("--cmap", default="bone", help="colour map (m cycles in the viewer)")
    g.add_argument("--cor-shift", type=float, default=0.0,
                   help="rotation-centre shift in (binned) detector pixels")
    g.add_argument("--cor-range", type=float, default=10.0, help="COR slider range +-px")
    g.add_argument("--auto-cor", action="store_true",
                   help="search +-3 px around the centre for the sharpest slice")
    g.add_argument("--save-slices", default=None, metavar="DIR",
                   help="save the 3 central slices (PNG + .npy) to DIR")
    g.add_argument("--snapshot", default=None, help="save a 3D snapshot PNG")
    g.add_argument("--zoom", type=float, default=1.0,
                   help="zoom for --save-slices/--snapshot (finer pixels, smaller area)")
    g.add_argument("--center", default="0,0,0", metavar="X,Y,Z",
                   help="zoom centre in mm, e.g. --center=-3,0,10 (use = when it starts with -)")


def add_sources(parent, extra):
    subs = parent.add_subparsers(dest="source", required=True)
    for name, adder, help_ in (("phantom", add_phantom_args, "synthetic 3D phantom"),
                               ("flexray", add_flexray_args, "FleX-ray cone beam (rat skull)"),
                               ("hdf5", add_hdf5_args, "Data Exchange HDF5 parallel beam")):
        sp = subs.add_parser(name, help=help_)
        adder(sp)
        for e in extra:
            e(sp)


# ---------------------------------------------------------------- run helpers
def view(recon, args, title, wait_for=None):
    center = tuple(float(v) for v in args.center.split(","))
    if len(center) != 3:
        raise SystemExit("--center needs three comma-separated numbers, e.g. --center 0,-3,10")
    if args.cor_shift:
        recon.set_cor_shift(args.cor_shift)
    fast = "numba" in str(recon.backend) or "remote" in str(recon.backend)
    preview = args.preview_res or (96 if fast else 48)

    if args.save_slices or args.snapshot or args.auto_cor:
        if wait_for is not None:                     # outputs need all the data
            print("[minirecast] waiting for the stream to finish ...")
            wait_for()
        if args.auto_cor:
            recon.estimate_cor()
        if args.save_slices:
            from .viewer_mpl import save_slices
            save_slices(recon, args.save_slices, res=max(args.res, 384), origin=center,
                        zoom=args.zoom)
            print(f"[minirecast] slices written to {args.save_slices}/")
        if args.snapshot:
            from .viewer_mpl import snapshot_3d
            snapshot_3d(recon, args.snapshot, res=min(args.res, 256), title=title,
                        origin=center, zoom=args.zoom)
            print(f"[minirecast] snapshot written to {args.snapshot}")

    if args.viewer == "pyvista":
        try:
            from .viewer_pyvista import run_viewer
        except ImportError as e:
            print(f"pyvista not available ({e}); falling back to matplotlib viewer")
            args.viewer = "mpl"
        else:
            run_viewer(recon, title=title, res=args.res, preview_res=preview,
                       panel=not args.no_panel, preview=args.preview,
                       preview_dim=args.preview_dim, cor_range=args.cor_range,
                       cmap=args.cmap)
    if args.viewer == "mpl":
        if wait_for is not None:
            wait_for()
        from .viewer_mpl import run_viewer_mpl
        run_viewer_mpl(recon, res=min(args.res, 256))
    elif args.viewer == "none" and wait_for is not None:
        wait_for()


def run_local(args):
    from .recon import SliceReconstructor
    proj, geom, title = load_source(args)
    opts = dict(window=args.filter, backend=args.backend, gaussian=args.gaussian,
                tilt=args.tilt)
    if not args.stream:
        recon = SliceReconstructor(proj, geom, **opts)
        t0 = time.perf_counter()
        recon.warmup()
        if recon.backend == "numba":
            print(f"[minirecast] numba kernels ready in {time.perf_counter() - t0:.1f}s")
        del proj
        return view(recon, args, title)
    from .stream import Feeder
    recon = SliceReconstructor(None, geom, **opts)
    feeder = Feeder(recon, proj, order=args.order, group=args.group, rate=args.rate)
    print(f"[stream] live scan: {geom.n_angles} projections, {args.order} order, "
          f"groups of {args.group}, {args.rate:g}/s")
    feeder.start()
    try:
        view(recon, args, title + "  [live]", wait_for=feeder.join)
    finally:
        feeder.stop()


def run_push(args):
    from .net import RemoteSink, parse_address
    from .stream import Feeder
    proj, geom, title = load_source(args)
    address = parse_address(args.server)
    sink = RemoteSink(address, geom)
    print(f"[push] {title} -> {address[0]}:{address[1]}: {geom.n_angles} projections, "
          f"{args.order} order, groups of {args.group}, {args.rate:g}/s")
    feeder = Feeder(sink, proj, order=args.order, group=args.group, rate=args.rate)
    feeder.run()                        # in this thread
    sink.end()


def add_model_args(sp):
    g = sp.add_argument_group("3D model")
    g.add_argument("--voxel", type=float, default=None,
                   help="voxel size in mm (default: detector pixel at the axis)")
    g.add_argument("--iso", default="auto",
                   help="surface threshold (attenuation) or 'auto' (Otsu)")
    g.add_argument("--smooth", type=int, default=15, help="smoothing iterations (0 = off)")
    g.add_argument("--min-piece", type=float, default=0.002,
                   help="drop loose pieces smaller than this fraction of the surface")
    g.add_argument("--decimate", type=float, default=0.0,
                   help="fraction of triangles to remove for smaller files (e.g. 0.7)")
    g.add_argument("--save-mesh", default=None, metavar="FILE",
                   help="export surface: .stl / .ply / .obj (mm)")
    g.add_argument("--save-volume", default=None, metavar="FILE",
                   help="export volume: .tif (Fiji/ImageJ) / .npz / .vti (ParaView)")
    g.add_argument("--snapshot", default=None, metavar="PNG", help="save a picture of the model")
    g.add_argument("--viewer", default="pyvista", choices=["pyvista", "none"])
    g.add_argument("--cor-shift", type=float, default=0.0,
                   help="rotation-centre shift in (binned) detector pixels")


def run_model(args):
    from .recon import SliceReconstructor
    from . import model as M
    proj, geom, title = load_source(args)
    recon = SliceReconstructor(proj, geom, window=args.filter, backend=args.backend,
                               gaussian=args.gaussian, tilt=args.tilt)
    del proj
    recon.warmup()
    if args.cor_shift:
        recon.set_cor_shift(args.cor_shift)
    h = args.voxel or recon.native_pixel
    dims = np.ceil(2 * recon.box / h).astype(int)
    print(f"[model] reconstructing {dims[0]}x{dims[1]}x{dims[2]} voxels of {h * 1000:.0f} um "
          f"({recon.n} projections, backend={recon.backend}) ...")
    t0 = time.perf_counter()
    last = [0.0]

    def progress(frac):
        if frac - last[0] >= 0.1 or frac >= 1.0:
            last[0] = frac
            el = time.perf_counter() - t0
            print(f"[model]   {100 * frac:3.0f}%  {el:5.1f}s  (~{el / max(frac, 1e-3) - el:.0f}s left)",
                  flush=True)

    vol, origin, h = recon.reconstruct_volume(voxel=h, progress=progress)
    del recon
    iso = M.otsu_threshold(vol) if args.iso == "auto" else float(args.iso)
    print(f"[model] volume done in {time.perf_counter() - t0:.1f}s; "
          f"iso threshold {iso:.4g} ({'Otsu' if args.iso == 'auto' else 'given'})")
    if args.save_volume:
        M.save_volume(vol, origin, h, args.save_volume)
        print(f"[model] volume written to {args.save_volume}")
    if args.save_mesh or args.snapshot:
        mesh = M.extract_surface(vol, origin, h, iso, smooth=args.smooth,
                                 min_piece=args.min_piece, decimate=args.decimate)
        pts, faces = M.mesh_arrays(mesh)
        print(f"[model] surface: {len(faces):,} triangles, {len(pts):,} vertices")
        if args.save_mesh:
            M.save_mesh(mesh, args.save_mesh)
            print(f"[model] mesh written to {args.save_mesh}")
        if args.snapshot:
            M.snapshot(mesh, args.snapshot, title=title)
            print(f"[model] picture written to {args.snapshot}")
    if args.viewer == "pyvista":
        M.view_model(vol, origin, h, iso, smooth=args.smooth, min_piece=args.min_piece,
                     title=f"3D model - {title}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="minirecast", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # local: python -m minirecast flexray PATH ...   (source is the command)
    for name, adder, help_ in (("phantom", add_phantom_args, "synthetic 3D phantom"),
                               ("flexray", add_flexray_args, "FleX-ray cone beam (rat skull)"),
                               ("hdf5", add_hdf5_args, "Data Exchange HDF5 parallel beam")):
        sp = sub.add_parser(name, help=help_ + " (local)")
        adder(sp)
        add_recon_args(sp)
        add_stream_args(sp)
        add_view_args(sp)
        sp.set_defaults(source=name, run=run_local)

    ss = sub.add_parser("serve", help="reconstruction server (receives projections, "
                                      "answers slice requests)")
    ss.add_argument("--host", default="127.0.0.1",
                    help="interface to listen on (keep 127.0.0.1 and use an SSH tunnel)")
    ss.add_argument("--port", type=int, default=5558)
    add_recon_args(ss)

    sv = sub.add_parser("view", help="viewer connected to a server")
    sv.add_argument("--server", default="127.0.0.1:5558", metavar="HOST:PORT")
    add_view_args(sv)

    sp = sub.add_parser("push", help="adapter: send a scan to a server as a live stream")
    sp.add_argument("--server", default="127.0.0.1:5558", metavar="HOST:PORT")
    add_sources(sp, [lambda s: add_stream_args(s, default_stream=True)])

    sm = sub.add_parser("model", help="full 3D reconstruction -> surface model "
                                      "(viewer, STL/PLY/OBJ, TIFF stack)")
    add_sources(sm, [add_recon_args, add_model_args])
    sm.set_defaults(run=run_model)

    sub.add_parser("selftest", help="check numba vs numpy, streaming and accuracy")
    args = p.parse_args(argv)

    if args.cmd == "selftest":
        from .selftest import run
        return run()
    if args.cmd == "serve":
        from .net import Server
        Server(args.host, args.port, window=args.filter, backend=args.backend,
               gaussian=args.gaussian, tilt=args.tilt).serve_forever()
        return 0
    if args.cmd == "view":
        from .net import RemoteReconstructor, parse_address
        recon = RemoteReconstructor(parse_address(args.server))
        view(recon, args, f"minirecast @ {args.server}")
        return 0
    if args.cmd == "push":
        run_push(args)
        return 0
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
