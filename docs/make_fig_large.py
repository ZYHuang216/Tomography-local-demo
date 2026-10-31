"""Large-font version of the minirecast architecture figure (3 rows, snake layout)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W, H = 100, 152
fig = plt.figure(figsize=(20, 30.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
fig.patch.set_facecolor("white")

FILL = dict(src="#E8F1FB", load="#E9F7EF", flow="#FFF4E0", filt="#F3EBFA", bp="#FDECEC",
            out="#EAF6F6", note="#F7F7F7")
EDGE = dict(src="#3B7DD8", load="#2E9D5B", flow="#D98C00", filt="#7B4FB8", bp="#C9453B",
            out="#178A8A", note="#34495E")
TXT, TITLE = 16.5, 21
LINE, MLINE = 2.05, 2.75          # line spacing for text / math lines


def box(x, y_top, w, title, lines, kind, fs=TXT):
    h = 4.6 + sum(MLINE if "$" in ln else (0.9 if ln == "" else LINE) for ln in lines)
    y = y_top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=FILL[kind], ec=EDGE[kind], lw=2.4))
    ax.text(x + 1.0, y_top - 0.9, title, fontsize=TITLE, fontweight="bold", va="top",
            color="#1F2D3D")
    ty = y_top - 4.2
    for ln in lines:
        if ln == "":
            ty -= 0.9
            continue
        ax.text(x + 1.2, ty, ln, fontsize=fs + (3 if "$" in ln else 0), va="top", color="#1d1d1d")
        ty -= MLINE if "$" in ln else LINE
    return x, y, w, h


def arrow(p, q, color, rad=0.0, ls="-", style="-|>", label=None, lpos=0.5, loff=(0, 0.6),
          lfs=15, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=28, color=color, lw=3,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if label:
        ax.text(p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1],
                label, fontsize=lfs, ha=ha, va="bottom", color=color, style="italic",
                fontweight="bold")


def stage(x, w, y, n, name):
    ax.text(x + 1.6, y, n, fontsize=19, ha="center", va="center", color="white",
            fontweight="bold", bbox=dict(boxstyle="circle,pad=0.3", fc="#34495E", ec="none"))
    ax.text(x + 4.0, y, name, fontsize=23, va="center", color="#1F2D3D", fontweight="bold")


ax.text(50, 150.3, "minirecast — algorithm & data flow", fontsize=36, fontweight="bold",
        ha="center", va="top", color="#1F2D3D")
ax.text(50, 146.6, "RECAST3D-style live tomographic reconstruction on a CPU (no CUDA GPU)",
        fontsize=19, ha="center", va="top", color="#5B6770")

X = [1.5, 36.5, 71.5]; WB = 27
# ============================ row 1 ============================
R1 = 139
stage(X[0], WB, R1, "1", "Data sources")
stage(X[1], WB, R1, "2", "Load & correct")
stage(X[2], WB, R1, "3", "Delivery")
top = R1 - 3
s1 = box(X[0], top, WB, "FleX-ray lab CT", ["scan_*.tif (1200 views)", "flat io*, dark di*",
         "scan settings.txt", "rat skull · Zenodo 1164088"], "src")
s2 = box(X[0], s1[1] - 2.2, WB, "Synchrotron HDF5", ["Data Exchange format",
         "/exchange/data, data_white,", "data_dark, theta", "APS / TomoBank"], "src")
s3 = box(X[0], s2[1] - 2.2, WB, "Analytic phantom", ["3D Shepp-Logan, exact",
         "line integrals → self-test"], "src")
l1 = box(X[1], top, WB, "flexray.py · hdf5.py", [
    "1  parse geometry +", "    CWI correction profile", "2  bin --sample (k×k mean)",
    "    and --skip views", "3  crop detector to the", "    object's shadow",
    "4  flat / dark + log:", r"     $p=-\ln\,\frac{I-D}{F-D}$",
    "5  cone_vec vectors per view", "    (source, detector, u, v)", "6  auto box around object"],
    "load")
ax.text(X[1] + WB / 2, l1[1] - 1.2, "→ proj[rows, angles, cols]\n+ geometry", ha="center",
        va="top", fontsize=16, color=EDGE["load"], fontweight="bold")
d1 = box(X[2], top, WB, "Batch", ["all views at once", "run_skull · model_skull"], "flow")
d2 = box(X[2], d1[1] - 2.2, WB, "Live stream", ["stream.py Feeder", "--group  --rate",
         "sequential | interleaved"], "flow")
d3 = box(X[2], d2[1] - 2.2, WB, "Network adapter", ["minirecast push → serve",
         "TCP + HMAC key, JSON + raw", "arrays, no pickle", "(SSH tunnel for remote)"], "flow")
row1_bottom = min(s3[1], l1[1] - 6, d3[1])

# ============================ row 2 ============================
R2 = row1_bottom - 9
stage(X[0], WB, R2, "4", "Ingest & filter (once)")
stage(X[1], WB, R2, "5", "On-demand reconstruction")
stage(X[2], WB, R2, "6", "Outputs")
top2 = R2 - 3
f1 = box(X[0], top2, WB, "add_projections()", [
    "for each arriving group:", "",
    "a  --tilt / --gaussian", "    (preprocess.py, optional)", "",
    "b  FDK cosine weight",
    r"     $w=\frac{SDD}{\sqrt{SDD^2+a^2+b^2}}$",
    "c  ramp filter |ω| per row", "    via FFT (Ram-Lak /", "    Shepp-Logan / Hann)", "",
    "d  store filtered q_k and", "    per-view constants",
    "    n += k ;  version += 1"], "filt")
b1 = box(X[1], top2, WB, "Back-projection kernel", [
    "for each requested point x", "and each received view k:",
    "  ray source → x hits the", "  detector → bilinear q_k", "",
    r"$f(x)=\frac{\pi}{N}\sum_k\left(\frac{SOD}{L_k(x)}\right)^{2}q_k$", "",
    "Numba, parallel on all cores", "(NumPy fallback)"], "bp")
b2 = box(X[1], b1[1] - 2.2, WB, "Requests", [
    "plane(origin, normal, n,", "      zoom, centre)",
    "volume(voxel) in z-slabs", "COR shift: move detector",
    "  in back-projection only", "COR search: min entropy"], "bp")
o1 = box(X[2], top2, WB, "3D viewer", ["3 movable planes + 2D panel", "COR & contrast sliders",
         "histogram · zoom · preview", "polls `version` → live"], "out")
o2 = box(X[2], o1[1] - 2.2, WB, "3D model (model.py)", ["Otsu threshold", "marching cubes",
         "clean + Taubin smooth", "→ STL / PLY / OBJ"], "out")
o3 = box(X[2], o2[1] - 2.2, WB, "Files", ["slices PNG / .npy", "volume TIFF / VTI"], "out")
row2_bottom = min(f1[1], b2[1], o3[1])

# ============================ arrows ============================
C = EDGE
for s in (s1, s2, s3):
    arrow((s[0] + s[2] + 0.4, s[1] + s[3] / 2),
          (X[1] - 0.4, min(max(s[1] + s[3] / 2, l1[1] + 4), top - 4)), C["src"])
for d in (d1, d2, d3):
    arrow((X[1] + WB + 0.4, l1[1] + l1[3] / 2), (X[2] - 0.4, d[1] + d[3] / 2), C["load"])
# delivery (row 1, right) -> ingest (row 2, left)
arrow((X[2] + WB / 2, d3[1] - 0.4), (X[0] + WB / 2, R2 + 2.4), C["flow"], rad=0.06,
      label="projections arrive (all at once, in groups, or over the network)", lpos=0.5,
      loff=(0, -3.4), lfs=16)
arrow((X[0] + WB + 0.4, f1[1] + f1[3] * 0.6), (X[1] - 0.4, b1[1] + b1[3] * 0.5), C["filt"],
      label="$q_k$", loff=(0, 0.6), lfs=20)
arrow((X[1] + WB / 2, b1[1] - 0.3), (X[1] + WB / 2, b2[1] + b2[3] + 0.3), C["bp"], style="<|-|>")
arrow((X[1] + WB + 0.4, b1[1] + b1[3] * 0.75), (X[2] - 0.4, o1[1] + o1[3] * 0.7), C["bp"],
      label="slices", loff=(0, 0.5))
arrow((X[2] - 0.4, o1[1] + o1[3] * 0.3), (X[1] + WB + 0.4, b1[1] + b1[3] * 0.3), C["out"],
      ls="--", label="requests", loff=(0, 0.3))
arrow((X[1] + WB + 0.4, b2[1] + b2[3] * 0.3), (X[2] - 0.4, o2[1] + o2[3] / 2), C["bp"],
      label="volume", loff=(-1.6, 1.2))
arrow((X[1] + WB + 0.4, b2[1] + 1.5), (X[2] - 0.4, o3[1] + o3[3] / 2), C["bp"])
# live loop under ingest
arrow((X[0] + WB * 0.75, f1[1] - 0.4), (X[0] + WB * 0.25, f1[1] - 0.4), C["flow"], rad=-0.6,
      ls="--", label="next group → slices refine live", lpos=0.5, loff=(0, -6.2), lfs=15)

# ============================ row 3: notes ============================
def note(x, y_top, w, title, lines, ec):
    h = 4.6 + LINE * 1.0 * len(lines)
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=FILL["note"], ec=ec, lw=2.2))
    ax.text(x + 1.0, y_top - 0.9, title, fontsize=TITLE, fontweight="bold", va="top",
            color="#1F2D3D")
    for i, ln in enumerate(lines):
        ax.text(x + 1.2, y_top - 4.2 - LINE * i, ln, fontsize=15, va="top", color="#262626")
    return y_top - h

R3 = row2_bottom - 6
note(X[0], R3, WB, "Run scripts", [
    "run_skull.sh", "   load all → interactive slices", "run_skull_hires.sh",
    "   82.5 µm, all views, Ram-Lak", "live_skull.sh", "   simulated live scan",
    "distributed_demo.sh", "   push → serve ⇄ view", "model_skull.sh",
    "   3D surface + STL / TIFF", "setup.sh", "   .venv + self-test"], EDGE["note"])
note(X[1], R3, WB, "Why no CUDA GPU is needed", [
    "• slice-on-demand: a 512² slice", "   from 1200 views ≈ 3·10⁸ lookups",
    "   (a 1024³ volume ≈ 10¹²)", "• each view filtered once;",
    "   COR / zoom never re-filter", "• Numba-compiled loops on", "   all CPU cores",
    "• bin / skip / crop keep RAM low", "   (1.1 GB at full resolution)",
    "• chunked work + kernel lock", "   keep the viewer responsive"], EDGE["bp"])
note(X[2], R3, WB, "vs RECAST3D (CWI)", [
    "same idea: slice-on-demand", "   FDK, cone_vec geometry,",
    "   live stream, adapter →", "   server → viewer, preview",
    "RECAST3D: C++, OpenGL, ZeroMQ,", "   ASTRA-CUDA (NVIDIA, Linux)",
    "minirecast: Python, NumPy,", "   Numba, VTK (any CPU, macOS)",
    "extra: re-reconstructing zoom,", "   entropy COR, 3D surface", "   model, STL / TIFF export"],
    EDGE["out"])

for ext in ("png", "svg"):
    fig.savefig(f"minirecast_architecture_large.{ext}", dpi=110 if ext == "png" else None,
                facecolor="white", bbox_inches="tight", pad_inches=0.3)
print("ok")
