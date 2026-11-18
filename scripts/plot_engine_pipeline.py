"""Figure for the paper's simulation-engine section: the workflow of one design.

    python scripts/plot_engine_pipeline.py [run_dir]  -> results/engine/pipeline.png

Top row, shared: (a) a laser-cut file is read into (b) a KiriSMILES string, which is
checked and keyed, and (c) the database is searched for a stored result.
Middle row, simulate (only when nothing is stored): (d) mesh, (e) FEniCSx solve,
(f) stress against strain. Bottom row, fabricate: (g) check the stored values against a
target, (h) write the cut files, (i) cut and test; the measured curve is stored as an
experiment result.

Uses a stored run of the alternating pattern P5x40S1/2O0[V36@0,0]N20E4 on a
120 x 60 mm PET sheet (h = 0.127 mm). The run was made with the 'alternating'
generator, whose cuts have the same geometry key as the string (checked here). The cut
file of (a) is written by kirigami.export and read back by kirigami.importer, and the
script checks that the imported string has the same geometry key and that the database
lookup finds the run. The target in (g) is an example.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import LineCollection     # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kirigami import export as ex                     # noqa: E402
from kirigami import geometry as geo                  # noqa: E402
from kirigami import importer as im                   # noqa: E402
from kirigami import jobs                             # noqa: E402
from kirigami import kirismiles as ks                 # noqa: E402
from kirigami import spec as S                        # noqa: E402

CODE = "P5x40S1/2O0[V36@0,0]N20E4"
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/runs/alternating_long_1029_150259"
OUT = ROOT / "results" / "engine"
SHEET, EDGE, CUT, LINE, MUTED, INK = "#dfe8f3", "#5a6b80", "#1d2b3a", "#1f6fd1", "#8a99ab", "#1d2b3a"
LASER, ARROW = "#d62828", "#6b7785"
LANE = {"shared": ("#f3f5f8", "#5a6b80"), "sim": ("#eef4fc", "#1f6fd1"), "fab": ("#fdf1ea", "#d9622b")}
STRAIN_SHOWN, W_TARGET = 0.05, 2.0                     # example target of (g): w_max >= 2 mm at 5%
FW, FH = 7.0, 5.0                                      # inches
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})


def box(x, y, w, h):
    return [x / FW, y / FH, w / FW, h / FH]


def stage(fig, x, y, w, h, title, lane, dashed=False):
    face, edge = LANE[lane]
    fig.add_artist(FancyBboxPatch((x / FW, y / FH), w / FW, h / FH, transform=fig.transFigure,
                                  boxstyle="round,pad=0,rounding_size=0.012", facecolor=face,
                                  edgecolor=edge, lw=0.9, ls=(0, (3, 2)) if dashed else "-",
                                  zorder=-10))
    fig.text((x + 0.08) / FW, (y + h - 0.08) / FH, title, ha="left", va="top", fontsize=7.5,
             fontweight="bold", color=INK)


def arrow(fig, pts, label=None, at=0.5, dy=0.06, ha="center", color=ARROW, ls="-"):
    """Polyline arrow through points given in inches; label near segment `at` of the path."""
    pts = [(x / FW, y / FH) for x, y in pts]
    for p, q in zip(pts[:-2], pts[1:-1]):
        fig.add_artist(plt.Line2D([p[0], q[0]], [p[1], q[1]], transform=fig.transFigure,
                                  color=color, lw=1.0, ls=ls))
    fig.add_artist(matplotlib.patches.FancyArrowPatch(pts[-2], pts[-1], transform=fig.transFigure,
                                                      arrowstyle="-|>", mutation_scale=8,
                                                      color=color, lw=1.0, ls=ls))
    if label:
        i = min(int(at), len(pts) - 2)
        f = at - i
        lx = pts[i][0] + f * (pts[i + 1][0] - pts[i][0])
        ly = pts[i][1] + f * (pts[i + 1][1] - pts[i][1])
        fig.text(lx, ly + dy / FH, label, ha=ha, va="bottom", fontsize=6.3, color="#3d4b5c")


def clean(ax):
    ax.set_axis_off()
    ax.patch.set_alpha(0)


def main():
    spec = json.loads((RUN / "spec.json").read_text())
    L, W = spec["sheet"]["L"], spec["sheet"]["W"]
    p = spec["pattern"]["params"]
    key = ks.geometry_key(CODE, L, W)
    assert key == ks.geometry_key(
        geo.alternating_pattern(L, W, p["b"], p["g"], p["d"], p["n_rows"]), L, W), "run != string"
    assert ks.is_valid(CODE, L, W)
    pat = ks.pattern(CODE, L, W)
    with tempfile.TemporaryDirectory() as tmp:          # (a) -> (b): write the file, read it back
        svg = Path(tmp) / "alternating.svg"
        ex.write_svg(ex.cut_paths(L, W, pat), L, W, svg)
        back = im.import_file(svg)
    assert ks.geometry_key(back["code"], back["L"], back["W"]) == key, "import changed the design"
    found = [r for r in jobs.find_runs(S.load(RUN / "spec.json"))
             if r["name"] == RUN.name and r["completed"] and r["same_settings"]]
    assert found, "database lookup did not find the run"

    d = np.load(RUN / "history.npz")
    x, cells, U, F, D = d["x"], d["cells"], d["U"], d["F"], d["Delta"]
    Dc, h = float(d["Delta_c"]), float(d["h"])
    sigma = F / (W * 1e-3 * h * 1e-3) / 1e3              # nominal stress F/(W h), kPa
    k = int(np.argmin(np.abs(D / L - STRAIN_SHOWN)))
    wmax = float(np.max(np.abs(U[k][:, 2])))
    k10 = int(np.argmin(np.abs(D / L - 0.10)))
    cuts = pat["cuts"]

    fig = plt.figure(figsize=(FW, FH))
    cw, gap = 2.12, 0.32                                 # stage width, gap between stages
    X = [0.0, cw + gap, 2 * (cw + gap)]
    rows = {"shared": (3.55, 1.35), "sim": (1.85, 1.45), "fab": (0.05, 1.35)}

    # ---- lane tags
    for lane, txt in (("shared", "design"), ("sim", "simulate"), ("fab", "fabricate")):
        y, hh = rows[lane]
        fig.text(-0.12 / FW, (y + hh / 2) / FH, txt, rotation=90, ha="center", va="center",
                 fontsize=7.5, color=LANE[lane][1], fontweight="bold")

    # ---- (a) cut file
    y, hh = rows["shared"]
    stage(fig, X[0], y, cw, hh, "(a) cut file (SVG, DXF)", "shared")
    ax = fig.add_axes(box(X[0] + 0.12, y + 0.18, cw - 0.24, hh - 0.5))
    ax.add_patch(Rectangle((0, 0), L, W, facecolor="white", edgecolor=LASER, lw=0.5))
    ax.add_collection(LineCollection([np.asarray(c) for c in cuts], colors=LASER, lw=0.5))
    ax.set(xlim=(-2, L + 2), ylim=(-2, W + 2), aspect="equal")
    clean(ax)
    fig.text((X[0] + cw / 2) / FW, (y + 0.06) / FH, "alternating.svg, laser-cut lines",
             ha="center", va="bottom", fontsize=6.3, color="#3d4b5c")

    # ---- (b) string
    stage(fig, X[1], y, cw, hh, "(b) KiriSMILES string", "shared")
    cx = X[1] + cw / 2
    fig.text(cx / FW, (y + 0.78) / FH, CODE, ha="center", va="center", family="monospace",
             fontsize=7.3, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#c9d2dd", lw=0.6))
    fig.text(cx / FW, (y + 0.42) / FH, "checked: valid", ha="center", va="center", fontsize=6.6,
             color="#2b8a3e")
    fig.text(cx / FW, (y + 0.2) / FH, f"geometry key {key}", ha="center", va="center",
             fontsize=6.6, color="#3d4b5c", family="monospace")

    # ---- (c) database
    stage(fig, X[2], y, cw, hh, "(c) database lookup", "shared")
    ax = fig.add_axes(box(X[2] + 0.14, y + 0.17, 0.62, 0.86))
    ax.set(xlim=(0, 1), ylim=(0, 1.3))
    for yy in (0.25, 0.55, 0.85):
        ax.add_patch(Ellipse((0.5, yy), 0.9, 0.28, facecolor="white", edgecolor=EDGE, lw=0.8,
                             zorder=1))
    ax.add_patch(Rectangle((0.05, 0.25), 0.9, 0.6, facecolor="white", edgecolor="none", zorder=2))
    for xx in (0.05, 0.95):
        ax.plot([xx, xx], [0.25, 0.85], color=EDGE, lw=0.8, zorder=3)
    ax.add_patch(Ellipse((0.5, 0.85), 0.9, 0.28, facecolor="#e7ecf2", edgecolor=EDGE, lw=0.8,
                         zorder=4))
    ax.add_patch(matplotlib.patches.Arc((0.5, 0.55), 0.9, 0.28, theta1=180, theta2=360,
                                        edgecolor=EDGE, lw=0.6, zorder=4))
    ax.add_patch(matplotlib.patches.Arc((0.5, 0.25), 0.9, 0.28, theta1=180, theta2=360,
                                        edgecolor=EDGE, lw=0.8, zorder=4))
    clean(ax)
    tx = X[2] + 0.86
    for i, t in enumerate(["same geometry key,", "sheet, material", "and loading?"]):
        fig.text(tx / FW, (y + 0.86 - 0.17 * i) / FH, t, ha="left", va="center", fontsize=6.6,
                 color=INK)
    fig.text(tx / FW, (y + 0.24) / FH, "found: 1 stored run", ha="left", va="center",
             fontsize=6.4, color="#2b8a3e")

    # ---- (d) mesh
    y, hh = rows["sim"]
    stage(fig, X[0], y, cw, hh, "(d) mesh (gmsh)", "sim")
    win = (52, 68, 22, 38)
    ax = fig.add_axes(box(X[0] + 0.22, y + 0.12, 1.02, hh - 0.42))
    keep = np.all((x[cells, 0] > win[0] - 2) & (x[cells, 0] < win[1] + 2) &
                  (x[cells, 1] > win[2] - 2) & (x[cells, 1] < win[3] + 2), axis=1)
    ax.triplot(x[:, 0], x[:, 1], cells[keep], color=LINE, lw=0.22)
    ax.add_collection(LineCollection([np.asarray(c) for c in cuts], colors=CUT, lw=1.1))
    ax.set(xlim=win[:2], ylim=win[2:], aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#9fb3c8"); s.set_linewidth(0.6)
    fig.text((X[0] + cw - 0.08) / FW, (y + 0.1) / FH, f"{len(cells):,} triangles\n(detail)".replace(",", r"$\,$"),
             ha="right", va="bottom", fontsize=6.1, color="#3d4b5c")

    # ---- (e) FEM
    stage(fig, X[1], y, cw, hh, "(e) FEM solve (FEniCSx)", "sim")
    P = np.column_stack([x, np.zeros(len(x))]) + U[k]
    ax = fig.add_axes(box(X[1] - 0.08, y - 0.12, cw - 0.15, hh + 0.02), projection="3d")
    wz = U[k][:, 2]
    lim = np.max(np.abs(wz))
    tri = ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=cells, cmap="RdBu_r",
                          vmin=-lim, vmax=lim, linewidth=0, antialiased=False, shade=False)
    tri.set_array(wz[cells].mean(axis=1))
    ax.set_box_aspect((L + D[k], W, 22), zoom=1.35)
    ax.view_init(elev=28, azim=-62)
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0, 0))
    cax = fig.add_axes(box(X[1] + cw - 0.2, y + 0.22, 0.05, 0.72))
    cb = fig.colorbar(tri, cax=cax)
    cax.set_title("$w$ (mm)", fontsize=6, pad=2)
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
    fig.text((X[1] + 0.1) / FW, (y + 0.1) / FH, rf"$\Delta/L={D[k] / L:.2f}$", ha="left",
             va="bottom", fontsize=6.3, color="#3d4b5c")

    # ---- (f) stress-strain
    stage(fig, X[2], y, cw, hh, "(f) stress–strain", "sim")
    ax = fig.add_axes(box(X[2] + 0.48, y + 0.33, cw - 0.62, hh - 0.66))
    ax.plot(D / L * 100, sigma, color=LINE, lw=1.4)
    ax.axvline(Dc / L * 100, color=MUTED, lw=0.7, ls="--")
    ax.annotate("buckling", xy=(Dc / L * 100, sigma.max() * 0.3), xytext=(5, sigma.max() * 0.12),
                fontsize=6, color="#3d4b5c", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set(xlim=(0, D.max() / L * 100), ylim=(0, None))
    ax.set_xlabel(r"$\Delta/L$ (%)", fontsize=6.5, labelpad=1)
    ax.set_ylabel(r"$\sigma$ (kPa)", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.grid(color="#dde3ea", lw=0.4)
    ax.set_facecolor((0, 0, 0, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- (g) check values (right), (h) cut files (middle), (i) cut and test (left)
    y, hh = rows["fab"]
    stage(fig, X[2], y, cw, hh, "(g) check values", "fab")
    lines = [(rf"$w_{{\max}}$ at $\Delta/L$ = {STRAIN_SHOWN:.0%}", f"{wmax:.1f} mm"),
             (r"$\sigma$ at $\Delta/L$ = 10%", f"{sigma[k10]:.1f} kPa"),
             (r"buckling at $\Delta_c$", f"{Dc:.2f} mm")]
    for i, (a, b) in enumerate(lines):
        yy = y + 0.92 - 0.2 * i
        fig.text((X[2] + 0.14) / FW, yy / FH, a, ha="left", va="center", fontsize=6.4, color=INK)
        fig.text((X[2] + cw - 0.14) / FW, yy / FH, b, ha="right", va="center", fontsize=6.4,
                 color=INK)
    ok = wmax >= W_TARGET
    fig.text((X[2] + cw / 2) / FW, (y + 0.2) / FH,
             rf"target $w_{{\max}}\geq{W_TARGET:.0f}$ mm: " + ("met" if ok else "not met"),
             ha="center", va="center", fontsize=6.4, color="#2b8a3e" if ok else LASER,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#c9d2dd", lw=0.6))

    stage(fig, X[1], y, cw, hh, "(h) fabrication files", "fab")
    chips = [("SVG", "laser cutter"), ("DXF", "laser cutter, CAD"), ("STL", "3D printer")]
    for i, (a, b) in enumerate(chips):
        yy = y + 0.88 - 0.24 * i
        fig.text((X[1] + 0.2) / FW, yy / FH, a, ha="left", va="center", fontsize=6.4,
                 family="monospace", fontweight="bold", color=LANE["fab"][1],
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor=LANE["fab"][1], lw=0.6))
        fig.text((X[1] + 0.68) / FW, yy / FH, b, ha="left", va="center", fontsize=6.4, color=INK)
    fig.text((X[1] + cw / 2) / FW, (y + 0.12) / FH, "cuts as slots of the kerf width",
             ha="center", va="center", fontsize=6.1, color="#3d4b5c")

    stage(fig, X[0], y, cw, hh, "(i) cut and test", "fab", dashed=True)
    for i, t in enumerate(["cut the sheet, stretch it", "and measure", "",
                           "the measured curve can be stored", "as an experiment result"]):
        fig.text((X[0] + cw / 2) / FW, (y + 0.9 - 0.15 * i) / FH, t, ha="center", va="center",
                 fontsize=6.4, color=INK if i < 2 else "#3d4b5c")

    # ---- arrows
    ys, hs = rows["shared"]
    ym, hm = rows["sim"]
    yf, hf = rows["fab"]
    mid = lambda yy, hh: yy + hh / 2
    arrow(fig, [(X[0] + cw, mid(ys, hs)), (X[1], mid(ys, hs))], "read", dy=0.04)
    arrow(fig, [(X[1] + cw, mid(ys, hs)), (X[2], mid(ys, hs))], "find", dy=0.04)
    yc = (ys + ym + hm) / 2                                          # gap between rows 1 and 2
    arrow(fig, [(X[2] + 0.5, ys), (X[2] + 0.5, yc), (X[0] + cw / 2, yc), (X[0] + cw / 2, ym + hm)],
          "not stored: simulate", at=1.5, dy=0.03)
    arrow(fig, [(X[2] + cw - 0.4, ys), (X[2] + cw - 0.4, ym + hm)], "  stored", at=0.5, dy=-0.03,
          ha="left")
    arrow(fig, [(X[0] + cw, mid(ym, hm)), (X[1], mid(ym, hm))], "solve", dy=0.04)
    arrow(fig, [(X[1] + cw, mid(ym, hm)), (X[2], mid(ym, hm))], "store", dy=0.04)
    arrow(fig, [(X[2] + cw / 2, ym), (X[2] + cw / 2, yf + hf)])
    arrow(fig, [(X[2], mid(yf, hf)), (X[1] + cw, mid(yf, hf))], "met", dy=0.04)
    arrow(fig, [(X[1], mid(yf, hf)), (X[0] + cw, mid(yf, hf))], "cut", dy=0.04)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", OUT / "pipeline.png", f"(step {k}, Delta/L = {D[k] / L:.3f}, Dc = {Dc:.4f} mm, "
          f"h = {h} mm, w_max = {wmax:.2f} mm, sigma(10%) = {sigma[k10]:.1f} kPa, "
          f"triangles = {len(cells)}, imported key = {key})")


if __name__ == "__main__":
    main()
