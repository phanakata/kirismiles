"""Figure for the paper's simulation-engine section: one string through the engine.

    python scripts/plot_engine_pipeline.py [run_dir]  -> results/engine/pipeline.png

Uses a stored run of the alternating pattern P5x40S1/2O0[V36@0,0]N20E4 on a
120 x 60 mm PET sheet (h = 0.127 mm). The run was made with the 'alternating'
generator, whose cuts have the same geometry key as the string (checked here).
Panels: (a) the pattern drawn from the string, (b) the mesh near a few slits,
(c) the deformed sheet, coloured by the deflection w, (d) force against strain.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import LineCollection     # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kirigami import geometry as geo                  # noqa: E402
from kirigami import kirismiles as ks                 # noqa: E402

CODE = "P5x40S1/2O0[V36@0,0]N20E4"
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/runs/alternating_long_1029_150259"
OUT = ROOT / "results" / "engine"
SHEET, EDGE, CUT, LINE, MUTED = "#dfe8f3", "#5a6b80", "#1d2b3a", "#1f6fd1", "#8a99ab"
STRAIN_SHOWN = 0.05
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})


def clamps(ax, L, W):
    for x0 in (-2.5, L):
        ax.add_patch(Rectangle((x0, -1.5), 2.5, W + 3, facecolor="#c3cddb", edgecolor=EDGE,
                               hatch="////", lw=0.6))


def main():
    spec = json.loads((RUN / "spec.json").read_text())
    L, W = spec["sheet"]["L"], spec["sheet"]["W"]
    p = spec["pattern"]["params"]
    assert ks.geometry_key(CODE, L, W) == ks.geometry_key(
        geo.alternating_pattern(L, W, p["b"], p["g"], p["d"], p["n_rows"]), L, W), "run != string"
    d = np.load(RUN / "history.npz")
    x, cells, U, F, D = d["x"], d["cells"], d["U"], d["F"], d["Delta"]
    Dc, h = float(d["Delta_c"]), float(d["h"])
    cuts = ks.pattern(CODE, L, W)["cuts"]

    # one row: (a) pattern, (b) mesh detail, (c) deformed sheet, (d) force
    FW, FH = 7.0, 1.8                                           # inches
    box = lambda x, y, w, h: [x / FW, y / FH, w / FW, h / FH]   # axes placed in inches
    fig = plt.figure(figsize=(FW, FH))

    # (a) pattern from the string
    ax = fig.add_axes(box(0.0, 0.2, 2.3, 1.35))
    ax.add_patch(Rectangle((0, 0), L, W, facecolor=SHEET, edgecolor=EDGE, lw=0.6))
    clamps(ax, L, W)
    ax.add_collection(LineCollection([np.asarray(c) for c in cuts], colors=CUT, lw=0.8))
    win = (52, 68, 22, 38)
    ax.add_patch(Rectangle((win[0], win[2]), win[1] - win[0], win[3] - win[2], fill=False,
                           edgecolor="#e8590c", lw=1.0))
    ax.set(xlim=(-4, L + 4), ylim=(-3, W + 3), aspect="equal")
    ax.set_axis_off()
    ax.set_title("(a) " + CODE, family="monospace", fontsize=6.5, loc="left")

    # (b) mesh near a few slits
    ax = fig.add_axes(box(2.42, 0.2, 0.8, 1.35))
    keep = np.all((x[cells, 0] > win[0] - 2) & (x[cells, 0] < win[1] + 2) &
                  (x[cells, 1] > win[2] - 2) & (x[cells, 1] < win[3] + 2), axis=1)
    ax.triplot(x[:, 0], x[:, 1], cells[keep], color=LINE, lw=0.25)
    ax.add_collection(LineCollection([np.asarray(c) for c in cuts], colors=CUT, lw=1.2))
    ax.set(xlim=win[:2], ylim=win[2:], aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#e8590c")
    ax.set_title("(b) mesh", fontsize=7.5, loc="left")

    # (c) deformed sheet
    k = int(np.argmin(np.abs(D / L - STRAIN_SHOWN)))
    P = np.column_stack([x, np.zeros(len(x))]) + U[k]
    ax = fig.add_axes(box(3.1, -0.05, 1.95, 1.75), projection="3d")
    wz = U[k][:, 2]
    lim = np.max(np.abs(wz))
    tri = ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=cells, cmap="RdBu_r",
                          vmin=-lim, vmax=lim, linewidth=0, antialiased=False, shade=False)
    tri.set_array(wz[cells].mean(axis=1))
    ax.set_box_aspect((L + D[k], W, 22), zoom=1.45)
    ax.view_init(elev=28, azim=-62)
    ax.set_axis_off()
    cax = fig.add_axes(box(5.0, 0.35, 0.06, 0.9))
    cb = fig.colorbar(tri, cax=cax)
    cax.set_title("$w$ (mm)", fontsize=6.5, pad=3)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(rf"(c) $\Delta/L={D[k] / L:.2f}$", fontsize=7.5, loc="left", y=0.86)

    # (d) force against strain
    ax = fig.add_axes(box(6.0, 0.4, 0.95, 1.1))
    ax.plot(D / L * 100, F * 1e3, color=LINE, lw=1.5)
    ax.axvline(Dc / L * 100, color=MUTED, lw=0.8, ls="--")
    ax.annotate("buckling", xy=(Dc / L * 100, F.max() * 1e3 * 0.35),
                xytext=(5, F.max() * 1e3 * 0.15), fontsize=6.5, color="#3d4b5c",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set(xlabel=r"$\Delta/L$ (%)", ylabel="force $F$ (mN)",
           xlim=(0, D.max() / L * 100), ylim=(0, None))
    ax.set_title("(d)", fontsize=7.5, loc="left")
    ax.grid(color="#e3e8ee", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", OUT / "pipeline.png", f"(step {k}, Delta/L = {D[k] / L:.3f}, Dc = {Dc:.4f} mm, "
          f"h = {h} mm, max |w| = {lim:.2f} mm, F_end = {F[-1] * 1e3:.1f} mN)")


if __name__ == "__main__":
    main()
