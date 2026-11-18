"""Paper figure of the kirigami shell (Rafsanjani et al. PNAS 2019 benchmark):
(a) the 8-row clamped shell, rolled and stretched, coloured by radial offset;
(b) stress against applied strain, ours (4 and 8 rows, clamped) against the
experiment and the rolled-unit-cell FE of that work (digitised).

  python scripts/plot_tube_paper.py   ->  results/tube/fig_shell.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "scripts")
import plot_tube_triangular as P  # noqa: E402

RUN8 = "results/tube/tri_n8_rows8_d0.75_k0.1_medium"
RUN4 = "results/tube/meshconv/rows4_medium"
OUT = "results/tube/fig_shell.png"
VMAX = 2.0                                           # mm, colour range of the radial offset
SHOW = [("rolled", "rolled"), ("eps0.0400", r"$\bar\varepsilon=0.04$"),
        ("eps0.1000", r"$\bar\varepsilon=0.10$")]


def main():
    z = np.load(RUN8 + ".npz")
    X, tri = z["X"], z["cells"][:, :3]
    Ps = [X + z[k] for k, _ in SHOW]
    allP = np.vstack(Ps)
    lim = [(allP[:, i].min(), allP[:, i].max()) for i in range(3)]
    ref = P.reference()
    with plt.rc_context({"font.size": 8}):
        FW, H = 7.0, 2.9
        fig = plt.figure(figsize=(FW, H))
        w3, S3 = 1.2, 2.35                      # slot width; square 3D axes, overlapping
        for j, (Pj, (_, ttl)) in enumerate(zip(Ps, SHOW)):
            cx = 0.1 + (j + 0.5) * w3
            ax = fig.add_axes([(cx - S3 / 2) / FW, 0.3 / H, S3 / FW, S3 / H], projection="3d",
                              proj_type="ortho")
            ax.patch.set_alpha(0)
            r = np.hypot(Pj[:, 0], Pj[:, 1])
            off = (r - r.mean())[tri].mean(1)
            col = plt.cm.coolwarm(np.clip(0.5 + 0.5 * off / VMAX, 0, 1))
            ax.add_collection3d(Poly3DCollection(Pj[tri], facecolors=col, edgecolors=col,
                                                 linewidths=0.05))
            (x0, x1), (y0, y1), (z0, z1) = lim
            ax.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(z0, z1))
            ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=1.0)
            ax.view_init(elev=12, azim=-60)
            ax.set_axis_off()
            fig.text(cx / FW, (H - 0.22) / H, ttl, ha="center", va="bottom")
        cax = fig.add_axes([0.75 / FW, 0.22 / H, 2.3 / FW, 0.07 / H])
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-VMAX, VMAX))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("radial offset (mm)", fontsize=7, labelpad=1)
        cb.set_ticks([-2, -1, 0, 1, 2]); cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
        fig.text(0.0, (H - 0.02) / H, "(a)", fontweight="bold", va="top")

        axb = fig.add_axes([4.35 / FW, 0.5 / H, 2.6 / FW, 2.1 / H])
        axb.plot(*ref["experiment"], color="#2ca02c", lw=1.8, label="experiment")
        axb.plot(*ref["unit_cell_FEM"], color="k", ls="-.", lw=1.1, label="FE, rolled unit cell")
        for run, lab, c, ls in [(RUN4, "ours, 4 rows clamped", "#2a6fdb", "--"),
                                (RUN8, "ours, 8 rows clamped", "#7b3fbf", "-")]:
            e, s = P.curve(json.load(open(run + ".json")))
            axb.plot(e, s, color=c, ls=ls, lw=1.6, label=lab)
        axb.set(xlabel=r"applied strain $\bar\varepsilon$", ylabel=r"nominal stress $\sigma$ (kPa)",
                xlim=(0, 0.12), ylim=(0, 125))
        axb.legend(frameon=False, fontsize=6.5, loc="lower right")
        axb.set_title("(b)", loc="left", fontweight="bold")
        axb.spines[["top", "right"]].set_visible(False)
        fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
