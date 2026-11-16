"""Figure for the rotating-squares check of BENCHMARK.md.

Reads the runs of specs/rs_thickness/ from results/runs/ and writes
results/rotating_squares/rotating_squares.png:
  (a) the in-plane biaxial run at its last step, from above;
  (b) panel rotation against strain of the cut band, for several hinge widths,
      with the rigid rotating-squares relation;
  (c) the buckling stretch of the clamped runs against thickness.

Usage: python scripts/plot_rotating_squares.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
OUT = Path(__file__).resolve().parents[1] / "results" / "rotating_squares"
INK, MUTED = "#0b0b0b", "#52514e"
A, NX = 8.0, 8                    # panel size and number of panel columns of the specs


def load(name):
    d = np.load(RUNS / name / "history.npz", allow_pickle=True)
    s = json.loads((RUNS / name / "summary.json").read_text())
    return d, s


def panels(d):
    """Masks of the nodes in the core of each panel, and the band's first column x."""
    X = d["x"][:, :2]
    L, W = float(d["L"]), float(d["W"])
    ny, x0 = int(round(W / A)), L / 2 - NX * A / 2
    m = {(i, j): np.all(np.abs(X - [x0 + (i + .5) * A, (j + .5) * A]) < A / 2 - 1.2, axis=1)
         for i in range(NX) for j in range(ny)}
    return m, ny


def band_strain_and_rotation(d):
    """Per load step: strain of the band along x (from panel centres) and the mean
    |in-plane rotation| of the inner panels (least-squares fit)."""
    X, U = d["x"][:, :2], d["U"]
    m, ny = panels(d)
    eps, rot = [], []
    for k in range(len(d["Delta"])):
        Y = X + U[k, :, :2]
        e = [(Y[m[NX - 1, j]].mean(0)[0] - Y[m[0, j]].mean(0)[0]) /
             (X[m[NX - 1, j]].mean(0)[0] - X[m[0, j]].mean(0)[0]) - 1 for j in range(ny)]
        th = []
        for i in range(2, NX - 2):
            for j in range(2, ny - 2):
                P = X[m[i, j]] - X[m[i, j]].mean(0)
                Q = Y[m[i, j]] - Y[m[i, j]].mean(0)
                H = P.T @ Q
                th.append(abs(np.degrees(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))))
        eps.append(np.mean(e))
        rot.append(np.mean(th))
    return np.array(eps), np.array(rot)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})

    # (a) deformed sheet, in-plane biaxial, hinge 2 mm
    d, _ = load("cho_l1_biaxial_inplane")
    X, U, cells = d["x"][:, :2], d["U"][-1, :, :2], d["cells"][:, :3]
    Y = X + U
    ax[0].add_collection(PolyCollection(Y[cells], facecolor="#cfd8e3", edgecolor="none"))
    ax[0].plot(*np.c_[[0, 0], [float(d["L"]), 0], [float(d["L"]), float(d["W"])],
                      [0, float(d["W"])], [0, 0]], color=MUTED, lw=0.8, ls="--")
    ax[0].autoscale()
    ax[0].set_aspect("equal")
    ax[0].set_axis_off()
    ax[0].set_title(f"(a) equal-biaxial, in-plane, Δ = {d['Delta'][-1]:.0f} mm\n"
                    "dashed: undeformed sheet", fontsize=10)

    # (b) rotation vs band strain
    th = np.linspace(0, 45, 200)
    ax[1].plot(np.cos(np.radians(th)) + np.sin(np.radians(th)) - 1, th, color=MUTED, ls="--",
               label="rigid squares, point hinges")
    for name, lab, c in [("cho_l1_biaxial_inplane", "hinge 2 mm", "#1f4e79"),
                         ("cho_l1_biaxial_inplane_t3", "hinge 1.5 mm", "#3b7dbf"),
                         ("cho_l1_biaxial_inplane_t2", "hinge 1 mm", "#8fb8e3"),
                         ("cho_l1_biaxial_inplane_t1", "hinge 0.5 mm", "#c9ddf2")]:
        if (RUNS / name / "summary.json").exists():
            e, r = band_strain_and_rotation(load(name)[0])
            ax[1].plot(e, r, color=c, lw=2, label=lab)
    ax[1].set(xlabel="strain of the cut band", ylabel="panel rotation (deg)", xlim=(0, 0.5),
              ylim=(0, 45))
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].set_title("(b) in-plane biaxial: rotation vs strain", fontsize=10)

    # (c) buckling stretch vs thickness, clamped
    h, Dc = [], []
    for name in ["cho_l1_h0p1", "cho_l1_h0p5", "cho_l1_h2"]:
        dd, s = load(name)
        h.append(float(dd["h"]))
        Dc.append(s["Delta_c_mm"])
    h, Dc = np.array(h), np.array(Dc)
    c = np.exp(np.mean(np.log(Dc / h**2)))
    hh = np.geomspace(0.07, 3, 50)
    ax[2].loglog(hh, c * hh**2, color=MUTED, ls="--", label=f"Δ_c = {c:.2f} h²  (mm)")
    ax[2].loglog(h, Dc, "o", color="#1f4e79", ms=7, label="linear buckling, clamped")
    ax[2].set(xlabel="thickness h (mm)", ylabel="buckling stretch Δ_c (mm)")
    ax[2].legend(frameon=False, fontsize=8)
    ax[2].set_title("(c) clamped: out-of-plane buckling vs thickness", fontsize=10)

    for a in ax[1:]:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "rotating_squares.png", dpi=150)
    print("wrote", OUT / "rotating_squares.png")


def rotation_and_tilt(d):
    """Per load step: band strain, mean |in-plane rotation| and mean out-of-plane tilt of
    the inner panels, from a 3D rigid fit (Kabsch) of each panel core (degrees)."""
    X = np.column_stack([d["x"][:, :2], np.zeros(len(d["x"]))])
    m, ny = panels(d)
    out = []
    for k in range(len(d["Delta"])):
        Y = X + d["U"][k]
        e = np.mean([(Y[m[NX - 1, j]].mean(0)[0] - Y[m[0, j]].mean(0)[0]) /
                     (X[m[NX - 1, j]].mean(0)[0] - X[m[0, j]].mean(0)[0]) - 1 for j in range(ny)])
        rot, tilt = [], []
        for i in range(2, NX - 2):
            for j in range(2, ny - 2):
                P = X[m[i, j]] - X[m[i, j]].mean(0)
                Q = Y[m[i, j]] - Y[m[i, j]].mean(0)
                u, _, vt = np.linalg.svd(P.T @ Q)
                s = np.eye(3)
                s[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
                Rm = vt.T @ s @ u.T
                tilt.append(np.degrees(np.arccos(min(1.0, abs((Rm @ [0, 0, 1.0])[2])))))
                ex = Rm @ [1.0, 0, 0]
                rot.append(abs(np.degrees(np.arctan2(ex[1], ex[0]))))
        out.append((e, np.mean(rot), np.mean(tilt)))
    return np.array(out)


def fig_paper():
    """Rotation against band strain (in-plane, equal-biaxial, four hinge widths) and rotation
    and tilt under clamped stretching for three thicknesses, without titles (for the paper)."""
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    with plt.rc_context({"font.size": 8}):
        fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.0), gridspec_kw=dict(wspace=0.28))
        th = np.linspace(0, 45, 200)
        ax[0].plot(np.cos(np.radians(th)) + np.sin(np.radians(th)) - 1, th, color=MUTED, ls="--",
                   lw=1, label="rigid squares")
        for name, lab, c in [("cho_l1_biaxial_inplane", "hinge 2 mm", "#0b2f5c"),
                             ("cho_l1_biaxial_inplane_t3", "1.5 mm", "#2a78d6"),
                             ("cho_l1_biaxial_inplane_t2", "1 mm", "#79aee6"),
                             ("cho_l1_biaxial_inplane_t1", "0.5 mm", "#b9d3f1")]:
            e, r = band_strain_and_rotation(load(name)[0])
            ax[0].plot(e, r, color=c, lw=1.6, label=lab)
        ax[0].set(xlabel="band strain", ylabel=r"panel rotation $|\theta|$ (deg)", xlim=(0, 0.5),
                  ylim=(0, 45))
        ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
        ax[0].set_title("(a)", loc="left", fontweight="bold")
        for name, c in (("cho_l1_h0p1", AQUA), ("cho_l1_h0p5", ORANGE), ("cho_l1_h2", BLUE)):
            d, _ = load(name)
            o = np.vstack([np.zeros(3), rotation_and_tilt(d)])      # start at the flat sheet
            h = float(d["h"])
            ax[1].plot(o[:, 0], o[:, 1], "-", color=c, lw=1.6, label=f"$h$ = {h:g} mm, rotation")
            ax[1].plot(o[:, 0], o[:, 2], "--", color=c, lw=1.2, label=f"$h$ = {h:g} mm, tilt")
        ax[1].set(xlabel="band strain", ylabel="angle (deg)", xlim=(0, 0.4), ylim=(0, 30))
        ax[1].legend(frameon=False, fontsize=6, loc="upper left", ncol=1)
        ax[1].set_title("(b)", loc="left", fontweight="bold")
        for a in ax:
            a.spines[["top", "right"]].set_visible(False)
        fig.savefig(OUT / "fig_rotating_squares.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("wrote", OUT / "fig_rotating_squares.png")


if __name__ == "__main__":
    main()
    fig_paper()
