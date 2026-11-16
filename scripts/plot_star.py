"""Figure for the multistable star pattern (Yang et al. 2023), in-plane equal-biaxial runs.

Reads results/runs/yang_star_3x3_biaxial_inplane{_s028,,_s050}_nh/ and writes
results/star/fig_star.png:
  (a) the sheet with s/l = 0.039 at four stretches, from above: the cut outlines in grey and,
      as for the rotating squares, each triangle drawn as its rigid-body fit, coloured by the
      sense in which it turns;
  (b) force against nominal strain Delta/L for the three hinge widths;
  (c) mean rotation of the triangles around the four inner cell corners (rigid fits), with
      the range over those triangles.
It also prints the numbers quoted in the paper.

The triangles are found from the mesh: elements that have turned by more than 15 deg at the
last step, grouped by sense into connected pieces; a piece is a triangle if it has a vertex
within 1 mm of a cell corner and covers more than 5 mm^2 (48 triangles lie whole in the
sheet). Each triangle is drawn as the convex hull of its nodes, moved by the rotation and
translation that best match their displacements.

Usage: python scripts/plot_star.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "results" / "runs", ROOT / "results" / "star"
RUNS_S = [("_s028", 0.028, "#b9d3f1"), ("", 0.039, "#2a78d6"), ("_s050", 0.050, "#0b2f5c")]
PANEL_COLORS = ("#e8590c", "#1f6fd1")      # counterclockwise, clockwise (as for the squares)
GREY = "#a9a8a3"
CELL, PAD, NC = 14.0, 4.0, 3               # cell side, pad width, cells per side


def load(tag):
    return np.load(RUNS / f"yang_star_3x3_biaxial_inplane{tag}_nh" / "history.npz")


def rotations(d):
    """In-plane rotation (deg) of every element at every step."""
    X, C, U = d["x"][:, :2], d["cells"][:, :3], d["U"]
    a, b, c = X[C[:, 0]], X[C[:, 1]], X[C[:, 2]]
    inv = np.linalg.inv(np.stack([b - a, c - a], -1))
    out = []
    for k in range(len(d["Delta"])):
        P = X + U[k][:, :2]
        F = np.stack([P[C[:, 1]] - P[C[:, 0]], P[C[:, 2]] - P[C[:, 0]]], -1) @ inv
        out.append(np.degrees(np.arctan2(F[:, 1, 0] - F[:, 0, 1], F[:, 0, 0] + F[:, 1, 1])))
    return np.array(out)


def triangles(d):
    """List of (sense, node indices, hull of the reference nodes, nearest corner)."""
    X, C = d["x"][:, :2], d["cells"][:, :3]
    r = rotations(d)[-1]
    corners = np.array([(PAD + CELL * i, CELL * j) for i in range(NC + 1) for j in range(NC + 1)])
    out = []
    for sgn in (1, -1):
        m = np.where(sgn * r > 15)[0]
        e = np.sort(np.concatenate([C[m][:, [0, 1]], C[m][:, [1, 2]], C[m][:, [2, 0]]]), 1)
        owner = np.tile(np.arange(len(m)), 3)
        key = e[:, 0] * 10**7 + e[:, 1]
        o = np.argsort(key)
        same = np.where(key[o][1:] == key[o][:-1])[0]
        g = coo_matrix((np.ones(len(same)), (owner[o][same], owner[o][same + 1])),
                       shape=(len(m), len(m)))
        _, lab = connected_components(g, directed=False)
        for i in range(lab.max() + 1):
            nodes = np.unique(C[m[lab == i]])
            if len(nodes) < 10:
                continue
            P = X[nodes]
            h = ConvexHull(P)
            dist = np.linalg.norm(P[:, None] - corners[None], axis=2)
            if dist.min() < 1.0 and h.volume > 5.0:
                out.append((sgn, nodes, P[np.r_[h.vertices, h.vertices[0]]],
                            corners[np.unravel_index(dist.argmin(), dist.shape)[1]]))
    return out


def rigid(X, Y):
    """Rotation angle (rad) and centroids of the best rigid fit of points X onto Y."""
    cx, cy = X.mean(0), Y.mean(0)
    H = (X - cx).T @ (Y - cy)
    return np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]), cx, cy


def traced(d, tris, k):
    """Outlines of the fitted triangles at step k (None = flat) and their angles (deg)."""
    X = d["x"][:, :2]
    loops, ang = [], []
    for _, nodes, hull, _ in tris:
        Y = X[nodes] + (0 if k is None else d["U"][k][nodes, :2])
        t, cx, cy = rigid(X[nodes], Y)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        loops.append(cy + (hull - cx) @ R.T)
        ang.append(np.degrees(t))
    return loops, np.array(ang)


def boundary_edges(C):
    e = np.sort(np.concatenate([C[:, [0, 1]], C[:, [1, 2]], C[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    return u[cnt == 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for tag, s, _ in RUNS_S:
        d = load(tag)
        tris = triangles(d)
        inner = np.array([PAD + CELL <= c[0] <= PAD + 2 * CELL and CELL <= c[1] <= 2 * CELL
                          for *_, c in tris])
        th = np.array([np.abs(traced(d, tris, k)[1][inner]) for k in range(len(d["Delta"]))])
        data[s] = (d, tris, th)
        D, F = d["Delta"], d["F"]
        k0 = np.diff(F) / np.diff(D)
        print(f"s/l={s}: {len(tris)} triangles ({inner.sum()} inner); stiffness {k0[0]:.4f} N/mm, "
              f"min tangent {k0.min():.4f} at Delta/L={D[1:][k0.argmin()] / 50:.2f}, "
              f"F(25)={F[-1]:.2f} N; mean |theta| at Delta/L=0.2, 0.4, 0.5: "
              + ", ".join(f"{np.interp(e * 50, D, th.mean(1)):.1f}" for e in (0.2, 0.4, 0.5))
              + f" (range at end {th[-1].min():.1f}-{th[-1].max():.1f}); "
              f"senses {sorted(set(int(np.sign(a)) for a in traced(d, tris, len(D) - 1)[1]))}")

    with plt.rc_context({"font.size": 8}):
        d, tris, th = data[0.039]
        X, C, D, L = d["x"][:, :2], d["cells"][:, :3], d["Delta"], float(d["L"])
        B = boundary_edges(C)
        # (a): every frame gets its own tight limits but one scale (inches per mm), so the
        # frames are packed edge to edge and still drawn to the same scale
        frames = []
        for f in (0.0, 0.2, 0.4, 0.5):
            k = None if f == 0 else int(np.argmin(np.abs(D / L - f)))
            P = X if k is None else X + d["U"][k][:, :2]
            frames.append((k, P, P.min(0) - 0.3, P.max(0) + 0.3))
        FW, GAP, TTL = 7.0, 0.08, 0.2                                  # inches
        ext = np.array([hi - lo for *_, lo, hi in frames])
        scale = (FW - GAP * 3) / ext[:, 0].sum()                       # inches per mm
        ha = ext[:, 1].max() * scale                                   # height of row (a)
        HB, GB = 1.55, 0.5                                            # plots (b, c), gap
        H = TTL + ha + GB + HB + 0.42
        fig = plt.figure(figsize=(FW, H))
        x0 = 0.0
        for j, ((k, P, lo, hi), (ex, ey)) in enumerate(zip(frames, ext)):
            w, h = ex * scale, ey * scale
            ax = fig.add_axes([x0 / FW, (H - TTL - ha + (ha - h) / 2) / H, w / FW, h / H])
            x0 += w + GAP
            ax.add_collection(LineCollection(P[B], colors=GREY, linewidths=0.3))
            loops, ang = traced(d, tris, k)
            for (sgn, *_), lp in zip(tris, loops):
                ax.plot(*lp.T, color=PANEL_COLORS[0 if sgn > 0 else 1], lw=0.8)
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal")
            ax.set_axis_off()
            ttl = r"$\Delta=0$" if k is None else rf"$\Delta/L={D[k] / L:.1f}$, $|\theta|={th[k].mean():.0f}^\circ$"
            fig.text((x0 - GAP - w / 2) / FW, (H - TTL + 0.04) / H, ttl, ha="center",
                     va="bottom", fontsize=7.5)
        fig.text(0.0, (H - 0.02) / H, "(a)", fontweight="bold", va="top")

        yb = 0.42
        axb = fig.add_axes([0.5 / FW, yb / H, 2.85 / FW, HB / H])
        axc = fig.add_axes([4.05 / FW, yb / H, 2.85 / FW, HB / H])
        for tag, s, c in RUNS_S:
            d, _, th = data[s]
            e = np.r_[0, d["Delta"] / float(d["L"])]
            axb.plot(e, np.r_[0, d["F"]], color=c, lw=1.6, label=f"$s/l$ = {s:.3f}")
            axc.fill_between(e, np.r_[0, th.min(1)], np.r_[0, th.max(1)], color=c, alpha=0.2, lw=0)
            axc.plot(e, np.r_[0, th.mean(1)], color=c, lw=1.6, label=f"$s/l$ = {s:.3f}")
        axb.set(xlabel=r"$\Delta/L$", ylabel="force $F$ (N)", xlim=(0, 0.5), ylim=(0, None))
        axc.set(xlabel=r"$\Delta/L$", ylabel=r"triangle rotation $|\theta|$ (deg)", xlim=(0, 0.5),
                ylim=(0, 75))
        for a, lab in ((axb, "(b)"), (axc, "(c)")):
            a.legend(frameon=False, fontsize=6.5, loc="upper left")
            a.set_title(lab, loc="left", fontweight="bold")
            a.spines[["top", "right"]].set_visible(False)
        fig.savefig(OUT / "fig_star.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("wrote", OUT / "fig_star.png")


if __name__ == "__main__":
    main()
