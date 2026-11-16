"""Snapshot figures for the paper: the GIFs as rows of still frames.

    python scripts/plot_snapshots.py            -> results/snapshots/snapshots_{single,actuators,rotating}.png

Each row is one simulation, drawn at four applied displacements: the flat sheet
before stretching (Delta = 0), the stored steps nearest to 1/3 and 2/3 of the
largest displacement, and the last step. The rendering follows
scripts/make_gif.py: colour is the out-of-plane displacement w (blue down, red
up), black lines are the cuts, and, as in the GIFs, the sign of w is chosen so
that the largest deflection of the last step points up (buckling up or down is
equivalent by symmetry). Every panel of a row shares one scale and one view,
and w is drawn to scale unless the row label says it is exaggerated.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from make_gif import CMAP, INK, cut_edges                              # noqa: E402

OUT = ROOT / "results" / "snapshots"
CFG, RUNS = ROOT / "results" / "configs", ROOT / "results" / "runs"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans"})

# (file, row label, z exaggeration). Exaggeration only where w is small next to the sheet.
GROUPS = {
    "single": [
        (CFG / "single_cut.npz", "single cut", 2),
        (CFG / "two_cuts_ls10.npz", "two cuts, 10 mm apart", 3),
        (CFG / "two_cuts_ls30.npz", "two cuts, 30 mm apart", 3),
    ],
    "actuators": [
        (CFG / "pitch_concave_fine.npz", "pitch, concave", 1),
        (CFG / "pitch_convex.npz", "pitch, convex", 1),
        (CFG / "roll.npz", "roll", 1),
        (CFG / "yaw.npz", "yaw", 1),
        (CFG / "pitch_chain.npz", "pitch chain", 1),
    ],
    "rotating": [
        (RUNS / "cho_l1_biaxial_inplane" / "history.npz", "equal-biaxial,\nin plane (top view)", 1,
         dict(panels=(8.0, 8))),
        (RUNS / "cho_l1_h0p5" / "history.npz", "uniaxial, clamped ends", 1),
        (RUNS / "cho_l1_h2" / "history.npz", "uniaxial, clamped ends", 1),
    ],
}
FRACTIONS = (0.0, 1 / 3, 2 / 3, 1.0)
# three frames per design (flat, halfway, last) where four were redundant
GROUP_FRACTIONS = {"single": (0.0, 0.5, 1.0), "actuators": (0.0, 0.5, 1.0)}
EDGE = "#6e6d69"                           # outline of the sheet
RENDER_DPI = 300                           # resolution of the off-screen frames
PANEL_COLORS = ("#e8590c", "#1f6fd1")      # the two checkerboard sublattices of panels


def panel_cores(x, L, W, a, nx):
    """Rotating squares: for each panel (i, j), the mask of mesh nodes in its core
    (more than 1.2 mm inside the panel), as in scripts/plot_rotating_squares.py."""
    ny, x0 = int(round(W / a)), L / 2 - nx * a / 2
    return {(i, j): np.all(np.abs(x - [x0 + (i + .5) * a, (j + .5) * a]) < a / 2 - 1.2, axis=1)
            for i in range(nx) for j in range(ny)}


def fit_panels(x, U2, cores, a):
    """Rigid fit (centre, rotation) of every panel core; returns the outlines of the
    fitted squares of side a and the rotation of each panel in degrees (CCW > 0)."""
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * a / 2
    loops, angles = {}, {}
    for key, m in cores.items():
        P = x[m] - x[m].mean(0)
        Y = x[m] + U2[m]
        Q = Y - Y.mean(0)
        H = P.T @ Q
        t = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        loops[key] = Y.mean(0) + sq @ R.T
        angles[key] = np.degrees(t)
    return loops, angles


def pick(Delta, fractions=FRACTIONS):
    """Step indices for the fractions of the largest displacement; None = flat sheet."""
    out = []
    for f in fractions:
        if f == 0:
            out.append(None)
        else:
            out.append(int(np.argmin(np.abs(Delta - f * Delta.max()))))
    return out


def outer_edges(cells, cuts):
    """Boundary edges of the mesh that are not cut faces: the outline of the sheet."""
    e = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    b = uniq[count == 1]
    cut = {tuple(c) for c in np.sort(cuts, axis=1)}
    return np.array([q for q in b if tuple(q) not in cut])


def render(draw, elev, azim, box, zoom):
    """Draw one frame on its own 3D axes, off screen, and return it as an RGB array."""
    f = plt.figure(figsize=(4.0, 4.0), dpi=RENDER_DPI)
    ax = f.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    draw(ax)
    ax.set_box_aspect(box, zoom=zoom)
    ax.view_init(elev, azim)
    ax.set_axis_off()
    f.canvas.draw()
    img = np.asarray(f.canvas.buffer_rgba())[..., :3].copy()
    plt.close(f)
    return img


def crop_common(imgs, pad=6):
    """Crop every image of a row to the union of their non-white pixels, so that the
    frames keep one scale and lose the empty space of the 3D axes."""
    mask = np.zeros(imgs[0].shape[:2], bool)
    for im in imgs:
        mask |= (im < 245).any(axis=2)
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, mask.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, mask.shape[1])
    return [im[r0:r1, c0:c1] for im in imgs]


def row(path, label, zx, opts=None, fractions=FRACTIONS):
    """Frames of one simulation (cropped RGB arrays), their titles, the row note, and the
    largest |w| in mm for the colour bar (None for in-plane runs)."""
    d = np.load(path)
    x, cells, U, Delta = d["x"], d["cells"], d["U"].astype(float), d["Delta"]
    L, W = float(d["L"]), float(d["W"])
    X = np.column_stack([x, np.zeros(len(x))])
    w_true = np.abs(U[:, :, 2]).max()
    planar = w_true < 1e-3 * W
    sign = 1.0
    last = U[-1, :, 2]
    if not planar and last[np.argmax(np.abs(last))] < 0:
        sign = -1.0
    U[:, :, 2] *= sign * zx
    zlim = max(np.abs(U[:, :, 2]).max(), 1e-6)
    xs, ys = X[None, :, 0] + U[:, :, 0], X[None, :, 1] + U[:, :, 1]
    xmin, xmax = min(0.0, xs.min()), max(L, xs.max())
    ymin, ymax = min(0.0, ys.min()), max(W, ys.max())
    edges = cut_edges(x, cells, L, W)
    rim = outer_edges(cells, edges)
    elev, azim = (90, -90) if planar else (24, -58)
    box = (xmax - xmin, ymax - ymin, 2 * zlim if not planar else 1)
    imgs, titles = [], []
    for i in pick(Delta, fractions):
        P = X if i is None else X + U[i]
        D = 0.0 if i is None else Delta[i]
        title = rf"$\Delta/L={D / L:.3f}$" if D > 0 else r"$\Delta=0$"
        loops = None
        if opts and "panels" in opts:                 # trace the rigid panels
            a, nx = opts["panels"]
            cores = panel_cores(x, L, W, a, nx)
            loops, ang = fit_panels(x, np.zeros_like(x) if i is None else U[i][:, :2], cores, a)
            inner = [abs(v) for (pi, pj), v in ang.items()
                     if 0 < pi < nx - 1 and 0 < pj < int(round(W / a)) - 1]
            title += rf", $|\theta|={np.mean(inner):.0f}^\circ$"
            print(f"   frame Delta={D:.2f}: inner |theta| mean {np.mean(inner):.1f}, "
                  f"min {np.min(inner):.1f}, max {np.max(inner):.1f}; signs by sublattice "
                  f"{sorted(set((int(np.sign(round(v, 3))), (pi + pj) % 2) for (pi, pj), v in ang.items()))}")

        def draw(ax, P=P, loops=loops):
            rgba = CMAP(plt.Normalize(-zlim, zlim)(P[cells][:, :, 2].mean(axis=1)))
            # edges in the face colour hide the seams between triangles
            ax.add_collection3d(Poly3DCollection(P[cells], facecolors=rgba, edgecolors=rgba,
                                                 linewidths=0.15))
            ax.add_collection3d(Line3DCollection(P[edges], colors=INK, linewidths=0.6, zorder=5))
            if not planar:
                ax.add_collection3d(Line3DCollection(P[rim], colors=EDGE, linewidths=0.6, zorder=5))
            if loops:
                for (pi, pj), lp in loops.items():
                    ax.add_collection3d(Line3DCollection(
                        [np.column_stack([lp, np.zeros(len(lp))])],
                        colors=PANEL_COLORS[(pi + pj) % 2], linewidths=1.2, zorder=6))
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(-zlim, zlim))

        imgs.append(render(draw, elev, azim, box, 0.9 if not planar else 1.0))
        titles.append(title)
    note = f"{label}\n{L:g} × {W:g} mm\nh = {float(d['h']):g} mm"
    if zx > 1:
        note += f"\nw exaggerated ×{zx:g}"
    return crop_common(imgs), titles, note, (None if planar else zlim / zx)


def make(group):
    """Lay the cropped frames out in inches: a label column, the frames, a colour bar."""
    rows = GROUPS[group]
    fr = GROUP_FRACTIONS.get(group, FRACTIONS)
    nf = len(fr)
    data = [row(path, label, zx, opts[0] if opts else None, fr)
            for path, label, zx, *opts in rows]
    FW, LAB, CB, GAP, TTL, VGAP = 7.0, 1.25, 0.55, 0.06, 0.17, 0.08     # inches
    fw = (FW - LAB - CB - GAP * (nf - 1)) / nf                           # width of one frame
    heights = [fw * im[0].shape[0] / im[0].shape[1] for im, *_ in data]
    H = sum(h + TTL for h in heights) + VGAP * (len(data) - 1)
    fig = plt.figure(figsize=(FW, H))
    y = H
    for (imgs, titles, note, wmax), h in zip(data, heights):
        y -= TTL + h
        for j, (im, t) in enumerate(zip(imgs, titles)):
            ax = fig.add_axes([(LAB + j * (fw + GAP)) / FW, y / H, fw / FW, h / H])
            ax.imshow(im, interpolation="lanczos")
            ax.set_axis_off()
            ax.set_title(t, fontsize=7.5, pad=2)
        fig.text(0.02 / FW, (y + h / 2) / H, note, ha="left", va="center", fontsize=7,
                 linespacing=1.35)
        if wmax is not None:
            ch = min(h * 0.8, 1.1)
            cax = fig.add_axes([(FW - CB + 0.08) / FW, (y + (h - ch) / 2) / H, 0.07 / FW, ch / H])
            sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-wmax, wmax))
            cb = fig.colorbar(sm, cax=cax)
            cb.set_label("$w$ (mm)", fontsize=7, labelpad=1)
            cb.ax.tick_params(labelsize=6, length=2)
        y -= VGAP
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"snapshots_{group}.png", dpi=220, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"snapshots_{group}.png")
    for path, label, *_ in rows:
        d = np.load(path)
        idx = [i for i in pick(d["Delta"], fr) if i is not None]
        print(f"   {label}: Delta = {np.round(d['Delta'][idx], 3).tolist()} mm "
              f"(Dmax {d['Delta'].max():g}, Dc {float(d['Delta_c']):.4f}, |w|max {np.abs(d['U'][:, :, 2]).max():.2f})")


if __name__ == "__main__":
    for g in (sys.argv[1:] or GROUPS):
        make(g)
