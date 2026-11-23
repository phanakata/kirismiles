"""Render the deformation history of a configuration as an animated GIF.

Usage (host Python):  python scripts/make_gif.py <name> [<name> ...]
Reads results/configs/<name>.npz and writes results/configs/<name>.gif.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

R = "results/configs"
INK = "#1f1f1d"
# Diverging: down (blue) - flat (grey) - up (red), as in the paper's colour maps
CMAP = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#d9d8d4", "#e34948"])


def frames_index(n: int, max_frames: int = 45) -> list[int]:
    """Pick at most max_frames (>= 1) steps, always including the last one."""
    if max_frames < 1:
        raise ValueError(f"max_frames must be at least 1, not {max_frames}")
    if n <= max_frames:
        return list(range(n))
    return sorted({int(i) for i in np.linspace(0, n - 1, max_frames).round()})


def cut_edges(x: np.ndarray, cells: np.ndarray, L: float, W: float, tol: float = 1e-6) -> np.ndarray:
    """Mesh edges lying on the cut faces: boundary edges (used by one triangle
    only) that are not on the outer rectangle. Returns an (n, 2) vertex array."""
    e = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    b = uniq[count == 1]
    p, q = x[b[:, 0]], x[b[:, 1]]
    outer = np.zeros(len(b), bool)
    for k, val in ((0, 0.0), (0, L), (1, 0.0), (1, W)):
        outer |= (np.abs(p[:, k] - val) < tol) & (np.abs(q[:, k] - val) < tol)
    return b[~outer]


def make(name: str, elev: float = 24, azim: float = -58, zscale: float | None = None,
         src: str | os.PathLike[str] | None = None, dst: str | os.PathLike[str] | None = None) -> None:
    """Write the GIF of a run. src/dst default to results/configs/<name>.npz / .gif; src needs
    the arrays x, cells, U, Delta, L and W (KeyError naming the missing one otherwise)."""
    if zscale is not None and not zscale > 0:
        raise ValueError(f"zscale must be positive, not {zscale}")
    src = src or f"{R}/{name}.npz"
    dst = dst or f"{R}/{name}.gif"
    with np.load(src) as d:
        x0, cells, U, Delta = d["x"], d["cells"], d["U"], d["Delta"]
        L, W = float(d["L"]), float(d["W"])
    if len(Delta) == 0:
        raise ValueError(f"{os.fspath(src)} has no load steps to animate")
    X = np.column_stack([x0, np.zeros(len(x0))])
    idx = frames_index(len(Delta))
    zlim = max(np.abs(U[:, :, 2]).max(), 1e-6)
    # extent of the deformed sheet over all frames (rollers and biaxial loading
    # let it move sideways and below x = 0)
    xs, ys = X[None, :, 0] + U[:, :, 0], X[None, :, 1] + U[:, :, 1]
    xmin, xmax = min(0.0, xs.min()), max(L + Delta.max(), xs.max())
    ymin, ymax = min(0.0, ys.min()), max(W, ys.max())
    planar = zlim < 1e-3 * W          # in-plane deformation: look from above
    if planar:
        elev, azim, zscale = 90, -90, 1.0
    if zscale is None:      # exaggerate small deflections so they stay visible
        zscale = max(1.0, round(0.12 * W / zlim))
    U = U.copy()
    # Buckling up or down is equivalent by symmetry; show the largest deflection up
    last = U[-1, :, 2]
    sign = 1.0 if last[np.argmax(np.abs(last))] >= 0 else -1.0
    U[:, :, 2] *= sign * zscale
    zlim *= zscale

    fig = plt.figure(figsize=(7.2, 4.6), dpi=110)
    fig.subplots_adjust(0, 0, 1, 1)
    # computed_zorder=False: draw the cut lines on top of the surface
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, 2 * zlim), zoom=1.15 if not planar else 1.5)
    title = fig.text(0.5, 0.93, "", ha="center", fontsize=11, color="#0b0b0b")
    note = "colour: out-of-plane displacement (blue down, red up); black lines: cuts"
    if planar:
        note = "top view: the sheet stays flat (in-plane deformation); black lines: cuts"
    if zscale > 1:
        note += f";  z exaggerated x{zscale:g}"
    fig.text(0.5, 0.04, note, ha="center", fontsize=8, color="#52514e")
    coll = Poly3DCollection([], cmap=CMAP, edgecolor="none", linewidth=0)
    coll.set_clim(-zlim, zlim)
    ax.add_collection3d(coll)
    edges = cut_edges(x0, cells, L, W)
    cuts = Line3DCollection(X[edges], colors=INK, linewidths=1.3, zorder=5)
    ax.add_collection3d(cuts)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(-zlim, zlim))
    ax.view_init(elev, azim)
    ax.set_axis_off()

    def update(k: int) -> tuple[Any, ...]:
        i = idx[k]
        P = X + U[i]
        coll.set_verts(P[cells])
        coll.set_array(P[cells][:, :, 2].mean(axis=1))
        cuts.set_segments(P[edges])
        title.set_text(f"{name.replace('_', ' ')}:   Δ/L = {Delta[i] / L:.2e}")
        return coll, cuts, title

    frames = list(range(len(idx))) + [len(idx) - 1] * 8      # hold the last frame
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(os.fspath(dst), writer=PillowWriter(fps=8))
    plt.close(fig)
    print("wrote", dst, file=sys.stderr)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        make(n)
