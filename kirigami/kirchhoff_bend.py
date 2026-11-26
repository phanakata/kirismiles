"""Corrected bending for MuJoCo's 2D flex: a discrete Kirchhoff plate, with the Poisson coupling.

MuJoCo stores the bending of a 2D flex as a fixed linear stencil precomputed from the flat mesh and
applied to every world coordinate of the nodes (force -K x, energy 1/2 x^T K x per coordinate). Its
stencil has no counterpart of the Poisson coupling of a Kirchhoff plate: any deformation with Gaussian
curvature (twist, saddle, sphere) costs the energy of a plate with nu ~ -1/3 whatever nu is given
(MuJoCo issue 3960). bending_stiffness keeps MuJoCo's structure and changes only K.

For each triangle t, a quadratic w = c0 + c1 x + c2 y + d x^2/2 + e x y + f y^2/2 is fitted to a
small patch of nodes: the triangle's three and the three across its edges (exact interpolation);
where an edge is free, the patch takes one more ring of edge-neighbours and the fit is least squares.
The curvatures (w_xx, w_xy, w_yy) = (d, e, f) are then linear in the patch values, h = G_t w_patch,
and the triangle's energy is the Kirchhoff plate energy

    U_t = A_t D/2 [(w_xx + w_yy)^2 - 2 (1 - nu) (w_xx w_yy - w_xy^2)],   D = E h^3 / (12 (1 - nu^2)),

so K is the sum of the blocks A_t G_t^T C G_t. Affine fields cost nothing (rigid motions and in-plane
stretches, left to the membrane). Patches follow shared edges only, so they never cross a slit, whose
two faces are different nodes. Exact for fields of constant curvature on any triangulation; linear in
the positions, so, like MuJoCo's own stencil, exact only for small rotations.

In MuJoCo, the flex keeps only its membrane (elastic2d="stretch") and kirchhoff_callback adds the
bending force through mujoco.set_mjcb_passive (scripts/mujoco_check.py --bending kirchhoff).
Units: mm, MPa, N.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp


def _cross2(u: np.ndarray, v: np.ndarray) -> float:
    """z component of the cross product of two 2D vectors."""
    return float(u[0] * v[1] - u[1] * v[0])


def edge_neighbours(cells: np.ndarray) -> list[list[tuple[int, int]]]:
    """For each triangle, its (neighbour triangle, neighbour's node opposite the shared edge) pairs."""
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for t, tri in enumerate(cells):
        for k in range(3):
            a, b = sorted((int(tri[k]), int(tri[(k + 1) % 3])))
            owners[a, b].append(t)
    out: list[list[tuple[int, int]]] = [[] for _ in cells]
    for (a, b), ts in owners.items():
        if len(ts) == 2:
            t0, t1 = ts
            out[t0].append((t1, int(next(v for v in cells[t1] if v not in (a, b)))))
            out[t1].append((t0, int(next(v for v in cells[t0] if v not in (a, b)))))
    return out


def patch(t: int, cells: np.ndarray, nbrs: list[list[tuple[int, int]]]) -> list[int]:
    """The nodes used to fit triangle t's quadratic: its own, those across its edges and, if a free
    edge leaves fewer than six, those of the next rings of edge-neighbours (up to nine nodes)."""
    verts = [int(v) for v in cells[t]] + [v for _, v in nbrs[t]]
    if len(nbrs[t]) == 3:
        return verts
    seen, ring = {t}, [n for n, _ in nbrs[t]]
    seen.update(ring)
    while len(set(verts)) < 9 and ring:
        nxt = []
        for s in ring:
            for n, _ in nbrs[s]:
                if n not in seen:
                    seen.add(n)
                    nxt.append(n)
                    verts.extend(int(u) for u in cells[n])
        ring = nxt
    return list(dict.fromkeys(verts))


def hessian_operator(P: np.ndarray, t: int, cells: np.ndarray, verts: list[int]) -> np.ndarray:
    """G (3 x len(verts)): (w_xx, w_xy, w_yy) (1/mm^2 per mm) of the quadratic fitted to the values at
    verts. Raises ValueError if the patch cannot determine a quadratic."""
    c = P[cells[t]].mean(axis=0)
    s = np.sqrt(abs(_cross2(P[cells[t][1]] - P[cells[t][0]], P[cells[t][2]] - P[cells[t][0]])) / 2)
    if not s > 0:
        raise ValueError(f"triangle {t} has zero area")
    xy = (P[verts] - c) / s                                       # scaled local coordinates
    x, y = xy[:, 0], xy[:, 1]
    V = np.c_[np.ones_like(x), x, y, x**2 / 2, x * y, y**2 / 2]
    if np.linalg.matrix_rank(V, tol=1e-8) < 6:
        raise ValueError(f"triangle {t}: its patch cannot determine a quadratic")
    return np.linalg.pinv(V)[3:6] / s**2                          # back to physical units


def bending_stiffness(P: np.ndarray, cells: np.ndarray, E: float, nu: float, h: float) -> sp.csr_matrix:
    """K (n x n, sparse, symmetric): bending energy 1/2 w^T K w (N mm) of the out-of-plane displacements
    w (mm) of the nodes P (n x 2, mm) of the triangles cells (m x 3). E > 0 (MPa), -1 < nu < 0.5, h > 0 (mm)."""
    if not -1 < nu < 0.5:
        raise ValueError(f"the Poisson ratio must be between -1 and 0.5, not {nu}")
    if not (np.isfinite(E) and E > 0 and np.isfinite(h) and h > 0):
        raise ValueError(f"E and h must be positive numbers (MPa, mm), not {E} and {h}")
    P = np.asarray(P, float)[:, :2]
    cells = np.asarray(cells, int)
    D = E * h**3 / (12 * (1 - nu**2))
    C = D * np.array([[1, 0, nu], [0, 2 * (1 - nu), 0], [nu, 0, 1]])
    nbrs = edge_neighbours(cells)
    rows, cols, vals = [], [], []
    for t, tri in enumerate(cells):
        verts = patch(t, cells, nbrs)
        G = hessian_operator(P, t, cells, verts)
        a, b, c = P[tri]
        Kt = abs(_cross2(b - a, c - a)) / 2 * G.T @ C @ G
        idx = np.array(verts)
        rows.append(np.repeat(idx, len(idx)))
        cols.append(np.tile(idx, len(idx)))
        vals.append(Kt.ravel())
    n = len(P)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def energy(K: sp.csr_matrix, w: np.ndarray) -> float:
    """Bending energy 1/2 w^T K w (N mm) of the out-of-plane displacements w (mm)."""
    return 0.5 * float(w @ (K @ w))


def kirchhoff_callback(K: sp.csr_matrix, x_rest: np.ndarray, free: np.ndarray, dof3: np.ndarray,
                       grip: np.ndarray, grip_dof: int) -> Callable[[Any, Any], None]:
    """A mujoco.set_mjcb_passive callback adding the force -K (x - x_rest) of every flex node (all three
    world coordinates) to MuJoCo's passive forces: on the free nodes' slide joints dof3 (n_free x 3, in
    the order of free) and, summed along x, on the grip's slide joint grip_dof (the grip nodes move with
    it). x_rest: the stress-free node positions (n x 3, mm), data.flexvert_xpos at the rest state."""
    def bend(model: Any, data: Any) -> None:
        f = -(K @ (data.flexvert_xpos - x_rest))
        data.qfrc_passive[dof3] += f[free]
        data.qfrc_passive[grip_dof] += f[grip, 0].sum()
    return bend
