"""Local material strain of a deformed sheet from P1 displacement snapshots
(numpy only, so it runs on the host and inside the solver).

  local strain = (max principal membrane strain) + (h/2) * |curvature|
      membrane strain: exact Green-Lagrange strain of each linear triangle;
      curvature: angle between the normals of two neighbouring triangles
      divided by the distance between their centroids.

The peak sits at crack tips, where the true elastic field is singular, so the
value is the strain averaged over one tip element: use it to rank designs, not
as an absolute failure prediction.
"""
from __future__ import annotations

import numpy as np


def triangle_geometry(x: np.ndarray, cells: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference inverse edge matrices and the interior-edge adjacency.
    x: (n_nodes, 2) flat positions; cells: (n_triangles, 3) node indices.
    Returns (X, Ginv, orient, pairs, dist), the arguments of local_strain."""
    x, cells = np.asarray(x, float), np.asarray(cells)
    if x.ndim != 2 or x.shape[1] != 2 or cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f"need x of shape (n, 2) and cells of shape (m, 3), got {x.shape} and {cells.shape}")
    X = np.column_stack([x, np.zeros(len(x))])
    G = np.stack([x[cells[:, 1]] - x[cells[:, 0]], x[cells[:, 2]] - x[cells[:, 0]]], axis=2)
    Ginv = np.linalg.inv(G)                                      # (n, 2, 2)
    # gmsh may orient fused sub-surfaces differently; flip those normals back
    orient = np.sign(np.linalg.det(G))
    e = np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    tri = np.tile(np.arange(len(cells)), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, tri = key[order], tri[order]
    same = np.all(key[1:] == key[:-1], axis=1)
    pairs = np.column_stack([tri[:-1][same], tri[1:][same]])     # triangles sharing an edge
    cent = X[cells].mean(axis=1)
    dist = np.linalg.norm(cent[pairs[:, 0]] - cent[pairs[:, 1]], axis=1)
    return X, Ginv, orient, pairs, dist


def local_strain(X: np.ndarray, U: np.ndarray, cells: np.ndarray, Ginv: np.ndarray, orient: np.ndarray,
                 pairs: np.ndarray, dist: np.ndarray, h: float) -> np.ndarray:
    """Per-triangle membrane + bending surface strain for one snapshot. U: (n_nodes, 3)
    displacements; h: thickness (mm, > 0); the rest from triangle_geometry."""
    if U.shape != X.shape:
        raise ValueError(f"U has shape {U.shape}; the mesh needs {X.shape}")
    if not h > 0:
        raise ValueError(f"the thickness h must be positive, not {h}")
    P = X + U
    Fd = np.stack([P[cells[:, 1]] - P[cells[:, 0]], P[cells[:, 2]] - P[cells[:, 0]]], axis=2)
    F = Fd @ Ginv                                                # (n, 3, 2)
    E = 0.5 * (np.transpose(F, (0, 2, 1)) @ F - np.eye(2))
    tr, det = E[:, 0, 0] + E[:, 1, 1], E[:, 0, 0] * E[:, 1, 1] - E[:, 0, 1] * E[:, 1, 0]
    e_mem = tr / 2 + np.sqrt(np.maximum(tr ** 2 / 4 - det, 0.0))  # max principal
    n = np.cross(Fd[:, :, 0], Fd[:, :, 1])
    n *= (orient / np.linalg.norm(n, axis=1))[:, None]
    cosang = np.clip(np.einsum("ij,ij->i", n[pairs[:, 0]], n[pairs[:, 1]]), -1, 1)
    kappa = np.arccos(cosang) / dist
    e_bend = np.zeros(len(cells))
    np.maximum.at(e_bend, pairs[:, 0], 0.5 * h * kappa)
    np.maximum.at(e_bend, pairs[:, 1], 0.5 * h * kappa)
    return np.abs(e_mem) + e_bend


def crossing(eps: np.ndarray, eloc: np.ndarray, eps_y: float) -> tuple[float, str]:
    """Applied strain at which eloc first reaches eps_y (log interpolation).
    eps and eloc: the applied and local strain at each step, positive. Returns (value, note);
    note is "" for a real crossing."""
    if not eps_y > 0:
        raise ValueError(f"the yield strain eps_y must be positive, not {eps_y}")
    if len(eps) != len(eloc) or not len(eps):
        raise ValueError(f"eps and eloc need the same, non-zero length ({len(eps)} and {len(eloc)})")
    k = np.where(eloc >= eps_y)[0]
    if len(k) == 0:
        return float(eps[-1]), "not reached within the simulated range"
    if k[0] == 0:
        return float(eps[0]), "already at the first step"
    i = k[0]
    t = (np.log(eps_y) - np.log(eloc[i - 1])) / (np.log(eloc[i]) - np.log(eloc[i - 1]))
    return float(np.exp(np.log(eps[i - 1]) + t * (np.log(eps[i]) - np.log(eps[i - 1])))), ""
