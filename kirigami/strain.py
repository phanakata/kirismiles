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
import numpy as np


def triangle_geometry(x, cells):
    """Reference inverse edge matrices and the interior-edge adjacency."""
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


def local_strain(X, U, cells, Ginv, orient, pairs, dist, h):
    """Per-triangle membrane + bending surface strain for one snapshot."""
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


def crossing(eps, eloc, eps_y):
    """Applied strain at which eloc first reaches eps_y (log interpolation).
    Returns (value, note); note is "" for a real crossing."""
    k = np.where(eloc >= eps_y)[0]
    if len(k) == 0:
        return float(eps[-1]), "not reached within the simulated range"
    if k[0] == 0:
        return float(eps[0]), "already at the first step"
    i = k[0]
    t = (np.log(eps_y) - np.log(eloc[i - 1])) / (np.log(eloc[i]) - np.log(eloc[i - 1]))
    return float(np.exp(np.log(eps[i - 1]) + t * (np.log(eps[i]) - np.log(eps[i - 1])))), ""
