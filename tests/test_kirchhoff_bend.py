"""The corrected bending (kirigami/kirchhoff_bend.py): exact Kirchhoff energy for constant curvature on any
triangulation, no energy for affine fields, three zero modes on a free plate, no coupling across a slit, and input
checks. Pure numpy: no MuJoCo needed."""
import numpy as np
import pytest

from kirigami import kirchhoff_bend as KB

E, H = 3500.0, 0.127
D0 = E * H**3 / 12


def grid(n=9, a=10.0, jitter=0.0, seed=0):
    """Right-triangle grid of a x a mm with n x n nodes; interior nodes moved by up to jitter of the spacing."""
    s = np.linspace(0, a, n)
    X, Y = np.meshgrid(s, s)
    P = np.c_[X.ravel(), Y.ravel()]
    if jitter:
        inner = (P[:, 0] > 0) & (P[:, 0] < a) & (P[:, 1] > 0) & (P[:, 1] < a)
        P[inner] += jitter * (a / (n - 1)) * np.random.default_rng(seed).uniform(-1, 1, (inner.sum(), 2))
    cells = []
    for j in range(n - 1):
        for i in range(n - 1):
            v = j * n + i
            cells += [[v, v + 1, v + n + 1], [v, v + n + 1, v + n]]
    return P, np.array(cells)


@pytest.mark.parametrize("nu", [0.0, 0.3, 0.45])
@pytest.mark.parametrize("jitter", [0.0, 0.25])
@pytest.mark.parametrize("k", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, -1.0), (1.0, 0.0, 1.0)])
def test_constant_curvature_is_exact(nu, jitter, k):
    # w = (kxx x^2 + 2 kxy x y + kyy y^2) / 2: cylinder, twist, saddle, sphere
    P, c = grid(jitter=jitter)
    kxx, kxy, kyy = k
    w = 0.5 * (kxx * P[:, 0] ** 2 + 2 * kxy * P[:, 0] * P[:, 1] + kyy * P[:, 1] ** 2) * 1e-3
    Dp = D0 / (1 - nu**2)
    exact = 100.0 * Dp / 2 * 1e-6 * ((kxx + kyy) ** 2 - 2 * (1 - nu) * (kxx * kyy - kxy**2))
    Kb = KB.bending_stiffness(P, c, E, nu, H)
    assert KB.energy(Kb, w) == pytest.approx(exact, rel=1e-8)


def test_affine_fields_cost_nothing_and_three_zero_modes():
    P, c = grid(jitter=0.25)
    Kb = KB.bending_stiffness(P, c, E, 0.38, H)
    for w in (np.ones(len(P)), P[:, 0], P[:, 1], 2 + 3 * P[:, 0] - P[:, 1]):
        assert abs(KB.energy(Kb, w)) < 1e-10 * KB.energy(Kb, P[:, 0] ** 2)
    A = Kb.toarray()
    assert np.allclose(A, A.T)
    ev = np.linalg.eigvalsh(A)
    assert (ev > 1e-9 * ev.max()).sum() == len(P) - 3 and ev.min() > -1e-9 * ev.max()


def test_no_coupling_across_a_slit():
    # a slit along part of x = 5: the nodes on it are duplicated, and the triangles on the right use the copies
    P, c = grid()
    n = 9
    on = [j * n + 4 for j in range(2, 7)]                      # nodes on x = 5, y = 2.5 .. 7.5
    tips = {on[0], on[-1]}
    copies = {v: len(P) + k for k, v in enumerate(v for v in on if v not in tips)}
    P2 = np.vstack([P, P[list(copies)]])
    c2 = c.copy()
    for t, tri in enumerate(c):
        if P[tri][:, 0].mean() > 5:                            # right of the slit
            c2[t] = [copies.get(int(v), int(v)) for v in tri]
    Kb = KB.bending_stiffness(P2, c2, E, 0.3, H).tocsr()
    for v, cp in copies.items():
        assert Kb[v, cp] == 0 and Kb[cp, v] == 0


@pytest.mark.parametrize("E_, nu, h", [(3500, 0.5, 0.1), (3500, -1.0, 0.1), (0, 0.3, 0.1), (3500, 0.3, -0.1),
                                       (np.nan, 0.3, 0.1)])
def test_bending_stiffness_checks_its_inputs(E_, nu, h):
    P, c = grid(n=4)
    with pytest.raises(ValueError):
        KB.bending_stiffness(P, c, E_, nu, h)
