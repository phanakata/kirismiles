"""kirigami/mjcf.py: the (young, poisson, thickness) handed to MuJoCo give the plate's constants.

Every node of a square flex gets a prescribed displacement field, and MuJoCo's stored energy (-1/2 qfrc_passive . q,
linear at these tiny amplitudes) is compared with the exact plane-stress / Kirchhoff energy. Skipped without mujoco.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from kirigami import mjcf as K  # noqa: E402

A_SIDE, N = 10.0, 41
E, H = 3500.0, 0.127


def square(n=N):
    g = np.linspace(0, A_SIDE, n)
    X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()]
    cells = []
    for j in range(n - 1):
        for i in range(n - 1):
            k = j * n + i
            cells += [[k, k + 1, k + n + 1], [k, k + n + 1, k + n]]
    return P, np.array(cells)


def energy(P, cells, nu, which, u):
    young, nu_in, t_in = K.shell_inputs(E, H, nu)
    bodies = "".join(f'<body name="v{i}" pos="{p[0]} {p[1]} 0"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
                     '<joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>'
                     '<joint type="slide" axis="0 0 1"/></body>' for i, p in enumerate(P))
    xml = f"""<mujoco><worldbody>{bodies}</worldbody><deformable>
      <flex name="s" dim="2" radius="0.01" body="{' '.join(f'v{i}' for i in range(len(P)))}"
            vertex="{' '.join('0 0 0' for _ in P)}" element="{' '.join(map(str, cells.ravel()))}">
        <contact contype="0" conaffinity="0"/>
        <elasticity young="{young}" poisson="{nu_in}" thickness="{t_in}" elastic2d="{which}"/>
      </flex></deformable></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = u.ravel()
    mujoco.mj_forward(m, d)
    return -0.5 * d.qfrc_passive @ d.qpos


@pytest.mark.parametrize("nu", [0.0, 0.2, 0.38, 0.45])
def test_membrane_is_plane_stress(nu):
    P, cells = square(11)
    e, area, z = 1e-6, A_SIDE**2, np.zeros(len(P))
    cases = [(np.c_[e * P[:, 0], z, z], 0.5 * E / (1 - nu**2) * e**2),          # eps_xx alone
             (np.c_[e * P[:, 0], e * P[:, 1], z], E / (1 - nu) * e**2),           # equal biaxial
             (np.c_[e * P[:, 1], z, z], 0.5 * E / (2 * (1 + nu)) * e**2)]         # simple shear
    for u, w in cases:
        assert energy(P, cells, nu, "stretch", u) == pytest.approx(w * H * area, rel=1e-6)


@pytest.mark.parametrize("nu", [0.0, 0.38])
def test_cylindrical_bending_stiffness(nu):
    P, cells = square()
    k, c = 1e-6, P - A_SIDE / 2
    w = 0.5 * k * c[:, 0] ** 2
    D = E * H**3 / (12 * (1 - nu**2))
    exact = 0.5 * D * k**2 * A_SIDE**2
    assert energy(P, cells, nu, "bend", np.c_[0 * w, 0 * w, w]) == pytest.approx(exact, rel=0.03)


@pytest.mark.xfail(reason="MuJoCo's 2D flex bending has no Kirchhoff Poisson coupling: twist is 1.3-3.7x too stiff "
                          "(measured on 3.17 and 4.0.1); kept as a record of the limitation", strict=True)
@pytest.mark.parametrize("nu", [0.0, 0.38])
def test_twist_stiffness(nu):
    P, cells = square()
    k, c = 1e-6, P - A_SIDE / 2
    w = k * c[:, 0] * c[:, 1]
    D = E * H**3 / (12 * (1 - nu**2))
    exact = D * (1 - nu) * k**2 * A_SIDE**2                              # Kirchhoff: 1/2 D 2(1-nu) kappa_xy^2
    assert energy(P, cells, nu, "bend", np.c_[0 * w, 0 * w, w]) == pytest.approx(exact, rel=0.1)


def test_mjcf_writes_poisson_for_the_installed_version():
    P, cells = square(5)
    xml, info = K.mjcf(P, cells, A_SIDE, nu=0.38)
    expected = 0.38 if K.mujoco_major() >= 4 else 0.19                 # 3.x: the measured nu / 2 mapping
    assert info["poisson_in"] == pytest.approx(expected)
    assert f'poisson="{expected:g}"' in xml
    assert K.shell_inputs(E, H, 0.38, major=4) == (E, 0.38, H)
    with pytest.raises(ValueError):
        K.shell_inputs(E, H, 0.5)
