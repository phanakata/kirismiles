"""Uncut sheet: MuJoCo's flex energy of the exact FEniCS fields (scripts/pristine_fenics.py), on the same P1 mesh.

Because both codes' energies are quadratic at small amplitude, the ratio of MuJoCo's energy to FEniCS's for the same
field is the ratio of their stiffnesses for that deformation:
  tension     the plane-stress displacement for Delta = 1 mm        -> MuJoCo / FEniCS in-plane stiffness
  buckling    FEniCS's first compression buckling mode (bending)      -> MuJoCo / FEniCS buckling load (Rayleigh
              quotient: the membrane prestress is the same, MuJoCo's membrane being exact plane stress)
  twist       the twisted-clamp Kirchhoff solution                    -> MuJoCo / FEniCS torsional stiffness
MuJoCo's own minimiser would have slightly lower energy than this imposed field (an upper bound), so the ratios are
upper bounds. Clamps as in kirigami/mjcf.py: a ghost strip outside each end that moves with the clamp.
The last line gives the same two bending ratios for the corrected bending (kirigami/kirchhoff_bend.py), which replaces
MuJoCo's built-in bending in scripts/mujoco_check.py --bending kirchhoff.

    python scripts/pristine_mujoco.py results/mujoco_check/pristine_nu0.38.npz
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kirigami import kirchhoff_bend as KB  # noqa: E402
from kirigami import mjcf as K  # noqa: E402

z = np.load(sys.argv[1])
x, cells, L, W, h, E, nu = z["x"], z["cells"], float(z["L"]), float(z["W"]), float(z["h"]), float(z["E"]), float(z["nu"])
e = np.linalg.norm(x[cells[:, 0]] - x[cells[:, 1]], axis=1)
xa, ca, side = K.ghost_layer(x, cells, L, float(np.median(e)))
n0 = len(x)
gsrc = np.array([np.argmin(np.abs(x[:, 0] - (0.0 if s < 0 else L)) + 1e3 * np.abs(x[:, 1] - xa[i, 1]))
                 if i >= n0 else i for i, s in enumerate(side)])   # each ghost node's clamp-edge partner


def model(which):
    young, nu_in, t_in = K.shell_inputs(E, h, nu)
    bodies = "".join(f'<body name="v{i}" pos="{p[0]} {p[1]} 0"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
                     '<joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>'
                     '<joint type="slide" axis="0 0 1"/></body>' for i, p in enumerate(xa))
    xml = f"""<mujoco><worldbody>{bodies}</worldbody><deformable>
      <flex name="s" dim="2" radius="0.01" body="{' '.join(f'v{i}' for i in range(len(xa)))}"
            vertex="{' '.join('0 0 0' for _ in xa)}" element="{' '.join(map(str, ca.ravel()))}">
        <contact contype="0" conaffinity="0"/>
        <elasticity young="{young}" poisson="{nu_in}" thickness="{t_in}" elastic2d="{which}"/>
      </flex></deformable></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def energy(m, U):
    d = mujoco.MjData(m)
    d.qpos[:] = U.ravel()
    mujoco.mj_forward(m, d)
    return -0.5 * d.qfrc_passive @ d.qpos


def on_all(U):
    """Field on the original nodes -> all nodes, ghosts copying their clamp edge (zero slope at the clamp)."""
    return U[gsrc]


a = 1e-4                                         # small amplitude: both energies quadratic
# tension (Delta = 1 mm field, scaled)
Ut = np.zeros((n0, 3)); Ut[:, :2] = z["u_tension"]
Ut = on_all(Ut); Ut[side == 1] = [1.0, 0, 0]
Em = energy(model("stretch"), a * Ut)
r_t = (2 * Em / a**2) / float(z["F1"])
# buckling mode (bending energy, Rayleigh quotient)
Ub = np.zeros((n0, 3)); Ub[:, 2] = z["phi"]
r_b = energy(model("bend"), a * on_all(Ub)) / (a**2 * float(z["Eb_phi"]))
# twist
Uw = np.zeros((n0, 3)); Uw[:, 2] = z["w_twist"]
r_w = energy(model("bend"), a * on_all(Uw) / float(z["theta"])) / (a**2 * float(z["E_twist"]) / float(z["theta"])**2)
print(f"uncut {L:g} x {W:g} mm, h {h}, E {E:g}, nu {nu:g}: {n0} nodes")
print(f"  MuJoCo / FEniCS   tension stiffness {r_t:.3f}   compression buckling load {r_b:.3f}   twist stiffness {r_w:.3f}")
Kb = KB.bending_stiffness(xa, ca, E, nu, h)                  # the corrected bending on the same mesh
c_b = KB.energy(Kb, a * on_all(Ub)[:, 2]) / (a**2 * float(z["Eb_phi"]))
c_w = KB.energy(Kb, a * on_all(Uw)[:, 2] / float(z["theta"])) / (a**2 * float(z["E_twist"]) / float(z["theta"])**2)
print(f"  corrected bending / FEniCS                            compression buckling load {c_b:.3f}   twist stiffness {c_w:.3f}")
