"""Uncut sheet: MuJoCo's own equilibria and buckling load against FEniCS, with the built-in and the corrected bending.

scripts/pristine_mujoco.py imposes FEniCS's exact fields on MuJoCo and compares energies (upper bounds). Here MuJoCo
finds its own states, on the mesh, clamps and material of results/mujoco_check/pristine_nu<nu>.npz
(scripts/pristine_fenics.py), each with MuJoCo's built-in bending (elastic2d="both") and with the corrected bending
(kirigami/kirchhoff_bend.py: MuJoCo's membrane only, the bending added through the passive-force callback):

  relaxed   the x = L clamp pulled by 0.01 mm (F / Delta against FEniCS's F1, N/mm) and turned by 1e-3 rad about the
            sheet's axis (torque / theta against FEniCS's 2 E_twist / theta^2, N mm/rad), each relaxed by the dynamic
            relaxation of scripts/mujoco_check.py (selective mass scaling, --factor; kinetic damping; tolerance 1e-5)
  buckling  the clamp pushed in by 1e-4 mm and the in-plane state relaxed with the sheet held flat; the out-of-plane
            stiffness matrix measured from MuJoCo's forces (finite differences), flat and compressed; Delta_c from the
            generalized eigenproblem, against FEniCS's Delta_c and the beam and plate values for clamped ends

    python scripts/mujoco_uncut.py relaxed [--nu 0.38] [--factor 12]
    python scripts/mujoco_uncut.py buckling [--nu 0.38]
Each takes a few minutes and holds the solver lock (one solve at a time).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kirigami import kirchhoff_bend as KB  # noqa: E402
from kirigami import mjcf as K  # noqa: E402
from kirigami.argtypes import positive  # noqa: E402
from kirigami.jobs import SolverLock  # noqa: E402

DT = 1e-4                    # explicit time step (s); the masses are scaled to it
BENDINGS = ("built-in", "corrected")


def build(z: Any, bending: str, hinge: bool = False) -> tuple[Any, Any, dict, tuple[float, ...]]:
    """MuJoCo model and data of the uncut sheet. hinge: the grip turns about the sheet's axis instead of sliding."""
    import mujoco
    x, c = z["x"], z["cells"]
    L, W, h, E, nu = (float(z[k]) for k in ("L", "W", "h", "E", "nu"))
    xml, info = K.mjcf(x, c, L, E=E, h=h, nu=nu, dt=DT)
    if bending == "corrected":
        xml = xml.replace('elastic2d="both"', 'elastic2d="stretch"')
    if hinge:
        xml = xml.replace('<joint name="pull" type="slide" axis="1 0 0"/>',
                          f'<joint name="pull" type="hinge" axis="1 0 0" pos="0 {W / 2:.8g} 0"/>')
    mujoco.set_mjcb_passive(None)
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    return m, mujoco.MjData(m), info, (L, W, h, E, nu)


def free_dof(m: Any, info: dict) -> np.ndarray:
    """dof addresses (n_free x 3, flattened) of the free nodes' x, y, z slides."""
    return np.array([[m.jnt_dofadr[m.joint(f"v{i}{a}").id] for a in "xyz"] for i in info["free"]]).ravel()


def relax(z: Any, test: str, bending: str, factor: float, tol: float = 1e-5,
          maxit: int = 200_000) -> tuple[float, float, int, float]:
    """MuJoCo's stiffness for test "tension" (N/mm) or "twist" (N mm/rad), from its relaxed state.
    Returns (stiffness, residual, iterations, seconds)."""
    import mujoco
    m, d, info, (L, W, h, E, nu) = build(z, bending, hinge=test == "twist")
    j, jd = m.joint("pull").qposadr[0], m.joint("pull").dofadr[0]
    adr = K.free_dofs(m, info)
    dof = free_dof(m, info)
    colour = K.colouring(info)
    free, grip = info["free"], np.flatnonzero(info["side"] == 1)
    xf = info["x"][free]
    cb, kdiag = None, None
    if bending == "corrected":
        Kb = KB.bending_stiffness(info["x"], info["cells"], E, nu, h)
        mujoco.mj_forward(m, d)
        x_rest = d.flexvert_xpos.copy()
        dof3 = dof.reshape(-1, 3)
        axis_pt = np.array([L, W / 2, 0.0])

        def cb(model: Any, data: Any) -> None:
            f = -(Kb @ (data.flexvert_xpos - x_rest))
            data.qfrc_passive[dof3] += f[free]
            if test == "tension":
                data.qfrc_passive[jd] += f[grip, 0].sum()
            else:                                          # moment of the grip nodes' forces about the axis
                r = data.flexvert_xpos[grip] - axis_pt
                data.qfrc_passive[jd] += float(np.sum(r[:, 1] * f[grip, 2] - r[:, 2] * f[grip, 1]))
        kdiag = Kb.diagonal()[free][:, None]
        mujoco.set_mjcb_passive(cb)

    def scale() -> np.ndarray:
        mujoco.set_mjcb_passive(None)
        Kd = K.stiffness_diagonal(m, d, info, colour)
        if cb is not None:
            Kd = Kd + kdiag
            mujoco.set_mjcb_passive(cb)
        return K.set_armature(m, d, info, Kd, DT, factor).ravel()

    load = 0.01 if test == "tension" else 1e-3                # mm or rad
    d.qpos[j] = load
    if test == "tension":
        d.qpos[adr[:, 0]] = load * xf[:, 0] / L                # start from the affine stretch
    else:
        d.qpos[adr[:, 2]] = load * (xf[:, 1] - W / 2) * xf[:, 0] / L   # start from a linear twist
    mujoco.mj_forward(m, d)
    meff, ke_prev, t0, res, R, it = scale(), 0.0, time.time(), np.inf, 0.0, 0
    try:
        for it in range(1, maxit + 1):
            mujoco.mj_step(m, d)
            d.qpos[j], d.qvel[jd] = load, 0
            if not np.isfinite(d.qpos).all():
                raise RuntimeError(f"{test}/{bending}: diverged at iteration {it}")
            ke = float(meff @ d.qvel[dof] ** 2)
            if ke < ke_prev:                                   # kinetic damping
                d.qvel[:] = 0
                ke = 0.0
            ke_prev = ke
            if it % 500 == 0:
                R = -d.qfrc_passive[jd]                        # reaction: force (N) or torque (N mm)
                ref = abs(R) if test == "tension" else abs(R) / (W / 2)
                res = float(np.linalg.norm(d.qfrc_passive[dof]) / max(ref, 1e-30))
                if res < tol:
                    break
            if it % 2000 == 0:                                 # refresh the masses
                d.qvel[:] = 0
                meff, ke_prev = scale(), 0.0
    finally:
        mujoco.set_mjcb_passive(None)
    return R / load, res, it, time.time() - t0


def buckling(z: Any, bending: str, factor: float, delta: float = 1e-4, eps: float = 1e-6) -> tuple[float, float]:
    """MuJoCo's linear buckling stretch Delta_c (mm) under compression. Returns (Delta_c, in-plane residual)."""
    import mujoco
    import scipy.linalg as sla
    m, d, info, (L, W, h, E, nu) = build(z, bending)
    j, jd = m.joint("pull").qposadr[0], m.joint("pull").dofadr[0]
    adr = K.free_dofs(m, info)
    dof = free_dof(m, info)
    dz, dxy = dof.reshape(-1, 3)[:, 2], dof.reshape(-1, 3)[:, :2].ravel()
    extra = None
    if bending == "corrected":         # the corrected bending's z-block on the free nodes, exact
        free = info["free"]
        extra = KB.bending_stiffness(info["x"], info["cells"], E, nu, h)[free][:, free].toarray()

    def kzz() -> np.ndarray:
        """Out-of-plane tangent stiffness on the free nodes (dense, symmetrized), one finite-difference column of
        MuJoCo's passive forces per node."""
        mujoco.mj_forward(m, d)
        q0, f0 = d.qpos.copy(), d.qfrc_passive[dz].copy()
        A = np.empty((len(dz), len(dz)))
        for k in range(len(dz)):
            d.qpos[:] = q0
            d.qpos[adr[k, 2]] += eps
            mujoco.mj_forward(m, d)
            A[:, k] = -(d.qfrc_passive[dz] - f0) / eps
        d.qpos[:] = q0
        mujoco.mj_forward(m, d)
        A = 0.5 * (A + A.T)
        return A if extra is None else A + extra

    K0 = kzz()                                                 # flat, unloaded: bending only
    D = -delta                                                 # compression, held flat; relax x, y
    d.qpos[j], d.qvel[:] = D, 0
    d.qpos[adr[:, 0]] = D * info["x"][info["free"], 0] / L
    mujoco.mj_forward(m, d)
    colour = K.colouring(info)
    meff = K.set_armature(m, d, info, K.stiffness_diagonal(m, d, info, colour), DT, factor).ravel()
    ke_prev, res = 0.0, np.inf
    for it in range(1, 200_001):
        mujoco.mj_step(m, d)
        d.qpos[j], d.qvel[jd] = D, 0
        d.qpos[adr[:, 2]], d.qvel[dz] = 0, 0
        ke = float(meff @ d.qvel[dof] ** 2)
        if ke < ke_prev:
            d.qvel[:] = 0
            ke = 0.0
        ke_prev = ke
        if it % 500 == 0:
            res = float(np.linalg.norm(d.qfrc_passive[dxy]) / max(abs(d.qfrc_passive[jd]), 1e-30))
            if res < 1e-9:
                break
    Kg = (kzz() - K0) / delta                                  # per mm of compression
    lam = sla.eigvals(K0, -Kg)
    lam = np.real(lam[np.isfinite(lam) & (np.abs(lam.imag) < 1e-6 * np.abs(lam.real) + 1e-30)])
    return float(lam[lam > 0].min()), res


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("what", choices=["relaxed", "buckling"])
    p.add_argument("--nu", default="0.38", help="selects results/mujoco_check/pristine_nu<nu>.npz")
    p.add_argument("--factor", type=positive, default=12.0, help="mass scaling: m_eff >= factor K_ii dt^2")
    a = p.parse_args()
    path = ROOT / "results" / "mujoco_check" / f"pristine_nu{a.nu}.npz"
    if not path.is_file():
        p.error(f"no such file: {path} (run scripts/pristine_fenics.py {a.nu} first)")
    z = np.load(path)
    lock = SolverLock()
    if not lock.try_acquire():
        raise SystemExit("another solve is running (results/runs/.solver.lock): try again later")
    try:
        import mujoco
        print(f"uncut sheet, {path.name}, nu = {float(z['nu']):g}, MuJoCo {mujoco.__version__}, mass factor {a.factor:g}")
        if a.what == "relaxed":
            ref = {"tension": float(z["F1"]), "twist": 2 * float(z["E_twist"]) / float(z["theta"]) ** 2}
            print(f"FEniCS: tension {ref['tension']:.5g} N/mm, twist {ref['twist']:.5g} N mm/rad")
            for test in ("tension", "twist"):
                for bending in BENDINGS:
                    k, res, it, sec = relax(z, test, bending, a.factor)
                    print(f"  {test:7s} {bending:9s} {k:.5g}  MuJoCo / FEniCS {k / ref[test]:.4f}  "
                          f"(residual {res:.1e}, {it} it, {sec:.0f} s)", flush=True)
        else:
            L, h, nu = float(z["L"]), float(z["h"]), float(z["nu"])
            beam, plate = 4 * np.pi**2 * h**2 / (12 * L), 4 * np.pi**2 * h**2 / (12 * (1 - nu**2) * L)
            Dfe = float(z["Delta_c"])
            print(f"clamped ends, free long edges: beam {beam:.4e} mm, plate {plate:.4e} mm; FEniCS {Dfe:.4e} mm")
            for bending in BENDINGS:
                Dc, res = buckling(z, bending, a.factor)
                print(f"  {bending:9s} Delta_c = {Dc:.4e} mm  MuJoCo / FEniCS {Dc / Dfe:.4f}  "
                      f"(in-plane residual {res:.1e})", flush=True)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
