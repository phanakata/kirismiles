"""Does a MuJoCo simulation of a KiriSMILES design give the same force-strain curve as the FEniCS shell?

The design's KiriSMILES mesh (mesh.vtk from the `mesh` command / export_mesh) becomes a MuJoCo 2D flex shell
(kirigami/mjcf.py), clamped at x = 0 and pulled at x = L through the load steps of the FEniCS simulate run
(0.5 Delta_c, then geometric from 1.05 Delta_c to --Dmax). At each step MuJoCo's own equilibrium is found by dynamic
relaxation: explicit Euler steps with selective mass scaling (each node just heavy enough for --dt to be stable, from
MuJoCo's stiffness diagonal; statics do not depend on mass) and kinetic damping (all velocities zeroed when the kinetic
energy passes a peak), until the net force on the free nodes is below --tol of the pull force. Past the linear
buckling load a still-flat sheet is started on the buckled branch with the predictor of the FEniCS run,
w = 1/2 sqrt(ell (Delta - Delta_c)) phi_1, as FEniCS does; the relaxation then finds MuJoCo's own state.

    python scripts/mujoco_check.py --mesh results/runs/<mesh run>/mesh.vtk --mode results/runs/<buckle run>/buckling.npz
        --fenics results/runs/<simulate run>/history.npz --nu 0.38 --L 120 --Dmax 24 --steps 20 --tag wf
    python scripts/mujoco_check.py --fenics ... --L 120 --tag wf --plot
Writes results/mujoco_check/<tag>.npz (Delta, F, w_max, w_rms, residual, iterations), every relaxed state in
<tag>_states/ (for scripts/mujoco_gif.py), and with --plot the comparison figure <tag>.png. --resume continues
after the last saved step, so a long run can go in chunks.
Material: E = 3500 MPa, h = 0.127 mm, nu = --nu in both codes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kirigami import mjcf as K  # noqa: E402
from kirigami.argtypes import count, nonneg, positive  # noqa: E402

OUT = ROOT / "results" / "mujoco_check"
E, H = 3500.0, 0.127


def relax(a: argparse.Namespace) -> None:
    """Quasi-static load path by dynamic relaxation: the grip steps to each load value and MuJoCo integrates the sheet
    (explicit, selective mass scaling recomputed at every load step) with kinetic damping -- all velocities zeroed
    whenever the kinetic energy passes a peak -- until the free nodes' net force is below --tol of the pull force."""
    import mujoco
    from scipy.spatial import cKDTree
    x, c = K.read_vtk(a.mesh)
    w0, phi = None, None
    if a.mode:
        z = np.load(a.mode)
        mode = z["modes"][0] / np.abs(z["modes"][0]).max()
        phi = mode[cKDTree(z["x"]).query(x)[1]]             # FEniCS mode 1 on this mesh's nodes, max 1
        w0 = a.imp * H * phi
    xml, info = K.mjcf(x, c, a.L, E=E, h=H, nu=a.nu, bf=a.bf, dt=a.dt, w0=w0)
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    d = mujoco.MjData(m)
    j, jd = m.joint("pull").qposadr[0], m.joint("pull").dofadr[0]
    adr = K.free_dofs(m, info)
    dof = np.array([[m.jnt_dofadr[m.joint(f"v{i}{ax}").id] for ax in "xyz"] for i in info["free"]]).ravel()
    colour = K.colouring(info)
    if a.fenics:      # the FEniCS run's load steps (kirigami/api.py simulate): 0.5 Dc, then geometric from 1.05 Dc
        Dc = float(np.load(a.fenics)["Delta_c"])
        Ds = np.concatenate([[0.5 * Dc], np.geomspace(1.05 * Dc, a.Dmax, a.steps)])
    else:
        Ds = np.linspace(a.Dmax / a.steps, a.Dmax, a.steps)
    rows: list[tuple[float, ...]] = []
    t0 = time.time()
    f_out = OUT / f"{a.tag}.npz"
    if a.resume and f_out.exists():          # continue after the last saved load step, from its state
        old = np.load(f_out)
        rows = [tuple(r) for r in np.column_stack([old[k] for k in
                ("Delta", "F", "w_max", "w_rms", "residual", "iterations")])]
        d.qpos[adr] = old["U"]
        Ds = Ds[Ds > rows[-1][0] * (1 + 1e-9)]
        print(f"resuming after Delta = {rows[-1][0]:.4f} ({len(rows)} steps saved)")

    def scale(factor: float) -> np.ndarray:
        return K.set_armature(m, d, info, K.stiffness_diagonal(m, d, info, colour), a.dt, factor).ravel()

    if a.stop:
        Ds = Ds[Ds <= a.stop * (1 + 1e-9)]
    for D in Ds:
        d.qpos[j], d.qvel[:] = D, 0
        # move the free nodes affinely with the grip first: the membrane then starts near equilibrium
        if rows:
            d.qpos[adr[:, 0]] *= D / rows[-1][0]
        else:
            d.qpos[adr[:, 0]] = D * info["x"][info["free"], 0] / a.L
        # the predictor of the FEniCS run (kirigami/api.py simulate, `seed`): past the linear threshold a still-flat
        # sheet is started on the buckled branch, w = 1/2 sqrt(ell (D - Dc)) phi; relaxation then finds MuJoCo's own
        # equilibrium, and the seed decays if MuJoCo's flat sheet is still stable there
        if (phi is not None and w0 is not None and a.fenics and D > Dc
                and np.abs(d.qpos[adr[:, 2]]).max() < 0.1 * H):
            ell = 30.0                                    # min(W, 30 mm), as in simulate
            d.qpos[adr[:, 2]] = 0.5 * np.sqrt(ell * (D - Dc)) * phi[info["free"]] - w0[info["free"]]
        mujoco.mj_forward(m, d)
        q_start = d.qpos.copy()
        factor = a.factor
        for attempt in range(4):             # on divergence: back to the step's start with heavier nodes
            meff = scale(factor)
            ke_prev, it, res, ok = 0.0, 0, np.inf, True
            for it in range(1, a.maxit + 1):
                mujoco.mj_step(m, d)
                d.qpos[j], d.qvel[jd] = D, 0
                if not np.isfinite(d.qacc).all() or not np.isfinite(d.qpos).all():
                    ok = False
                    break
                v = d.qvel[dof]
                ke = float(meff @ v**2)
                if ke < ke_prev:                 # past a peak of kinetic energy: stop all motion (kinetic damping)
                    d.qvel[:] = 0
                    ke = 0.0
                ke_prev = ke
                if it % 50 == 0:
                    F = -d.qfrc_passive[jd]
                    res = np.linalg.norm(d.qfrc_passive[dof]) / max(abs(F), 1e-9)
                    if res < a.tol:
                        break
                if it % a.rescale == 0:          # the sheet stiffens as the flaps rotate: refresh the masses
                    d.qvel[:] = 0
                    meff = scale(factor)
                    ke_prev = 0.0
            if ok:
                break
            factor *= 2
            print(f"  Delta {D:7.4f}: diverged at iteration {it}; retrying with factor {factor:g}", flush=True)
            d.qpos[:], d.qvel[:], d.warning.number[:] = q_start, 0, 0
            mujoco.mj_forward(m, d)
        else:
            raise RuntimeError(f"unstable at Delta = {D:.4f} even with factor {factor:g}")
        F = -d.qfrc_passive[jd]
        w = d.qpos[adr[:, 2]]
        rows.append((D, F, np.abs(w).max(), np.sqrt(np.mean(w**2)), res, it))
        print(f"  Delta {D:7.4f} mm  F {F:8.4f} N  max|w| {np.abs(w).max():6.3f} mm  residual {res:.1e}  {it:6d} it"
              f"  ({time.time() - t0:.0f} s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        r = np.array(rows)
        np.savez(f_out, Delta=r[:, 0], F=r[:, 1], w_max=r[:, 2], w_rms=r[:, 3], residual=r[:, 4],
                 iterations=r[:, 5], L=a.L, dt=a.dt, nodes=len(x), seconds=time.time() - t0, mesh=str(a.mesh),
                 U=d.qpos[adr], X=info["x"][info["free"]])        # last state: displacement of each free node
        # every relaxed state, for rendering (results/mujoco_check/<tag>_states/<step>.npy: free-node displacements)
        sdir = OUT / f"{a.tag}_states"
        sdir.mkdir(exist_ok=True)
        np.save(sdir / f"{len(rows):03d}_D{D:.4f}.npy", d.qpos[adr].astype(np.float32))
    print(f"done in {time.time() - t0:.0f} s")


def plot(a: argparse.Namespace) -> None:
    """Force and max |w| against strain, FEniCS (--fenics) and MuJoCo (<tag>.npz). Only MuJoCo's relaxed steps
    (residual <= 10 tol) are drawn; if the run stopped relaxing, both curves end a little past the last relaxed step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    mj, Dlim = None, np.inf
    f = OUT / f"{a.tag}.npz"
    if f.exists():
        r = np.load(f)
        ok = r["residual"] <= 10 * a.tol
        n = len(ok) if ok.all() else int(np.argmin(ok))
        mj = np.r_[0, r["Delta"][:n]], np.r_[0, r["F"][:n]], np.r_[0, r["w_max"][:n]]
        if n < len(ok):
            Dlim = 1.15 * mj[0][-1]
            print(f"plotting the {n} relaxed steps (to Delta = {mj[0][-1]:.4f} mm); step {n + 1} on did not relax")
    if a.fenics:
        z = np.load(a.fenics)
        D, F, uz = np.r_[0, z["Delta"]], np.r_[0, z["F"]], np.r_[0, z["uz_max"]]
        k = D <= Dlim
        ax[0].plot(100 * D[k] / a.L, F[k], "o-", color="k", ms=3, lw=1.2, label="FEniCSx (KL shell)")
        ax[1].plot(100 * D[k] / a.L, uz[k], "o-", color="k", ms=3, lw=1.2, label="FEniCSx")
    if mj is not None:
        D, F, w = mj
        ax[0].plot(100 * D / a.L, F, "s--", color="#D55E00", ms=3.5, lw=1.2, label="MuJoCo (same design, 2D flex)")
        ax[1].plot(100 * D / a.L, w, "s--", color="#D55E00", ms=3.5, lw=1.2, label="MuJoCo")
    ax[0].set(xlabel="strain Δ/L [%]", ylabel="force F [N]")
    ax[1].set(xlabel="strain Δ/L [%]", ylabel="max out-of-plane |w| [mm]")
    for x_ in ax:
        x_.set_ylim(bottom=0)
        x_.legend(frameon=False, fontsize=8)
        x_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(a.title or a.tag, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{a.tag}.png", dpi=150)
    print(f"wrote {OUT / (a.tag + '.png')}")


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--mesh")
    p.add_argument("--mode", help="buckling.npz of a FEniCS buckle run (imperfection = its mode 1)")
    p.add_argument("--fenics", help="history.npz of the FEniCS simulate run, for --plot")
    p.add_argument("--L", type=positive, default=120.0)
    p.add_argument("--Dmax", type=positive, default=12.0)
    p.add_argument("--steps", type=count, default=40, help="load steps (as the FEniCS run)")
    p.add_argument("--dt", type=positive, default=1e-4, help="explicit step (masses are scaled to it)")
    p.add_argument("--factor", type=positive, default=6.0, help="mass scaling: m_eff >= factor K_ii dt^2")
    p.add_argument("--tol", type=positive, default=1e-4, help="relaxed when |net force on free nodes| < tol |F|")
    p.add_argument("--rescale", type=count, default=2000, help="refresh the mass scaling every this many iterations")
    p.add_argument("--stop", type=positive, default=None, help="last load value to run (mm), keeping the step list")
    p.add_argument("--resume", action="store_true", help="continue from the saved <tag>.npz")
    p.add_argument("--maxit", type=count, default=40000, help="iterations per load step at most")
    p.add_argument("--imp", type=nonneg, default=0.01, help="imperfection amplitude / h")
    p.add_argument("--nu", type=float, default=0.0, help="Poisson ratio (as in the FEniCS run)")
    p.add_argument("--bf", type=positive, default=K.BENDING_FACTOR)
    p.add_argument("--tag", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--plot", action="store_true", help="only plot what is saved")
    a = p.parse_args()
    for f in (a.mode, a.fenics):
        if f and not Path(f).is_file():
            p.error(f"no such file: {f}")
    if a.plot:
        plot(a)
    elif not a.mesh or not Path(a.mesh).is_file():
        p.error("a run needs --mesh, the design's mesh.vtk")
    else:
        relax(a)


if __name__ == "__main__":
    main()
