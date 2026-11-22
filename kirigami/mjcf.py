"""KiriSMILES design -> MuJoCo model (MJCF): the solver's own triangle mesh as a MuJoCo 2D flex shell.

The mesh is the one the FEniCS solver uses (linear triangles in mm, nodes along every cut duplicated so the lips are
free): mesh.vtk from the `mesh` command / export_mesh, read with read_vtk(). Each free node becomes a body with three
slide joints; the nodes on x = 0 belong to the world and those on x = L to a grip body that slides along x (joint
"pull", driven by setting its qpos). The clamp is a clamp, not a hinge: a MuJoCo shell vertex has no rotation, so a
row of ghost nodes one element outside each end, moving with the clamp, holds the slope at zero (FEniCS: w = 0,
dw/dn = 0) while the free length stays exactly L.

Material. MuJoCo >= 4.0 takes the physical E, nu and h: measured by imposing exact strain and curvature fields
(tests/test_mjcf.py), its membrane is plane stress exactly and its cylindrical bending stiffness is the plate's
E h^3 / 12(1-nu^2) (to within 2.5% on 40 elements per side). MuJoCo 3.x had other constants (membrane shear x1/4 and
the 3D lambda x1/2; bending 3/4 (1-nu) D), which shell_inputs() maps exactly: poisson nu/2, young and thickness
rescaled. What no input fixes, in either version: the discrete bending has no Kirchhoff Poisson coupling, so twisting
is 1.3-2x too stiff (nu = 0 to 0.38) and bending with free edges (anticlastic curvature) up to 25% too stiff at
nu = 0.38 (scripts/pristine_mujoco.py, uncut sheet against FEniCS). Kirigami flaps bend and twist, so force-strain
curves inherit part of this.

Units: mm, N, s, mass in tonnes (E in MPa). The default density is not PET's (1.4e-9 t/mm^3) but 1e-6:
MuJoCo rejects body masses below 1e-15, and a quasi-static result does not depend on the density. Gravity is off.
This module writes XML with numpy alone; loading the model needs `pip install mujoco` (>= 4.0 recommended).
"""
from __future__ import annotations

import numpy as np

MEMBRANE_FACTOR = 0.25      # MuJoCo 3.x: membrane shear stiffness / (young * thickness / 2(1+poisson))
BENDING_FACTOR = 0.73       # MuJoCo 3.x: cylindrical bending stiffness / (young * thickness^3 / 12(1+poisson))


def mujoco_major() -> int:
    """Major version of the installed mujoco package (4 if it is not installed: the XML is then for the latest)."""
    try:
        from importlib.metadata import version
        return int(version("mujoco").split(".")[0])
    except Exception:
        return 4


def shell_inputs(E: float, h: float, nu: float = 0.0, mf: float = MEMBRANE_FACTOR,
                 bf: float = BENDING_FACTOR, major: int | None = None) -> tuple[float, float, float]:
    """(young, poisson, thickness) for MuJoCo's flex so that its membrane is the plane-stress law of (E, nu) with
    thickness h, and its cylindrical bending stiffness is E h^3 / 12 (1 - nu^2).

    MuJoCo >= 4 uses these constants itself (4.0.0 fixed the flex elastic constants): the physical values are returned.
    MuJoCo 3.x needs the mapping measured in tests/test_mjcf.py: poisson nu / 2, young and thickness rescaled."""
    if not 0.0 <= nu < 0.5:
        raise ValueError(f"Poisson ratio {nu} outside [0, 0.5)")
    if (mujoco_major() if major is None else major) >= 4:
        return E, nu, h
    nu_in = nu / 2
    Et = E * h * (1 + nu_in) / (mf * (1 + nu))                          # young * thickness for MuJoCo
    D = E * h**3 / (12 * (1 - nu**2))
    t_in = np.sqrt(12 * D * (1 + nu_in) / (bf * Et))                    # bf * young * t^3 / 12(1+nu') = D
    return Et / t_in, nu_in, t_in


def ghost_layer(x: np.ndarray, cells: np.ndarray, L: float, delta: float, tol: float = 1e-6):
    """Ghost nodes one strip outside x = 0 and x = L, and the triangles joining them to the boundary edges.
    Returns (x_all, cells_all, side) where side is -1 at x <= 0 (fixed), +1 at x >= L (grip), 0 free."""
    edges = {}
    for t in cells:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    xs, new_cells = [x], []
    ghost = {}
    n = len(x)
    for (a, b), count in edges.items():
        if count != 1:
            continue
        for x_end, sgn in ((0.0, -1.0), (L, 1.0)):
            if abs(x[a, 0] - x_end) < tol and abs(x[b, 0] - x_end) < tol:
                gs = []
                for v in (a, b):
                    if v not in ghost:
                        ghost[v] = n
                        xs.append(np.array([[x[v, 0] + sgn * delta, x[v, 1]]]))
                        n += 1
                    gs.append(ghost[v])
                # orientation does not matter to MuJoCo's flex energy; keep it consistent with the interior anyway
                new_cells += [[a, b, gs[1]], [a, gs[1], gs[0]]]
    x_all = np.vstack(xs)
    cells_all = np.vstack([cells, np.array(new_cells, dtype=cells.dtype)]) if new_cells else cells
    side = np.where(x_all[:, 0] < tol, -1, np.where(x_all[:, 0] > L - tol, 1, 0))
    return x_all, cells_all, side


def mjcf(x: np.ndarray, cells: np.ndarray, L: float, *, E: float = 3500.0, h: float = 0.127, rho: float = 1e-6,
         dt: float = 1e-6, damping: float = 0.0, ghost: float | None = None, w0: np.ndarray | None = None,
         armature: float = 0.0, mass_damping: float = 0.0, nu: float = 0.0,
         mf: float = MEMBRANE_FACTOR, bf: float = BENDING_FACTOR) -> tuple[str, dict]:
    """MJCF string of the sheet and an info dict (node bookkeeping for reading results back).

    x (n, 2) in mm, cells (m, 3); ghost = width of the ghost strip (default: median edge length).
    w0 (n,): initial out-of-plane offset of each mesh node (mm), the imperfection that picks the buckling pattern.
    damping: MuJoCo's flex (stiffness-proportional) damping, s.
    armature: extra inertia per node: a number (multiple of the node's mass, in-plane only) or an (n_nodes, 3) array
        of absolute values per direction (stable_armature). A quasi-static result does not depend on mass; heavier
        nodes where the mesh is fine or the membrane is stiff let explicit steps be longer (selective mass scaling).
    mass_damping: c / m (1/s) of a viscous force on the out-of-plane motion of every node (a velocity actuator per
        node, since MuJoCo does not allow joint damping on flex vertices): damps the slow flap modes that
        stiffness-proportional damping cannot reach. In-plane motion is not damped this way, which would drag on the
        pull; `damping` (stiffness-proportional) takes out the fast in-plane ringing.
    The grip (joint "pull", x = L end) has no actuator: drive it by setting its qpos/qvel; the sheet's pull on it is
    -qfrc_passive at its dof.
    """
    x = np.asarray(x, float)
    cells = np.asarray(cells, int)
    if ghost is None:
        e = np.linalg.norm(x[cells[:, 0]] - x[cells[:, 1]], axis=1)
        ghost = float(np.median(e))
    xa, ca, side = ghost_layer(x, cells, L, ghost)
    p = np.column_stack([xa, np.zeros(len(xa))])
    if w0 is not None:
        p[: len(x), 2] = np.where(side[: len(x)] == 0, w0, 0.0)
    tri = p[ca]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    mass = np.zeros(len(p))
    np.add.at(mass, ca.ravel(), np.repeat(area * rho * h / 3, 3))
    young, nu_in, t_in = shell_inputs(E, h, nu, mf, bf)
    grip_mass = float(mass[side == 1].sum())
    bodies, vbody, vpos, acts = [], [], [], []
    for i, q in enumerate(p):
        if side[i] == -1:
            vbody.append("world")
            vpos.append(q)
        elif side[i] == 1:
            vbody.append("grip")
            vpos.append(q - [L, 0, 0])
        else:
            m_i = mass[i]
            arm = (armature[i] if np.ndim(armature) else (armature * m_i, armature * m_i, 0.0))
            joints = "".join(f'<joint name="v{i}{a}" type="slide" axis="{ax}" armature="{arm[k]:.6g}"/>'
                             for k, (a, ax) in enumerate((("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1"))))
            bodies.append(f'<body name="v{i}" pos="{q[0]:.8g} {q[1]:.8g} {q[2]:.8g}">'
                          f'<inertial pos="0 0 0" mass="{m_i:.8g}" diaginertia="{m_i:.3g} {m_i:.3g} {m_i:.3g}"/>'
                          f'{joints}</body>')
            if mass_damping:
                acts.append(f'<velocity joint="v{i}z" kv="{mass_damping * (m_i + arm[2]):.6g}"/>')
            vbody.append(f"v{i}")
            vpos.append(np.zeros(3))
    xml = f"""<mujoco model="kirigami">
  <option timestep="{dt:.6g}" gravity="0 0 0" integrator="Euler"/>
  <size memory="2G"/>
  <worldbody>
    <body name="grip" pos="{L:.8g} 0 0">
      <inertial pos="0 0 0" mass="{max(grip_mass, 1e-12):.8g}" diaginertia="1 1 1"/>
      <joint name="pull" type="slide" axis="1 0 0"/>
    </body>
    {''.join(bodies)}
  </worldbody>
  <deformable>
    <flex name="sheet" dim="2" radius="{t_in / 2:.6g}" body="{' '.join(vbody)}"
          vertex="{' '.join(f'{v:.8g}' for q in vpos for v in q)}"
          element="{' '.join(map(str, ca.ravel()))}">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <elasticity young="{young:.8g}" poisson="{nu_in:.8g}" thickness="{t_in:.8g}" damping="{damping:.6g}" elastic2d="both"/>
    </flex>
  </deformable>
  <actuator>{''.join(acts)}</actuator>
</mujoco>"""
    info = {"n_nodes": len(x), "x": xa, "cells": ca, "side": side, "free": np.flatnonzero(side == 0),
            "mass": mass, "young": young, "poisson_in": nu_in, "t_in": t_in, "nu": nu, "ghost": ghost}
    return xml, info


def read_vtk(path) -> tuple[np.ndarray, np.ndarray]:
    """(x (n, 2), triangles (m, 3)) from the legacy ASCII mesh.vtk that export_mesh / the `mesh` command writes."""
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("POINTS"))
    n = int(lines[i].split()[1])
    pts = np.array(" ".join(lines[i + 1:i + 1 + 3 * n]).split()[: 3 * n], float).reshape(n, 3)
    j = next(k for k, l in enumerate(lines) if l.startswith("CELLS"))
    m = int(lines[j].split()[1])
    tris = [list(map(int, l.split()[1:])) for l in lines[j + 1:j + 1 + m] if l.split()[0] == "3"]
    return pts[:, :2], np.array(tris, int)


def free_dofs(model, info) -> np.ndarray:
    """qpos addresses (n_free, 3) of the free nodes' x, y, z slides, in info['free'] order."""
    adr = np.empty((len(info["free"]), 3), int)
    for k, i in enumerate(info["free"]):
        b = model.body(f"v{i}").id
        j0 = model.body_jntadr[b]
        adr[k] = model.jnt_qposadr[j0:j0 + 3]
    return adr


def colouring(info) -> np.ndarray:
    """A colour per mesh node such that nodes of one colour are more than two elements apart (no shared element or
    bending stencil), so perturbing them together gives each one's own stiffness diagonal. -1 for clamped nodes."""
    n = len(info["x"])
    nbr = [set() for _ in range(n)]
    for t in info["cells"]:
        for a in t:
            nbr[a].update(int(b) for b in t)
    nbr2 = [set().union(*(nbr[b] for b in nbr[a])) for a in range(n)]
    colour = -np.ones(n, int)
    for a in info["free"]:
        used = {colour[b] for b in nbr2[a] if colour[b] >= 0}
        colour[a] = next(c for c in range(len(used) + 1) if c not in used)
    return colour


def stiffness_diagonal(model, data, info, colour, eps: float = 1e-5) -> np.ndarray:
    """MuJoCo's own tangent stiffness diagonal K_ii (N/mm), (n_free, 3), at the current state of `data` (restored
    afterwards), from its flex forces: one force evaluation per colour and direction."""
    import mujoco
    adr = free_dofs(model, info)
    dof = np.array([[model.jnt_dofadr[model.joint(f"v{i}{a}").id] for a in "xyz"] for i in info["free"]])
    q0, v0 = data.qpos.copy(), data.qvel.copy()
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    f0 = data.qfrc_passive.copy()
    col = colour[info["free"]]
    K = np.zeros((len(info["free"]), 3))
    for c in range(col.max() + 1):
        k = np.flatnonzero(col == c)
        for a in range(3):
            data.qpos[:] = q0
            data.qpos[adr[k, a]] += eps
            mujoco.mj_forward(model, data)
            K[k, a] = -(data.qfrc_passive[dof[k, a]] - f0[dof[k, a]]) / eps
    data.qpos[:], data.qvel[:] = q0, v0
    mujoco.mj_forward(model, data)
    return K


def set_armature(model, data, info, K: np.ndarray, dt: float, factor: float = 2.0) -> np.ndarray:
    """Selective mass scaling: armature per free node and direction so that each has m_eff >= factor K_ii dt^2, which
    keeps explicit steps of dt stable (Gershgorin). Written into model.dof_armature, then mj_setConst (MuJoCo keeps
    the mass matrix's constant part compiled: without it a changed armature has no effect; it resets the state,
    which is restored). Returns m_eff (n_free, 3)."""
    import mujoco
    dof = np.array([[model.jnt_dofadr[model.joint(f"v{i}{a}").id] for a in "xyz"] for i in info["free"]])
    m = info["mass"][info["free"]][:, None]
    arm = np.clip(factor * np.abs(K) * dt**2 - m, 0.0, None)
    model.dof_armature[dof] = arm
    q, v = data.qpos.copy(), data.qvel.copy()
    mujoco.mj_setConst(model, data)              # works at qpos0: put the state back
    data.qpos[:], data.qvel[:] = q, v
    mujoco.mj_forward(model, data)
    return m + arm
