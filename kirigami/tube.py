"""Kirchhoff-Love shell on a closed cylindrical tube (experimental).

The tube is meshed as a closed surface in 3D (gmsh), so there is no seam: the
elements that meet where the sheet's two edges join are ordinary neighbours and
the C0 interior-penalty kink term acts there as everywhere else.

Kinematics follow kirigami/shell.py, written in the tube's orthonormal tangent
basis e1 = e_theta (around the tube), e2 = e_z (along it):
  F       = [d phi/ds, d phi/dz]                 (3 x 2), phi = X + u
  E       = (F^T F - I) / 2                      membrane strain (shell._membrane)
  K_ab    = n . d^2 phi / ds_a ds_b - c B_ab     bending strain
where B is the curvature of the undeformed tube. The constant c selects the
natural (stress-free) shape:
  c = 1   a tube made curved (cut from tubing): stress-free as meshed;
  c = 0   a flat sheet rolled into a tube (Rafsanjani et al., PNAS 2019): its
          natural shape is flat, so the rolled tube carries a bending prestress.
Ramping c from 1 to 0 at fixed ends is the rolling step.

Boundary conditions (the clamped ends are those of the super-cells of Rafsanjani
et al., PNAS 2019, SI section 4):
  Tube(ends="free")      rings held in z (top moved by Delta) and around (penalty
                         on u . e_theta); free radially and in rotation. Uncut
                         benchmark only.
  Tube(ends="clamped")   bottom ring fully fixed; top ring u_z = Delta, free
                         radially and around; neither ring may rotate (Nitsche
                         penalty on the slope across the ring). Finite tube with
                         end caps, as the experiments and the super-cells.
  PeriodicTube           one period of an infinitely long tube: u(ghost) =
                         u(master) + Delta e_z with slope continuity across the
                         periodic interface; Delta free (zero force) while rolling.
Strain is measured from the rolled length; the rolling step leaves the axial
length free in every case.

radial_imperfection seeds a small geometric imperfection, needed to get through
the pop-up when the cells are all alike (periodic tube).

Units: mm, MPa, N. Cuts come from tube_cut_mesh (zero-width cracks or kerf
holes); no KiriSMILES support yet.
"""
import numpy as np
import ufl
import basix.ufl
from dolfinx import fem
from dolfinx.io import gmshio
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

from kirigami.fvk import _moment
from kirigami.shell import _membrane

BOTTOM, TOP = 1, 2


def tube_mesh(R, Lz, h, order=2):
    """Closed lateral surface of a cylinder of radius R and length Lz (along z),
    element size h, geometry of the given order. Returns (mesh, facet tags)."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tube")
    c = gmsh.model.occ.addCircle(0, 0, 0, R)
    out = gmsh.model.occ.extrude([(1, c)], 0, 0, Lz)
    gmsh.model.occ.synchronize()
    surf = [t for d, t in out if d == 2]
    bot, top = [], []
    for d, t in gmsh.model.getEntities(1):
        bb = gmsh.model.getBoundingBox(d, t)
        tol = 1e-5 * max(1.0, Lz)                 # OCC bounding boxes are padded
        if abs(bb[5]) < tol:
            bot.append(t)
        elif abs(bb[2] - Lz) < tol:
            top.append(t)
    if not bot or not top:
        raise RuntimeError("tube_mesh: end rings not found")
    gmsh.model.addPhysicalGroup(2, surf, 1)
    gmsh.model.addPhysicalGroup(1, bot, BOTTOM)
    gmsh.model.addPhysicalGroup(1, top, TOP)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    msh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return msh, ft


def _clip(poly, C):
    """Pieces of a polyline (and its copies shifted by multiples of C) that lie in
    0 <= x <= C."""
    P = np.asarray(poly, float)
    out = []
    for k in range(int(np.floor(P[:, 0].min() / C)) - 1, int(np.ceil(P[:, 0].max() / C)) + 1):
        Q = P - [k * C, 0.0]
        pts = [Q[0]]
        for a, b in zip(Q[:-1], Q[1:]):
            ts = sorted((xc - a[0]) / (b[0] - a[0]) for xc in (0.0, C) if (a[0] - xc) * (b[0] - xc) < 0)
            pts += [a + t * (b - a) for t in ts] + [b]
        pts = np.array(pts)
        cur = []
        for a, b in zip(pts[:-1], pts[1:]):
            if -1e-9 <= 0.5 * (a[0] + b[0]) <= C + 1e-9:
                if not cur:
                    cur = [a]
                cur.append(b)
            elif cur:
                out.append(np.array(cur))
                cur = []
        if cur:
            out.append(np.array(cur))
    return out


def slot(poly, w):
    """Closed outline of width w around an open polyline (square ends, mitred
    corners): a cut with kerf w."""
    P = np.asarray(poly, float)
    d = np.diff(P, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    off = [nrm[0]]
    for k in range(1, len(P) - 1):
        m = nrm[k - 1] + nrm[k]
        off.append(m / np.dot(m, nrm[k]))          # mitre: unit offset from both segments
    off.append(nrm[-1])
    off = np.array(off) * w / 2
    return [tuple(q) for q in np.vstack([P + off, (P - off)[::-1]])]


def tube_cut_mesh(C, Lz, cuts, tips=(), h_far=1.2, h_cut=0.5, h_tip=0.15, r_tip=1.0,
                  cut_dist=(0.3, 2.0), holes=(), periodic_z=False):
    """Closed tube of circumference C and length Lz with zero-width cuts and holes.

    holes  closed polygons in (s, z) removed from the sheet (e.g. slot(cut, kerf));
           they may also cross the seam.

    cuts  polylines in the unrolled sheet (s, z), s around the tube; they may cross
          the seam s = 0 = C and are wrapped. tips: points (s, z) around which the
          mesh is refined to h_tip within r_tip (cut ends, hinges).
    The flat strip [0, C] x [0, Lz] is meshed with matching nodes on s = 0 and
    s = C (gmsh periodic meshing), those nodes are merged, nodes on the cuts are
    split so that the two faces of each cut are free edges, and the P2 mesh is
    wrapped onto the cylinder of radius C / 2 pi. Returns (mesh, facet tags, info).

    periodic_z  the pattern repeats along the tube with period Lz (holes only, no
          zero-width cuts). The strip is meshed periodically in z as well, and a
          one-element "ghost" band, a copy of the cells along z = 0 shifted by Lz, is
          attached above z = Lz. info["ghost_cells"] lists the ghost cells and
          info["master"] pairs (node, node it copies) for every ghost node, including
          the nodes on z = Lz; PeriodicTube ties each to its partner plus Delta e_z."""
    import gmsh
    from dolfinx import mesh as dmesh
    from dolfinx.io.gmshio import cell_perm_array
    R = C / (2 * np.pi)
    pieces = [p for c in cuts for p in _clip(c, C)]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tube_flat")
    occ = gmsh.model.occ
    sheet = [(2, occ.addRectangle(0, 0, 0, C, Lz))]
    if periodic_z and len(cuts):
        raise NotImplementedError("tube_cut_mesh: periodic_z supports holes (kerf cuts) only")
    zsh = (-Lz, 0.0, Lz) if periodic_z else (0.0,)
    tools = []
    for h in holes:
        H = np.asarray(h, float)
        for sh, shz in ((a, b) for a in (-C, 0.0, C) for b in zsh):
            if (H[:, 0].max() + sh > 0 and H[:, 0].min() + sh < C
                    and H[:, 1].max() + shz > 0 and H[:, 1].min() + shz < Lz):
                t = [occ.addPoint(x + sh, y + shz, 0) for x, y in H]
                ls = [occ.addLine(a, b) for a, b in zip(t, t[1:] + t[:1])]
                tools.append((2, occ.addPlaneSurface([occ.addCurveLoop(ls)])))
    if tools:
        sheet, _ = occ.cut(sheet, tools)
    lines = []
    for p in pieces:
        t = [occ.addPoint(x, y, 0) for x, y in p]
        lines += [occ.addLine(a, b) for a, b in zip(t[:-1], t[1:])]
    if lines:
        occ.fragment(sheet, [(1, l) for l in lines])
    occ.synchronize()
    surfs = [s for _, s in gmsh.model.getEntities(2)]
    bnd = {abs(c) for _, c in gmsh.model.getBoundary([(2, s) for s in surfs], combined=False,
                                                      oriented=False)}
    cut_curves = [c for _, c in gmsh.model.getEntities(1) if c not in bnd]
    tol = 1e-6 * max(C, Lz)

    def side(x0):
        cs = []
        for c in bnd:
            bb = gmsh.model.getBoundingBox(1, c)
            if abs(bb[0] - x0) < 1e-4 and abs(bb[3] - x0) < 1e-4:
                cs.append((bb[1], c))
        return [c for _, c in sorted(cs)]
    left, right = side(0.0), side(C)

    def outer(c):
        bb = gmsh.model.getBoundingBox(1, c)
        return any(abs(bb[i] - v) < 1e-4 and abs(bb[i + 3] - v) < 1e-4
                   for i, v in ((0, 0.0), (0, C), (1, 0.0), (1, Lz)))
    hole_curves = [c for c in bnd if not outer(c)]
    if len(left) != len(right) or not left:
        raise RuntimeError("tube_cut_mesh: seam edges do not match")
    for cl, cr in zip(left, right):
        gmsh.model.mesh.setPeriodic(1, [cr], [cl], [1, 0, 0, C, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    if periodic_z:
        def row(y0):
            cs = []
            for c in bnd:
                bb = gmsh.model.getBoundingBox(1, c)
                if abs(bb[1] - y0) < 1e-4 and abs(bb[4] - y0) < 1e-4:
                    cs.append((bb[0], c))
            return [c for _, c in sorted(cs)]
        bot_c, top_c = row(0.0), row(Lz)
        if len(bot_c) != len(top_c) or not bot_c:
            raise RuntimeError("tube_cut_mesh: z = 0 and z = Lz edges do not match")
        for cb, ct in zip(bot_c, top_c):
            gmsh.model.mesh.setPeriodic(1, [ct], [cb], [1, 0, 0, 0, 0, 1, 0, Lz, 0, 0, 1, 0, 0, 0, 0, 1])
    f = gmsh.model.mesh.field
    fields = []
    if cut_curves or hole_curves:
        d = f.add("Distance")
        f.setNumbers(d, "CurvesList", cut_curves + hole_curves)
        f.setNumber(d, "Sampling", 200)
        t = f.add("Threshold")
        f.setNumber(t, "InField", d)
        f.setNumber(t, "SizeMin", h_cut)
        f.setNumber(t, "SizeMax", h_far)
        f.setNumber(t, "DistMin", cut_dist[0])
        f.setNumber(t, "DistMax", cut_dist[1])
        fields.append(t)
    for xs, ys in tips:
        for sh, shz in ((a, b) for a in (-C, 0.0, C) for b in zsh):
            if -3 * r_tip < xs + sh < C + 3 * r_tip and -3 * r_tip < ys + shz < Lz + 3 * r_tip:
                b = f.add("Ball")
                f.setNumber(b, "XCenter", xs + sh)
                f.setNumber(b, "YCenter", ys + shz)
                f.setNumber(b, "ZCenter", 0)
                f.setNumber(b, "Radius", r_tip)
                f.setNumber(b, "Thickness", 2 * r_tip)
                f.setNumber(b, "VIn", h_tip)
                f.setNumber(b, "VOut", h_far)
                fields.append(b)
    if fields:
        fm = f.add("Min")
        f.setNumbers(fm, "FieldsList", fields)
        f.setAsBackgroundMesh(fm)
    for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(opt, 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_far)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)

    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    idx = np.full(int(tags.max()) + 1, -1)
    idx[tags.astype(int)] = np.arange(len(tags))
    tri = []
    for s in surfs:
        types, _, nodes = gmsh.model.mesh.getElements(2, s)
        for ty, nd in zip(types, nodes):
            if ty != 9:
                raise RuntimeError(f"tube_cut_mesh: unexpected element type {ty}")
            tri.append(idx[nd.astype(int)].reshape(-1, 6))
    tri = np.vstack(tri)
    cut_edges = []
    for c in cut_curves:
        types, _, nodes = gmsh.model.mesh.getElements(1, c)
        for ty, nd in zip(types, nodes):
            cut_edges.append(idx[nd.astype(int)].reshape(-1, 3))
    cut_edges = np.vstack(cut_edges) if cut_edges else np.zeros((0, 3), int)
    gmsh.finalize()

    # merge the seam: nodes on s = C onto their partners on s = 0
    used = np.unique(tri)
    rep = np.arange(len(xyz))
    on_l = used[np.abs(xyz[used, 0]) < tol]
    on_r = used[np.abs(xyz[used, 0] - C) < tol]
    if len(on_l) != len(on_r):
        raise RuntimeError(f"tube_cut_mesh: seam node counts differ ({len(on_l)}, {len(on_r)})")
    yl = xyz[on_l, 1]
    order = np.argsort(yl)
    for i in on_r:
        k = np.searchsorted(yl[order], xyz[i, 1])
        cand = [order[j] for j in (k - 1, k) if 0 <= j < len(order)]
        j = min(cand, key=lambda q: abs(yl[q] - xyz[i, 1]))
        if abs(yl[j] - xyz[i, 1]) > 1e-6:
            raise RuntimeError(f"tube_cut_mesh: seam node at z = {xyz[i, 1]} has no partner")
        rep[i] = on_l[j]
    tri, cut_edges = rep[tri], rep[cut_edges]

    n_real = len(tri)
    master = []
    if periodic_z:
        # nodes on z = Lz are tied to their partners on z = 0
        used = np.unique(tri)
        on_b = used[np.abs(xyz[used, 1]) < tol]
        on_t = used[np.abs(xyz[used, 1] - Lz) < tol]
        if len(on_b) != len(on_t):
            raise RuntimeError(f"tube_cut_mesh: ring node counts differ ({len(on_b)}, {len(on_t)})")
        xb = xyz[on_b, 0]
        ob = np.argsort(xb)
        partner = {}
        for i in on_t:
            k = np.searchsorted(xb[ob], xyz[i, 0])
            cand = [ob[j] for j in (k - 1, k) if 0 <= j < len(ob)]
            j = min(cand, key=lambda q: abs(xb[q] - xyz[i, 0]))
            if abs(xb[j] - xyz[i, 0]) > 1e-6:
                raise RuntimeError(f"tube_cut_mesh: top node at s = {xyz[i, 0]} has no partner")
            partner[on_b[j]] = i
            master.append((i, on_b[j]))
        # ghost band: copies of the cells with an edge on z = 0, shifted by Lz
        corner = np.abs(xyz[tri[:, :3], 1]) < tol
        band = tri[corner.sum(1) >= 2]
        new_id, extra = {}, []
        for n in np.unique(band):
            if n in partner:
                new_id[n] = partner[n]
            else:
                new_id[n] = len(xyz) + len(extra)
                extra.append(xyz[n] + [0.0, Lz, 0.0])
                master.append((new_id[n], n))
        xyz = np.vstack([xyz, np.array(extra).reshape(-1, 3)])
        tri = np.vstack([tri, np.vectorize(new_id.get)(band)])

    # split nodes along the cuts: slots (triangle, local node) are joined across
    # every shared edge that is not a cut edge; each joined class becomes one node
    T = len(tri)
    parent = np.arange(6 * T)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    cut_set = {(min(a, b), max(a, b)) for a, b, _ in cut_edges}
    edges = {}
    for t in range(T):
        for la, lb, lm in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
            a, b = tri[t, la], tri[t, lb]
            edges.setdefault((min(a, b), max(a, b)), []).append((t, la, lb, lm))
    n_cut = 0
    for k, occs in edges.items():
        if len(occs) > 2:
            raise RuntimeError("tube_cut_mesh: non-manifold edge")
        if len(occs) < 2:
            continue
        if k in cut_set:
            n_cut += 1
            continue
        (t1, a1, b1, m1), (t2, a2, b2, m2) = occs
        slot2 = {tri[t2, a2]: 6 * t2 + a2, tri[t2, b2]: 6 * t2 + b2}
        union(6 * t1 + a1, slot2[tri[t1, a1]])
        union(6 * t1 + b1, slot2[tri[t1, b1]])
        union(6 * t1 + m1, 6 * t2 + m2)
    if n_cut != len(cut_set):
        raise RuntimeError(f"tube_cut_mesh: {len(cut_set) - n_cut} cut edges are not interior")
    roots = np.array([find(a) for a in range(6 * T)])
    _, new = np.unique(roots, return_inverse=True)
    cells = new.reshape(T, 6)
    first = np.empty(new.max() + 1, dtype=int)
    first[new] = np.arange(6 * T)                      # any slot of a class: same original node
    src = tri.reshape(-1)[first]
    s_, z_ = xyz[src, 0], xyz[src, 1]
    X = np.column_stack([R * np.cos(s_ / R), R * np.sin(s_ / R), z_])

    cells = cells[:, cell_perm_array(dmesh.CellType.triangle, 6)].astype(np.int64)
    el = basix.ufl.element("Lagrange", "triangle", 2, shape=(3,))
    msh = dmesh.create_mesh(MPI.COMM_WORLD, cells, X, ufl.Mesh(el))
    msh.topology.create_connectivity(1, 2)
    bot = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[2], 0.0))
    top = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[2], Lz))
    fac = np.concatenate([bot, top]).astype(np.int32)
    val = np.concatenate([np.full(len(bot), BOTTOM), np.full(len(top), TOP)]).astype(np.int32)
    o = np.argsort(fac)
    ft = dmesh.meshtags(msh, 1, fac[o], val[o])
    info = {"triangles": int(n_real), "nodes": int(new.max() + 1), "cut_pieces": len(pieces),
            "cut_edges": int(len(cut_set)), "seam_nodes": int(len(on_l)), "R": float(R)}
    if periodic_z:
        cls = np.empty(len(xyz), dtype=np.int64)
        cls[tri.reshape(-1)] = new                       # no cuts: one class per node
        info["ghost_cells"] = np.arange(n_real, T)
        info["master"] = np.array([(cls[g], cls[m]) for g, m in master], dtype=np.int64)
    return msh, ft, info


def radial_imperfection(msh, Lz, amp, seed=0, m_max=16, k_max=8):
    """Perturb the tube's geometry radially, in place: x -> x + w0(theta, z) e_r with

        w0 = sum_{m=0..m_max, k=0..k_max, (m,k) != (0,0)} a_mk cos(m theta + 2 pi k z / Lz + p_mk)

    a_mk ~ N(0, 1) and p_mk ~ U(0, 2 pi) drawn from `seed`, then scaled so that
    max |w0| over the mesh nodes is amp. Every term has period Lz in z, so a ghost
    node of a PeriodicTube (a copy shifted by Lz) moves exactly as the node it
    copies. The perturbed tube is the reference (stress-free when c = 1) shape: a
    geometric imperfection, as Abaqus' *IMPERFECTION, not a load. It breaks the
    symmetry between the cells around and along the tube, so that Newton does not
    stall at the bifurcation where identical cells could buckle in many ways.
    Returns the applied max |w0|."""
    x = msh.geometry.x
    th = np.arctan2(x[:, 1], x[:, 0])
    z = x[:, 2]
    rng = np.random.default_rng(seed)
    w = np.zeros(len(x))
    for m in range(m_max + 1):
        for k in range(k_max + 1):
            if m == 0 and k == 0:
                continue
            w += rng.normal() * np.cos(m * th + 2 * np.pi * k * z / Lz + rng.uniform(0, 2 * np.pi))
    w *= amp / np.abs(w).max()
    r = np.hypot(x[:, 0], x[:, 1])
    x[:, 0] += w * x[:, 0] / r
    x[:, 1] += w * x[:, 1] / r
    return float(np.abs(w).max())


class Tube:
    """ends  "free":    rings held in z and around (penalty on u . e_theta), free
                       radially and in rotation (uncut benchmark);
          "clamped": as Rafsanjani et al.'s super-cells: the bottom ring fully
                       fixed, the top ring moved in z and free radially and around,
                       and neither ring allowed to rotate.
    mesh  optional (mesh, facet tags) from tube_cut_mesh; h is then the smallest
          element size (used in the interior-penalty terms)."""

    def __init__(self, R, Lz, mat, h=0.5, alpha=10.0, order=2, mesh=None, ends="free",
                 imperfection=None):
        self.R, self.Lz, self.mat, self.alpha, self.h = R, Lz, mat, alpha, h
        self.ends = ends
        self.msh, self.ft = mesh if mesh is not None else tube_mesh(R, Lz, h, order)
        msh = self.msh
        # geometric imperfection {"amp": mm, "seed": int}: radial_imperfection, applied
        # before anything is computed from the geometry
        self.imperfection = None
        if imperfection and imperfection.get("amp", 0) > 0:
            self.imperfection = dict(imperfection)
            self.imperfection["applied"] = radial_imperfection(msh, Lz, **imperfection)
        msh.topology.create_connectivity(1, 2)
        self.V = fem.functionspace(msh, basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(3,)))
        self.ds = ufl.Measure("ds", domain=msh, subdomain_data=self.ft)
        self.c = fem.Constant(msh, 1.0)          # 1: stress-free tube, 0: rolled flat sheet
        self.eta = fem.Constant(msh, 0.0)        # artificial (pseudo-viscous) damping, off by default
        # element size per cell (longest corner-to-corner distance): CellDiameter is
        # not available on curved cells, and one size for a graded mesh would make
        # the interior-penalty terms far too stiff in the coarse elements
        g = msh.geometry
        P = g.x[g.dofmap[:, :3]]
        hc = np.max([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)
        self.hcell = fem.Function(fem.functionspace(msh, ("DG", 0)))
        self.hcell.x.array[self.hcell.function_space.dofmap.list[:, 0]] = hc
        x = ufl.SpatialCoordinate(msh)
        r = ufl.sqrt(x[0]**2 + x[1]**2)
        self.e1 = ufl.as_vector([-x[1] / r, x[0] / r, 0.0])
        self.e2 = ufl.as_vector([0.0, 0.0, 1.0])
        self.x = x

    # ------------------------------------------------------------- kinematics
    def _kin(self, phi):
        e1, e2 = self.e1, self.e2
        G = ufl.grad(phi)                                    # 3 x 3, tangential
        F1, F2 = G * e1, G * e2
        F = ufl.as_matrix([[F1[i], F2[i]] for i in range(3)])
        nd = ufl.cross(F1, F2)
        nd = nd / ufl.sqrt(ufl.dot(nd, nd))
        k11 = ufl.dot(nd, ufl.grad(F1) * e1)
        k22 = ufl.dot(nd, ufl.grad(F2) * e2)
        k12 = 0.5 * (ufl.dot(nd, ufl.grad(F1) * e2) + ufl.dot(nd, ufl.grad(F2) * e1))
        K = ufl.as_matrix([[k11, k12], [k12, k22]])
        return G, F, nd, K

    def energy(self, u):
        mat, a, c = self.mat, self.alpha, self.c
        phi = self.x + u
        G, F, nd, K = self._kin(phi)
        G0, _, _, B = self._kin(self.x)
        psi_m, N, E = _membrane(F, mat)
        Kb = K - c * B
        M = _moment(Kb, mat)

        nr = ufl.FacetNormal(self.msh)                       # conormal, tangent to the surface
        hE = self.hcell
        jdn = G("+") * nr("+") + G("-") * nr("-")
        jdn0 = G0("+") * nr("+") + G0("-") * nr("-")         # kink of the meshed tube itself
        n_avg = ufl.avg(nd)
        n_avg = n_avg / ufl.sqrt(ufl.dot(n_avg, n_avg))
        kink = ufl.dot(jdn - c * jdn0, n_avg)
        n2 = ufl.as_vector([ufl.dot(nr, self.e1), ufl.dot(nr, self.e2)])
        Mnn_avg = ufl.avg(ufl.dot(n2, M * n2))

        md = {"quadrature_degree": 4}
        dx, dS = ufl.dx(metadata=md), ufl.dS(metadata=md)
        ring = self.ds((BOTTOM, TOP), metadata=md)
        w = getattr(self, "w", None)                         # 0 on ghost cells (PeriodicTube)
        wc = 1.0 if w is None else w
        wf = 1.0 if w is None else ufl.max_value(w("+"), w("-"))
        Pi = (wc * (psi_m + 0.5 * ufl.inner(M, Kb)) * dx
              + wf * (- Mnn_avg * kink + 0.5 * a * mat.D / ufl.avg(hE) * kink**2) * dS)
        if self.ends == "periodic":
            pass
        elif self.ends == "free":
            Pi += 0.5 * 10 * mat.Y * ufl.dot(u, self.e1)**2 * ring
        else:                                                # rings kept from rotating
            rot = ufl.dot(ufl.grad(u) * nr, nd)
            Mnn = ufl.dot(n2, M * n2)
            Pi += - Mnn * rot * ring + 0.5 * a * mat.D / hE * rot**2 * ring
        self._M, self._N, self._phi = M, N, phi
        return Pi

    # ------------------------------------------------------------------ solve
    def setup(self):
        V = self.V
        u = fem.Function(V)
        self.u_prev = fem.Function(V)
        Pi = self.energy(u)
        du, ut = ufl.TestFunction(V), ufl.TrialFunction(V)
        R_el = ufl.derivative(Pi, u, du)
        damp = 0.5 * self.eta * ufl.inner(u - self.u_prev, u - self.u_prev) * ufl.dx
        R = R_el + ufl.derivative(damp, u, du)
        J = ufl.derivative(R, u, ut)
        Vz, _ = V.sub(2).collapse()
        self.uz_top = fem.Function(Vz)
        uz_bot = fem.Function(Vz)
        dbot = fem.locate_dofs_topological((V.sub(2), Vz), 1, self.ft.find(BOTTOM))
        dtop = fem.locate_dofs_topological((V.sub(2), Vz), 1, self.ft.find(TOP))
        if len(dbot[0]) == 0 or len(dtop[0]) == 0:
            raise RuntimeError("Tube: no dofs on an end ring")
        bcs = [fem.dirichletbc(uz_bot, dbot, V.sub(2)), fem.dirichletbc(self.uz_top, dtop, V.sub(2))]
        if self.ends == "clamped":                           # bottom ring fully fixed
            dall = fem.locate_dofs_topological(V, 1, self.ft.find(BOTTOM))
            bcs = [fem.dirichletbc(fem.Function(V), dall), bcs[1]]
        self._top_sub = dtop[1]                              # top-ring dofs in the collapsed z space
        self._Vz_map = V.sub(2).collapse()[1]

        def newton(bc_list):
            s = NewtonSolver(self.msh.comm, NonlinearProblem(R, u, bcs=bc_list, J=J))
            s.convergence_criterion = "residual"
            s.rtol, s.atol, s.max_it = 1e-9, 1e-10, 30
            ksp = s.krylov_solver
            ksp.setType("preonly"); ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps")
            return s
        self.u, self.solver = u, newton(bcs)
        self.solver_free_top = newton(bcs[:1])               # top ring free in z (rolling step)
        x = self.x
        self.react = fem.form(ufl.replace(R_el, {du: ufl.as_vector([0.0, 0.0, x[2] / self.Lz])}))

    def top_uz(self):
        """Mean z displacement of the top ring."""
        return float(self.u.x.array[np.asarray(self._Vz_map)[self._top_sub]].mean())

    def step(self, Delta, c, free_top=False, guess=None):
        """Newton solve at end displacement Delta and natural-shape parameter c.
        free_top: leave the top ring free in z (Delta unused), as when a rolled
        sheet is allowed to change length before its ends are fixed.
        guess: starting point of Newton (default: the last converged state). It
        changes only where Newton starts, not the equations it solves."""
        self.uz_top.x.array[:] = Delta
        self.c.value = c
        if guess is not None:
            self.u.x.array[:] = guess
        try:
            n, conv = (self.solver_free_top if free_top else self.solver).solve(self.u)
        except RuntimeError:
            conv = False
        if not conv:
            self.u.x.array[:] = self.u_prev.x.array
            return False
        self.u.x.scatter_forward()
        self.u_prev.x.array[:] = self.u.x.array
        return True

    # ------------------------------------------------------------- observables
    def mid_band(self, half):
        """Mean radius, mean hoop moment m_ss and mean membrane strains over
        |z - Lz/2| < half."""
        x = self.x
        ind = ufl.conditional(ufl.lt(abs(x[2] - self.Lz / 2), half), 1.0, 0.0)
        md = {"quadrature_degree": 4}
        dx = ufl.dx(domain=self.msh, metadata=md)
        area = fem.assemble_scalar(fem.form(ind * dx))
        phi = self._phi
        rad = ufl.sqrt(phi[0]**2 + phi[1]**2)
        vals = {}
        for k, expr in (("radius", rad), ("m_ss", self._M[0, 0]), ("m_zz", self._M[1, 1]),
                        ("n_zz", self._N[1, 1]), ("n_ss", self._N[0, 0])):
            vals[k] = fem.assemble_scalar(fem.form(ind * expr * dx)) / area
        vals["force"] = fem.assemble_scalar(self.react)
        return vals


class PeriodicTube(Tube):
    """Tube periodic along its axis with period Lz: one period of an infinitely long
    tube (as Rafsanjani et al.'s periodic cells, but the whole circumference and
    several rows). Mesh from tube_cut_mesh(..., periodic_z=True).

    Every ghost node is tied to the node it copies: u_ghost = u_master + Delta e_z.
    The ghost band carries no energy of its own, but the facets where it meets the
    top of the period keep the kink (interior-penalty) terms, so slopes are periodic
    as well as displacements. The constraints are eliminated, u = T v, and Newton is
    solved for v. Rigid motions are removed by pinning u_z and u_y at one node near
    the x axis and u_x at one node near the y axis, half way up the period. With
    Delta free (rolling step) Delta is one more unknown, i.e. zero axial force."""

    def __init__(self, R, Lz, mat, mesh, info, h=0.5, alpha=10.0, imperfection=None):
        super().__init__(R, Lz, mat, h=h, alpha=alpha, mesh=mesh, ends="periodic",
                         imperfection=imperfection)
        msh = self.msh
        ghost_in = np.zeros(info["triangles"] + len(info["ghost_cells"]), dtype=bool)
        ghost_in[info["ghost_cells"]] = True
        oc = np.asarray(msh.topology.original_cell_index)
        self.ghost_cell = ghost_in[oc]
        self.w = fem.Function(fem.functionspace(msh, ("DG", 0)))
        self.w.x.array[self.w.function_space.dofmap.list[:, 0]] = np.where(self.ghost_cell, 0.0, 1.0)
        # input node -> V block (V is P2 like the geometry: same local layout per cell)
        gin = np.asarray(msh.geometry.input_global_indices)
        gdm, vdm = msh.geometry.dofmap, self.V.dofmap.list
        blk = np.full(gin.max() + 1, -1)
        blk[gin[gdm.reshape(-1)]] = vdm.reshape(-1)
        X = self.V.tabulate_dof_coordinates()
        if not np.allclose(X[blk[gin]], msh.geometry.x, atol=1e-8):
            raise RuntimeError("PeriodicTube: function space and geometry layouts differ")
        g, m = blk[info["master"][:, 0]], blk[info["master"][:, 1]]
        d = X[g] - X[m]
        if not (np.allclose(d[:, :2], 0, atol=1e-8) and np.allclose(d[:, 2], Lz, atol=1e-8)):
            raise RuntimeError("PeriodicTube: ghost nodes are not copies shifted by Lz")
        if np.intersect1d(g, m).size:
            raise RuntimeError("PeriodicTube: a ghost node is also a master")
        self.gblk, self.mblk = g, m
        self.real_cells = np.flatnonzero(~self.ghost_cell)

    def setup(self):
        from petsc4py import PETSc
        from dolfinx.fem import petsc as fp
        V = self.V
        u = fem.Function(V)
        self.u_prev = fem.Function(V)
        Pi = self.energy(u)
        du, ut = ufl.TestFunction(V), ufl.TrialFunction(V)
        R_el = ufl.derivative(Pi, u, du)
        damp = 0.5 * self.eta * ufl.inner(u - self.u_prev, u - self.u_prev) * ufl.dx
        R = R_el + ufl.derivative(damp, u, du)
        self.u = u
        self.Rf, self.Ref = fem.form(R), fem.form(R_el)
        self.Jf = fem.form(ufl.derivative(R, u, ut))
        self.Jmat = fp.create_matrix(self.Jf)
        self.Rvec, self.Revec = fp.create_vector(self.Rf), fp.create_vector(self.Ref)

        n = V.dofmap.index_map.size_local * 3
        X = V.tabulate_dof_coordinates()
        ok = np.ones(len(X), dtype=bool)
        ok[self.gblk] = ok[self.mblk] = False
        mid = np.abs(X[:, 2] - self.Lz / 2)
        th = np.arctan2(X[:, 1], X[:, 0])
        idx = np.flatnonzero(ok)
        pa = idx[np.argmin((mid + 20 * np.abs(th))[ok])]
        pb = idx[np.argmin((mid + 20 * np.abs(th - np.pi / 2))[ok])]
        pinned = np.array([3 * pa + 2, 3 * pa + 1, 3 * pb + 0])
        self.pinned = pinned
        col = np.zeros(n, dtype=np.int64)
        drop = np.zeros(n, dtype=bool)
        drop[pinned] = True
        for k in range(3):
            drop[3 * self.gblk + k] = True
        free = np.flatnonzero(~drop)
        col[:] = -1
        col[free] = np.arange(len(free))
        for k in range(3):
            col[3 * self.gblk + k] = col[3 * self.mblk + k]
        nf = len(free)
        self.gz, self.mz = 3 * self.gblk + 2, 3 * self.mblk + 2

        def build(extra):
            rows = np.flatnonzero(col >= 0)
            nnz = np.ones(n, dtype=np.int32)
            nnz[self.gz] += extra
            T = PETSc.Mat().createAIJ([n, nf + extra], nnz=nnz)
            for i in rows:
                T.setValue(int(i), int(col[i]), 1.0)
            if extra:
                for i in self.gz:
                    T.setValue(int(i), nf, 1.0)
            T.assemble()
            return T
        self.T = {False: build(0), True: build(1)}
        self.A = {False: None, True: None}
        self.ksp = {False: None, True: None}
        self.nfree = nf

    def delta(self):
        a = self.u.x.array
        return float(np.mean(a[self.gz] - a[self.mz]))

    def _residual(self, form, vec):
        from petsc4py import PETSc
        from dolfinx.fem import petsc as fp
        with vec.localForm() as loc:
            loc.set(0.0)
        fp.assemble_vector(vec, form)
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return vec

    def _newton(self, free_delta, rtol=1e-9, atol=1e-10, max_it=30):
        from petsc4py import PETSc
        from dolfinx.fem import petsc as fp
        T = self.T[free_delta]
        r, dv, du = T.createVecRight(), T.createVecRight(), T.createVecLeft()
        r0 = None
        a = self.u.x.array
        for it in range(max_it + 1):
            T.multTranspose(self._residual(self.Rf, self.Rvec), r)
            nr = r.norm()
            if not np.isfinite(nr):
                return False
            if r0 is None:
                r0 = max(nr, 1e-300)
            if nr < atol or nr / r0 < rtol:
                return True
            if it == max_it:
                return False
            self.Jmat.zeroEntries()
            fp.assemble_matrix(self.Jmat, self.Jf)
            self.Jmat.assemble()
            if self.A[free_delta] is None:
                self.A[free_delta] = self.Jmat.PtAP(T)
            else:
                self.Jmat.PtAP(T, result=self.A[free_delta])
            # one KSP per T, kept between iterations and steps: A keeps its nonzero
            # pattern, so MUMPS redoes only the numeric factorisation
            ksp = self.ksp.get(free_delta)
            if ksp is None:
                ksp = PETSc.KSP().create(self.msh.comm)
                ksp.setType("preonly")
                pc = ksp.getPC()
                pc.setType("lu")
                pc.setFactorSolverType("mumps")
                self.ksp[free_delta] = ksp
            ksp.setOperators(self.A[free_delta])
            r.scale(-1.0)
            try:
                ksp.solve(r, dv)
                reason = ksp.getConvergedReason()
            except Exception:
                reason = -1
            if reason < 0:                       # a failed factorisation: start afresh
                ksp.destroy()
                self.ksp[free_delta] = None
                return False
            T.mult(dv, du)
            a[:] += du.getArray()
            self.u.x.scatter_forward()
        return False

    def step(self, Delta, c, free_top=False, guess=None):
        """Newton solve at period stretch Delta (Delta free if free_top: zero axial
        force) and natural-shape parameter c. guess: starting point of Newton
        (default: the last converged state); it is made admissible by a rigid
        translation that puts the pinned dofs back, and by re-tying the ghosts."""
        self.c.value = c
        a = self.u.x.array
        if guess is not None:
            a[:] = guess
            prev = self.u_prev.x.array
            for p in self.pinned:                       # pins: u_z, u_y at pa; u_x at pb
                a[p % 3::3] -= a[p] - prev[p]
        for k in range(3):
            a[3 * self.gblk + k] = a[3 * self.mblk + k]
        a[self.gz] += getattr(self, "delta_prev", 0.0) if free_top else Delta
        try:
            conv = self._newton(free_top)
        except RuntimeError:
            conv = False
        if not conv:
            a[:] = self.u_prev.x.array
            return False
        self.u_prev.x.array[:] = a
        self.delta_prev = self.delta()
        return True

    def top_uz(self):
        return self.delta()

    def force(self):
        """Axial force carried by the tube, dPi/dDelta (without damping)."""
        v = self._residual(self.Ref, self.Revec).getArray()
        return float(v[self.gz].sum())
