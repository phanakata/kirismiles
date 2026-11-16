"""Geometrically exact Kirchhoff-Love shell for initially flat sheets with cuts.

Unlike the FvK model (kirigami/fvk.py), this handles large rotations, which
the pitch/roll/yaw actuators need (rotations of tens of degrees).

Unknown: 3D displacement u (P2) of the flat reference X = (x, y, 0);
deformed position phi = X + u.
  membrane strain   E = (grad(phi)^T grad(phi) - I) / 2      (Green-Lagrange)
  bending strain    K_ab = phi_,ab . n,  n = unit normal of the deformed surface
with St Venant-Kirchhoff plane-stress constitutive laws (or, for the membrane, an
optional compressible neo-Hookean law; Material.model). The C1 requirement of
the bending term is handled by C0 interior penalty, as in the FvK model: on
interior facets, the component of the jump [[d phi/dn]] along the surface
normal (the kink between two elements, zero for a smooth surface) is
penalised, with a consistency term pairing it with avg(M_nn). On the clamped
edges, the rotation of d phi/dn away from n_ref is suppressed the same way.
"""
import numpy as np
import ufl
import basix.ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

from kirigami.fvk import Sheet, _membrane_stress, _moment

MEMBRANE_MODELS = ("svk", "neo_hookean")


def _membrane(F, mat):
    """Membrane energy density (per unit reference area), stress resultant and
    Green-Lagrange strain for the deformation gradient F (3 x 2).

    svk          N = E h / (1 - nu^2) [(1 - nu) E + nu tr(E) I]. Under uniaxial
                 compression its stress peaks at a stretch of 1/sqrt(3) and then falls
                 to zero, so a hinge squeezed further loses its stiffness.
    neo_hookean  2D compressible neo-Hookean, h [mu/2 (I1 - 2 - 2 ln J) + lam/2 (ln J)^2]
                 with C = F^T F, I1 = tr C, J = sqrt(det C), mu = E / 2(1 + nu) and the
                 plane-stress lam = E nu / (1 - nu^2). It reduces to the svk law at small
                 strain and its energy grows without bound as J -> 0.
    """
    I = ufl.Identity(2)
    C = F.T * F
    E = 0.5 * (C - I)
    if getattr(mat, "model", "svk") == "neo_hookean":
        mu = mat.E / (2 * (1 + mat.nu))
        lam = mat.E * mat.nu / (1 - mat.nu**2)
        lnJ = 0.5 * ufl.ln(ufl.det(C))
        Ci = ufl.inv(C)
        psi = mat.h * (0.5 * mu * (ufl.tr(C) - 2 - 2 * lnJ) + 0.5 * lam * lnJ**2)
        S = mat.h * (mu * (I - Ci) + lam * lnJ * Ci)            # second Piola-Kirchhoff
        return psi, S, E
    N = _membrane_stress(E, mat)
    return 0.5 * ufl.inner(N, E), N, E


class ShellSheet(Sheet):

    def __init__(self, msh, L, mat, alpha=10.0, boundary=None):
        """boundary: {"type": "clamped" | "roller" | "biaxial", "in_plane": bool};
        see kirigami/spec.py BOUNDARY_TYPES."""
        super().__init__(msh, L, mat, alpha)
        cell = msh.basix_cell()
        self.V3 = fem.functionspace(msh, basix.ufl.element("Lagrange", cell, 2, shape=(3,)))
        self.boundary = {"type": "clamped", "in_plane": False, **(boundary or {})}
        xg = msh.geometry.x
        self.Wsheet = float(xg[:, 1].max())
        self.bottom_facets = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 0.0))
        self.top_facets = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], self.Wsheet))
        # vertex on x = 0 closest to mid-height: pinned in y for the roller condition
        on_left = np.flatnonzero(np.isclose(xg[:, 0], 0.0))
        self.pin_vertex = None
        if len(on_left):
            k = on_left[np.argmin(np.abs(xg[on_left, 1] - self.Wsheet / 2))]
            msh.topology.create_connectivity(0, msh.topology.dim)
            verts = dmesh.locate_entities(msh, 0, lambda x, p=xg[k]: np.isclose(x[0], p[0]) &
                                          np.isclose(x[1], p[1]))
            self.pin_vertex = verts[:1]

    def _bcs(self, V, u_left, u_right):
        """Dirichlet conditions for the chosen boundary type. u_left / u_right are
        Functions in V whose values (set by solve each step) are imposed on the
        loaded edges; for 'roller' and 'biaxial' only their normal components are
        used."""
        t, bcs = self.boundary["type"], []
        if t == "clamped" and not self.boundary["in_plane"]:
            dl = fem.locate_dofs_topological(V, 1, self.left_facets)
            dr = fem.locate_dofs_topological(V, 1, self.right_facets)
            return [fem.dirichletbc(u_left, dl), fem.dirichletbc(u_right, dr)]

        def comp(facets, i, func, dim=1):
            Vi, _ = V.sub(i).collapse()
            dofs = fem.locate_dofs_topological((V.sub(i), Vi), dim, facets)
            fi = fem.Function(Vi)
            return fem.dirichletbc(fi, dofs, V.sub(i)), fi, func

        edges = [(self.left_facets, 0, u_left), (self.right_facets, 0, u_right)]
        if t == "clamped":                      # in plane: the ends also hold y; z is fixed below
            edges += [(self.left_facets, 1, u_left), (self.right_facets, 1, u_right)]
        if t == "biaxial":
            edges += [(self.bottom_facets, 1, u_left), (self.top_facets, 1, u_right)]
        self._linked = []                       # (bc function, source Function, component)
        for facets, i, src in edges:
            bc, fi, _ = comp(facets, i, src)
            bcs.append(bc)
            self._linked.append((fi, src, i))
        if not self.boundary["in_plane"]:       # loaded edges stay at z = 0 (simply supported)
            loaded = np.concatenate([e[0] for e in edges])
            bc, fi, _ = comp(loaded, 2, None)
            bcs.append(bc)
        if t == "roller" and self.pin_vertex is not None and len(self.pin_vertex):
            bc, fi, _ = comp(self.pin_vertex, 1, None, dim=0)
            bcs.append(bc)
        if self.boundary["in_plane"]:           # every point at z = 0
            tdim = self.msh.topology.dim
            self.msh.topology.create_connectivity(tdim, tdim)   # needed to locate dofs on cells
            cells =np.arange(self.msh.topology.index_map(tdim).size_local, dtype=np.int32)
            bc, fi, _ = comp(cells, 2, None, dim=tdim)
            bcs.append(bc)
        return bcs

    def _update_bcs(self):
        for fi, src, i in getattr(self, "_linked", []):
            fi.x.array[:] = src.sub(i).collapse().x.array

    def energy(self, u, f_imp=None):
        msh, mat, a = self.msh, self.mat, self.alpha
        x = ufl.SpatialCoordinate(msh)
        phi = ufl.as_vector([x[0], x[1], 0.0]) + u
        F = ufl.grad(phi)                                   # 3 x 2
        psi_m, N, E = _membrane(F, mat)
        g1, g2 = F[:, 0], F[:, 1]
        nd = ufl.cross(g1, g2)
        nd = nd / ufl.sqrt(ufl.dot(nd, nd))                # deformed unit normal
        H = ufl.grad(F)                                     # 3 x 2 x 2
        i, al, be = ufl.indices(3)
        K = ufl.as_tensor(nd[i] * H[i, al, be], (al, be))
        M = _moment(K, mat)

        nr = ufl.FacetNormal(msh)                           # reference (in-plane) normal
        hE = ufl.CellDiameter(msh)
        # Only the part of the jump [[d phi / dn]] along the surface normal is a
        # kink (a rotation jump); its in-plane part is a jump of the membrane
        # strain, which the membrane energy already controls. Penalising the whole
        # vector would stiffen the in-plane response by a factor of order
        # (h / element size)^2, which is not small for thick sheets.
        jdn = F("+") * nr("+") + F("-") * nr("-")           # [[d phi / dn]]
        Mnn_avg = ufl.inner(ufl.avg(M) * nr("+"), nr("+"))
        n_avg = ufl.avg(nd)
        n_avg = n_avg / ufl.sqrt(ufl.dot(n_avg, n_avg))
        kink = ufl.dot(jdn, n_avg)
        clamp = self.ds((1, 2))
        dn = F * nr - ufl.as_vector([nr[0], nr[1], 0.0])    # deviation from no rotation
        Mnn = ufl.inner(M * nr, nr)

        # The unit normal makes the integrands non-polynomial, for which UFL
        # would pick a very high quadrature degree (huge kernels, slow JIT).
        md = {"quadrature_degree": 4}
        dx, dS, clamp = ufl.dx(metadata=md), self.dS(metadata=md), self.ds((1, 2), metadata=md)
        if self.boundary["in_plane"]:                       # membrane only: no bending, no CIP
            Pi = psi_m * dx
            return Pi, N, phi
        Pi = (psi_m * dx + 0.5 * ufl.inner(M, K) * dx
              - Mnn_avg * kink * dS
              + 0.5 * a * mat.D / ufl.avg(hE) * kink**2 * dS)
        if self.boundary["type"] == "clamped":              # no rotation at the grips
            Pi += (- Mnn * ufl.dot(dn, nd) * clamp
                   + 0.5 * a * mat.D / hE * ufl.dot(dn, nd)**2 * clamp)
        if f_imp is not None:                               # small transverse load
            Pi -= f_imp * u[2] * dx
        return Pi, N, phi

    def solve(self, Deltas, f_imp=None, observers=(), predictor=None, newton_kw=None):
        """Displacement-controlled solve over Deltas (same contract as
        Sheet.postbuckling). f_imp: optional transverse load density (UFL
        expression) acting as an imperfection."""
        V = self.V3
        u = fem.Function(V)
        Pi, N, phi = self.energy(u, f_imp)
        du, ut = ufl.TestFunction(V), ufl.TrialFunction(V)
        R = ufl.derivative(Pi, u, du)
        J = ufl.derivative(R, u, ut)

        u_left, u_right = fem.Function(V), fem.Function(V)
        bcs = self._bcs(V, u_left, u_right)
        biax = self.boundary["type"] == "biaxial"
        ey = 0.5 * self.Wsheet / self.L                          # y displacement per unit Delta

        problem = NonlinearProblem(R, u, bcs=bcs, J=J)
        solver = NewtonSolver(self.msh.comm, problem)
        solver.convergence_criterion = "residual"
        solver.rtol, solver.atol, solver.max_it = 1e-9, 1e-9, 25
        for k, val in (newton_kw or {}).items():
            setattr(solver, k, val)
        ksp = solver.krylov_solver
        ksp.setType("preonly"); ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        # Reaction force: virtual work of the residual along v = (x/L, 0, 0)
        x = ufl.SpatialCoordinate(self.msh)
        react = fem.form(ufl.replace(R, {du: ufl.as_vector([x[0] / self.L, 0.0, 0.0])}))
        self.u = u

        out, prev, u_prev = [], 0.0, u.x.array.copy()
        Deltas, i = list(Deltas), 0
        use_predictor = True
        while i < len(Deltas):
            D = Deltas[i]
            if biax:      # symmetric: x = 0, L by -/+ D/2;  y = 0, W by -/+ D W/(2L)
                u_left.interpolate(lambda x, D=D: np.vstack([np.full(x.shape[1], -D / 2),
                                                             np.full(x.shape[1], -D * ey),
                                                             np.zeros(x.shape[1])]))
                u_right.interpolate(lambda x, D=D: np.vstack([np.full(x.shape[1], D / 2),
                                                              np.full(x.shape[1], D * ey),
                                                              np.zeros(x.shape[1])]))
            else:
                u_right.interpolate(lambda x, D=D: np.vstack([np.full(x.shape[1], D),
                                                              np.zeros((2, x.shape[1]))]))
            self._update_bcs()
            seeded = False
            if predictor is not None and use_predictor:
                predictor(D, u, self)
                seeded = not np.array_equal(u.x.array, u_prev)
            try:
                _, conv = solver.solve(u)
            except RuntimeError:
                conv = False
            if not conv:
                u.x.array[:] = u_prev
                if seeded:          # the seeded guess did not converge: try this step without it;
                    use_predictor = False       # the imperfection then selects the branch
                    continue
                if abs(D - prev) < 1e-7:
                    raise RuntimeError(f"Newton failed at Delta={D}")
                Deltas.insert(i, 0.5 * (prev + D))
                continue
            u.x.scatter_forward()
            u_prev[:] = u.x.array
            prev = D
            use_predictor = True
            Fx = self.msh.comm.allreduce(fem.assemble_scalar(react), op=MPI.SUM)
            self.force = Fx                       # available to observers as s.force
            out.append((D, Fx, [obs(D, u, self) for obs in observers]))
            print(f"  Delta={D:.5f}  F={Fx:.4f}", flush=True)
            i += 1
        return out
