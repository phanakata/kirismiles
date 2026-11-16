"""Foppl-von Karman plate model of a thin sheet with cuts, in FEniCSx.

Unknowns: in-plane displacement u (P2 vector) and deflection w (P2).
Bending uses a C0 interior-penalty (CIP) discretisation of the Kirchhoff plate,
so plain Lagrange elements can be used for w. Cut faces are exterior facets of
the mesh and are therefore free edges automatically.

Boundary conditions (paper, Methods): x=0 clamped, x=L displaced by Delta along
x; neither edge may rotate (w = 0 and dw/dn = 0, the latter imposed weakly by
Nitsche). The edges y=0, y=W and the cut faces are free.

Units: mm, MPa (N/mm^2), N.
"""
from dataclasses import dataclass

import numpy as np
import ufl
import basix.ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import (LinearProblem, NonlinearProblem, assemble_matrix)
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc


@dataclass
class Material:
    E: float = 3500.0      # MPa
    nu: float = 0.38
    h: float = 0.127       # mm
    # membrane law of the shell model (kirigami/shell.py): "svk" (St Venant-Kirchhoff,
    # the default) or "neo_hookean" (compressible; stiffens instead of softening under
    # strong compression, for hinges that are squeezed far). The FvK model is always SVK.
    model: str = "svk"

    @property
    def Y(self):           # stretching rigidity E h
        return self.E * self.h

    @property
    def D(self):           # bending rigidity E h^3 / 12(1 - nu^2)
        return self.E * self.h**3 / (12 * (1 - self.nu**2))


def _membrane_stress(eps, mat):
    I = ufl.Identity(2)
    return mat.Y / (1 - mat.nu**2) * ((1 - mat.nu) * eps + mat.nu * ufl.tr(eps) * I)


def _moment(kappa, mat):
    I = ufl.Identity(2)
    return mat.D * ((1 - mat.nu) * kappa + mat.nu * ufl.tr(kappa) * I)


class Sheet:
    """Holds mesh, function spaces, measures and boundary data of one problem."""

    def __init__(self, msh, L, mat=Material(), alpha=10.0):
        self.msh, self.L, self.mat, self.alpha = msh, L, mat, alpha
        tdim = msh.topology.dim
        msh.topology.create_connectivity(tdim - 1, tdim)

        left = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[0], 0.0))
        right = dmesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[0], L))
        facets = np.concatenate([left, right])
        tags = np.concatenate([np.full_like(left, 1), np.full_like(right, 2)])
        order = np.argsort(facets)
        self.ft = dmesh.meshtags(msh, 1, facets[order], tags[order])
        self.ds = ufl.Measure("ds", domain=msh, subdomain_data=self.ft)
        self.dS = ufl.Measure("dS", domain=msh)
        self.left_facets, self.right_facets = left, right

        # Mixed space for the full FvK problem
        cell = msh.basix_cell()
        Pu = basix.ufl.element("Lagrange", cell, 2, shape=(2,))
        Pw = basix.ufl.element("Lagrange", cell, 2)
        self.Q = fem.functionspace(msh, basix.ufl.mixed_element([Pu, Pw]))
        # Stand-alone spaces for the linear (in-plane / buckling) problems
        self.U = fem.functionspace(msh, Pu)
        self.W = fem.functionspace(msh, Pw)
        # Test field used to extract the reaction force: v = (x/L, 0)
        x = ufl.SpatialCoordinate(msh)
        self.v_react = ufl.as_vector([x[0] / L, 0.0])

    # ---------------------------------------------------------------- bending
    def bending_energy(self, w):
        """CIP Kirchhoff bending energy, with dw/dn = 0 imposed weakly on the
        clamped edges (ds(1), ds(2))."""
        mat, a = self.mat, self.alpha
        n = ufl.FacetNormal(self.msh)
        hE = ufl.CellDiameter(self.msh)
        kappa = ufl.grad(ufl.grad(w))
        M = _moment(kappa, mat)
        jdn = ufl.jump(ufl.grad(w), n)                       # [[dw/dn]]
        Mnn_avg = ufl.inner(ufl.avg(M) * n("+"), n("+"))
        dn = ufl.dot(ufl.grad(w), n)
        Mnn = ufl.inner(M * n, n)
        clamp = self.ds(1) + self.ds(2)
        return (0.5 * ufl.inner(M, kappa) * ufl.dx
                - Mnn_avg * jdn * self.dS
                + 0.5 * a * mat.D / ufl.avg(hE) * jdn**2 * self.dS
                - Mnn * dn * clamp
                + 0.5 * a * mat.D / hE * dn**2 * clamp)

    # --------------------------------------------------------------- in-plane
    def solve_inplane(self, Delta=1.0):
        """Linear plane-stress problem. Returns (u, N, reaction force)."""
        U = self.U
        u, v = ufl.TrialFunction(U), ufl.TestFunction(U)
        eps = lambda z: ufl.sym(ufl.grad(z))
        a = ufl.inner(_membrane_stress(eps(u), self.mat), eps(v)) * ufl.dx
        f = ufl.inner(fem.Constant(self.msh, (0.0, 0.0)), v) * ufl.dx
        bcs = self._u_bcs(U, Delta)
        problem = LinearProblem(a, f, bcs=bcs, petsc_options={
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
        uh = problem.solve()
        N = _membrane_stress(eps(uh), self.mat)
        F = self.reaction(N)
        return uh, N, F

    def _u_bcs(self, U, Delta):
        """Dirichlet conditions u=0 on x=0 and u=(Delta,0) on x=L."""
        dl = fem.locate_dofs_topological(U, 1, self.left_facets)
        dr = fem.locate_dofs_topological(U, 1, self.right_facets)
        zero = fem.Constant(self.msh, (0.0, 0.0))
        delta = fem.Constant(self.msh, (float(Delta), 0.0))
        return [fem.dirichletbc(zero, dl, U), fem.dirichletbc(delta, dr, U)]

    def reaction(self, N):
        """Force on the x=L edge, computed as the virtual work of the membrane
        stress against v = (x/L, 0) (valid at equilibrium)."""
        form = fem.form(ufl.inner(N, ufl.sym(ufl.grad(self.v_react))) * ufl.dx)
        return self.msh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)

    # --------------------------------------------------------- linear buckling
    def linear_buckling(self, N0, nev=6):
        """Solve K_b phi = lambda (-K_g(N0)) phi for the smallest positive
        lambda; N0 is the membrane stress for a unit applied Delta, so the
        lambdas are critical displacements. Returns (lambdas, modes)."""
        from slepc4py import SLEPc
        W = self.W
        w, v = ufl.TrialFunction(W), ufl.TestFunction(W)
        wf = fem.Function(W)
        Eb = self.bending_energy(wf)
        kb = ufl.derivative(ufl.derivative(Eb, wf, v), wf, w)
        kg = -ufl.inner(N0 * ufl.grad(w), ufl.grad(v)) * ufl.dx

        dofs = fem.locate_dofs_topological(W, 1, np.concatenate([self.left_facets, self.right_facets]))
        bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, W)
        A = assemble_matrix(fem.form(kg), bcs=[bc], diagonal=0.0); A.assemble()
        B = assemble_matrix(fem.form(kb), bcs=[bc], diagonal=1.0); B.assemble()

        # (-K_g) phi = mu K_b phi, largest mu -> lambda = 1/mu
        eps = SLEPc.EPS().create(self.msh.comm)
        eps.setOperators(A, B)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setDimensions(nev, max(2 * nev, 20))
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        st = eps.getST(); st.setType("shift")
        ksp = st.getKSP(); ksp.setType("preonly")
        pc = ksp.getPC(); pc.setType("lu"); pc.setFactorSolverType("mumps")
        eps.setTolerances(1e-10, 500)
        eps.solve()
        lams, modes = [], []
        for i in range(eps.getConverged()):
            vr = B.createVecRight()
            mu = eps.getEigenpair(i, vr).real
            if mu <= 0:
                continue
            phi = fem.Function(W)
            phi.x.array[:] = vr.array
            phi.x.scatter_forward()
            amax = self.msh.comm.allreduce(np.max(np.abs(phi.x.array)), op=MPI.MAX)
            phi.x.array[:] /= amax
            lams.append(1.0 / mu); modes.append(phi)
        idx = np.argsort(lams)
        return [lams[i] for i in idx], [modes[i] for i in idx]

    # ----------------------------------------------------------- post-buckling
    def postbuckling(self, Deltas, w_imp=None, observers=(), predictor=None, newton_kw=None):
        """Displacement-controlled FvK solve over the sequence Deltas.

        w_imp: initial geometric imperfection (Function on self.W); the strain
        is measured relative to the imperfect flat-ish reference so that the
        imperfect sheet is stress free. observers: callables f(Delta, q, sheet)
        called after each converged step; their returns are collected.
        predictor: optional callable f(Delta, q, sheet) that may modify the
        initial guess q before each Newton solve (e.g. to seed the buckled
        branch, since Newton started near w = 0 tends to stay on the unstable
        near-flat branch past the buckling threshold).
        """
        mat, Q = self.mat, self.Q
        q = fem.Function(Q)
        u, w = ufl.split(q)
        w0 = fem.Function(self.W)
        if w_imp is not None:
            w0.interpolate(w_imp)
        wt = w + w0
        eps = (ufl.sym(ufl.grad(u))
               + 0.5 * (ufl.outer(ufl.grad(wt), ufl.grad(wt)) - ufl.outer(ufl.grad(w0), ufl.grad(w0))))
        N = _membrane_stress(eps, mat)
        Pi = 0.5 * ufl.inner(N, eps) * ufl.dx + self.bending_energy(w)
        dq, qt = ufl.TestFunction(Q), ufl.TrialFunction(Q)
        F = ufl.derivative(Pi, q, dq)
        J = ufl.derivative(F, q, qt)

        Uq, _ = Q.sub(0).collapse()
        Wq, _ = Q.sub(1).collapse()
        dl = fem.locate_dofs_topological((Q.sub(0), Uq), 1, self.left_facets)
        dr = fem.locate_dofs_topological((Q.sub(0), Uq), 1, self.right_facets)
        wdofs = fem.locate_dofs_topological((Q.sub(1), Wq), 1,
                                            np.concatenate([self.left_facets, self.right_facets]))
        u_left, u_right, w_zero = fem.Function(Uq), fem.Function(Uq), fem.Function(Wq)
        bcs = [fem.dirichletbc(u_left, dl, Q.sub(0)),
               fem.dirichletbc(u_right, dr, Q.sub(0)),
               fem.dirichletbc(w_zero, wdofs, Q.sub(1))]

        problem = NonlinearProblem(F, q, bcs=bcs, J=J)
        solver = NewtonSolver(self.msh.comm, problem)
        solver.convergence_criterion = "residual"
        solver.rtol, solver.atol, solver.max_it = 1e-9, 1e-10, 30
        for k, val in (newton_kw or {}).items():
            setattr(solver, k, val)
        ksp = solver.krylov_solver
        ksp.setType("preonly"); ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        # Reaction force form (virtual work against v = (x/L, 0))
        react = fem.form(ufl.inner(N, ufl.sym(ufl.grad(self.v_react))) * ufl.dx)
        self.q, self.w0, self.N = q, w0, N

        out = []
        prev = 0.0
        q_prev = q.x.array.copy()
        Deltas = list(Deltas)
        i = 0
        use_predictor = True
        while i < len(Deltas):
            D = Deltas[i]
            u_right.interpolate(lambda x, D=D: np.vstack([np.full(x.shape[1], D), np.zeros(x.shape[1])]))
            seeded = False
            if predictor is not None and use_predictor:
                predictor(D, q, self)
                seeded = not np.array_equal(q.x.array, q_prev)
            try:
                n_it, conv = solver.solve(q)
            except RuntimeError:
                conv = False
            if not conv:
                q.x.array[:] = q_prev
                if seeded:          # the seeded guess did not converge: try this step without it;
                    use_predictor = False       # the imperfection then selects the branch
                    continue
                # bisect the step
                Deltas.insert(i, 0.5 * (prev + D))
                if abs(D - prev) < 1e-9:
                    raise RuntimeError(f"Newton failed at Delta={D}")
                continue
            q.x.scatter_forward()
            q_prev[:] = q.x.array
            prev = D
            use_predictor = True
            Fx = self.msh.comm.allreduce(fem.assemble_scalar(react), op=MPI.SUM)
            out.append((D, Fx, [obs(D, q, self) for obs in observers]))
            i += 1
        return out
