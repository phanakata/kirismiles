"""Fig. 2b/2d: post-buckling of a sheet with a single cut (FvK, displacement control).

Following the paper's Methods, the first (symmetric) linear buckling mode is
added as a small geometric imperfection, then Delta is ramped up.

Usage: python3 scripts/single_cut_postbuckling.py [--L=<mm>] h b [h b ...]
Writes results/single_cut/post_h{h}_b{b}.npz (post_h{h}_b{b}_L{L}.npz for a sheet
length other than the default 182 mm) with Delta, F, delta0 and the
deflection field at the final step.
"""
import sys
import time

import numpy as np
from dolfinx import fem, plot
from mpi4py import MPI

from kirigami.fvk import Material, Sheet
from kirigami.geometry import single_cut

L, W = 182.0, 100.0
E, NU = 3500.0, 0.38
OUT = "results/single_cut"
IMP = 1e-3          # imperfection amplitude, in units of h
DELTA_MAX = 0.1 * W


def run(h, b, n_steps=40, L=L):
    t0 = time.time()
    msh = single_cut(L, W, b, h_far=4.0, h_cut=0.35)
    sh = Sheet(msh, L, Material(E, NU, h))
    _, N1, _ = sh.solve_inplane(1.0)
    lams, modes = sh.linear_buckling(N1, nev=2)
    Dc = lams[0]
    w_imp = fem.Function(sh.W)
    w_imp.x.array[:] = IMP * h * modes[0].x.array

    Wq, wmap = sh.Q.sub(1).collapse()
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    w1 = fem.Function(V1)

    def delta0(D, q, s):
        wt = q.x.array[wmap] + s.w0.x.array
        return msh.comm.allreduce(np.max(np.abs(wt)), op=MPI.MAX)

    def seed(D, q, s):
        # Past the threshold, if still (nearly) flat, put w on the symmetric
        # mode with the amplitude suggested by delta0 ~ sqrt(b (Delta - Dc)).
        if D > Dc and np.max(np.abs(q.x.array[wmap])) < 0.1 * h:
            q.x.array[wmap] = 0.5 * np.sqrt(b * (D - Dc)) * modes[0].x.array

    snapshots = {}

    def snapshot(D, q, s):
        wt = fem.Function(Wq)
        wt.x.array[:] = q.x.array[wmap] + s.w0.x.array
        w1.interpolate(wt)
        snapshots[D] = w1.x.array.copy()

    Deltas = np.concatenate([[0.5 * Dc], np.geomspace(1.05 * Dc, DELTA_MAX, n_steps)])
    res = sh.postbuckling(Deltas, w_imp=w_imp, observers=(delta0, snapshot),
                          predictor=seed)
    D = np.array([r[0] for r in res])
    F = np.array([r[1] for r in res])
    d0 = np.array([r[2][0] for r in res])
    cells, _, x = plot.vtk_mesh(V1)
    keep = [D[0], D[len(D) // 3], D[2 * len(D) // 3], D[-1]]
    tag = "" if L == 182.0 else f"_L{L:g}"
    np.savez(f"{OUT}/post_h{h:g}_b{b:g}{tag}.npz", h=h, b=b, w=W, L=L, Delta_c=Dc, Delta=D, F=F,
             delta0=d0, x=x[:, :2], cells=cells.reshape(-1, 4)[:, 1:],
             snap_Delta=np.array(keep), snap_w=np.array([snapshots[k] for k in keep]))
    print(f"h={h} b={b}: Dc={Dc:.5f}  steps={len(D)}  delta0(end)={d0[-1]:.3f} mm  "
          f"F(end)={F[-1]:.2f} N  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    opts = [a for a in sys.argv[1:] if a.startswith("--L=")]
    Lsheet = float(opts[-1][4:]) if opts else L
    args = [float(a) for a in sys.argv[1:] if not a.startswith("--")] or [0.127, 80.0]
    for h, b in zip(args[::2], args[1::2]):
        run(h, b, L=Lsheet)
