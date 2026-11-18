"""Applied strain measured like the experiment (Rafsanjani et al. PNAS 2019, SI Eq. S1).

The experiment tracks markers at the base of the triangular cuts, one per 12 mm row on a
vertical line, and takes eps = (z_q - z_p)/(Z_q - Z_p) - 1 between interior markers
(p, q) = (3, 18) of 20, away from the end caps. Here a marker is the mean position of
the mesh nodes within --radius of the base mid-point of a cut (on the lattice rows of
one parity), averaged over the n columns around the tube. Z is the rolled state (the
experiment's undeformed sample is already rolled). Uses the snapshots saved in the run's
.npz (every 0.005 of applied strain) and the force from its .json.

  python scripts/tube_marker_strain.py RUN [--p 2 --q 7] [--radius 1.0]
RUN is the output prefix (without .json/.npz); p, q are 1-based marker numbers, marker 1
at the bottom row. Prints and returns (eps_end_to_end, eps_markers, sigma_kPa).
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from tube_triangular import triangular_cuts  # noqa: E402


def marker_curve(run, p, q, radius=1.0):
    d = json.load(open(run + ".json"))
    z = np.load(run + ".npz")
    a = d["args"]
    if a.get("periodic"):
        raise SystemExit("marker strain is for clamped (finite) tubes")
    cuts, _, C, Lz = triangular_cuts(a["n"], a["rows"], delta=a["delta"])
    X = z["X"]
    R = C / (2 * np.pi)
    s = (np.arctan2(X[:, 1], X[:, 0]) % (2 * np.pi)) * R
    # base mid-point of each cut on even lattice rows (one marker per 12 mm row)
    n, rows = a["n"], a["rows"]
    groups = []                                     # groups[i] = node ids of marker row i
    for i in range(rows):
        k = 2 * i
        ids = []
        for j in range(n):
            P1, _, P5 = cuts[k * n + j]
            bs, bz = (P1[0] + P5[0]) / 2, (P1[1] + P5[1]) / 2
            ds = (s - bs + C / 2) % C - C / 2
            ids.append(np.where(np.hypot(ds, X[:, 2] - bz) < radius)[0])
        groups.append(ids)
    if min(len(g) for ids in groups for g in ids) == 0:
        raise SystemExit("empty marker: increase --radius")

    def zm(u, i):                                   # mean z of marker row i over columns
        x = X + u
        return np.mean([x[g, 2].mean() for g in groups[i]])

    Z0 = zm(z["rolled"], p - 1), zm(z["rolled"], q - 1)
    rows_j = [r for r in d["rows"] if not r.get("damped")]
    eJ = np.array([r["eps"] for r in rows_j])
    sJ = np.array([r["sigma_kPa"] for r in rows_j])
    out = [(0.0, 0.0, 0.0)]
    for key in sorted(k for k in z.files if k.startswith("eps")):
        e = float(key[3:])
        zp, zq = zm(z[key], p - 1), zm(z[key], q - 1)
        em = (zq - zp) / (Z0[1] - Z0[0]) - 1.0
        out.append((e, em, float(np.interp(e, eJ, sJ))))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--q", type=int, default=7)
    ap.add_argument("--radius", type=float, default=1.0)
    a = ap.parse_args()
    c = marker_curve(a.run, a.p, a.q, a.radius)
    print(f"{'eps_end':>8s} {'eps_mk':>8s} {'ratio':>6s} {'sigma':>7s}")
    for e, em, s in c[1:]:
        print(f"{e:8.4f} {em:8.4f} {em / e:6.3f} {s:7.2f}")


if __name__ == "__main__":
    main()
