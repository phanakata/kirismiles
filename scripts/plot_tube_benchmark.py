"""Compare the uncut-tube benchmark (results/tube/benchmark.json) with the
analytical solutions: exact SVK uniaxial stretch for the stress-free tube, and
Rafsanjani et al. (PNAS 2019) SI Eqs. S11 and S13 for the rolled sheet."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("results/tube/benchmark.json"))
E, nu, t = d["material"]["E"], d["material"]["nu"], d["material"]["h"]
L, H = d["L"], d["H"]
Dm = E * t**3 / (12 * (1 - nu**2))


def svk(eps):
    l1 = 1 + eps
    E11 = 0.5 * (l1**2 - 1)
    return np.sqrt(1 - 2 * nu * E11), l1 * E * E11       # lambda_2, nominal stress


fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
s = d["stress_free_n4"]
eps = np.array([r["eps"] for r in s["rows"]])
lam2, P = svk(eps)
ax[0].plot(eps, lam2, "k-", lw=1, label="exact (SVK)")
ax[0].plot(eps, [r["radius"] / s["R"] for r in s["rows"]], "o", ms=4, mfc="none", label="tube FE")
ax[0].set(xlabel=r"$\varepsilon$", ylabel=r"$R/R_0$", title="stress-free tube, radius")
ax[1].plot(eps, 2 * np.pi * s["R"] * t * P, "k-", lw=1, label="exact (SVK)")
ax[1].plot(eps, [r["force"] for r in s["rows"]], "o", ms=4, mfc="none", label="tube FE")
ax[1].set(xlabel=r"$\varepsilon$", ylabel="axial force (N)", title="stress-free tube, force")
for n, col in ((4, "C3"), (8, "C2")):
    r = d[f"rolled_n{n}"]
    e = np.array([q["eps"] for q in r["rows"]])
    M0 = E * H * t**3 * (2 * np.pi / n) / (12 * L * (1 - nu**2))
    ee = np.linspace(0, e.max(), 50)
    ax[2].plot(ee, M0 * (1 + ee) * (1 - nu * ee)**3, "-", color=col, lw=1, label=f"Eq. S13, n={n}")
    ax[2].plot(e, [abs(q["m_ss"]) * H for q in r["rows"]], "o", color=col, ms=4, mfc="none",
               label=f"tube FE, n={n}")
ax[2].set(xlabel=r"$\varepsilon$", ylabel=r"$M_z$ (N mm)", title="rolled sheet, moment", ylim=(0, 0.35))
for a in ax:
    a.legend(fontsize=6, frameon=False)
fig.savefig("results/tube/benchmark.png", dpi=200)

print("stress-free: eps  R/R0(FE)  exact  F(FE)  exact")
for r, l2, p in zip(s["rows"], lam2, P):
    print(f"  {r['eps']:.3f}  {r['radius'] / s['R']:.5f}  {l2:.5f}  {r['force']:.4f}  {2 * np.pi * s['R'] * t * p:.4f}")
for n in (4, 8):
    r = d[f"rolled_n{n}"]
    M0 = E * H * t**3 * (2 * np.pi / n) / (12 * L * (1 - nu**2))
    print(f"rolled n={n}: R={r['R']:.3f}  Eq.S11 M0={M0:.4f}")
    print("  eps  M_z(FE)  Eq.S13  KL-const-thickness  m_zz/m_ss")
    for q in r["rows"]:
        e = q["eps"]
        l2, _ = svk(e)
        print(f"  {e:.3f}  {abs(q['m_ss']) * H:.4f}  {M0 * (1 + e) * (1 - nu * e)**3:.4f}  "
              f"{Dm * H * l2 / r['R']:.4f}  {q['m_zz'] / q['m_ss']:.3f}")
