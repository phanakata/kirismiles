"""Compare tube runs (scripts/tube_triangular.py JSON) against a reference run:
E_eff (eps <= 0.01), peak stress and its strain (eps <= 0.06), sigma at chosen strains,
max relative difference of sigma on the common undamped strains, dofs/triangles, time.

  python scripts/compare_tube_runs.py REF.json RUN.json [RUN.json ...] [--png out.png]
"""
import json
import sys

import numpy as np


def load(p):
    d = json.load(open(p))
    r = [q for q in d["rows"] if not q.get("damped")]
    e = np.r_[0.0, [q["eps"] for q in r]]
    s = np.r_[0.0, [q["sigma_kPa"] for q in r]]
    return d, e, s


def stats(e, s):
    m = (e > 0) & (e <= 0.01 + 1e-9)
    E = float(np.sum(e[m] * s[m]) / np.sum(e[m] ** 2)) / 1e3          # MPa
    w = e <= 0.06
    i = int(np.argmax(s[w]))
    at = {x: (float(np.interp(x, e, s)) if e[-1] >= x - 1e-9 else np.nan) for x in (0.02, 0.05, 0.10)}
    return E, float(s[w][i]), float(e[w][i]), at


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    png = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None
    if png:
        args.remove(png)
    ref, runs = args[0], args[1:]
    dr, er, sr = load(ref)
    Er, pr, epr, atr = stats(er, sr)
    print(f"{'run':40s} {'tri':>7s} {'E(MPa)':>7s} {'peak':>7s} {'eps_pk':>7s} "
          f"{'s.02':>7s} {'s.05':>7s} {'s.10':>7s} {'maxdiff':>8s}")
    def row(name, d, e, s, ref=False):
        E, p, ep, at = stats(e, s)
        if ref:
            md = "ref"
        else:
            hi = min(e[-1], er[-1])
            g = np.linspace(0.0025, hi, 200)
            md = f"{np.max(np.abs(np.interp(g, e, s) - np.interp(g, er, sr)) / np.interp(g, er, sr)) * 100:.1f}%"
        print(f"{name:40s} {d['mesh'].get('triangles', 0):7d} {E:7.3f} {p:7.1f} {ep:7.4f} "
              f"{at[0.02]:7.1f} {at[0.05]:7.1f} {at[0.10]:7.1f} {md:>8s}")
    row(ref.split('/')[-1][:-5], dr, er, sr, True)
    curves = [(ref, er, sr)]
    for p in runs:
        d, e, s = load(p)
        row(p.split('/')[-1][:-5], d, e, s)
        curves.append((p, e, s))
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
        for k, (p, e, s) in enumerate(curves):
            ax.plot(e, s, "-" if k == 0 else "--", lw=2.2 if k == 0 else 1.6,
                    label=p.split('/')[-1][:-5] + (" (reference)" if k == 0 else ""))
        ax.set_xlabel(r"applied strain $\bar\varepsilon$")
        ax.set_ylabel(r"nominal stress $\sigma$ (kPa)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(png)
        print("wrote", png)


if __name__ == "__main__":
    main()
