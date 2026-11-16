"""Host-side plotting of the single-cut results (Fig. 2 of the paper).

Run with the host Python (needs numpy + matplotlib):  python scripts/plot_single_cut.py
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: E402
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

R = "results/single_cut"
INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "figure.dpi": 150})
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#d9d8d4", "#e34948"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#e8f0fb", "#2a78d6", "#0b2f5c"])


def fig2a_2b():
    d = np.load(f"{R}/inplane_b80.npz")
    x, cells = d["x"], d["cells"]
    tri = mtri.Triangulation(x[:, 0], x[:, 1], cells)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    s = d["trace_stress"]
    lim = np.percentile(np.abs(s), 99)
    tc = axes[0].tripcolor(tri, s, shading="gouraud", cmap=DIVERGING,
                           norm=TwoSlopeNorm(0, -lim, lim))
    axes[0].plot([91, 91], [10, 90], color=INK, lw=1)
    fig.colorbar(tc, ax=axes[0], label=r"$w(\sigma_1+\sigma_2)/Eh$,  $\Delta=h$")
    axes[0].set(aspect="equal", xlabel="x (mm)", ylabel="y (mm)",
                title="In-plane stress, b = 80 mm")
    axes[0].grid(False)
    m = d["mode0"]
    tc = axes[1].tripcolor(tri, m, shading="gouraud", cmap=SEQ)
    axes[1].plot([91, 91], [10, 90], color=INK, lw=1)
    fig.colorbar(tc, ax=axes[1], label=r"$\delta/\delta_0$")
    axes[1].set(aspect="equal", xlabel="x (mm)",
                title=f"First buckling mode, $\\Delta_c$ = {d['Delta_c'][0]*1e3:.2f} µm")
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(f"{R}/fig2a_inplane_mode.png")


def fig2b_3d():
    f = f"{R}/post_h0.127_b80.npz"
    if not os.path.exists(f):
        return
    d = np.load(f)
    x, cells = d["x"], d["cells"]
    tri = mtri.Triangulation(x[:, 0], x[:, 1], cells)
    fig = plt.figure(figsize=(10, 4))
    for k, i in enumerate((1, 3)):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        wz = d["snap_w"][i]
        ax.plot_trisurf(tri, wz, cmap=SEQ, linewidth=0, antialiased=True,
                        vmin=0, vmax=np.abs(wz).max())
        ax.set_box_aspect((182, 100, 40))
        ax.set(xlabel="x", ylabel="y", zlabel="w (mm)",
               title=f"Δ = {d['snap_Delta'][i]:.2f} mm,  δ0 = {np.abs(wz).max():.1f} mm")
        ax.zaxis.set_major_locator(plt.MaxNLocator(3))
        ax.view_init(28, -60)
    fig.suptitle("Post-buckled shape, h = 0.127 mm, b = 80 mm")
    fig.tight_layout()
    fig.savefig(f"{R}/fig2b_postbuckled.png")


def fig2c():
    rows = list(csv.DictReader(open(f"{R}/buckling_sweep.csv")))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    markers = {"0.1": "o", "0.12": "s", "0.127": "^", "0.14": "D"}
    for h, mk in markers.items():
        r = [row for row in rows if row["h"] == h]
        ax.plot([float(t["b_over_w"]) for t in r], [float(t["Fc_over_EhDc"]) for t in r],
                mk, ms=5, mfc="none", color=BLUE, label=f"h = {h} mm")
    ax.set(xlabel="b / w", ylabel=r"$F_c / (E h \Delta_c)$,  $\Delta_c=h^2/b$",
           xlim=(0.25, 0.85), ylim=(0, 20), title="Critical force")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig2c_critical_force.png")


def fig2d():
    files = sorted(f for f in glob.glob(f"{R}/post_h*_b*.npz") if "_L" not in f)
    if not files:
        return
    data = [np.load(f) for f in files]
    bws = sorted({float(d["b"]) / float(d["w"]) for d in data})
    cmap = LinearSegmentedColormap.from_list("s", ["#9ec3ee", "#0b2f5c"])
    styles = {0.127: ":", 0.15: "-", 0.21: "--"}
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for d in data:
        h, b, w = float(d["h"]), float(d["b"]), float(d["w"])
        bw = b / w
        c = cmap(bws.index(round(bw, 6)) / max(len(bws) - 1, 1)) if round(bw, 6) in bws else BLUE
        ax.loglog(d["Delta"] / w, (d["delta0"] / b) ** 2, styles.get(h, "-"), color=c, lw=1.6,
                  label=f"h={h:g}, b/w={bw:.1f}")
    xs = np.geomspace(1e-4, 1, 50)
    ax.loglog(xs, xs, "--", color=ORANGE, lw=1.2, label=r"$(\delta_0/b)^2=\Delta/w$")
    ax.set(xlabel=r"$\Delta / w$", ylabel=r"$\delta_0^2 / b^2$", xlim=(1e-4, 0.2), ylim=(1e-4, 1),
           title="Post-buckling amplitude")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{R}/fig2d_scaling.png")


def prefactor_at(p, x):
    """Local prefactor (delta0/b)^2 / ((Delta - Delta_c)/W) of one run at Delta/W = x."""
    D, Dc, b, w = p["Delta"], float(p["Delta_c"]), float(p["b"]), float(p["w"])
    m = D > 1.05 * Dc
    return float(np.interp(x, D[m] / w, ((p["delta0"] / b) ** 2 / ((D - Dc) / w))[m]))


def fig_paper():
    """The five single-cut panels in one figure, without titles (for the paper)."""
    d = np.load(f"{R}/inplane_b80.npz")
    x, cells = d["x"], d["cells"]
    tri = mtri.Triangulation(x[:, 0], x[:, 1], cells)
    with plt.rc_context({"font.size": 8}):
        fig = plt.figure(figsize=(7.0, 4.0))
        outer = fig.add_gridspec(2, 1, hspace=0.45, height_ratios=[1, 1.05])
        top = outer[0].subgridspec(1, 2, wspace=0.32)
        bot = outer[1].subgridspec(1, 3, wspace=0.55)
        ax = fig.add_subplot(top[0])
        s = d["trace_stress"]
        lim = np.percentile(np.abs(s), 99)
        tc = ax.tripcolor(tri, s, shading="gouraud", cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
        ax.plot([91, 91], [10, 90], color=INK, lw=1)
        fig.colorbar(tc, ax=ax, label=r"$W(\sigma_1+\sigma_2)/(Eh)$", shrink=0.8)
        ax.set(aspect="equal", xlabel="$x$ (mm)", ylabel="$y$ (mm)")
        ax.grid(False)
        ax.set_title("(a)", loc="left", fontweight="bold")
        ax = fig.add_subplot(top[1])
        tc = ax.tripcolor(tri, d["mode0"], shading="gouraud", cmap=DIVERGING,
                          norm=TwoSlopeNorm(0, -1, 1))     # the mode is negative near the free edges
        ax.plot([91, 91], [10, 90], color=INK, lw=1)
        fig.colorbar(tc, ax=ax, label=r"$w/\delta_0$", shrink=0.8)
        ax.set(aspect="equal", xlabel="$x$ (mm)", ylabel="$y$ (mm)")
        ax.grid(False)
        ax.set_title("(b)", loc="left", fontweight="bold")
        ax = fig.add_subplot(bot[0])
        rows = list(csv.DictReader(open(f"{R}/buckling_sweep.csv")))
        for h, mk in {"0.1": "o", "0.12": "s", "0.127": "^", "0.14": "D"}.items():
            r = [row for row in rows if row["h"] == h]
            ax.plot([float(t["b_over_w"]) for t in r], [float(t["Fc_over_EhDc"]) for t in r],
                    mk, ms=4, mfc="none", color=BLUE, label=f"$h$ = {h} mm")
        ax.set(xlabel="$b/W$", ylabel=r"$F_c/(Eh\,h^2/b)$", xlim=(0.25, 0.85), ylim=(0, 20))
        ax.legend(frameon=False, fontsize=6)
        ax.set_title("(c)", loc="left", fontweight="bold")

        # (d) every post-buckling run against the weakly nonlinear (Koiter) amplitude
        # delta_0^2 = c h^2 (Delta/Delta_c - 1), with Delta_c and c from linear solves
        # (scripts/single_cut_koiter.py): no fitted parameter. Near the threshold only.
        ax = fig.add_subplot(bot[1])
        koi = {(round(float(r["b_over_W"]), 3), round(float(r["L_over_W"]), 3)): r
               for r in csv.DictReader(open(f"{R}/koiter.csv"))}
        c_of = lambda p: float(koi[(round(float(p["b"]) / float(p["w"]), 3),
                                    round(float(p["L"]) / float(p["w"]), 3))]["c"])
        data = [np.load(f) for f in sorted(glob.glob(f"{R}/post_h*_b*.npz")) if "_L" not in f]
        long_sheet = sorted([np.load(f) for f in glob.glob(f"{R}/post_h0.127_b*_L350.npz")],
                            key=lambda p: float(p["b"]))
        # one distinct color per simulated b/W: discrete steps of viridis, stopping short of its
        # pale yellow end, so neighbouring values differ in both hue and lightness
        bw_all = sorted({round(float(p["b"]) / float(p["w"]), 6) for p in data})
        colors = [plt.cm.viridis(0.9 * i / (len(bw_all) - 1)) for i in range(len(bw_all))]
        color_of = dict(zip(bw_all, colors))
        markers = {0.127: "o", 0.15: "s", 0.21: "^"}
        RMAX = 4.0           # Delta/Delta_c range of (d)
        for runs_, filled in ((sorted(data, key=lambda p: float(p["b"])), False), (long_sheet, True)):
            for p in runs_:
                h, b, w, Dc = float(p["h"]), float(p["b"]), float(p["w"]), float(p["Delta_c"])
                r = p["Delta"] / Dc
                m = (r > 1.04) & (r <= RMAX)
                col = color_of[round(b / w, 6)]
                ax.plot(r[m] - 1, (p["delta0"][m] / h) ** 2 / c_of(p), "D" if filled else markers[h],
                        ms=2.6 if filled else 3.0, mfc=col if filled else "none", mew=0.8, color=col)
        ax.plot([0, RMAX - 1], [0, RMAX - 1], color=INK, lw=0.8, zorder=0)
        ax.set(xlabel=r"$\Delta/\Delta_c-1$", ylabel=r"$\delta_0^2/(c\,h^2)$",
               xlim=(0, RMAX - 1), ylim=(0, 4.2))
        from matplotlib.colors import BoundaryNorm, ListedColormap
        dcmap = ListedColormap(colors)
        bounds = np.arange(len(bw_all) + 1) - 0.5
        sm = plt.cm.ScalarMappable(cmap=dcmap, norm=BoundaryNorm(bounds, dcmap.N))
        cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.03, ticks=range(len(bw_all)))
        cb.ax.set_yticklabels([f"{v:g}" for v in bw_all], fontsize=6)
        cb.ax.tick_params(length=0)
        cb.ax.set_title("$b/W$", fontsize=7, pad=3)
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], color=INK, marker=mk, ls="none", mfc="none", ms=3.5) for mk in markers.values()]
        hd += [Line2D([], [], color=INK, marker="D", ls="none", ms=3), Line2D([], [], color=INK, lw=0.8)]
        ax.legend(hd, [f"$h$ = {v:g}" for v in markers] + ["$L/W$ = 3.5", "theory"], frameon=False,
                  fontsize=5.5, loc="upper left", handletextpad=0.3, labelspacing=0.25)
        ax.set_title("(d)", loc="left", fontweight="bold")

        # (e) theory against simulation for the one number the theory predicts: the coefficient c.
        # Lines: c from the linear solves (scripts/single_cut_koiter.py). Symbols: the nonlinear
        # runs (h = 0.127 mm), delta_0^2 / [h^2 (Delta/Delta_c - 1)] at Delta = 1.3 Delta_c.
        ax = fig.add_subplot(bot[2])
        thin = sorted([p for p in data if float(p["h"]) == 0.127], key=lambda p: float(p["b"]))
        table = []

        def c_run(p, at=1.3):
            r = p["Delta"] / float(p["Delta_c"])
            return float(np.interp(at, r, (p["delta0"] / float(p["h"])) ** 2 / (r - 1)))

        for runs_, col, mk, lw in ((thin, BLUE, "o", 1.82), (long_sheet, ORANGE, "s", 3.5)):
            kc = sorted((k[0], float(r["c"])) for k, r in koi.items() if k[1] == lw)
            kb = sorted((k[0], float(r["beta1_0"])) for k, r in koi.items() if k[1] == lw)
            ax.plot(*zip(*kc), "-", lw=1.0, color=col)
            bws_ = [float(p["b"]) / float(p["w"]) for p in runs_]
            ax.plot(bws_, [c_run(p) for p in runs_], mk, ms=4, mfc="none", mew=0.9, color=col)
            table.append((lw, kb, bws_, runs_))
        hd = [Line2D([], [], color=BLUE, lw=1.0, marker="o", mfc="none", ms=4),
              Line2D([], [], color=ORANGE, lw=1.0, marker="s", mfc="none", ms=4)]
        ax.legend(hd, ["$L/W$ = 1.82", "$L/W$ = 3.5"], frameon=False, fontsize=6, loc="upper left",
                  title="lines: theory\nsymbols: simulation", title_fontsize=6)
        ax.set(xlabel="$b/W$", ylabel="$c$", xlim=(0.25, 1.0), ylim=(0, 14))
        ax.set_title("(e)", loc="left", fontweight="bold")
        fig.savefig(f"{R}/fig_single_cut.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    # numbers quoted in the paper
    for lw, kb, bws_, runs_ in table:
        print(f"L/W = {lw}: beta_1 at threshold (Koiter): " + "  ".join(f"{b:g}: {v:.3f}" for b, v in kb))
        for v, p in zip(bws_, runs_):
            Dc, h = float(p["Delta_c"]), float(p["h"])
            r = p["Delta"] / Dc
            cm = (p["delta0"] / h) ** 2 / (r - 1)
            print(f"   b/W={v:.2f}: c = {c_of(p):.2f}; nonlinear c at Delta/Dc = 1.3, 2, 4: "
                  + ", ".join(f"{np.interp(t, r, cm):.2f} ({100 * (np.interp(t, r, cm) / c_of(p) - 1):+.1f}%)"
                              for t in (1.3, 2.0, 4.0))
                  + f"; at Delta/W = 1e-3: Delta/Dc = {1e-3 * float(p['w']) / Dc:.1f}, "
                  f"beta_1 = {prefactor_at(p, 1e-3):.2f}")


def fig_shape():
    """Critical force against b/W and L/W (shape_sweep.csv), and the post-buckling
    amplitude for three sheet lengths (b/W = 0.4 and 0.8, h = 0.127 mm)."""
    path = f"{R}/shape_sweep.csv"
    if not os.path.exists(path):
        return
    E, h = 3500.0, 0.127
    rows = [r for r in csv.DictReader(open(path)) if float(r["W"]) == 100]
    LW = sorted({float(r["L_over_W"]) for r in rows})
    BW = sorted({float(r["b_over_W"]) for r in rows})

    def fc(r):        # Fc / (E h h^2 / b)
        return float(r["F_unit"]) * float(r["Delta_c"]) / (E * h * h**2 / float(r["b"]))
    seq = LinearSegmentedColormap.from_list("s", ["#9ec3ee", "#0b2f5c"])
    with plt.rc_context({"font.size": 8}):
        fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.0), gridspec_kw=dict(wspace=0.38))
        ax = axs[0]
        for i, lw in enumerate(LW):
            r = sorted([x for x in rows if float(x["L_over_W"]) == lw], key=lambda x: float(x["b_over_W"]))
            ax.plot([float(x["b_over_W"]) for x in r], [fc(x) for x in r], "o-", ms=3, lw=1.2,
                    color=seq(i / (len(LW) - 1)), label=f"$L/W$ = {lw:g}")
        ax.set(xlabel="$b/W$", ylabel=r"$F_c/(Eh\,h^2/b)$", ylim=(0, None))
        ax.legend(frameon=False, fontsize=6)
        ax.set_title("(a)", loc="left", fontweight="bold")
        ax = axs[1]
        for i, bw in enumerate(BW):
            r = sorted([x for x in rows if float(x["b_over_W"]) == bw], key=lambda x: float(x["L_over_W"]))
            f = np.array([fc(x) for x in r])
            ax.plot([float(x["L_over_W"]) for x in r], f / f[-1], "o-", ms=3, lw=1.2,
                    color=seq(i / (len(BW) - 1)), label=f"$b/W$ = {bw:g}")
        ax.axhline(1, color=MUTED, lw=0.6, ls=":")
        ax.set(xlabel="$L/W$", ylabel=r"$F_c\,/\,F_c[L/W=3.5]$", yscale="log")
        ax.set_yticks([1, 2, 4, 6])
        ax.set_yticklabels(["1", "2", "4", "6"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(frameon=False, fontsize=6)
        ax.set_title("(b)", loc="left", fontweight="bold")
        ax = axs[2]
        styles = {100: ":", 182: "-", 350: "--"}
        for b, c in ((40, seq(0.0)), (80, seq(1.0))):
            for L, st in styles.items():
                f = f"{R}/post_h0.127_b{b}.npz" if L == 182 else f"{R}/post_h0.127_b{b}_L{L}.npz"
                if not os.path.exists(f):
                    continue
                p = np.load(f)
                ax.loglog(p["Delta"] / float(p["w"]), (p["delta0"] / b) ** 2, st, color=c, lw=1.2)
        xs = np.geomspace(1e-4, 1, 20)
        ax.loglog(xs, xs, "--", color=ORANGE, lw=1.0)
        from matplotlib.lines import Line2D
        l1 = ax.legend([Line2D([], [], color=seq(0.0), lw=2), Line2D([], [], color=seq(1.0), lw=2),
                        Line2D([], [], color=ORANGE, ls="--")],
                       ["$b/W$ = 0.4", "$b/W$ = 0.8", r"$\Delta/W$"], frameon=False, fontsize=6,
                       loc="upper left")
        ax.add_artist(l1)
        ax.legend([Line2D([], [], color=INK, ls=st) for st in styles.values()],
                  [f"$L/W$ = {L / 100:g}" for L in styles], frameon=False, fontsize=6, loc="lower right")
        ax.set(xlabel=r"$\Delta/W$", ylabel=r"$(\delta_0/b)^2$", xlim=(1e-4, 0.2), ylim=(1e-4, 1))
        ax.set_title("(c)", loc="left", fontweight="bold")
        fig.savefig(f"{R}/fig_single_cut_shape.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def fit_slope(runs, xmax=5e-3):
    """One-parameter fit of (delta0/b)^2 = beta1 (Delta - Delta_c)/W, with Delta_c the computed
    threshold of each run, by least squares on the relative error over 1.05 Dc < Delta and
    Delta/W <= xmax. Returns beta1."""
    xs, ys = [], []
    for p in runs:
        D, Dc, b, w = p["Delta"], float(p["Delta_c"]), float(p["b"]), float(p["w"])
        m = (D > 1.05 * Dc) & (D / w <= xmax)
        xs.append((D[m] - Dc) / w)
        ys.append((p["delta0"][m] / b) ** 2)
    q = np.concatenate(xs) / np.concatenate(ys)
    return float(np.sum(q) / np.sum(q**2))


def fit_amplitude(runs, xmax=5e-3):
    """Least-squares fit of delta0^2 = beta1 b^2 Delta/W + beta2 h^2 (relative residuals),
    over the steps with 1.05 Dc < Delta and Delta/W <= xmax. Returns (beta1, beta2)."""
    A = []
    for r in runs:
        m = (r["Delta"] > 1.05 * float(r["Delta_c"])) & (r["Delta"] / float(r["w"]) <= xmax)
        d2 = r["delta0"][m] ** 2
        A.append(np.column_stack([float(r["b"]) ** 2 * r["Delta"][m] / float(r["w"]),
                                  np.full(m.sum(), float(r["h"]) ** 2)]) / d2[:, None])
    A = np.vstack(A)
    beta, *_ = np.linalg.lstsq(A, np.ones(len(A)), rcond=None)
    return beta


def fig_fit(xmax=5e-3):
    """Collapse of the nine L = 182 mm runs with one global fit of beta1, beta2."""
    runs = [np.load(f) for f in sorted(glob.glob(f"{R}/post_h*_b*.npz")) if "_L" not in f]
    b1, b2 = fit_amplitude(runs, xmax)
    bws = sorted({round(float(p["b"]) / float(p["w"]), 6) for p in runs})
    cmap = LinearSegmentedColormap.from_list("s", ["#9ec3ee", "#0b2f5c"])
    styles = {0.127: "-", 0.15: "--", 0.21: ":"}
    from matplotlib.lines import Line2D
    with plt.rc_context({"font.size": 8}):
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(wspace=0.3))
        for p in runs:
            h, b, w = float(p["h"]), float(p["b"]), float(p["w"])
            m = p["Delta"] > 1.05 * float(p["Delta_c"])
            X = p["Delta"][m] / w
            Y = (p["delta0"][m] ** 2 - b2 * h**2) / b**2
            c = cmap(bws.index(round(b / w, 6)) / (len(bws) - 1))
            axs[0].loglog(X, Y, styles[h], color=c, lw=1.2)
            axs[1].semilogx(X, Y / X, styles[h], color=c, lw=1.2)
        xs = np.geomspace(1e-5, 0.2, 20)
        for ax in axs:
            ax.axvspan(1e-5, xmax, color="#eef2f7", zorder=0, lw=0)
        axs[0].loglog(xs, b1 * xs, "--", color=ORANGE, lw=1.1)
        axs[1].axhline(b1, color=ORANGE, ls="--", lw=1.1)
        axs[0].set(xlabel=r"$\Delta/W$", ylabel=r"$(\delta_0^2-\beta_2h^2)/b^2$", xlim=(1e-5, 0.12),
                   ylim=(1e-5, 0.2))
        axs[1].set(xlabel=r"$\Delta/W$", ylabel=r"$(\delta_0^2-\beta_2h^2)/(b^2\,\Delta/W)$",
                   xlim=(1e-5, 0.12), ylim=(0, 1.2))
        cs = [Line2D([], [], color=cmap(i / (len(bws) - 1)), lw=2) for i in range(len(bws))]
        hs = [Line2D([], [], color=INK, ls=st, lw=1.2) for st in styles.values()]
        fitl = Line2D([], [], color=ORANGE, ls="--")
        l1 = axs[1].legend(cs + [fitl], [f"$b/W$ = {v:.1f}" for v in bws] +
                           [rf"fit: $\beta_1$ = {b1:.2f}, $\beta_2$ = {b2:.1f}"],
                           frameon=False, fontsize=6.5, loc="lower left")
        axs[1].add_artist(l1)
        axs[1].legend(hs, [f"$h$ = {v:g} mm" for v in styles], frameon=False, fontsize=6.5,
                      loc="upper left")
        axs[0].set_title("(a)", loc="left", fontweight="bold")
        axs[1].set_title("(b)", loc="left", fontweight="bold")
        fig.savefig(f"{R}/fig_single_cut_fit.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return b1, b2


if __name__ == "__main__":
    fig2a_2b(); fig2b_3d(); fig2c(); fig2d(); fig_paper(); fig_shape()
    print("figures written to", R)
