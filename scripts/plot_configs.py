"""Host-side plots of the multi-cut configurations (Figs. 3c, 4b, 5a).

Usage: python scripts/plot_configs.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "results/configs"
INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "figure.dpi": 150})


def load(name):
    f = f"{R}/{name}.npz"
    return np.load(f) if os.path.exists(f) else None


def probe(d, name):
    """Probe history (n_steps x 3 displacement) for a named probe."""
    return d["probes"][:, list(d["probe_names"]).index(name), :]


def theta_y(d):
    """Pitch angle of the inner ribbon from its two end probes (degrees)."""
    uL, uR = probe(d, "rib_left"), probe(d, "rib_right")
    dx = 88.0 + uR[:, 0] - uL[:, 0]
    return np.degrees(np.arctan2(uR[:, 2] - uL[:, 2], dx))


def theta_x(d, half="up"):
    """Roll: rotation about x of one half of the central panel (above or below
    the horizontal slit), from an outer and an inner probe (degrees). The two
    halves fold up symmetrically, like a roof, so each half rolls by the same
    angle in opposite directions."""
    y_out, y_in = {"up": (52.0, 34.0), "lo": (8.0, 26.0)}[half]   # probe positions
    o, i = probe(d, f"{half}_out"), probe(d, f"{half}_in")
    dy = np.abs((y_out + o[:, 1]) - (y_in + i[:, 1]))
    return np.degrees(np.arctan2(i[:, 2] - o[:, 2], dy))


def theta_z(d):
    """Yaw: in-plane rotation of the leaf's major axis (degrees)."""
    a, b = probe(d, "leaf_a"), probe(d, "leaf_b")
    v0 = np.array([2 * 9.21, 2 * 3.35])
    v = v0 + (b - a)[:, :2]
    return np.degrees(np.arctan2(v[:, 1], v[:, 0]) - np.arctan2(v0[1], v0[0]))


def fig3c():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for name, c, lab in (("two_cuts_ls10", BLUE, r"$\ell_s$ = 10 mm (3a-i)"),
                         ("two_cuts_ls30", ORANGE, r"$\ell_s$ = 30 mm (3a-ii)")):
        d = load(name)
        if d is None:
            continue
        ax.plot(d["Delta"] / d["L"] * 1e3, np.abs(probe(d, "centre")[:, 2]), "o-", ms=3,
                color=c, lw=1.5, label=lab)
    ax.set(xlabel=r"$\Delta/L\ \ (\times 10^{-3})$", ylabel=r"$\delta_0$ at centre (mm)",
           title="Two parallel cuts: lift")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig3c_lift.png")


def fig_pitch():
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    for name, c, lab in (("pitch_concave", BLUE, r"concave, $(2L_c+L_\parallel)/w$ = 1.16"),
                         ("pitch_convex", ORANGE, r"convex, $(2L_c+L_\parallel)/w$ = 0.80")):
        d = load(name)
        if d is None:
            continue
        ax.semilogx(d["Delta"] / d["L"], np.abs(theta_y(d)), "o-", ms=3, color=c, lw=1.5,
                    label=lab)
    ax.set(xlabel=r"$\Delta/L$", ylabel=r"$\theta_y$ (deg)", ylim=(0, None),
           title="Pitch actuator")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig4_pitch.png")


def fig5a():
    """Summary in the style of Fig. 5a: lift and the three rotations vs
    Delta/L (two panels instead of the paper's twin y-axes)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    d = load("two_cuts_ls10")
    if d is not None:
        a1.semilogx(d["Delta"][1:] / d["L"], np.abs(probe(d, "centre")[1:, 2]), color=INK,
                    lw=1.8, ls="--", label=r"lift $\delta_0$ (two cuts)")
    d = load("pitch_convex")
    if d is not None:
        a1.semilogx(d["Delta"][1:] / d["L"], d["uz_max"][1:], color=MUTED, lw=1.5,
                    label="max lift, convex pitch cell")
    a1.set(xlabel=r"$\Delta/L$", ylabel="out-of-plane displacement (mm)", title="Lift")
    a1.legend(frameon=False, fontsize=8)
    for name, fn, c, lab in (("pitch_concave", theta_y, "#e34948", r"pitch $\theta_y$"),
                             ("roll", theta_x, ORANGE, r"roll $\theta_x$ (half-panel)"),
                             ("yaw", theta_z, BLUE, r"yaw $\theta_z$")):
        d = load(name)
        if d is None:
            continue
        a2.semilogx(d["Delta"][1:] / d["L"], np.abs(fn(d))[1:], color=c, lw=1.8, label=lab)
    a2.set(xlabel=r"$\Delta/L$", ylabel=r"rotation $\theta$ (deg)", ylim=(0, None),
           title="Rotations")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig5a_summary.png")


def fig_alternating():
    """Alternating array: force-stretch response against an uncut sheet, and
    the out-of-plane tilt of two neighbouring facets."""
    import sys
    sys.path.insert(0, ".")
    from kirigami.api import _pair_angles
    d = load("alternating")
    if d is None or "F" not in d:
        return
    L, W, h = float(d["L"]), float(d["W"]), float(d["h"])
    eps = d["Delta"] / L
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.loglog(eps[1:], d["F"][1:], color=BLUE, lw=1.8, label="alternating cuts")
    a1.loglog(eps[1:], 3500.0 * h * W * eps[1:], color=MUTED, lw=1.2, ls="--",
              label=r"uncut sheet, $EhW\,\Delta/L$")
    a1.set(xlabel=r"stretch $\Delta/L$", ylabel="force (N)", title="Force vs stretch")
    a1.legend(frameon=False, fontsize=8)
    names = {"facet1_a": (58.3, 24.0), "facet1_b": (61.7, 24.0),
             "facet2_a": (63.3, 36.0), "facet2_b": (66.7, 36.0)}
    ang = _pair_angles(d["probes"], names)
    for k, c in (("facet1", BLUE), ("facet2", ORANGE)):
        a2.plot(eps * 100, np.abs(ang[k]["tilt_deg"]), color=c, lw=1.8, label=k)
    a2.set(xlabel=r"stretch $\Delta/L$ (%)", ylabel="facet tilt (deg)", ylim=(0, None),
           title="Facet tilt (centre of the array)")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_alternating.png")


def fig_paper():
    """Lift and rotations of the multi-cut configurations, without titles (for the paper)."""
    with plt.rc_context({"font.size": 8}):
        fig, axs = plt.subplots(1, 3, figsize=(5.6, 2.1), gridspec_kw=dict(wspace=0.55))
        ax = axs[0]
        for name, c, lab in (("two_cuts_ls10", BLUE, r"$\ell_s=10$ mm"),
                             ("two_cuts_ls30", ORANGE, r"$\ell_s=30$ mm")):
            d = load(name)
            ax.plot(d["Delta"] / d["L"] * 1e3, np.abs(probe(d, "centre")[:, 2]), "o-", ms=2.5,
                    lw=1, color=c, label=lab)
        ax.set(xlabel=r"$\Delta/L\ (\times10^{-3})$", ylabel=r"lift $\delta_0$ (mm)", ylim=(0, None))
        ax.legend(frameon=False, fontsize=6)
        ax.set_title("(a)", loc="left", fontweight="bold")
        ax = axs[1]
        for name, c, lab in (("pitch_concave_fine", BLUE, "concave"), ("pitch_convex", ORANGE, "convex")):
            d = load(name)
            ax.plot(d["Delta"] / d["L"], np.abs(theta_y(d)), "-", lw=1.4, color=c, label=lab)
        ax.set(xlabel=r"$\Delta/L$", ylabel=r"pitch $|\theta_y|$ (deg)", ylim=(0, 75), xlim=(0, 0.08))
        ax.legend(frameon=False, fontsize=6)
        ax.set_title("(b)", loc="left", fontweight="bold")
        ax = axs[2]
        d = load("roll")
        ax.plot(d["Delta"] / d["L"], np.abs(theta_x(d, "up")), "-", lw=1.4, color=BLUE,
                label=r"roll $|\theta_x|$")
        d = load("yaw")
        ax.plot(d["Delta"] / d["L"], np.abs(theta_z(d)), "-", lw=1.4, color=ORANGE,
                label=r"yaw $|\theta_z|$")
        ax.set(xlabel=r"$\Delta/L$", ylabel=r"rotation (deg)", ylim=(0, 60), xlim=(0, 0.15))
        ax.legend(frameon=False, fontsize=6)
        ax.set_title("(c)", loc="left", fontweight="bold")
        fig.savefig(f"{R}/fig_actuators.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def chain_pitch(d):
    """Pitch of the three ribbons of the pitch chain (degrees), from their end probes."""
    names = list(d["probe_names"])
    out = []
    for k in range(3):
        a = d["probes"][:, names.index(f"rib{k}_l"), :]
        b = d["probes"][:, names.index(f"rib{k}_r"), :]
        out.append(np.degrees(np.arctan2(b[:, 2] - a[:, 2], 88.0 + b[:, 0] - a[:, 0])))
    return out


PR_RUN = f"{R}/pitch_roll_2x2.npz"          # copy of results/runs/pitch_roll_2x2/history.npz
# pitch_roll_2x2 (scripts/build_pitch_roll.py), sheet coordinates in mm: the centre of the
# middle panel of each pitch cell (A), and the roll probes of each roll cell (B) at the
# positions used for the single roll cell (15 mm right of the cell centre, 52 and 34 mm
# above the bottom of its 60 mm row).
PR_A = {"top": (82.5, 91.0), "bottom": (207.5, 30.0)}
PR_B = {"top": (207.5, 61.0), "bottom": (82.5, 0.0)}


def _node(x, p):
    return int(np.argmin((x[:, 0] - p[0]) ** 2 + (x[:, 1] - p[1]) ** 2))


def pitch_roll_angles(d):
    """Pitch of each A panel (rotation about y of the segment from 40 mm left to 40 mm right of
    its centre) and roll of the upper half of each B panel (degrees), for every step."""
    x, U = d["x"], d["U"].astype(float)
    th = {}
    for row, (xc, yc) in PR_A.items():
        i, j = _node(x, (xc - 40, yc)), _node(x, (xc + 40, yc))
        th[f"A {row}"] = np.degrees(np.arctan2(U[:, j, 2] - U[:, i, 2], 80.0 + U[:, j, 0] - U[:, i, 0]))
    for row, (xc, y0) in PR_B.items():
        o, i = _node(x, (xc + 15, y0 + 52)), _node(x, (xc + 15, y0 + 34))
        dy = np.abs(18.0 + U[:, o, 1] - U[:, i, 1])
        th[f"B {row}"] = np.degrees(np.arctan2(U[:, i, 2] - U[:, o, 2], dy))
    return th


def fig_series():
    """Cells in series: (a) the pitch chain, (b) pitch (A) and roll (B) in the 2 x 2 design,
    (c) the force carried by each row of the 2 x 2 design (for the paper, no titles)."""
    d2 = np.load(PR_RUN)
    th = pitch_roll_angles(d2)
    L2 = float(d2["L"])
    x2 = d2["Delta"] / L2
    onset = x2[np.argmax(np.abs(th["A top"]) > 5.0)]           # A has tilted by 5 degrees
    with plt.rc_context({"font.size": 8}):
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.55))
        ax = axs[0]
        d = load("pitch_chain")
        for k, (t, c) in enumerate(zip(chain_pitch(d), (BLUE, ORANGE, AQUA))):
            ax.plot(d["Delta"] / d["L"], t, "-", lw=1.4, color=c, label=f"cell {k + 1}")
        ax.set(xlabel=r"$\Delta/L$", ylabel=r"pitch $\theta_y$ (deg)", ylim=(0, 75), xlim=(0, 0.075),
               xticks=[0, 0.03, 0.06])
        ax.legend(frameon=False, fontsize=6, loc="lower right")
        ax.set_title("(a)", loc="left", fontweight="bold")
        ax = axs[1]
        for key, c, ls in (("B top", BLUE, "-"), ("B bottom", BLUE, "--"),
                           ("A top", ORANGE, "-"), ("A bottom", ORANGE, "--")):
            lab = (r"B roll $|\theta_x|$" if key[0] == "B" else r"A pitch $|\theta_y|$") + f", {key.split()[1]}"
            ax.plot(x2, np.abs(th[key]), ls, lw=1.3, color=c, label=lab)
        ax.axvline(onset, color=MUTED, lw=0.8, ls=":")
        ax.set(xlabel=r"$\Delta/L$", ylabel="rotation (deg)", ylim=(0, 80), xlim=(0, x2.max()),
               xticks=[0, 0.05, 0.1])
        ax.legend(frameon=False, fontsize=5.5, loc="upper left", handlelength=1.8,
                  labelspacing=0.3)
        ax.set_title("(b)", loc="left", fontweight="bold")
        ax = axs[2]
        ax.plot(x2, d2["F"] / 2, "-", lw=1.4, color=INK)
        ax.axvline(onset, color=MUTED, lw=0.8, ls=":")
        ax.set(xlabel=r"$\Delta/L$", ylabel="force per row (N)", ylim=(0, None), xlim=(0, x2.max()),
               xticks=[0, 0.05, 0.1])
        ax.set_title("(c)", loc="left", fontweight="bold")
        fig.savefig(f"{R}/fig_series.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    i = int(np.argmax(np.abs(th["A top"]) > 5.0))
    print(f"A tilts past 5 deg at Delta/L = {onset:.4f} (Delta = {d2['Delta'][i]:.2f} mm, "
          f"force per row {d2['F'][i] / 2:.3f} N)")
    for k, v in th.items():
        print(f"   {k}: {np.abs(v[-1]):.1f} deg at the last step (Delta = {d2['Delta'][-1]:.2f} mm)")


if __name__ == "__main__":
    fig3c(); fig_pitch(); fig5a(); fig_alternating(); fig_paper(); fig_series()
    print("figures written to", R)
