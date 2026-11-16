"""Figures for KIRISMILES.md: every panel is drawn from its string by the parser.

    python scripts/plot_kirismiles.py          -> results/kirismiles/*.png
    python scripts/plot_kirismiles.py --paper  -> the same figures without titles,
                                                  in paper/figures/ (captions say it)

Each panel shows the L x W sheet (clamped ends hatched, pulled along x), the
cuts as dark lines, holes as white openings and solids in the sheet colour.
Where a figure explains an operation, the piece that was written ("seed") is
drawn in orange and the copies the operation adds in dark blue.
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kirigami import grammar                       # noqa: E402
from kirigami import kirismiles as ks              # noqa: E402

OUT = ROOT / "results" / "kirismiles"
PAPER_MODE = "--paper" in sys.argv
if PAPER_MODE:
    OUT = ROOT / "paper" / "figures"
SHEET, EDGE, CUT, SEED, MUTED, BAD, GEN = "#dfe8f3", "#5a6b80", "#1d2b3a", "#e8590c", "#8a99ab", "#d6336c", "#1f6fd1"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})


def sheet(ax, L, W, clamps=True):
    ax.add_patch(Rectangle((0, 0), L, W, fc=SHEET, ec=EDGE, lw=0.8, zorder=0))
    if clamps:
        c = 0.035 * L
        for x in (-c, L):
            ax.add_patch(Rectangle((x, -0.04 * W), c, 1.08 * W, fc="#c7ced8", ec=EDGE, lw=0.6,
                                   hatch="////", zorder=0))
    ax.set_xlim(-0.06 * L, 1.06 * L)
    ax.set_ylim(-0.1 * W, 1.1 * W)
    ax.set_aspect("equal")
    ax.axis("off")


def draw(ax, pat, color=CUT, lw=1.4, z=3):
    for h in pat.get("holes", ()):
        ax.add_patch(Polygon(h, closed=True, fc="white", ec=color, lw=0.8, zorder=z))
    for s in pat.get("solids", ()):
        ax.add_patch(Polygon(s, closed=True, fc=SHEET, ec=color, lw=0.6, ls="--", zorder=z + 0.5))
    for c in pat.get("cuts", ()):
        a = np.asarray(c)
        ax.plot(a[:, 0], a[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=z + 1)


def wrap(code, width=34):
    """Break a long string after ';' so it fits above its panel."""
    lines, cur = [], ""
    for tok in re.findall(r"[^;]*;?", code):
        if cur and len(cur) + len(tok) > width:
            lines.append(cur)
            cur = ""
        cur += tok
    return "\n".join(lines + [cur] if cur else lines)


def panel(ax, code, L, W, seed=None, title=None, sub=None, clamps=True):
    """The pattern of `code`; `seed` (a string) is overdrawn in orange and the
    rest, which the operation generated, in blue."""
    sheet(ax, L, W, clamps)
    draw(ax, ks.pattern(code, L, W), color=GEN if seed else CUT)
    if seed:
        draw(ax, ks.pattern(seed, L, W), color=SEED, lw=2.0, z=6)
    ax.set_title(title if title is not None else wrap(code), family="monospace", fontsize=8.5,
                 pad=4)
    if sub:
        ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top", fontsize=8,
                color=EDGE)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    if PAPER_MODE:                                   # the caption carries the title
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        for ax in fig.axes:
            if ax.get_title().startswith("Free-form coordinates"):
                ax.set_title("")
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def grid(rows, cols, w=3.2, h=1.9):
    fig, axs = plt.subplots(rows, cols, figsize=(w * cols, h * rows))
    return fig, np.atleast_1d(axs).ravel()


# ------------------------------------------------------------------ figures

def fig_coordinates():
    L, W = 120, 60
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sheet(ax, L, W)
    ax.set_xlim(-14, L + 14)
    ax.set_ylim(-12, W + 12)
    draw(ax, ks.pattern("V20@-30,5", L, W), color=SEED, lw=2.2)
    draw(ax, ks.pattern("H24@25,t-8", L, W), color=SEED, lw=2.2)
    draw(ax, ks.pattern("K15@25", L, W), color=SEED, lw=2.2)
    cx, cy = L / 2, W / 2
    kw = dict(color=EDGE, lw=0.8)
    ax.annotate("", (cx + 14, cy), (cx, cy), arrowprops=dict(arrowstyle="->", **kw))
    ax.annotate("", (cx, cy + 14), (cx, cy), arrowprops=dict(arrowstyle="->", **kw))
    ax.plot([0, L], [cy, cy], ls=":", **kw)
    ax.plot(cx, cy, "o", color=EDGE, ms=4)
    ax.text(cx + 15, cy, "x", va="center", color=EDGE)
    ax.text(cx, cy + 15, "y", ha="center", color=EDGE)
    ax.text(cx - 1.5, cy - 2.5, "(0, 0)", ha="right", va="top", fontsize=8, color=EDGE)
    ax.text(3, cy + 1.2, "centre line, y = 0", fontsize=7.5, color=EDGE)
    ax.text(L / 2, W + 1.6, "free edge  t  (y = +W/2)", ha="center", fontsize=8, color=EDGE)
    ax.text(L / 2, -1.6, "free edge  b  (y = -W/2)", ha="center", va="top", fontsize=8, color=EDGE)
    for x, lab in ((-0.035 * L / 2, "clamped"), (L + 0.035 * L / 2, "clamped")):
        ax.text(x, W / 2, lab, rotation=90, ha="center", va="center", fontsize=8, color=EDGE)
    ax.annotate("", (L + 12, -6), (L - 10, -6), arrowprops=dict(arrowstyle="->", color=BAD, lw=1.2))
    ax.annotate("", (-12, -6), (10, -6), arrowprops=dict(arrowstyle="->", color=BAD, lw=1.2))
    ax.text(L - 12, -8, "pull", color=BAD, ha="right", va="top", fontsize=8)
    ax.text(cx - 30 + 2, cy + 5, "V20@-30,5", family="monospace", fontsize=8, color=SEED)
    ax.text(cx + 25 - 12, W - 8 - 3.5, "H24@25,t-8", family="monospace", fontsize=8,
            color=SEED, va="top")
    ax.text(cx + 25 + 2, 9, "K15@25", family="monospace", fontsize=8, color=SEED)
    ax.set_title("Free-form coordinates on an L × W = 120 × 60 sheet (mm)", fontsize=9)
    save(fig, "coordinates")


def fig_pieces():
    L, W = 80, 40
    cases = [
        ("V20@0,0", "slit across the pull"),
        ("H30@0,0", "slit along the pull"),
        ("D20@-22,0;A20@0,0;R30:20@22,0", "45°, 135° and 30° slits"),
        ("C(-24,0;-12,8;0,-8;12,8;24,0)", "open polyline cut"),
        ("C(-20,-8;-20,8;20,8;20,-8)", "U-shaped polyline cut"),
        ("K16@-15;K16@15t", "cracks from b and t"),
        ("X(-20,-10;20,-10;20,10;-20,10)", "hole"),
        ("X(-20,-12;20,-12;20,12;-20,12);+(-6,-6;6,-6;6,6;-6,6)", "hole + solid island"),
        ("X(-24,-14;24,-14;24,14;-24,14);+E(12,6,20)", "hole + ellipse"),
        ("X(-24,-14;24,-14;24,14;-24,14);+B(-24,-14;-20,10;20,-10;24,14)w2",
         "hole + Bézier strip"),
        ("C(-20,-8;-20,8;20,8;20,-8)k1.5", "kerf k: cut -> slot"),
        ("K16@-15k2;K16@15tk2", "slotted cracks"),
    ]
    fig, axs = grid(3, 4, 3.0, 1.9)
    for ax, (code, sub) in zip(axs, cases):
        panel(ax, code, L, W, sub=sub)
    fig.suptitle("Building blocks (80 × 40 sheet; holes white, solids dashed)", fontsize=10, y=1.0)
    fig.tight_layout(h_pad=2.2)
    save(fig, "pieces")


SEED_CUT = "C(8,4;24,4;24,14)"          # an L-shaped cut with no symmetry of its own


def fig_symmetry():
    L, W = 80, 44
    s = SEED_CUT
    cases = [
        (s, "the seed: written once"),
        (f"({s}){{mx}}", "mirror x -> -x"),
        (f"({s}){{my}}", "mirror y -> -y"),
        (f"({s}){{i}}", "inversion (x, y) -> (-x, -y)"),
        (f"({s}){{mx}}{{my}}", "both mirrors: 4 copies"),
        (f"({s}){{mx}}{{i}}", "{mx} then {i}: same 4 copies"),
        (f"(C(-4,4;4,4;4,14)){{t20x3}}", "3 copies 20 mm apart"),
        (f"((C(-4,4;4,4;4,14)){{my}}){{t20x3}}", "groups nest: {my} then {t}"),
    ]
    seeds = [s] * 6 + ["C(-24,4;-16,4;-16,14)"] * 2
    fig, axs = grid(2, 4, 3.0, 2.0)
    for ax, (code, sub), sd in zip(axs, cases, seeds):
        panel(ax, code, L, W, seed=sd, sub=sub)
        ax.plot([L / 2, L / 2], [0, W], ls=":", color=MUTED, lw=0.6)
        ax.plot([0, L], [W / 2, W / 2], ls=":", color=MUTED, lw=0.6)
    fig.suptitle("Symmetry and repetition: orange is written, blue is generated", fontsize=10,
                 y=1.0)
    fig.tight_layout(h_pad=2.2)
    save(fig, "symmetry")


def lattice_points(code, L, W):
    d = ks.parse(code)
    d = d.base if isinstance(d, ks.Lattice) else d
    x0 = grammar.band(d, L)[0]
    pts = []
    for i in range(d.n):
        xi, yb = x0 + i * d.px, W / 2 + d.offset + i * float(d.shift) * d.py
        for j in range(int(np.floor(-yb / d.py)) - 1, int(np.ceil((W - yb) / d.py)) + 2):
            if -1e-6 <= yb + j * d.py <= W + 1e-6:
                pts.append((xi, yb + j * d.py))
    return d, np.array(pts)


def lattice_panel(ax, code, L, W, sub, cell=True):
    panel(ax, code, L, W, sub=sub)
    d, P = lattice_points(code, L, W)
    ax.plot(P[:, 0], P[:, 1], "o", ms=2.2, color=SEED, zorder=8)
    x0, x1 = grammar.band(d, L)
    ax.axvspan(x0, x1, ymin=0.06, ymax=0.94, color=SEED, alpha=0.07, lw=0)
    if cell:                                     # one lattice cell, from the first point
        first = P[np.isclose(P[:, 0], P[:, 0].min())]    # column 0; cell centred in y
        o = first[np.argmin(np.abs(first[:, 1] + (1 + float(d.shift)) * d.py / 2 - W / 2))]
        a, b = np.array([d.px, float(d.shift) * d.py]), np.array([0, d.py])
        quad = [o, o + a, o + a + b, o + b]
        ax.add_patch(Polygon(quad, closed=True, fill=False, ec=SEED, lw=1.1, ls="--", zorder=7))


def fig_lattice():
    L, W = 120, 60
    cases = [
        ("P10x20S0O0[V14@0,0]N9E2", "S0: columns in register"),
        ("P10x20S1/2O0[V14@0,0]N9E2", "S1/2: staggered columns"),
        ("P10x20S1/4O0[V14@0,0]N9E2", "S1/4: a sheared lattice"),
        ("P10x20S1/2O5[V14@0,0]N9E2", "O5: whole lattice moved up 5 mm"),
        ("P10x20S1/2O0[V14@0,0]N5E2", "N5: fewer columns, same centre"),
        ("P10x20S1/2O0[H8@0,0]N9E2", "H8: slit along the pull"),
        ("P10x20S1/2O0[V14@0,0;H6@1/2,1/2]N9E2", "two slits in the motif"),
        ("P12x20S0O0[D12@0,0;A12@1/2,1/2]N8E2", "diagonal slits"),
    ]
    fig, axs = grid(2, 4, 3.2, 2.0)
    for ax, (code, sub) in zip(axs, cases):
        lattice_panel(ax, code, L, W, sub)
    fig.suptitle("Lattice form: dots are lattice points, the dashed box is one cell "
                 "(px across, shift·py up), the band is shaded", fontsize=10, y=1.0)
    fig.tight_layout(h_pad=2.2)
    save(fig, "lattice")


def fig_trim():
    L, W = 120, 60
    cases = [
        ("P8x45S1/2O0[V40@0,0]N12E0", "E0: keep every clipped piece"),
        ("P8x45S1/2O0[V40@0,0]N12E10", "E10: drop pieces under 10 mm"),
        ("P8x45S1/2O0[V40@0,0]N12E28", "E28: the 27.5 mm cracks go too"),
    ]
    fig, axs = grid(1, 3, 3.3, 2.1)
    for ax, (code, sub) in zip(axs, cases):
        lattice_panel(ax, code, L, W, sub, cell=False)
    fig.suptitle("Edge trim E: slits clipped by a free edge become edge cracks; "
                 "E drops the short ones", fontsize=10, y=1.02)
    fig.tight_layout(h_pad=2.2)
    save(fig, "trim")


def fig_motif():
    L, W = 120, 60
    cases = [
        ("P10x20S1/2O10[H18@0,0;V18@0,1/2]N9E2", "rotating squares (slits)"),
        ("P10x20S1/2O10[(C(-8.5,0;8.5,0);C(0,1.5;0,18.5))k0.5]N9E2", "same, as slots (free pieces)"),
        ("P12x24S1/2O0[X(-3,-3;3,-3;3,3;-3,3)]N9E0", "a square hole as the motif"),
        ("P14x20S1/2O0[(C(-5,2;0,8;5,2)){my}]N8E2", "a mirrored chevron pair"),
    ]
    fig, axs = grid(1, 4, 3.2, 2.0)
    for ax, (code, sub) in zip(axs, cases):
        lattice_panel(ax, code, L, W, sub, cell=False)
    fig.suptitle("Motifs with free-form pieces (mm from each lattice point)", fontsize=10, y=1.02)
    fig.tight_layout(h_pad=2.2)
    save(fig, "motif")


def fig_build_pitch():
    L, W = 180, 60
    u = "C(2,16;47.5,16;47.5,-16;2,-16)"
    steps = [
        (u, u, "1. one U-shaped cut"),
        (f"({u}){{mx}}", u, "2. {mx}: its mirror image"),
        (f"({u}){{mx}};K18.75@0", "K18.75@0", "3. a crack from the bottom edge"),
        (f"({u}){{mx}};(K18.75@0){{my}}", "K18.75@0", "4. {my}: the crack from the top too"),
    ]
    fig, axs = grid(2, 2, 4.6, 2.0)
    for ax, (code, seed, sub) in zip(axs, steps):
        panel(ax, code, L, W, seed=seed, sub=sub)
    fig.suptitle("Building the pitch cell of the paper in four steps", fontsize=10, y=1.0)
    fig.tight_layout(h_pad=2.2)
    save(fig, "build_pitch")


PAPER = [
    ("single cut", "V80@0,0", 182, 100),
    ("two cuts", "(V40@5,0){mx}", 182, 100),
    ("alternating", "P5x40S1/2O0[V36@0,0]N20E4", 120, 60),
    ("rotating squares", "P8x16S1/2O0[H12@0,0;V12@0,1/2]N9E4", 72, 64),
    ("pitch cell", "(C(2,16;47.5,16;47.5,-16;2,-16)){mx};(K18.75@0){my}", 180, 60),
    ("pitch chain", "P125x200S0O0[(C(0,-30;0,-11.25)){my};(C(2,16;47.5,16;47.5,-16;2,-16)){mx}]N3E0",
     415, 60),
    ("roll cell", "((K23.25@49){my};(K7.5@34.4){my};C(41.6,0;5,0);C(41.6,b+2.9;41.6,t-3.5);"
                  "X(1.25,-2.9;5.5,0;1.25,2.9)){mx}k1", 150, 60),
    ("yaw cell", "(+B(-24,t-2;-24,2;-16,14;-2,4)w3;K30@-31.5t){i};+E(14,8,20);"
                 "X(-30,b+4;30,b+4;30,t-4;-30,t-4)", 150, 60),
]


def fig_paper():
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(3, 3)
    slots = [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1], gs[2, 1:], gs[1, 2], gs[2, 0]]
    for g, (name, code, L, W) in zip(slots, PAPER):
        ax = fig.add_subplot(g)
        panel(ax, code, L, W, title=f"{name}  ({L} × {W} mm)")
    fig.tight_layout(h_pad=2.2)
    save(fig, "paper")


def fig_identity():
    L, W = 120, 60
    codes = ["(V40@6,0){mx}", "V40@-6,0;V40@6,0", "C(6,-20;6,20);C(-6,20;-6,-20)",
             "(V40@-6,0){mx}"]
    fig, axs = grid(1, 4, 3.0, 2.0)
    for ax, code in zip(axs, codes):
        panel(ax, code, L, W, sub=f"key {ks.geometry_key(code, L, W)}")
    fig.suptitle("Four spellings, one design, one geometry key", fontsize=10, y=1.03)
    fig.tight_layout(h_pad=2.2)
    save(fig, "identity")


_NEAR = re.compile(r"near \(([-\d.]+), ([-\d.]+)\)")


def fig_check():
    L, W = 120, 60
    cases = [
        ("V40@0,0;V40@0.8,0", "ligament 0.8 mm < 1 mm"),
        ("V30@0,0;H30@0,0", "zero-width cuts cross"),
        ("(V30@0,0;H30@0,0)k1", "fixed: slots may meet"),
        ("H30@0,t-0.5", "0.5 mm sliver at the edge"),
        ("H50@-40,0", "reaches a clamped end"),
        ("K60@0", "cuts the sheet in two"),
        ("C(-15,-10;15,-10;15,10;-15,10;-15,-10)", "cuts a piece loose"),
        ("P5x40S1/2O0[V36@0,0]N20E4", "valid: check returns []"),
    ]
    fig, axs = grid(2, 4, 3.0, 2.1)
    for ax, (code, sub) in zip(axs, cases):
        errs = ks.check(code, L, W)
        panel(ax, code, L, W, sub=sub)
        for e in errs:
            m = _NEAR.search(e)
            if m:
                ax.plot(float(m[1]), float(m[2]), "o", ms=13, mfc="none", mec=BAD, mew=1.4, zorder=9)
        ax.text(0.86, 0.9, "✗" if errs else "✓", transform=ax.transAxes, ha="right", va="top",
                fontsize=13, color=BAD if errs else "#2b8a3e", fontweight="bold")
        print(f"  {code}: {errs}")
    fig.suptitle("ks.check(s, 120, 60): what makes a design impossible to cut or clamp",
                 fontsize=10, y=1.0)
    fig.tight_layout(h_pad=2.2)
    save(fig, "check")


if __name__ == "__main__":
    for f in (fig_coordinates, fig_pieces, fig_symmetry, fig_lattice, fig_trim, fig_motif,
              fig_build_pitch, fig_paper, fig_identity, fig_check):
        f()
