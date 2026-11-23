"""Step-by-step schematic of how a KiriSMILES string is built, for the paper.

    python scripts/plot_notation_buildup.py   -> results/kirismiles/buildup.png

Every panel is drawn by the parser from the string above it. Orange is the
piece that was written, blue the copies an operation or the lattice adds;
orange dots are lattice points.
"""

import plot_kirismiles as pk
from kirigami import kirismiles as ks

L, W = 120, 60
SEED = "V20@0,0"
STEPS = [
    # (string, seed drawn in orange, caption under the panel, lattice?)
    ("V20@0,0", "V20@0,0", "one slit at the centre", False),
    ("V20@-30,10", "V20@-30,10", "moved: position after @", False),
    ("(V20@-30,10){mx}", "V20@-30,10", "mirrored: {mx}", False),
    ("(V20@0,0){t10x5}", "V20@-20,0", "repeated: {t10x5}", False),
    ("((V20@0,12){my}){t10x5}", "V20@-20,12", "groups nest: {my}, then {t}", False),
    ("P10x30S0O0[V20@0,0]N9E2", None, "on a lattice: P...[motif]N9", True),
    ("P10x30S1/2O0[V20@0,0]N9E2", None, "staggered columns: S1/2", True),
    ("P8x16S1/2O2[H12@0,0;V12@0,1/2]N9E4", None, "two-slit motif: rotating squares", True),
]


def main():
    fig, axs = pk.grid(2, 4, 3.2, 2.05)
    for k, (ax, (code, seed, sub, lat)) in enumerate(zip(axs, STEPS)):
        assert not ks.check(code, L, W), code
        if lat:
            pk.lattice_panel(ax, code, L, W, sub, cell=True)
        else:
            pk.panel(ax, code, L, W, seed=seed, sub=sub)
        ax.text(-0.02, 1.13, f"({chr(97 + k)})", transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", va="bottom")
    fig.tight_layout(h_pad=2.4)
    pk.save(fig, "buildup")


if __name__ == "__main__":
    main()
