"""The API workflow figure for TOOLS.md: one design, three ways in, four steps.

    python scripts/plot_api_workflow.py        -> results/tools/api_workflow.png

The design is the alternating pattern P5x40S1/2O0[V36@0,0]N20E4 on a 120 x 60
sheet. The pattern and cut file are computed here; the simulation panels show
the stored run results/runs/alternating_long_1029_150259, which has the same
geometry key (the same cuts, written with the 'alternating' generator).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image                                           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kirigami import export as X                                # noqa: E402
from kirigami import kirismiles as ks                           # noqa: E402
from kirigami import spec as S                                  # noqa: E402

CODE, L, W = "P5x40S1/2O0[V36@0,0]N20E4", 120, 60
RUN = ROOT / "results" / "runs" / "alternating_long_1029_150259"
OUT = ROOT / "results" / "tools" / "api_workflow.png"
INK, MUTED, LINE, ACCENT, CUT, SHEET = "#1d1d1b", "#6b6a66", "#d6d5cf", "#2a78d6", "#e34948", "#d9d8d4"
CARD = "#f6f6f4"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})


def card(ax, title, lines, sub):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0,rounding_size=0.04",
                                transform=ax.transAxes, fc=CARD, ec=LINE, lw=1))
    ax.text(0.05, 0.86, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK)
    ax.text(0.05, 0.74, sub, transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.text(0.05, 0.62, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace",
            fontsize=7.6, color=INK, linespacing=1.55)


def sheet(ax, pat, cut_color=CUT, fill=SHEET):
    ax.add_patch(Rectangle((0, 0), L, W, fc=fill, ec="#52514e", lw=0.8))
    for c in pat["cuts"]:
        ax.plot(*zip(*c), color=cut_color, lw=0.9)
    ax.set(xlim=(-3, L + 3), ylim=(-3, W + 3), aspect="equal")
    ax.axis("off")


def step(ax, n, title, cli, http):
    """A step's header, in its own axes so that all four line up."""
    ax.axis("off")
    ax.text(0.02, 0.9, f"{n}. {title}", transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=INK, va="top")
    ax.text(0.02, 0.42, f"{cli}\n{http}", transform=ax.transAxes, fontsize=7.6,
            family="monospace", color=MUTED, va="top")


def main():
    summary = json.loads((RUN / "summary.json").read_text())
    pat = ks.pattern(CODE, L, W)
    run_pat = S.pattern(S.load(RUN / "spec.json"))
    assert ks.geometry_key(pat, L, W) == ks.geometry_key(run_pat, L, W), "the run is another design"

    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.05, 0.26, 0.26, 1.3], hspace=0.1, wspace=0.18,
                          left=0.03, right=0.97, top=0.92, bottom=0.06)
    for k, args in enumerate([
            (1, "Validate and preview", "cli: validate, preview", "http: POST /api/pattern"),
            (2, "Quick check (buckle)", "cli: buckle", "http: POST /api/buckle"),
            (3, "Simulate", "cli: simulate", "http: POST /api/simulate, GET /api/jobs/<id>"),
            (4, "Export for fabrication", "cli: export, stl", "http: POST /api/export")]):
        step(fig.add_subplot(gs[2, k]), *args)
    fig.suptitle(f"One design, three ways in:  {CODE}  on a {L} × {W} mm sheet",
                 fontsize=12, fontweight="bold", color=INK, x=0.03, ha="left")

    # ways in
    card(fig.add_subplot(gs[0, 0]), "Python", [
        "from kirigami import kirismiles as ks",
        f'code = "{CODE}"',
        "ks.check(code, 120, 60)",
        "  -> []          (valid)",
        "ks.geometry_key(code, 120, 60)",
        f"  -> '{ks.geometry_key(CODE, L, W)}'",
        "ks.pattern(code, L=120, W=60)",
        "  -> {'cuts': [50 cuts], ...}"], "kirigami/kirismiles.py, spec.py, api.py")
    card(fig.add_subplot(gs[0, 1]), "Command line", [
        "python scripts/kirigami_cli.py \\",
        "       validate spec.json",
        '-> {"name": "api_example",',
        '    "ok": true, "errors": [],',
        '    "counts": {"cuts": 50,',
        '               "holes": 0, ...}}',
        "",
        "one JSON line per command"], "for scripts and agents")
    card(fig.add_subplot(gs[0, 2]), "HTTP (web app API)", [
        "python webapp/server.py",
        "POST /api/pattern  {spec}",
        '-> {"ok": true, "errors": [],',
        '    "pattern": {"cuts": [...]},',
        '    "spec": {...}}',
        "",
        "the browser app uses these",
        "same endpoints"], "localhost:8000, JSON in and out")
    card(fig.add_subplot(gs[0, 3]), "The spec (JSON)", [
        "{",
        '  "name": "api_example",',
        '  "sheet": {"L": 120, "W": 60},',
        '  "pattern": {',
        '    "generator": "kirismiles",',
        '    "params": {"code":',
        f'      "{CODE}"}}}}',
        "}"], "material, mesh and load take defaults")

    # arrow band
    band = fig.add_subplot(gs[1, :])
    band.axis("off")
    band.add_patch(FancyArrowPatch((0.02, 0.5), (0.98, 0.5), transform=band.transAxes,
                                   arrowstyle="-|>", mutation_scale=18, color=ACCENT, lw=2))
    band.text(0.5, 0.82, "each way in runs the same four steps; files land in results/runs/<name>/",
              transform=band.transAxes, ha="center", color=ACCENT, fontsize=9.5)
    for x, t in ((0.125, "host only, instant"), (0.375, "solver, seconds"),
                 (0.625, "solver, minutes"), (0.875, "host only, instant")):
        band.text(x, 0.12, t, transform=band.transAxes, ha="center", color=MUTED, fontsize=8)

    # 1. validate / preview
    ax = fig.add_subplot(gs[3, 0])
    sheet(ax, pat)
    for x in (0, L):
        ax.plot([x, x], [0, W], color=ACCENT, lw=3)
    ax.text(L / 2, -9, "50 cuts · valid · key " + ks.geometry_key(CODE, L, W), ha="center",
            fontsize=8, color=MUTED)

    # 2. buckle
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    rows = [("stiffness before buckling", f"{summary['stiffness_N_per_mm']:.3f} N/mm"),
            ("buckles at Δ_c", f"{summary['Delta_c_mm']:.3f} mm"),
            ("Δ_c / L", f"{summary['Delta_c_mm'] / L:.2e}")]
    for k, (a, b) in enumerate(rows):
        y = 0.78 - 0.16 * k
        ax.text(0.02, y, a, transform=ax.transAxes, color=MUTED, fontsize=9)
        ax.text(0.98, y, b, transform=ax.transAxes, ha="right", fontsize=9.5, color=INK,
                fontweight="bold")
        ax.plot([0.02, 0.98], [y - 0.05, y - 0.05], transform=ax.transAxes, color=LINE, lw=0.8)
    ax.text(0.02, 0.2, "Linear analysis of the stretched flat\nsheet: where it first buckles out of\n"
            "plane, and how stiff it is before that.", transform=ax.transAxes, fontsize=8,
            color=MUTED, va="top")

    # 3. simulate: last GIF frame over the stress-strain curve
    sub = gs[3, 2].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.02)
    ax = fig.add_subplot(sub[0])
    gif = Image.open(RUN / "deformation.gif")
    gif.seek(gif.n_frames - 1)
    frame = gif.convert("RGB")
    w, h = frame.size
    ax.imshow(frame.crop((int(0.2 * w), int(0.3 * h), int(0.8 * w), int(0.75 * h))))
    ax.axis("off")
    ax = fig.add_subplot(sub[1])
    ss = Image.open(RUN / "stress_strain.png").convert("RGB")
    ax.imshow(ss.crop((0, 0, ss.size[0] // 2, ss.size[1])))
    ax.axis("off")
    ax.text(0.5, -0.06, f"post-buckling to Δ/L = {summary['strain_final']:.0%} "
            f"({summary['status']}); max lift {summary['uz_max_final_mm']:.1f} mm",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=MUTED)

    # 4. export
    ax = fig.add_subplot(gs[3, 3])
    paths = X.cut_paths(L, W, pat, exact=False)
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color="black", lw=0.7)
    ax.set(xlim=(-3, L + 3), ylim=(-3, W + 3), aspect="equal")
    ax.axis("off")
    ax.text(L / 2, -9, f"cut.svg + cut.dxf: {len(paths)} paths (outline + cuts)", ha="center",
            fontsize=8, color=MUTED)

    fig.text(0.03, 0.012, f"Steps 2 and 3 show the stored run {RUN.name}, which has the same "
             "geometry key: the same cuts, written with the 'alternating' generator.",
             fontsize=7.5, color=MUTED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
