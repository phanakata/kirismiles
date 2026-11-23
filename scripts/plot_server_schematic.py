"""Schematic of the KiriSMILES platform: who uses it, what its server offers, and where the work goes.

    python scripts/plot_server_schematic.py
Writes results/engine/server.png (a figure of the paper).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "results" / "engine" / "server.png"
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FILL = {BLUE: "#e8f0fb", ORANGE: "#fdeee6", AQUA: "#e4f6ef", MUTED: "#f3f2ee"}


def box(ax, x, y, w, h, title, lines=(), color=MUTED, title_size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=FILL[color], ec=color, lw=1.1))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    for k, t in enumerate(lines):
        ax.text(x + 0.16, y + h - 0.68 - 0.36 * k, t, ha="left", va="top", fontsize=6.3, color=INK)


def arrow(ax, p, q, label="", color=LINE, dy=0.14, rad=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text((p[0] + q[0]) / 2 - 0.12, (p[1] + q[1]) / 2 + dy, label, ha=ha, va="bottom", fontsize=6.5,
                color=MUTED)


def main():
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.set(xlim=(0, 14.3), ylim=(-0.1, 7.9))
    ax.axis("off")

    # clients
    box(ax, 0.1, 4.2, 3.0, 2.5, "Researcher", ["web application", "command line", "Python"], BLUE)
    box(ax, 0.1, 0.5, 3.0, 2.9, "AI agent", ["any client of the", "Model Context Protocol,",
                                             "one or several at once"], ORANGE)

    # the platform: server, queue, engine and store inside one frame
    ax.add_patch(FancyBboxPatch((5.13, 0.2), 9.02, 6.85, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="none", ec=LINE, lw=1.0, ls=(0, (4, 3))))
    ax.text(9.55, 7.42, "KiriSMILES platform", ha="center", va="bottom", fontsize=9.5,
            fontweight="bold", color=INK)
    ax.text(9.55, 7.13, "notation, engine, job queue and results store", ha="center", va="bottom",
            fontsize=6.5, color=MUTED)

    # server: the entry point of the platform
    box(ax, 5.3, 0.5, 3.9, 6.2, "Server", [
        "instant", "   check a string", "   draw it", "   find stored results",
        "seconds", "   stiffness and threshold",
        "minutes", "   start a simulation", "   follow or cancel a job", "   show a result",
        "files", "   cut files for a laser cutter",
        "documents", "   notation and spec format"], MUTED)

    # back end
    box(ax, 10.9, 4.9, 3.0, 1.8, "Job queue", ["one solve at a time;", "jobs outlive sessions"], MUTED)
    box(ax, 10.9, 2.7, 3.0, 1.6, "Engine", ["plate and shell models;", "FEniCSx in a container"], AQUA)
    box(ax, 10.9, 0.5, 3.0, 1.6, "Results store", ["one record per run,", "found by geometry key"], MUTED)

    # arrows: requests go right (upper), answers come back (lower)
    arrow(ax, (3.1, 5.75), (5.3, 5.75), "string + context")
    arrow(ax, (5.3, 5.0), (3.1, 5.0), "problems, key,\nimages, results", dy=-0.62)
    arrow(ax, (3.1, 2.35), (5.3, 2.35), "tool call")
    arrow(ax, (5.3, 1.6), (3.1, 1.6), "result or job id", dy=-0.42)
    arrow(ax, (9.2, 5.8), (10.9, 5.8), "job")
    arrow(ax, (12.4, 4.9), (12.4, 4.3))
    arrow(ax, (12.4, 2.7), (12.4, 2.1))
    arrow(ax, (10.9, 1.3), (9.2, 1.3), "stored\nresult", dy=0.14)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
