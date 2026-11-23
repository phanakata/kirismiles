"""Check and draw the literature corpus (specs/corpus/literature.json).

For every entry with a string: parse it, run the validity checks, compute the
canonical string and geometry key, and, for entries that name a generator,
compare the cuts with that generator of kirigami/geometry.py (same geometry key). Writes
results/corpus/corpus.md (a table) and results/corpus/corpus.png (every
expressed design drawn from its string). Needs only numpy and matplotlib.

Usage: python scripts/corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kirigami import geometry as geo          # noqa: E402
from kirigami import kirismiles as ks         # noqa: E402

SRC = ROOT / "specs" / "corpus" / "literature.json"
OUT = ROOT / "results" / "corpus"
INK, PAPER, HOLE = "#0b0b0b", "#e9eef4", "#ffffff"


def evaluate(e: dict[str, Any]) -> dict[str, Any]:
    """Canonical form, key, problems and size of one corpus entry; with a generator, whether
    the string cuts the same geometry. KeyError naming a missing field (code, L, W)."""
    code, L, W = e["code"], e["L"], e["W"]
    if not (L > 0 and W > 0):
        raise ValueError(f"{e.get('id', code)}: the sheet needs L > 0 and W > 0")
    pat = ks.pattern(code, L, W)
    r: dict[str, Any] = dict(canonical=ks.canonical(code), key=ks.geometry_key(code, L, W),
             problems=ks.check(code, L, W, min_gap=e.get("min_gap", 1.0)), chars=len(code),
             pieces=len(pat["cuts"]) + len(pat["holes"]) + len(pat["solids"]), pattern=pat)
    if "generator" in e:
        name, args = e["generator"]
        ref = getattr(geo, name)(*args)
        r["matches_generator"] = ks.geometry_key(ref, L, W) == r["key"]
    return r


def draw(ax: Any, pat: dict[str, Any], L: float, W: float) -> None:
    ax.add_patch(Rectangle((0, 0), L, W, facecolor=PAPER, edgecolor="none"))
    for h in pat["holes"]:
        ax.add_patch(Polygon(h, closed=True, facecolor=HOLE, edgecolor=INK, lw=0.25))
    for s in pat["solids"]:
        ax.add_patch(Polygon(s, closed=True, facecolor=PAPER, edgecolor="none"))
    for cut in pat["cuts"]:
        c = np.asarray(cut)
        ax.plot(c[:, 0], c[:, 1], color=INK, lw=0.35, solid_capstyle="butt")
    ax.set(xlim=(0, L), ylim=(0, W), aspect="equal")
    ax.set_axis_off()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = json.loads(SRC.read_text(encoding="utf-8"))["entries"]
    rows: list[str] = []
    shown: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failed: list[tuple[Any, ...]] = []
    for e in entries:
        refs = ", ".join(e["refs"])
        if "code" not in e:
            rows.append(f"| {e['design']} | {refs} | – | – | – | {e['note']} |")
            continue
        r = evaluate(e)
        ok = not r["problems"] and r.get("matches_generator", True)
        if not ok:
            failed.append((e["id"], r["problems"][:2], r.get("matches_generator")))
        same = "✓" if r.get("matches_generator") else ""
        rows.append(f"| {e['design']} | {refs} | {same} | `{e['code']}` | {r['chars']} | "
                    f"{e.get('note', '')} |")
        shown.append((e, r))

    md = ["| Design | Papers | Same as generator | KiriSMILES | Characters | Note |",
          "|---|---|---|---|---|---|", *rows]
    (OUT / "corpus.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    n = len(shown)
    cols = 5
    nrow = -(-n // cols)
    # drawn at its printed size (7 in, the width of a two-column figure), so the text prints
    # at its true size: titles 6.5 pt, strings 5 pt
    import textwrap
    fig, axes = plt.subplots(nrow, cols, figsize=(7.0, 1.3 * nrow),
                             gridspec_kw=dict(hspace=0.78, wspace=0.12))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (e, r) in zip(axes.flat, shown):
        draw(ax, r["pattern"], e["L"], e["W"])
        ax.set_title(textwrap.fill(e["design"], 22), fontsize=6.5, pad=3, linespacing=1.1)
        ax.text(0.5, -0.05, e["code"] if len(e["code"]) <= 26 else e["code"][:25] + "…",
                transform=ax.transAxes, ha="center", va="top", fontsize=5, family="monospace")
    fig.savefig(OUT / "corpus.png", dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")

    print(f"{len(entries)} entries, {n} with strings, {len(entries) - n} not yet expressed")
    for e, r in shown:
        print(f"  {e['id']:24s} chars={r['chars']:4d} pieces={r['pieces']:4d} key={r['key']} "
              f"generator={r.get('matches_generator', '-')} problems={len(r['problems'])}")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print("wrote", OUT / "corpus.md", "and", OUT / "corpus.png")


if __name__ == "__main__":
    main()
