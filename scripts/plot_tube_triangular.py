"""Plots and GIFs for the triangular-cut tube runs (scripts/tube_triangular.py).

  results/tube/tri_stress_strain.png   sigma vs strain of every cut run, with Rafsanjani
                                       et al. (PNAS 2019) Fig. S17A digitised
  results/tube/tri_cut_vs_uncut.png    cut runs against the same tube without cuts
  results/tube/tri_benchmark.txt       numbers behind the comparison with S17A
  results/tube/<run>.gif               deformed tube, coloured by radial offset
  results/tube/<run>_frames.png        three frames side by side
  results/tube/tri_compare_rows2.gif   uncut | kerf | zero-width, same strain per frame

Points the solver reached only with artificial damping (the "damped" flag) carry
spurious viscous force; they are drawn hollow and left out of the benchmark numbers.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

OUT = "results/tube"
COL = {"uncut": "#555555", "k0.1_rows2": "#2a6fdb", "k0.1_rows4": "#7b3fbf", "k0.0_rows2": "#d9622b",
       "k0.1_rows4_per": "#c2185b", "k0.1_rows8": "#0b8a8a", "k0.1_rows4_per_imp": "#e08e0b",
       "k0.1_rows8_per_imp": "#6d4c41", "k0.1_rows8_imp": "#0b8a8a"}


def runs():
    for j in sorted(glob.glob(f"{OUT}/tri_n*.json")):
        yield j[:-5], json.load(open(j))


def key(d):
    a = d["args"]
    per = ("_per" if a.get("periodic") else "") + ("_imp" if a.get("imperfection") else "")
    return "uncut" + per if a.get("uncut") else f"k{a.get('kerf', 0.0)}_rows{a['rows']}{per}"


def label(d):
    a = d["args"]
    if a.get("uncut"):
        return f"no cuts, {a['rows']} rows long"
    k = a.get("kerf", 0.0)
    bc = ", periodic" if a.get("periodic") else ", clamped"
    if a.get("imperfection"):
        bc += f", imperfection {a['imperfection']:g}t"
    return f"{a['rows']} rows{bc}, " + ("zero-width cuts" if not k else f"kerf {k} mm")


def curve(d, damped=False):
    r = [q for q in d["rows"] if q.get("damped", False) == damped]
    e = np.array([q["eps"] for q in r])
    s = np.array([q["sigma_kPa"] for q in r])
    return (np.r_[0.0, e], np.r_[0.0, s]) if not damped else (e, s)


def reference():
    ref = json.load(open(f"{OUT}/rafsanjani_s17a_digitised.json"))["eps_sigma_kPa"]
    return {k: np.array(v).T for k, v in ref.items()}


def stress_strain():
    ref = reference()
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    ax.plot(*ref["experiment"], "-", color="#2e9e44", lw=1.6, label="experiment (S17A)")
    ax.plot(*ref["unit_cell_FEM"], "k--", lw=1.0, label="Rafsanjani FE: rolled periodic unit cell (S17A)")
    for stem, d in runs():
        if key(d).startswith("uncut"):
            continue
        c = COL.get(key(d), "C0")
        ax.plot(*curve(d), "-", color=c, lw=1.4, label="tube FE, " + label(d))
        e, s = curve(d, damped=True)
        if len(e):
            ax.plot(e, s, "o", mfc="none", mec=c, ms=3, mew=0.6,
                    label="  (damped steps, not physical)")
    ax.set(xlabel=r"applied strain $\bar\varepsilon$", ylabel=r"nominal stress $\sigma$ (kPa)",
           xlim=(0, 0.3), ylim=(0, 260))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.savefig(f"{OUT}/tri_stress_strain.png", dpi=200)
    print("wrote", f"{OUT}/tri_stress_strain.png")


def cut_vs_uncut():
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    for stem, d in runs():
        e, s = curve(d)
        ax.semilogy(e[1:], s[1:], "-", color=COL.get(key(d), "C0"), lw=1.4, label=label(d))
    e, s = reference()["experiment"]
    ax.semilogy(e, s, "-", color="#2e9e44", lw=1.0, alpha=0.7, label="experiment (S17A, cut)")
    ax.set(xlabel=r"applied strain $\bar\varepsilon$", ylabel=r"nominal stress $\sigma$ (kPa)",
           xlim=(0, 0.1))
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.savefig(f"{OUT}/tri_cut_vs_uncut.png", dpi=200)
    print("wrote", f"{OUT}/tri_cut_vs_uncut.png")


def slope(e, s, emax):
    m = (e > 0) & (e <= emax + 1e-9)
    return np.sum(e[m] * s[m]) / np.sum(e[m] ** 2)          # fit through the origin


def benchmark():
    ref = reference()
    lines = ["Triangular-cut tube vs Rafsanjani et al. PNAS 2019 SI Fig. S17A (digitised)",
             "E_eff = slope of sigma vs eps for eps <= 0.01 (MPa, fit through the origin)",
             "sigma at eps (kPa); undamped points only, linear interpolation", ""]
    hdr = f"{'case':38s} {'E_eff (MPa)':>11s} {'s(0.02)':>8s} {'s(0.05)':>8s} {'s(0.10)':>8s} {'last undamped eps':>18s}"
    lines.append(hdr)
    rows = [("experiment (S17A)", *ref["experiment"]), ("Rafsanjani rolled unit-cell FE (S17A)", *ref["unit_cell_FEM"])]
    uncut = None
    for stem, d in runs():
        e, s = curve(d)
        rows.append(("tube FE, " + label(d), e, s))
        if key(d) == "uncut":
            uncut = slope(e, s, 0.01)
    for name, e, s in rows:
        at = [np.interp(x, e, s) if x <= e.max() + 1e-9 else np.nan for x in (0.02, 0.05, 0.10)]
        lines.append(f"{name:38s} {slope(e, s, 0.01) / 1e3:11.2f} {at[0]:8.1f} {at[1]:8.1f} {at[2]:8.1f} {e.max():18.4f}")
    if uncut:
        ex = slope(*ref["experiment"], 0.01)
        lines += ["", f"uncut tube E_eff = {uncut / 1e3:.0f} MPa (sheet E = 4330 MPa); "
                  f"the cuts make the tube {uncut / ex:.0f}x softer than uncut (experiment)"]
    txt = "\n".join(lines)
    open(f"{OUT}/tri_benchmark.txt", "w").write(txt + "\n")
    print(txt)


def frames(stem, d):
    z = np.load(stem + ".npz")
    keys = ["rolled"] + sorted([k for k in z.files if k.startswith("eps")], key=lambda k: float(k[3:]))
    damped = {round(q["eps"], 4) for q in d["rows"] if q.get("damped")}
    return z["X"], z["cells"][:, :3], [(k, z[k]) for k in keys], damped


def frame_title(k, damped):
    if k == "rolled":
        return "rolled, before stretching"
    e = float(k[3:])
    return rf"$\bar\varepsilon$ = {e:.3f}" + ("  (damped)" if round(e, 4) in damped else "")


def draw(ax, X, tri, U, lim, vmax):
    P = X + U
    r = np.hypot(P[:, 0], P[:, 1])
    off = r - r.mean()                           # radial offset from the mean radius
    c = off[tri].mean(1)
    col = plt.cm.coolwarm(np.clip(0.5 + 0.5 * c / vmax, 0, 1))
    ax.add_collection3d(Poly3DCollection(P[tri], facecolors=col, edgecolors=col, linewidths=0.05))
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(z0, z1))
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=12, azim=-60)
    ax.set_axis_off()


def to_img(fig):
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return img


def limits(Xs):
    allP = np.vstack(Xs)
    return [(allP[:, i].min(), allP[:, i].max()) for i in range(3)]


def gif(stem, d):
    X, tri, fr, damped = frames(stem, d)
    lim = limits([X + U for _, U in fr])
    imgs = []
    for k, U in fr:
        fig = plt.figure(figsize=(5.0, 3.6))
        draw(fig.add_axes([0, 0, 1, 0.92], projection="3d"), X, tri, U, lim, 1.0)
        fig.suptitle(f"{label(d)}: {frame_title(k, damped)}", fontsize=9, y=0.97)
        imgs.append(to_img(fig))
    imgs[0].save(stem + ".gif", save_all=True, append_images=imgs[1:] + [imgs[-1]] * 3,
                 duration=350, loop=0)
    pick = [imgs[0], imgs[len(imgs) // 2], imgs[-1]]
    w, h = pick[0].size
    strip = Image.new("RGB", (3 * w, h), "white")
    for i, im in enumerate(pick):
        strip.paste(im, (i * w, 0))
    strip.save(stem + "_frames.png")
    print("wrote", stem + ".gif", f"({len(imgs)} frames)")


def compare(rows, out):
    """Side-by-side GIF of runs with the same tube length, frame by frame in strain."""
    sets = []
    for stem, d in runs():
        if d["args"]["rows"] == rows and not d["args"].get("periodic") and os.path.exists(stem + ".npz"):
            sets.append((d, *frames(stem, d)))
    order = ["uncut", "k0.1_rows%d" % rows, "k0.0_rows%d" % rows]
    sets.sort(key=lambda s: order.index(key(s[0])) if key(s[0]) in order else 9)
    if len(sets) < 2:
        return
    common = [k for k, _ in sets[0][3] if all(k in dict(s[3]) for s in sets[1:])]
    lim = limits([s[1] + U for s in sets for _, U in s[3]])
    imgs = []
    for k in common:
        fig = plt.figure(figsize=(3.4 * len(sets), 3.4))
        for i, (d, X, tri, fr, damped) in enumerate(sets):
            ax = fig.add_axes([i / len(sets), 0, 1 / len(sets), 0.86], projection="3d")
            draw(ax, X, tri, dict(fr)[k], lim, 1.0)
            ax.set_title(label(d) + ("\n(damped)" if k != "rolled" and round(float(k[3:]), 4) in damped else ""),
                         fontsize=8, y=0.98)
        fig.suptitle(frame_title(k, set()) + "   colour: radial offset, blue -1 mm, red +1 mm",
                     fontsize=9, y=0.98)
        imgs.append(to_img(fig))
    imgs[0].save(out, save_all=True, append_images=imgs[1:] + [imgs[-1]] * 3, duration=350, loop=0)
    imgs[-1].save(out.replace(".gif", "_final.png"))
    print("wrote", out, f"({len(imgs)} frames)")


if __name__ == "__main__":
    stress_strain()
    cut_vs_uncut()
    benchmark()
    for stem, d in runs():
        if os.path.exists(stem + ".npz"):
            gif(stem, d)
    compare(2, f"{OUT}/tri_compare_rows2.gif")
