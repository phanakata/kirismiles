"""Mesh a configuration from run_config.CONFIGS and save it for a quick look.

Usage: python3 scripts/preview_mesh.py <name> [...]   (in the container)
Writes results/configs/mesh_<name>.npz (x, cells); plot it on the host with
    python scripts/preview_mesh.py --plot <name>
"""
import sys

import numpy as np


def mesh(name):
    from dolfinx import plot, fem
    from run_config import CONFIGS
    gfun, gkw = CONFIGS[name]["mesh"]
    msh = gfun(**gkw, **CONFIGS[name].get("mesh_kw", dict(h_far=4.0, h_cut=0.5)))
    cells, _, x = plot.vtk_mesh(fem.functionspace(msh, ("Lagrange", 1)))
    np.savez(f"results/configs/mesh_{name}.npz", x=x[:, :2], cells=cells.reshape(-1, 4)[:, 1:])
    print(name, "cells:", len(cells) // 4)


def show(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(f"results/configs/mesh_{name}.npz")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.triplot(d["x"][:, 0], d["x"][:, 1], d["cells"], lw=0.2, color="#2a78d6")
    ax.set(aspect="equal", title=f"{name}: {len(d['cells'])} triangles")
    fig.tight_layout()
    fig.savefig(f"results/configs/mesh_{name}.png", dpi=150)


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        for n in sys.argv[2:]:
            show(n)
    else:
        sys.path.insert(0, "scripts")
        for n in sys.argv[1:]:
            mesh(n)
