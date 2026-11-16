"""Figure of the single-cut mesh: full sheet plus a zoom on one crack tip.

Run scripts/preview_mesh.py single_cut in the container first, then (host):
    python scripts/plot_mesh_zoom.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"
d = np.load("results/configs/mesh_single_cut.npz")
x, cells = d["x"], d["cells"]
tip = (91.0, 10.0)                      # lower crack tip (cut from y=10 to y=90 at x=91)
zoom = 8.0

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
for ax in (a, b):
    ax.triplot(x[:, 0], x[:, 1], cells, lw=0.25, color=BLUE)
    ax.plot([91, 91], [10, 90], color=INK, lw=1.2)
    ax.set_aspect("equal")
    ax.tick_params(colors=MUTED, labelsize=8)
a.add_patch(Rectangle((tip[0] - zoom, tip[1] - zoom), 2 * zoom, 2 * zoom,
                      fill=False, ec="#e34948", lw=1.2))
a.set(title=f"Single cut, {len(cells)} triangles (fixed for the whole run)",
      xlabel="x (mm)", ylabel="y (mm)")
b.set(xlim=(tip[0] - zoom, tip[0] + zoom), ylim=(tip[1] - zoom, tip[1] + zoom),
      title="Zoom on the lower crack tip", xlabel="x (mm)")
b.annotate("crack tip\n(element size ≈ 0.3 mm)", xy=tip, xytext=(tip[0] + 2.5, tip[1] - 5.5),
           fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
fig.tight_layout()
fig.savefig("results/configs/mesh_single_cut.png", dpi=150)
print("wrote results/configs/mesh_single_cut.png")
