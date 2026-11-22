"""GIF: a parallel-jaw gripper pulling a KiriSMILES sheet in MuJoCo, from the relaxed states of scripts/mujoco_check.py.

Left: the MuJoCo scene, rendered by MuJoCo: the design's KiriSMILES mesh as a 2D flex, its x = 0 edge clamped in one
jaw and its x = L edge in the other, the jaws hanging from a rail as on a two-finger gripper. Right: the sheet alone
from an oblique view, coloured by out-of-plane displacement (blue down, red up), so the pop-up of the flaps is visible.
Each frame is a quasi-static equilibrium from mujoco_check.py (dynamic relaxation of the same flex) posed in the scene:
the right jaw has moved by Delta and the sheet's nodes take their relaxed displacements; frames between two load
steps are interpolated linearly. True scale unless --zscale is given (then said in the caption).

    python scripts/mujoco_gif.py --mesh results/runs/<mesh run>/mesh.vtk --states results/mujoco_check/wf_states
        --L 120 --W 60 --out results/mujoco_check/wf_gripper.gif --title "P5x40S1/2O0[V36@0,0]N20E4"
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kirigami import mjcf as K  # noqa: E402


def look(pos, target):
    """MJCF camera attributes looking from pos at target, with world z up."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = (target - pos) / np.linalg.norm(target - pos)
    xa = np.cross(f, [0, 0, 1.0])
    xa /= np.linalg.norm(xa)
    ya = np.cross(xa, f)
    return f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" xyaxes="{" ".join(f"{v:.4f}" for v in (*xa, *ya))}"'


def scene_xml(x, cells, L, W, Dmax):
    """Gripper + sheet, in metres. Every sheet node is a body with three slides (posed, not simulated)."""
    s = 1e-3
    Lm, Wm, Dm = L * s, W * s, Dmax * s
    jaw, t, rail_z = 0.012, 0.004, 0.09                     # jaw depth along x, plate thickness, rail height (m)
    bodies = "".join(
        f'<body name="n{i}" pos="{p[0] * s:.6f} {p[1] * s:.6f} 0">'
        '<inertial pos="0 0 0" mass="1e-3" diaginertia="1e-9 1e-9 1e-9"/>'
        '<joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/></body>'
        for i, p in enumerate(x))

    def finger(name, x0, sgn, joint):
        # jaw plates above and below the sheet end, a finger up to the rail, a carriage on the rail
        cx = x0 + sgn * jaw / 2
        return f"""<body name="{name}" pos="{cx:.5f} {Wm / 2:.5f} 0">{joint}
        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>
        <geom type="box" size="{jaw / 2} {Wm / 2 + 0.006} {t / 2}" pos="0 0 {t / 2 + 0.0004}" material="metal"/>
        <geom type="box" size="{jaw / 2} {Wm / 2 + 0.006} {t / 2}" pos="0 0 {-t / 2 - 0.0004}" material="metal"/>
        <geom type="box" size="{jaw / 4} 0.005 {rail_z / 2}" pos="{sgn * jaw / 4} {Wm / 2 + 0.011} {rail_z / 2}" material="dark"/>
        <geom type="box" size="0.014 0.012 0.008" pos="{sgn * jaw / 4} {Wm / 2 + 0.011} {rail_z}" material="dark"/>
      </body>"""
    centre = np.array([Lm / 2 + Dm / 2, Wm / 2, 0.0])
    return f"""<mujoco model="kirigami gripper">
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.94 0.96" rgb2="0.86 0.88 0.91" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12"/>
    <material name="metal" rgba="0.55 0.58 0.62 1" specular="0.6" shininess="0.6"/>
    <material name="dark" rgba="0.18 0.2 0.24 1"/>
    <material name="sheet" rgba="0.93 0.55 0.2 1"/>
  </asset>
  <worldbody>
    <light pos="{centre[0]} {centre[1] - 0.2} 0.5" dir="0 0.3 -1" directional="true" castshadow="true"/>
    <light pos="{centre[0]} {centre[1] - 0.1} -0.3" dir="0 0.3 1" diffuse="0.35 0.35 0.35"/>
    <geom type="plane" size="1 1 0.01" pos="{centre[0]} {centre[1]} -0.06" material="grid"/>
    <geom type="box" size="{Lm / 2 + Dm / 2 + 0.05} 0.006 0.006" pos="{centre[0]} {Wm + 0.017} {rail_z + 0.014}" material="dark"/>
    <camera name="wide" {look(centre + [0.07, -0.19, 0.12], centre + [0, 0.005, 0.01])} fovy="45"/>
    {finger("jaw0", 0.0, -1, "")}
    {finger("jaw1", Lm, 1, '<joint name="pull" type="slide" axis="1 0 0"/>')}
    {bodies}
  </worldbody>
  <deformable>
    <flex name="sheet" dim="2" radius="0.0002" material="sheet" body="{' '.join(f'n{i}' for i in range(len(x)))}"
          vertex="{' '.join('0 0 0' for _ in x)}" element="{' '.join(map(str, cells.ravel()))}">
      <contact contype="0" conaffinity="0"/>
    </flex>
  </deformable>
</mujoco>"""


def sheet_panel(x, cells, U_all, size, lim, vmax, zscale, elev=32, azim=-58):
    """The sheet alone, deformed, coloured by out-of-plane displacement, from an oblique view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    P = np.column_stack([x + U_all[:, :2], zscale * U_all[:, 2]])
    wc = U_all[cells, 2].mean(1)
    col = plt.cm.coolwarm(np.clip(0.5 + 0.5 * wc / vmax, 0, 1))
    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([-0.1, -0.15, 1.2, 1.3], projection="3d", proj_type="ortho")
    ax.add_collection3d(Poly3DCollection(P[cells], facecolors=col, edgecolors="none"))
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(z0, z1))
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--states", required=True, help="folder of <step>_D<Delta>.npy from mujoco_check.py")
    p.add_argument("--L", type=float, default=120.0)
    p.add_argument("--W", type=float, default=60.0)
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--steps", type=int, default=0, help="use only the first this many states (e.g. the relaxed ones)")
    p.add_argument("--zscale", type=float, default=1.0, help="out-of-plane scale in the right panel (1 = true)")
    a = p.parse_args()
    import mujoco
    from PIL import Image, ImageDraw

    x, c = K.read_vtk(a.mesh)
    side = np.where(x[:, 0] < 1e-6, -1, np.where(x[:, 0] > a.L - 1e-6, 1, 0))
    free = np.flatnonzero(side == 0)
    files = sorted(Path(a.states).glob("*_D*.npy"))[:a.steps or None]
    Ds = np.array([0.0] + [float(f.stem.split("_D")[1]) for f in files])
    Us = [np.zeros((len(free), 3))] + [np.load(f).astype(float) for f in files]   # mm: (u_x, u_y, w) per free node

    m = mujoco.MjModel.from_xml_string(scene_xml(x, c, a.L, a.W, Ds[-1]))
    d = mujoco.MjData(m)
    slides = np.array([[m.jnt_qposadr[m.body_jntadr[m.body(f"n{i}").id] + k] for k in range(3)] for i in range(len(x))])
    pull = m.joint("pull").qposadr[0]
    r = mujoco.Renderer(m, height=360, width=480)

    U_full = np.zeros((len(x), 3))
    wmax = max(np.abs(u[:, 2]).max() for u in Us) or 1.0
    lim = ((-2, a.L + Ds[-1] + 2), (-2, a.W + 2), (-a.zscale * wmax - 1, a.zscale * wmax + 1))
    frames = []
    for D in np.linspace(0, Ds[-1], a.frames):
        k = min(np.searchsorted(Ds, D, side="right") - 1, len(Ds) - 2)
        t = (D - Ds[k]) / (Ds[k + 1] - Ds[k])
        U = (1 - t) * Us[k] + t * Us[k + 1]
        U_full[:] = 0
        U_full[side == 1, 0] = D
        U_full[free] = U
        d.qpos[:] = 0
        d.qpos[pull] = D * 1e-3
        d.qpos[slides] = U_full * 1e-3
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="wide")
        top = np.hstack([r.render().copy(), sheet_panel(x, c, U_full, (480, 360), lim, wmax, a.zscale)])
        img = Image.new("RGB", (top.shape[1], top.shape[0] + 24), "white")
        img.paste(Image.fromarray(top), (0, 0))
        ImageDraw.Draw(img).text((8, top.shape[0] + 6),
                                 f"{a.title}   pull {D:.2f} mm (strain {100 * D / a.L:.1f}%)"
                                 f"   max out-of-plane {np.abs(U[:, 2]).max():.2f} mm"
                                 + (f" (right: height x{a.zscale:g})" if a.zscale != 1 else ""), fill=(0, 0, 0))
        frames.append(img)
    frames += [frames[-1]] * 15
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)
    frames[-16].save(str(Path(a.out).with_suffix(".png")))
    print(f"wrote {a.out} ({len(frames)} frames) and {Path(a.out).with_suffix('.png')}")


if __name__ == "__main__":
    main()
