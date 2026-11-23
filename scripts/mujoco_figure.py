"""Paper figure of the MuJoCo export: a robot arm holding the gripper and the sheet, a close-up of the sheet in the
gripper, the sheet coloured by out-of-plane displacement, and the force and deflection against FEniCS.

The arm is an illustration of where such a sheet would sit on a robot: it is drawn, not simulated. Its wrist holds the
gripper's rail; the jaws and the sheet are posed exactly as in scripts/mujoco_gif.py, from a relaxed state of
scripts/mujoco_check.py. The comparison plots are read from <tag>.png of mujoco_check.py --plot.

    python scripts/mujoco_figure.py --mesh results/runs/mjc_wfnu38_mesh_1_2/mesh.vtk
        --states results/mujoco_check/wfnu38_v4_states --plots results/mujoco_check/wfnu38_v4.png --step 6 --zscale 10
        --out results/mujoco_check/fig_mujoco.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from kirigami import mjcf as K  # noqa: E402
from kirigami.argtypes import count, positive  # noqa: E402
from mujoco_gif import look, scene_xml, sheet_panel  # noqa: E402

RENDER = (960, 720)                                   # the scene's offscreen buffer (scene_xml)
ARM = 'rgba="0.86 0.87 0.89 1"'        # the arm's links
JOINT = 'rgba="0.32 0.36 0.42 1"'      # its joint housings and base
SKY = '<texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.88 0.9 0.93" width="512" height="512"/>'


def arm_xml(L: float, W: float, Dmax: float) -> str:
    """A six-axis arm on the table behind the sheet, its wrist flange on top of the gripper rail, plus the camera
    "overview" that sees all of it. Metres; the rail of scene_xml is at y = W + 17 mm, z = 104 mm."""
    s = 1e-3
    cx, yr, zr = (L + Dmax) * s / 2, W * s + 0.017, 0.104 + 0.006      # rail centre top
    floor = -0.06                                                       # the table top of scene_xml
    base = np.array([cx, yr + 0.34, floor])
    sh = base + [0, 0, 0.16]                                            # shoulder
    el = np.array([cx, yr + 0.24, 0.42])                                # elbow
    wr = np.array([cx, yr, 0.27])                                       # wrist
    fl = np.array([cx, yr, zr + 0.012])                                 # flange

    def seg(a: np.ndarray, b: np.ndarray, r: float) -> str:
        return (f'<geom type="capsule" fromto="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} {b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" '
                f'size="{r}" {ARM}/>')

    def joint(p: np.ndarray, r: float, half: float) -> str:             # a joint housing, its axis along x
        return (f'<geom type="cylinder" fromto="{p[0] - half:.4f} {p[1]:.4f} {p[2]:.4f} {p[0] + half:.4f} {p[1]:.4f} '
                f'{p[2]:.4f}" size="{r}" {JOINT}/>')

    centre = np.array([cx, yr + 0.12, 0.17])
    return f"""
    <geom type="cylinder" fromto="{base[0]} {base[1]} {floor} {base[0]} {base[1]} {floor + 0.05}" size="0.07" {JOINT}/>
    {seg(base + [0, 0, 0.05], sh, 0.05)}
    {joint(sh, 0.058, 0.06)}
    {seg(sh, el, 0.042)}
    {joint(el, 0.05, 0.052)}
    {seg(el, wr, 0.034)}
    {joint(wr, 0.04, 0.045)}
    {seg(wr, fl, 0.028)}
    <geom type="cylinder" fromto="{fl[0]} {fl[1]} {fl[2]} {fl[0]} {fl[1]} {zr}" size="0.032" {JOINT}/>
    <geom type="box" size="0.03 0.012 0.004" pos="{cx} {yr} {zr - 0.004}" material="dark"/>
    <camera name="overview" {look(centre + [0.62, -0.95, 0.42], centre)} fovy="40"/>"""


def project(m: Any, d: Any, cam: str, pts: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Pixel (column, row) of world points pts (n, 3) in the image of camera `cam` (MjModel m, MjData d after
    mj_forward)."""
    i = m.camera(cam).id
    R = d.cam_xmat[i].reshape(3, 3)                     # columns: camera right, up, back
    v = (pts - d.cam_xpos[i]) @ R
    w, h = size
    f = (h / 2) / np.tan(np.radians(m.cam_fovy[i]) / 2)
    return np.column_stack([w / 2 + f * v[:, 0] / -v[:, 2], h / 2 - f * v[:, 1] / -v[:, 2]])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--states", required=True, help="folder of <step>_D<Delta>.npy from mujoco_check.py")
    p.add_argument("--plots", required=True, help="<tag>.png of mujoco_check.py --plot")
    p.add_argument("--step", type=count, default=6, help="the state shown: 1 is the first file in --states")
    p.add_argument("--L", type=positive, default=120.0)
    p.add_argument("--W", type=positive, default=60.0)
    p.add_argument("--zscale", type=positive, default=10.0, help="out-of-plane scale of the coloured sheet")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for f in (a.mesh, a.plots):
        if not Path(f).is_file():
            p.error(f"no file {f}")
    files = sorted(Path(a.states).glob("*_D*.npy")) if Path(a.states).is_dir() else []
    if len(files) < a.step:
        p.error(f"{a.states} has {len(files)} states, fewer than --step {a.step}")
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    x, c = K.read_vtk(a.mesh)
    side = np.where(x[:, 0] < 1e-6, -1, np.where(x[:, 0] > a.L - 1e-6, 1, 0))
    f = files[a.step - 1]
    D = float(f.stem.split("_D")[1])
    U = np.zeros((len(x), 3))
    U[side == 1, 0] = D
    U[side == 0] = np.load(f).astype(float)

    m = mujoco.MjModel.from_xml_string(scene_xml(x, c, a.L, a.W, D, extra=arm_xml(a.L, a.W, D), assets=SKY,
                                                  floor=4.0))
    d = mujoco.MjData(m)
    slides = np.array([[m.jnt_qposadr[m.body_jntadr[m.body(f"n{i}").id] + k] for k in range(3)] for i in range(len(x))])
    d.qpos[m.joint("pull").qposadr[0]] = D * 1e-3
    d.qpos[slides] = U * 1e-3
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=RENDER[1], width=RENDER[0])
    shots = {}
    for cam in ("overview", "wide"):
        r.update_scene(d, camera=cam)
        shots[cam] = Image.fromarray(r.render().copy())
    # the part of the overview shown in the close-up: the sheet, the jaws and the rail
    Lm, Wm = (a.L + D) * 1e-3, a.W * 1e-3
    box = np.array([[xx, yy, zz] for xx in (-0.014, Lm + 0.014) for yy in (-0.005, Wm + 0.025) for zz in (-0.006, 0.12)])
    uv = project(m, d, "overview", box, RENDER)
    (u0, v0), (u1, v1) = uv.min(0) - 6, uv.max(0) + 6

    wmax = float(np.abs(U[:, 2]).max()) or 1.0
    lim = ((-2, a.L + D + 2), (-2, a.W + 2), (-a.zscale * wmax - 1, a.zscale * wmax + 1))
    sheet = Image.fromarray(sheet_panel(x, c, U, (960, 720), lim, wmax, a.zscale))
    plots = Image.open(a.plots).convert("RGB")
    plots = plots.crop((0, 40, plots.width, plots.height))            # drop the blank title band

    # top row: overview | close-up | coloured sheet, all of one height, as wide as the plots together
    over = shots["overview"].crop((150, 60, 810, 700))                  # square-ish crop around the arm
    u0, u1, v0, v1 = u0 - 150, u1 - 150, v0 - 60, v1 - 60
    top = [over, shots["wide"], sheet]
    gap = 24
    H = int((plots.width - 2 * gap) / sum(im.width / im.height for im in top))
    sc = H / over.height
    top = [im.resize((int(im.width * H / im.height), H), Image.Resampling.LANCZOS) for im in top]
    W_out = plots.width
    out = Image.new("RGB", (W_out, H + plots.height), "white")
    xs = [0, top[0].width + gap, top[0].width + top[1].width + 2 * gap]
    for im, x0 in zip(top, xs):
        out.paste(im, (x0, 0))
    out.paste(plots, (0, H))

    dr = ImageDraw.Draw(out)
    col = (213, 94, 0)
    rect = (u0 * sc, v0 * sc, u1 * sc, v1 * sc)
    dr.rectangle(rect, outline=col, width=3)
    x1 = xs[1]
    dr.line((rect[2], rect[1], x1, 2), fill=col, width=2)                # zoom lines to the close-up's corners
    dr.line((rect[2], rect[3], x1, H - 3), fill=col, width=2)
    dr.rectangle((x1, 1, x1 + top[1].width - 1, H - 2), outline=col, width=3)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("arialbd.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    for t, xy in [("(a)", (xs[0] + 8, 6)), ("(b)", (xs[1] + 10, 8)), ("(c)", (xs[2] + 8, 6)),
                  ("(d)", (10, H + 4)), ("(e)", (W_out // 2 - 30, H + 4))]:
        dr.rectangle((xy[0] - 4, xy[1] - 2, xy[0] + 58, xy[1] + 40), fill="white")
        dr.text(xy, t, fill="black", font=font)
    out.save(a.out)
    print(f"wrote {a.out} {out.size}; state {f.name} (pull {D} mm)")


if __name__ == "__main__":
    main()
