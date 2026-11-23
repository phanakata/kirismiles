"""Export a KiriSMILES design as a MuJoCo model (MJCF), ready to load or to drop into a robot scene.

The input is the design's finite element mesh (mesh.vtk, from `python scripts/kirigami_cli.py mesh <spec>` or the
export_mesh MCP tool); the output is an MJCF file with the sheet as a MuJoCo 2D flex shell, clamped at x = 0 and held
at x = L by a grip body that slides along x (joint "pull"). See kirigami/mjcf.py for units, the clamp, and what
MuJoCo's flex does and does not reproduce of a Kirchhoff plate.

    python scripts/export_mujoco.py --mesh results/runs/<run>/mesh.vtk --L 120 --nu 0.38 --out design.xml
    python -c "import mujoco; m = mujoco.MjModel.from_xml_path('design.xml'); print(m.nflexvert)"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kirigami import mjcf as K  # noqa: E402
from kirigami.argtypes import positive  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--mesh", required=True, help="mesh.vtk of the design (mm)")
    p.add_argument("--L", type=positive, required=True, help="sheet length along the pull (mm)")
    p.add_argument("--E", type=positive, default=3500.0, help="Young's modulus (MPa)")
    p.add_argument("--nu", type=float, default=0.38, help="Poisson ratio")
    p.add_argument("--h", type=positive, default=0.127, help="thickness (mm)")
    p.add_argument("--dt", type=positive, default=1e-6, help="MuJoCo time step (s)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not Path(a.mesh).is_file():
        p.error(f"no mesh file {a.mesh}")
    x, cells = K.read_vtk(a.mesh)
    xml, info = K.mjcf(x, cells, a.L, E=a.E, h=a.h, nu=a.nu, dt=a.dt)
    Path(a.out).write_text(xml)
    print(f"wrote {a.out}: {info['n_nodes']} nodes, {len(info['free'])} free, MuJoCo "
          f"{'>= 4' if K.mujoco_major() >= 4 else '3.x'} inputs young={info['young']:.6g} "
          f"poisson={info['poisson_in']:.6g} thickness={info['t_in']:.6g}")


if __name__ == "__main__":
    main()
