"""Fabrication export of a pattern: laser-cut paths (SVG, DXF) and STL.

Cut paths are the curves a laser cutter must follow: the outer outline, every
hole edge and every slit. For patterns without solids they are taken straight
from the pattern data (overlapping slot outlines only cut through material that
is removed anyway). Patterns with solids (e.g. a leaf inside a window) need a
boolean union, which is done with the same gmsh/OpenCASCADE geometry that the
simulation meshes, so what is cut is exactly what was simulated (requires gmsh,
i.e. run in the FEniCSx container).

Units are mm throughout. SVG uses y pointing down, so y is flipped.
"""
import numpy as np


def kerf_pattern(pattern, kerf):
    """Replace zero-width slits by slots of width `kerf` (needed for 3D printing,
    and a closer model of a real laser cut). Returns a new pattern."""
    from kirigami.geometry import _slot
    holes = list(pattern.get("holes", []))
    for c in pattern.get("cuts", []):
        holes += [_slot(p, q, kerf) for p, q in zip(c[:-1], c[1:])]
    return dict(cuts=[], holes=holes, solids=list(pattern.get("solids", [])))


def cut_paths(L, W, pattern, exact=None):
    """List of polylines (arrays of shape (n, 2)) to cut. exact=None picks the
    gmsh route only when the pattern has solids."""
    if exact is None:
        exact = bool(pattern.get("solids"))
    if exact:
        return _cut_paths_gmsh(L, W, pattern)
    paths = [np.array([(0, 0), (L, 0), (L, W), (0, W), (0, 0)], float)]
    for h in pattern.get("holes", []):
        h = np.asarray(h, float)
        paths.append(np.vstack([h, h[:1]]))
    paths += [np.asarray(c, float) for c in pattern.get("cuts", [])]
    return paths


def _cut_paths_gmsh(L, W, pattern, n_curve=60):
    import gmsh
    from kirigami.geometry import build_geometry
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("export")
    try:
        build_geometry(L, W, pattern.get("cuts", ()), pattern.get("holes", ()),
                       pattern.get("solids", ()))
        # Cut along the boundary of the union of all faces (fuse can leave
        # several faces sharing internal edges, which must NOT be cut) plus the
        # embedded slits (curves that bound no face).
        faces = gmsh.model.getEntities(2)
        outer = {abs(c) for _, c in gmsh.model.getBoundary(faces, combined=True, oriented=False)}
        any_bnd = {abs(c) for _, c in gmsh.model.getBoundary(faces, combined=False, oriented=False)}
        keep = [c for _, c in gmsh.model.getEntities(1) if c in outer or c not in any_bnd]
        paths = []
        for c in keep:
            lo, hi = gmsh.model.getParametrizationBounds(1, c)
            n = 2 if gmsh.model.getType(1, c) == "Line" else n_curve
            t = np.linspace(lo[0], hi[0], n)
            xyz = np.asarray(gmsh.model.getValue(1, c, t)).reshape(-1, 3)
            paths.append(xyz[:, :2])
        return paths
    finally:
        gmsh.finalize()


def write_svg(paths, L, W, filename, stroke="#ff0000", stroke_mm=0.05):
    """SVG in mm, hairline red strokes (the usual 'cut' convention)."""
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{L}mm" height="{W}mm" '
             f'viewBox="0 0 {L} {W}">',
             f'<g fill="none" stroke="{stroke}" stroke-width="{stroke_mm}">']
    for p in paths:
        pts = " ".join(f"{x:.4f},{W - y:.4f}" for x, y in p)
        lines.append(f'<polyline points="{pts}"/>')
    lines += ["</g>", "</svg>"]
    with open(filename, "w") as f:
        f.write("\n".join(lines))


def write_dxf(paths, filename):
    """Minimal ASCII DXF (R12, ENTITIES only) of LINE segments, in mm."""
    out = ["0", "SECTION", "2", "ENTITIES"]
    for p in paths:
        for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
            out += ["0", "LINE", "8", "CUT", "10", f"{x0:.4f}", "20", f"{y0:.4f}", "30", "0.0",
                    "11", f"{x1:.4f}", "21", f"{y1:.4f}", "31", "0.0"]
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(filename, "w") as f:
        f.write("\n".join(out) + "\n")


def write_stl(L, W, pattern, thickness, filename, kerf=0.4, h=1.0):
    """Extrude the sheet to `thickness` (mm) and write an STL for 3D printing.
    Slits are widened to slots of width `kerf` first. Requires gmsh."""
    import gmsh
    from kirigami.geometry import build_geometry
    pat = kerf_pattern(pattern, kerf)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("stl")
    try:
        build_geometry(L, W, (), pat["holes"], pat["solids"])
        gmsh.model.occ.extrude(gmsh.model.getEntities(2), 0, 0, thickness)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)
        gmsh.write(filename)
    finally:
        gmsh.finalize()
