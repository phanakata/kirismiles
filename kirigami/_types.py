"""Type aliases shared by the host-side modules (notation, specs, API).

Lengths are in mm, angles in degrees, moduli in MPa, unless a name says otherwise.
"""
from __future__ import annotations

from typing import Any

Point = tuple[float, float]               # (x, y), mm; x along the pull, y across it
Polyline = list[Point]                    # a cut, or the outline of a hole or a solid


# The geometry of a sheet, as made by the generators of kirigami/geometry.py:
# {"cuts": zero-width cuts (open polylines), "holes": removed polygons,
#  "solids": polygons added back after the holes}
Pattern = dict[str, list[Polyline]]


Spec = dict[str, Any]                     # a JSON design spec (kirigami/spec.py)
Summary = dict[str, Any]                  # a JSON result summary printed by kirigami/api.py
