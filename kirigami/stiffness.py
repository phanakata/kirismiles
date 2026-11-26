"""Stiffness of a stretched sheet from its force history: before and after buckling.

A kirigami sheet pulled at its ends is stiff until its ligaments buckle out of plane at the
stretch Delta_c, then much softer. fit_stiffness fits one straight line to each regime of
the force F (N) against the stretch Delta (mm):

  before buckling   F = k0 * Delta, through the origin, over the steps with Delta < Delta_c
  after buckling    F = k * Delta + F0, over the steps whose strain 100 * Delta / L lies
                    between lo and hi percent
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any

import numpy as np


def fit_stiffness(delta: Sequence[float] | np.ndarray, force: Sequence[float] | np.ndarray,
                  delta_c: float, L: float, lo: float = 1.0, hi: float | None = None) -> dict[str, Any]:
    """Fit the stiffness before and after buckling (N/mm).

    delta, force: stretch (mm) and force (N) at each load step, same length.
    delta_c: buckling stretch (mm, > 0); L: sheet length (mm, > 0).
    lo, hi: strain range of the after-buckling fit, in percent (0 <= lo < hi); hi = None
    means up to the last step.
    Returns k_before_N_per_mm (None without a step before Delta_c), k_after_N_per_mm and
    F0_N (None with fewer than 2 steps in the range), the step counts and the range used,
    and before_spread: the largest relative difference between a pre-buckling step's
    secant F/Delta and k0 (0 for a straight line; None with fewer than 2 steps, when
    linearity cannot be judged).
    """
    D, F = np.asarray(delta, float), np.asarray(force, float)
    if D.ndim != 1 or D.shape != F.shape or len(D) == 0:
        raise ValueError(f"delta and force must be equal-length lists, not {D.shape} and {F.shape}")
    for name, v in (("delta_c", delta_c), ("L", L)):
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"{name} must be a positive number of mm, not {v}")
    strain = 100 * D / L
    hi_ = float(strain[-1]) if hi is None else hi
    if not (math.isfinite(lo) and lo >= 0 and math.isfinite(hi_) and hi_ > lo):
        raise ValueError(f"the fit range must satisfy 0 <= lo < hi (percent), not {lo} to {hi_}")
    pre = (D > 0) & (D < delta_c)
    k0 = float(np.sum(F[pre] * D[pre]) / np.sum(D[pre] ** 2)) if pre.any() else None
    spread = (float(np.max(np.abs(F[pre] / D[pre] / k0 - 1)))
              if k0 is not None and k0 != 0 and pre.sum() >= 2 else None)
    post = (strain >= lo) & (strain <= hi_)
    k, F0 = (float(v) for v in np.polyfit(D[post], F[post], 1)) if post.sum() >= 2 else (None, None)
    return dict(k_before_N_per_mm=k0, n_before=int(pre.sum()), before_spread=spread,
                k_after_N_per_mm=k, F0_N=F0, n_after=int(post.sum()), fit_from_pct=lo, fit_to_pct=hi_)


def from_history(path: str | os.PathLike[str], lo: float = 1.0, hi: float | None = None) -> dict[str, Any]:
    """fit_stiffness on a run's history.npz (Delta, F, Delta_c and L are read from it)."""
    with np.load(path) as d:
        return fit_stiffness(d["Delta"], d["F"], float(d["Delta_c"]), float(d["L"]), lo, hi)
