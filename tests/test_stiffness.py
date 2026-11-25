"""The stiffness fit (kirigami/stiffness.py): before and after buckling, and its input checks."""
import numpy as np
import pytest

from kirigami.stiffness import fit_stiffness

L, DC, K0, K = 120.0, 0.1, 0.35, 0.01


def bilinear(D):
    """Force of a sheet with stiffness K0 up to DC and K after it (N, for D in mm)."""
    D = np.asarray(D, float)
    return np.where(D < DC, K0 * D, K0 * DC + K * (D - DC))


def test_fit_recovers_both_stiffnesses():
    D = np.array([0.05, 0.09, 0.2, 0.5, 1.2, 2.4, 3.6, 4.8, 6.0])
    r = fit_stiffness(D, bilinear(D), DC, L)
    assert r["k_before_N_per_mm"] == pytest.approx(K0) and r["n_before"] == 2
    assert r["k_after_N_per_mm"] == pytest.approx(K) and r["F0_N"] == pytest.approx(K0 * DC - K * DC)
    assert r["n_after"] == 5 and r["fit_from_pct"] == 1.0 and r["fit_to_pct"] == pytest.approx(5.0)


def test_fit_range_and_missing_regimes():
    D = np.array([0.2, 1.2, 2.4, 3.6, 4.8, 6.0])
    r = fit_stiffness(D, bilinear(D), DC, L, lo=2, hi=3.5)         # 2.4 and 3.6 mm only
    assert r["k_before_N_per_mm"] is None and r["n_before"] == 0
    assert r["n_after"] == 2 and r["k_after_N_per_mm"] == pytest.approx(K)
    r = fit_stiffness(D, bilinear(D), DC, L, lo=4.5)                # one step left
    assert r["k_after_N_per_mm"] is None and r["F0_N"] is None


@pytest.mark.parametrize("kw, match", [
    (dict(delta_c=0.0), "delta_c"), (dict(delta_c=float("nan")), "delta_c"),
    (dict(L=-1.0), "L must"), (dict(lo=-1.0), "fit range"), (dict(lo=3.0, hi=2.0), "fit range"),
    (dict(hi=float("inf")), "fit range"),
])
def test_bad_inputs_are_refused(kw, match):
    D = [0.05, 1.2, 6.0]
    args = dict(delta=D, force=bilinear(D), delta_c=DC, L=L) | kw
    with pytest.raises(ValueError, match=match):
        fit_stiffness(**args)


def test_mismatched_lists_are_refused():
    with pytest.raises(ValueError, match="equal-length"):
        fit_stiffness([0.1, 0.2], [1.0], DC, L)
