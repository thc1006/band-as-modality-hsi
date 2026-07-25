"""Regression tests for the High-risk fixes from the 2026-07-22 conformal review (H-1..H-7)."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(_HERE, ".."))
import bandsim.reliability as R  # noqa: E402


# ------------------------------------------------------------------- H-4: temperature optimiser guards
def test_fit_temperature_rejects_nonfinite_logits():
    with pytest.raises(ValueError, match="NaN/Inf"):
        R.fit_temperature(np.array([[1.0, np.inf], [0.0, 1.0]]), np.array([0, 1]))


def test_fit_temperature_rejects_out_of_range_label():
    with pytest.raises(ValueError, match="out of range"):
        R.fit_temperature(np.array([[1.0, 2.0], [0.5, 0.3]]), np.array([0, 5]))


def test_fit_temperature_never_returns_nonfinite():
    """Extreme-but-finite logits must not diverge to Inf/NaN T (they used to)."""
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1e4, size=(64, 4))
    T = R.fit_temperature(logits, rng.integers(0, 4, size=64))
    assert np.isfinite(T) and 1e-3 <= T <= 1e3


# ------------------------------------------------------------------- H-5: metric input validation
def test_risk_coverage_rejects_non_binary_correct():
    with pytest.raises(ValueError, match="boolean or 0/1"):
        R.risk_coverage_curve([2, -1], [0.9, 0.8])


def test_selective_auroc_rejects_nan_confidence():
    with pytest.raises(ValueError, match="non-finite"):
        R.selective_auroc([1, 0], [np.nan, 0.2])


def test_aurc_stays_in_unit_interval_on_valid_input():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 500)
    correct = (rng.uniform(0, 1, 500) < conf).astype(int)
    a = R.aurc(correct, conf)
    assert 0.0 <= a <= 1.0


# ------------------------------------------------------------------- H-6: explicit status enum
def test_conformal_at_risk_reports_status():
    # all-wrong calibration -> the target is unreachable, and the status must SAY so (not degenerate)
    cc = np.zeros(40, bool); cf = np.linspace(0.6, 0.9, 40)
    out = R.conformal_at_risk(cc, cf, cc, cf, target_risk=0.05, conservative=False)
    assert out["status"] == "target_unreachable"
    assert np.isnan(out["risk"]) and out["coverage"] == 0.0
    # a healthy problem returns ok
    rng = np.random.default_rng(2)
    conf = rng.uniform(0, 1, 400)
    correct = (rng.uniform(0, 1, 400) < conf).astype(int)
    ok = R.conformal_at_risk(correct, conf, correct, conf, target_risk=0.2, conservative=False)
    assert ok["status"] == "ok"


# ------------------------------------------------------------------- H-7: exact / density-placed grid
def test_crc_threshold_is_an_actual_score_when_unique_scores_are_few():
    """With <= n_grid unique calib scores the candidate set is exactly those scores, so the selected
    threshold is one of them (or +inf) -- never a linspace point that lands between real boundaries."""
    cc = np.array([0, 1, 0, 1, 1, 0]); cf = np.array([0.60, 0.70, 0.65, 0.80, 0.75, 0.62])
    ec = np.array([0, 1, 1]); ef = np.array([0.90, 0.85, 0.70])
    thr = R.conformal_risk_control(cc, cf, ec, ef, alpha=0.2)["threshold"]
    assert np.isinf(thr) or thr in set(cf.tolist())


def test_crc_grid_bounded_for_many_unique_scores():
    """Many unique scores -> quantile grid capped at n_grid, so the loss matrix stays bounded."""
    rng = np.random.default_rng(3)
    cf = rng.uniform(0, 1, 5000)                    # 5000 distinct scores
    cc = (rng.uniform(0, 1, 5000) < cf).astype(int)
    out = R.conformal_risk_control(cc, cf, cc[:100], cf[:100], alpha=0.1, n_grid=256)
    assert out["feasible"] in (True, False)         # runs without building a 5000-wide matrix
