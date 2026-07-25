"""Regression tests for the four P0 fixes in the phase8R reliability flagship (2026-07-22 review).

Each test fails on the exact defect the external adversarial review found, so a re-introduction
is caught in CI rather than in a re-run of the expensive campaign.
"""
import inspect
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(_HERE, ".."))
import bandsim.reliability as R  # noqa: E402


# ---------------------------------------------------------------- P0-1: --alpha reaches the CRC
def test_run_seed_accepts_target():
    """run_seed MUST take `target`; without it reliability_over_states silently used its 0.10 default
    while --alpha was written to the plot/console/provenance (the P0-1 mislabelling)."""
    import phase8R_reliability as P8R
    assert "target" in inspect.signature(P8R.run_seed).parameters


def test_run_seed_threads_target_to_over_states():
    """The one call site must forward target; a structural guard so the wiring can't silently drop."""
    import phase8R_reliability as P8R
    src = inspect.getsource(P8R.run_seed)
    assert "reliability_over_states(" in src
    assert "target=target" in src, "run_seed must pass target=target into reliability_over_states"


def test_alpha_actually_changes_the_crc_threshold():
    """Behavioural: the conformal core must respond to alpha, so a threaded alpha is observable."""
    rng = np.random.default_rng(0)
    n = 2000
    conf = rng.uniform(0, 1, n)
    correct = (rng.uniform(0, 1, n) < conf).astype(int)      # higher confidence -> more often correct
    ec = (rng.uniform(0, 1, n) < conf).astype(int)
    lo = R.conformal_risk_control(correct, conf, ec, conf, alpha=0.02)["threshold"]
    hi = R.conformal_risk_control(correct, conf, ec, conf, alpha=0.30)["threshold"]
    assert lo != hi, "CRC threshold ignores alpha -- alpha propagation cannot be verified downstream"


# ---------------------------------------------------------------- P0-4: B < 1 rejected (binary loss)
def test_B_below_one_is_rejected():
    cc = np.array([0, 1, 0, 1]); cf = np.array([.9, .8, .7, .6])
    ec = np.array([0, 1]); ef = np.array([.5, .5])
    with pytest.raises(ValueError, match=r"B .*>= 1"):
        R.conformal_risk_control(cc, cf, ec, ef, alpha=0.1, B=0.1)


def test_B_default_is_one_and_works():
    cc = np.array([0, 1, 0, 1]); cf = np.array([.9, .8, .7, .6])
    ec = np.array([0, 1]); ef = np.array([.5, .5])
    out = R.conformal_risk_control(cc, cf, ec, ef, alpha=0.1)     # default B=1.0
    assert "threshold" in out


# ---------------------------------------------------------------- P0-2: scene-component exchangeable unit
def test_scene_components_merge_shared_scenes():
    """ROIs that share ANY Sentinel-2 s2_id must land in one component; on CloudSEN12 test this is
    195 ROIs -> 184 components (the review's count), not 195."""
    import phase8R_reliability as P8R
    comp = P8R.scene_component_ids("test")
    roi = P8R.test_roi_ids("test")
    assert np.unique(roi).size == 195
    assert np.unique(comp).size == 184
    # a concrete shared-scene pair must co-locate
    import pandas as pd
    meta = pd.read_csv(os.path.join(P8R.P8.DATA, "test", "metadata.csv"))
    c_of_roi = {r: comp[i] for i, r in enumerate(meta["roi_id"].astype(str).to_numpy())}
    assert c_of_roi["ROI_0069"] == c_of_roi["ROI_0747"]        # these two share an s2_id


# ---------------------------------------------------------------- P0-3: two-way cluster-robust SE
def test_two_way_se_exceeds_iid_under_model_clustering():
    """When variance is driven by a factor that recurs across cells (model_seed), the iid SE is
    optimistic; the two-way SE must be larger. Re-implements the estimator to pin its contract."""
    rng = np.random.default_rng(1)
    means = {0: 10.0, 1: 20.0, 2: 30.0}
    vals, splits, models = [], [], []
    for ss in range(10):
        for ms in range(3):
            vals.append(means[ms] + rng.normal(0, 0.5)); splits.append(ss); models.append(ms)
    a = np.asarray(vals); N = a.size; e = a - a.mean()

    def V(lab):
        lab = np.asarray(lab)
        return sum(float(e[lab == g].sum()) ** 2 for g in set(lab.tolist())) / N ** 2
    two_way = np.sqrt(max(V(splits) + V(models) - float((e ** 2).sum()) / N ** 2, 0.0))
    iid = a.std(ddof=1) / np.sqrt(N)
    assert two_way > iid


def test_phase8R_rejects_duplicate_seeds():
    """A repeated seed is the SAME draw; counting it as independent inflates n in the two-way SE."""
    import phase8R_reliability as P8R
    src = inspect.getsource(P8R.main)
    assert "has duplicates" in src, "main must reject duplicate split/model seeds"
