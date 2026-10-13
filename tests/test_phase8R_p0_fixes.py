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


@pytest.fixture(autouse=True)
def _restore_global_rngs():
    """test_target_reaches_every_crc_call trains models, which reseeds the process-global RNGs
    (hw.seed_model); hand the CPU generators (random, numpy, torch) back as they were, so no later test
    runs on a state left behind here. The CUDA generators, which torch.manual_seed also seeds when CUDA is
    available, are not restored: restoring them would initialise CUDA, and these tests draw nothing from
    them."""
    import random
    import torch
    saved = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    yield
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    torch.random.set_rng_state(saved[2])


# ---------------------------------------------------------------- P0-1: --alpha reaches the CRC
def test_run_seed_accepts_target():
    """run_seed MUST take `target`; without it reliability_over_states silently used its 0.10 default
    while --alpha was written to the plot/console/provenance (the P0-1 mislabelling)."""
    import phase8R_reliability as P8R
    assert "target" in inspect.signature(P8R.run_seed).parameters


def test_run_seed_threads_target_to_over_states():
    """Structural guard on the wiring from --alpha to the CRC. Since H-10 the evaluation lives in
    evaluate_cell, reached two ways -- run_seed (the single-cell path) and eval_job (main's per-cell
    stage, via run_two_stage) -- so both hops into it are pinned, and main must hand --alpha (and the
    scene-component unit) to the evaluation stage. test_target_reaches_every_crc_call below checks the
    remaining hops behaviourally."""
    import phase8R_reliability as P8R
    src = inspect.getsource(P8R.evaluate_cell)
    assert "reliability_over_states(" in src
    assert "target=target" in src, "evaluate_cell must pass target=target into reliability_over_states"
    for fn in (P8R.run_seed, P8R.eval_job):
        assert "target" in inspect.signature(fn).parameters, f"{fn.__name__} must take target"
        s = inspect.getsource(fn)
        assert "evaluate_cell(" in s and "target=target" in s, \
            f"{fn.__name__} must pass target=target into evaluate_cell"
    main_src = inspect.getsource(P8R.main)
    assert "run_two_stage(" in main_src and "target=args.alpha" in main_src, \
        "main must hand --alpha to the evaluation stage"
    assert "rid_te=cid_te" in main_src, "the evaluation stage must group by scene component (P0-2)"


def test_target_has_no_default_on_any_hop():
    """The P0-1 defect was a DEFAULT: a hop that forgot to forward target silently fell back to 0.10.
    With target required on every hop, forgetting it is a TypeError instead of a mislabelled run."""
    import phase8R_reliability as P8R
    for fn in (P8R.run_seed, P8R.eval_job, P8R.evaluate_cell, P8R.reliability_over_states):
        p = inspect.signature(fn).parameters["target"]
        assert p.default is inspect.Parameter.empty, f"{fn.__name__}: target must have no default"


def test_target_reaches_every_crc_call(monkeypatch):
    """Behavioural: whatever target enters run_seed or eval_job is the alpha / target_risk of EVERY
    conformal call underneath -- the hops reliability_over_states -> conformal_risk_control /
    conformal_at_risk included, which no source check sees."""
    import pickle
    import phase8R_reliability as P8R
    import phase8_cloudsen12 as P8
    from bandsim.grouping import group_center_wavelengths
    monkeypatch.setenv("BANDSIM_DEVICE", "cpu")
    seen = []
    real_crc, real_at = P8R.conformal_risk_control, P8R.conformal_at_risk

    def crc(*a, **k):
        seen.append(("crc", k.get("alpha")))
        return real_crc(*a, **k)

    def at(*a, **k):
        seen.append(("at", k.get("target_risk")))
        return real_at(*a, **k)
    monkeypatch.setattr(P8R, "conformal_risk_control", crc)
    monkeypatch.setattr(P8R, "conformal_at_risk", at)

    rng = np.random.default_rng(0)
    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    Xtr = rng.standard_normal((600, 13)).astype(np.float32)
    ytr = rng.integers(0, 4, 600)
    X1 = rng.standard_normal((2400, 13)).astype(np.float32)
    Xte = {"l1c": X1, "l2a": (X1 * 1.1 + 0.2).astype(np.float32)}
    yte = rng.integers(0, 4, 2400)
    unit = np.repeat(np.arange(24), 100)
    target = 0.37                                              # anything but the old 0.10 default
    P8R.run_seed((0, 0), Xtr, ytr, Xte, yte, unit, groups, cwl, 0.8, 1, target=target)
    blob = P8R.train_job(0, Xtr, ytr, groups, cwl, 0.8, 1)
    assert isinstance(pickle.loads(blob), dict)
    P8R.eval_job((1, 0), {0: blob}, Xte, yte, unit, groups, target=target)
    kinds = {k for k, _ in seen}
    assert kinds == {"crc", "at"}, f"expected both CRC entry points to be exercised, saw {kinds}"
    assert all(v == target for _, v in seen), \
        f"{sum(v != target for _, v in seen)} of {len(seen)} conformal calls did not receive target"


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
    if not os.path.exists(os.path.join(P8R.P8.DATA, "test", "metadata.csv")):
        pytest.skip("CloudSEN12 data not present")
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
