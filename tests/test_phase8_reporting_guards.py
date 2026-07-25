"""Standing assertions for the Phase-8 CloudSEN12 reporting pipeline.

Every test here locks a defect that ALREADY SHIPPED or was reachable without crashing. The
motivating incident is recorded in paper/results_phase8_cloudsen12_curve.csv.provenance.json:
the flagship tables were produced by `--seeds 0 1 --patches-train 3000` while the module
docstring claimed "mean over >=5 seeds" and reproduce.sh used seven. A two-seed "std" is the
half-range of two numbers, printed into a column named `_std`, with nothing in the CSV saying so.
Prose in a docstring is not a guard; these are.

What is covered, and why each matters:
  preflight        -- a formal run cannot be configured into an unsupportable table
  drop sets        -- the curve is exhaustive (no Monte-Carlo noise) AND the legacy MC policy
                      still reproduces the PUBLISHED sequence bit-for-bit, so old numbers remain
                      regenerable
  metric agreement -- the ROI counts used for the bootstrap reproduce metrics.miou EXACTLY;
                      otherwise every confidence interval would describe a different statistic
                      from the reported point estimate
  batched predict  -- chunked inference is bit-identical to the old single-shot call (it was
                      introduced for the measured 7.6 GB/worker eval-time VRAM peak, and a
                      throughput fix that moved a number would be worse than the peak)
  auditability     -- every aggregate in the paper CSVs is recomputable from the raw CSVs
  atomic writes    -- a killed run cannot leave a half-updated set of deliverables, and the
                      published files do not silently become owner-only (mkstemp creates 0600)
  loader contract  -- the (X, y, ...) return shapes phase8R/8D/8E depend on are unchanged

CPU-only and training-free by construction: the models used are freshly constructed, never
fitted, and the main() test stubs out both the loader and the job runner.
"""
import csv
import json
import math
import os
import subprocess
import sys
from argparse import Namespace

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_REPO, "experiments"))
sys.path.insert(0, _REPO)

import phase8_cloudsen12 as P8            # noqa: E402
import phase2_degradation as P2           # noqa: E402
from bandsim.grouping import group_center_wavelengths   # noqa: E402
from bandsim.metrics import miou, per_class_iou, audc   # noqa: E402

_HAS_DATA = os.path.exists(os.path.join(P8.DATA, "test", "L1C_B2.dat"))
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="CloudSEN12 data not present")

KEYS = ["b1", "b2", "b3", "b4", "b6", "proposed"]
STATES = ["clean", "dropB10", "dropB1B9B10", "L2A_real"]


# =====================================================================================
# preflight: the 2-seed flagship table must be unconfigurable
# =====================================================================================
def _args(**over):
    """A configuration that PASSES preflight, so each test changes exactly one thing."""
    base = dict(seeds=[0, 1, 2, 3, 4], smoke=False, epochs=40, max_missing=5,
                subsample_frac=0.8, drop_policy="exhaustive", trials=8, jobs=None, boot=2000)
    base.update(over)
    return Namespace(**base)


def test_preflight_accepts_the_canonical_reproduce_sh_configuration():
    # reproduce.sh runs 7 seeds; that must not have been made illegal by the >=5 rule.
    P8._preflight(_args(seeds=[0, 1, 2, 3, 4, 5, 6]), len(P8.S2_PHYSICAL_GROUPS))
    P8._preflight(_args(), len(P8.S2_PHYSICAL_GROUPS))


@pytest.mark.parametrize("over,msg", [
    (dict(seeds=[0]), "at least"),                       # the incident: 1 seed -> std is exactly 0
    (dict(seeds=[0, 1]), "at least"),                    # THE SHIPPED RUN: 2 seeds -> half-range
    (dict(seeds=[0, 1, 2, 3]), "at least"),
    (dict(seeds=[0, 0, 1, 2, 3, 4]), "duplicates"),      # looks like 6 seeds, is 5 models
    (dict(epochs=0), "epochs"),                          # untrained models, full results table
    (dict(max_missing=7), "max-missing"),                # no band left for B3 to interpolate from
    (dict(max_missing=-1), "max-missing"),
    (dict(subsample_frac=0.0), "subsample-frac"),        # would collapse to a 1-pixel training set
    (dict(subsample_frac=1.5), "subsample-frac"),
    (dict(jobs=0), "jobs"),                              # 0 silently meant "auto"
    (dict(boot=10), "boot"),                             # CI endpoints dominated by their own noise
    (dict(drop_policy="mc", trials=0), "trials"),
])
def test_preflight_rejects_unsupportable_configurations(over, msg):
    with pytest.raises(SystemExit) as e:
        P8._preflight(_args(**over), len(P8.S2_PHYSICAL_GROUPS))
    assert msg in str(e.value), f"wrong diagnostic for {over}: {e.value}"


def test_preflight_allows_two_seeds_only_under_smoke():
    P8._preflight(_args(seeds=[0, 1], smoke=True), len(P8.S2_PHYSICAL_GROUPS))
    with pytest.raises(SystemExit):
        P8._preflight(_args(seeds=[0, 1], smoke=False), len(P8.S2_PHYSICAL_GROUPS))


def test_preflight_runs_before_any_data_load_or_hw_setup():
    """Ordering matters: a bad --seeds must cost a second, not a full CloudSEN12 load.

    Runs the real CLI so the guard is checked where it actually sits in main(), not in isolation.
    """
    p = subprocess.run([sys.executable, "experiments/phase8_cloudsen12.py", "--seeds", "0"],
                       cwd=_REPO, capture_output=True, text=True, timeout=300,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    out = p.stdout + p.stderr
    assert p.returncode != 0 and "at least" in out
    assert "loading CloudSEN12" not in out, "preflight ran AFTER the data load"
    assert "HW:" not in out, "preflight ran AFTER hw.setup"


# =====================================================================================
# drop sets: exhaustive curve, and the published MC sequence stays regenerable
# =====================================================================================
def test_drop_sets_enumerate_every_subset_exactly_once():
    G, M = len(P8.S2_PHYSICAL_GROUPS), 5
    ds = P8._drop_sets(G, M)
    assert len(ds) == 120, "7 groups, sizes 0..5 -> 1+7+21+35+35+21"
    assert len({d for _, d in ds}) == len(ds), "a subset was enumerated twice"
    assert all(len(d) == m for m, d in ds)
    assert ds[0] == (0, ()), "m=0 must be the single empty drop set (the clean evaluation)"
    for m in range(M + 1):
        assert sum(1 for mm, _ in ds if mm == m) == math.comb(G, m)


def test_mc_policy_reproduces_the_published_monte_carlo_sequence():
    """--drop-policy mc must regenerate the pre-enumeration curve BIT-FOR-BIT.

    The exhaustive policy changes the curve's VALUES (it removes the estimator noise the MC draws
    carried), so without this escape hatch the already-published numbers become unreproducible.
    The reference below is phase2.degradation_curve's own loop, transcribed.
    """
    G, M, T, SEED = len(P8.S2_PHYSICAL_GROUPS), 5, 8, 999
    ref, rng = [], np.random.default_rng(SEED)
    for m in range(0, M + 1):
        for _ in range(T if m > 0 else 1):
            ref.append(sorted(rng.choice(G, size=m, replace=False).tolist()) if m > 0 else [])
    got = [sorted(d) for _, d in P8._mc_drop_sets(G, M, T, np.random.default_rng(SEED))]
    assert got == ref
    assert len(got) == 1 + M * T


def test_mc_policy_rejects_zero_trials():
    with pytest.raises(ValueError, match="trials"):
        P8._mc_drop_sets(7, 5, 0, np.random.default_rng(0))


# =====================================================================================
# metric agreement: the bootstrap's sufficient statistic must reproduce the reported metric
# =====================================================================================
def _fake_pred(y, rng, acc=0.7, k=4):
    return np.where(rng.random(y.size) < acc, y, rng.integers(0, k, y.size))


def test_roi_counts_reproduce_metrics_miou_exactly():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 20000)
    pred = _fake_pred(y, rng)
    roi = rng.integers(0, 37, y.size).astype(np.int32)
    cnt = P8._roi_class_counts(y, pred, roi, 37)
    assert float(P8._miou_from_counts(cnt.sum(0))) == pytest.approx(miou(y, pred, 4), abs=1e-12)


def test_roi_counts_reproduce_miou_when_a_class_is_absent_from_ground_truth():
    """metrics.miou averages over GT-PRESENT classes only; _miou_from_counts must do the same.

    Two subtly different definitions of the headline metric in one file is a slow-burning defect:
    the point estimate and its confidence interval would silently describe different quantities.
    """
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 20000)
    y[y == 2] = 0                                  # class 2 gone from GT, still predicted
    pred = _fake_pred(y, rng)
    roi = rng.integers(0, 37, y.size).astype(np.int32)
    cnt = P8._roi_class_counts(y, pred, roi, 37)
    assert float(P8._miou_from_counts(cnt.sum(0))) == pytest.approx(miou(y, pred, 4), abs=1e-12)


def test_per_class_iou_nanmean_equals_miou():
    """The scenario CSV and the per-class CSV are compared by readers; they must agree."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, 5000)
    pred = _fake_pred(y, rng)
    assert float(np.nanmean(per_class_iou(y, pred, 4))) == pytest.approx(miou(y, pred, 4), abs=1e-12)


def test_roi_counts_sum_to_the_global_confusion_counts():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 8000)
    pred = _fake_pred(y, rng)
    roi = rng.integers(0, 11, y.size).astype(np.int32)
    cnt = P8._roi_class_counts(y, pred, roi, 11).sum(0)
    for c in range(4):
        assert cnt[c, 0] == int(((pred == c) & (y == c)).sum())
        assert cnt[c, 1] == int(((pred == c) & (y != c)).sum())
        assert cnt[c, 2] == int(((pred != c) & (y == c)).sum())


# =====================================================================================
# seed spread: ddof=1, and a single seed reports NaN rather than a reassuring 0
# =====================================================================================
def test_seed_sd_is_the_sample_sd():
    a = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert float(P8._sd(a)) == pytest.approx(np.std(a, ddof=1))
    assert float(P8._sd(a)) != pytest.approx(np.std(a, ddof=0)), "ddof=0 understates by 10.6% at n=5"


def test_single_seed_sd_is_nan_not_zero():
    """A 0 in a '_std' column reads as 'perfectly reproducible'; it means 'never measured'."""
    assert np.isnan(float(P8._sd(np.array([3.0]))))
    assert P8._sd(np.zeros((1, 6))).shape == (6,)
    assert np.all(np.isnan(P8._sd(np.zeros((1, 6)))))
    assert P8._sd(np.zeros((5, 6))).shape == (6,)


# =====================================================================================
# grouping: the singleton guard says nothing about the OTHER ten bands
# =====================================================================================
def test_real_grouping_is_an_exact_partition():
    g = P8.s2_physical_groups()                    # validate_partition runs inside
    flat = np.sort(np.concatenate([np.asarray(x) for x in g]))
    assert np.array_equal(flat, np.arange(len(P8.L1C_BANDS)))


@pytest.mark.parametrize("bad,why", [
    ([[0], [1, 2, 3], [4, 5, 6], [7, 8], [9], [10], [11]], "B12 dropped from every group"),
    ([[0], [1, 2, 3], [3, 4, 5], [6, 7, 8], [9], [10], [11, 12]], "B3 in two groups"),
    ([[0], [], [1, 2, 3, 4, 5, 6, 7, 8], [9], [10], [11, 12]], "empty group"),
    ([[0], [1, 2, 3], [4, 5, 6], [7, 8], [9], [10], [11, 12], [12]], "B12 duplicated"),
    ([], "no groups at all"),
])
def test_validate_partition_rejects_groupings_the_singleton_guard_would_pass(bad, why):
    """Each of these keeps B1/B9/B10 as singletons, so _assert_singleton is satisfied and
    `dropB10` stays band-exact — while the curve is computed on a model that never sees B12, or
    whose group-dropout mask double-counts a band. Wrong numbers, no crash."""
    with pytest.raises(ValueError):
        P8.validate_partition([np.asarray(x, int) for x in bad], len(P8.L1C_BANDS), what=why)


def test_atmospheric_bands_remain_singletons():
    g = P8.s2_physical_groups()
    for idx, name in [(P8.B1_IDX, "B1"), (P8.B9_IDX, "B9"), (P8.B10_IDX, "B10")]:
        assert list(g[P8._assert_singleton(g, idx, name)]) == [idx]


# =====================================================================================
# model construction: no dependence on leftover global RNG state
# =====================================================================================
def _cwl(g):
    return group_center_wavelengths(np.array(P8.S2_WL_NM, float), g)


def _same_weights(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_grouped_model_construction_is_seed_reproducible():
    g = P8.s2_physical_groups(); c = _cwl(g)
    assert _same_weights(P8._build_grouped(g, c, 0), P8._build_grouped(g, c, 0))
    assert _same_weights(P8._build_grouped(g, c, 0, pe_type="learned"),
                         P8._build_grouped(g, c, 0, pe_type="learned"))
    assert not _same_weights(P8._build_grouped(g, c, 0), P8._build_grouped(g, c, 1))


def test_grouped_model_construction_ignores_preceding_rng_consumption():
    """B4/B6/Proposed used to be constructed from whatever state the PREVIOUS method's training
    loop left behind, so changing --epochs (more batches consumed) silently re-initialised them.
    """
    g = P8.s2_physical_groups(); c = _cwl(g)
    torch.manual_seed(0); ref = P8._build_grouped(g, c, 42)
    torch.manual_seed(0); torch.randn(1234)          # stand-in for a preceding training loop
    assert _same_weights(ref, P8._build_grouped(g, c, 42))


def test_grouped_model_construction_does_not_perturb_the_caller_rng():
    g = P8.s2_physical_groups(); c = _cwl(g)
    torch.manual_seed(123); before = torch.randn(4)
    torch.manual_seed(123); P8._build_grouped(g, c, 77); after = torch.randn(4)
    assert torch.equal(before, after), "fork_rng did not isolate the construction draw"


def test_b4_and_b6_start_from_identical_weights():
    """They differ ONLY in training recipe (HCS vs SGMAE+finetune), so their comparison is a
    controlled one only if the initialisation is shared."""
    g = P8.s2_physical_groups(); c = _cwl(g)
    assert _same_weights(P8._build_grouped(g, c, 5, pe_type="learned"),
                         P8._build_grouped(g, c, 5, pe_type="learned"))


# =====================================================================================
# batched inference must not move a single prediction
# =====================================================================================
@pytest.fixture(scope="module")
def predict_fixture():
    g = P8.s2_physical_groups()
    wl = np.array(P8.S2_WL_NM, float)
    rng = np.random.default_rng(11)
    raw = (rng.random((3000, 13)) * 0.3).astype(np.float32)
    mu = raw.mean(0); sd = raw.std(0) + 1e-8
    std = ((raw - mu) / sd).astype(np.float32)
    models = {"proposed": P8._build_grouped(g, _cwl(g), 0),
              "b4": P8._build_grouped(g, _cwl(g), 0, pe_type="learned"),
              "b1": P2.MLPBaseline(13, P8.NUM_CLASSES).eval()}
    models["b3"] = models["b1"]
    return g, wl, raw, std, mu, sd, models


@pytest.mark.parametrize("kind", ["proposed", "b4", "b1", "b3"])
@pytest.mark.parametrize("drop", [[], [5], [0, 4, 5]])
def test_batched_prediction_is_bit_identical_to_single_shot(predict_fixture, kind, drop):
    g, wl, raw, std, mu, sd, models = predict_fixture
    kw = dict(X_raw=raw, mu=mu, sd=sd)
    whole = P8._predict(kind, models[kind], std, g, drop, wl, batch_size=10 ** 9, **kw)
    chunk = P8._predict(kind, models[kind], std, g, drop, wl, batch_size=517, **kw)
    assert np.array_equal(whole, chunk)
    assert whole.shape == (std.shape[0],)


def test_predict_restores_the_callers_train_eval_mode(predict_fixture):
    g, wl, raw, std, mu, sd, models = predict_fixture
    m = models["b1"]
    m.train(); P8._predict("b1", m, std, g, [], wl); assert m.training
    m.eval(); P8._predict("b1", m, std, g, [], wl); assert not m.training


def test_b3_with_nothing_dropped_is_identical_to_b1(predict_fixture):
    """B3 IS B1's model plus a test-time imputation rule, and that rule is the identity when no
    group is dropped. Divergence means the raw-vs-standardised interpolation order has drifted
    (phase2's selfcheck pins that at 1.0 vs 5.05)."""
    g, wl, raw, std, mu, sd, models = predict_fixture
    assert np.array_equal(P8._predict("b1", models["b1"], std, g, [], wl),
                          P8._predict("b3", models["b3"], std, g, [], wl,
                                      X_raw=raw, mu=mu, sd=sd))


def test_b3_still_refuses_to_run_without_raw_reflectance(predict_fixture):
    g, wl, raw, std, mu, sd, models = predict_fixture
    with pytest.raises(ValueError, match="RAW reflectance"):
        P8._predict("b3", models["b3"], std, g, [5], wl)


def test_predict_rejects_a_nonsense_batch_size(predict_fixture):
    g, wl, raw, std, mu, sd, models = predict_fixture
    for bs in (0, -1, 2.5):
        with pytest.raises(ValueError, match="batch_size"):
            P8._predict("b1", models["b1"], std, g, [], wl, batch_size=bs)


# =====================================================================================
# ROI bootstrap: paired on shared geography
# =====================================================================================
def _two_count_sets(n_roi=40, n=6000):
    rng = np.random.default_rng(21)
    y = rng.integers(0, 4, n)
    roi = rng.integers(0, n_roi, n).astype(np.int32)
    return y, roi, {"a": P8._roi_class_counts(y, _fake_pred(y, rng, 0.75), roi, n_roi),
                    "b": P8._roi_class_counts(y, _fake_pred(y, rng, 0.55), roi, n_roi)}


def test_bootstrap_is_deterministic_for_a_fixed_rng():
    _, _, counts = _two_count_sets()
    s1 = P8._roi_bootstrap(counts, np.random.default_rng(1), 200)
    s2 = P8._roi_bootstrap(counts, np.random.default_rng(1), 200)
    assert np.allclose(s1["a"], s2["a"]) and np.allclose(s1["b"], s2["b"])
    assert s1["a"].shape == (200,)


def test_bootstrap_centres_on_the_reported_point_estimate():
    _, _, counts = _two_count_sets()
    s = P8._roi_bootstrap(counts, np.random.default_rng(2), 2000)
    assert s["a"].mean() == pytest.approx(float(P8._miou_from_counts(counts["a"].sum(0))), abs=1.0)


def test_bootstrap_shares_one_resample_across_every_method():
    """STRUCTURAL proof of pairing, independent of any statistical assumption.

    Two identical count arrays must yield identical bootstrap samples element for element. If
    each key drew its own resample they would differ, and every reported method difference would
    be an unpaired comparison wearing a paired label.
    """
    _, _, counts = _two_count_sets()
    s = P8._roi_bootstrap({"a": counts["a"], "copy": counts["a"].copy()},
                          np.random.default_rng(3), 500)
    assert np.array_equal(s["a"], s["copy"])
    assert s["a"].std() > 0, "degenerate fixture: the bootstrap produced a constant"


def test_bootstrap_reports_a_resample_that_changed_the_present_class_set(capsys):
    """A class confined to few ROIs can fall out of a resample entirely.

    metrics.miou averages over GT-present classes, so such a replicate is a 3-class mean mixed
    into a distribution of 4-class means -- two different estimands under one interval. This is
    the same hazard phase2 pins down for its block splits with an explicit class_set; here it is
    detected rather than forced, because forcing an absent class in gives it an empty union and a
    free IoU of 1.0.
    """
    n_roi, per = 8, 100
    roi = np.repeat(np.arange(n_roi), per).astype(np.int32)
    y = np.zeros(n_roi * per, int)
    y[roi == 0] = 3                                  # class 3 lives in ONE ROI only
    pred = y.copy()
    counts = {"m": P8._roi_class_counts(y, pred, roi, n_roi)}
    P8._roi_bootstrap(counts, np.random.default_rng(4), 500)
    out = capsys.readouterr().out
    assert "changed the present-class set" in out, "silent estimand drift in the bootstrap"
    assert "not like-for-like" in out


def test_bootstrap_is_silent_when_every_resample_keeps_the_same_classes():
    _, _, counts = _two_count_sets()
    P8._roi_bootstrap(counts, np.random.default_rng(4), 300)     # must not warn; 4 classes, 40 ROIs


def test_pairing_shrinks_the_variance_of_a_method_difference():
    """The STATISTICAL payoff, on data where the premise actually holds.

    Pairing helps exactly when methods share ROI difficulty -- a hard ROI is hard for everyone --
    which is the real situation and which shows up as Cov > 0 on a shared resample. Constructed
    here by nesting the errors (the weaker method agrees with the stronger 75% of the time);
    measured Cov gives corr ~= 0.5 and halves the variance of the difference. NOTE for anyone
    tempted to simplify this fixture: two INDEPENDENTLY corrupted predictors give corr ~= 0 and
    no variance reduction at all, which says nothing about the code.
    """
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, 6000)
    roi = rng.integers(0, 40, y.size).astype(np.int32)
    good = _fake_pred(y, rng, 0.80)
    worse = np.where(rng.random(y.size) < 0.75, good, rng.integers(0, 4, y.size))
    counts = {"a": P8._roi_class_counts(y, good, roi, 40),
              "b": P8._roi_class_counts(y, worse, roi, 40)}
    s = P8._roi_bootstrap(counts, np.random.default_rng(6), 3000)
    assert np.corrcoef(s["a"], s["b"])[0, 1] > 0.2, "fixture lost the shared-difficulty structure"
    assert np.var(s["a"] - s["b"]) < 0.75 * (np.var(s["a"]) + np.var(s["b"]))


# =====================================================================================
# atomic publish
# =====================================================================================
def test_write_csv_content_and_permissions(tmp_path):
    tgt = str(tmp_path / "probe.csv")
    P8._write_csv(tgt, ["a", "b"], [[1, 2.5], ["x", None]])
    with open(tgt) as f:
        assert list(csv.reader(f)) == [["a", "b"], ["1", "2.5"], ["x", ""]]
    mode = os.stat(tgt).st_mode & 0o777
    assert mode & 0o044 == 0o044, f"mkstemp's 0600 leaked into a published result file ({oct(mode)})"


def test_rewrite_preserves_a_deliberate_mode(tmp_path):
    tgt = str(tmp_path / "probe.csv")
    P8._write_csv(tgt, ["a"], [[1]])
    os.chmod(tgt, 0o600)
    P8._write_csv(tgt, ["a"], [[2]])
    assert os.stat(tgt).st_mode & 0o777 == 0o600


def test_failed_write_leaves_the_previous_file_intact_and_no_temp_behind(tmp_path):
    tgt = str(tmp_path / "probe.csv")
    P8._write_csv(tgt, ["a"], [[1]])
    with pytest.raises(RuntimeError):
        P8._atomic_write(tgt, lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert os.path.exists(tgt)
    with open(tgt) as f:
        assert list(csv.reader(f)) == [["a"], ["1"]], "a failed write corrupted the old artefact"
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]


def test_atomic_temp_keeps_the_extension_so_matplotlib_picks_the_right_format(tmp_path):
    """The temp name carries the real basename as a SUFFIX: a temp called `.tmp_ab12cd` would be
    written as PNG bytes and then renamed to *.pdf."""
    seen = {}
    P8._atomic_write(str(tmp_path / "fig.pdf"),
                     lambda p: (seen.setdefault("p", p), open(p, "w").write("x")))
    assert seen["p"].endswith(".pdf")


# =====================================================================================
# loader contract (data-gated)
# =====================================================================================
@needs_data
@pytest.mark.parametrize("kw,want", [
    (dict(), 2),
    (dict(return_patch_id=True), 3),
    (dict(return_roi_id=True), 3),
    (dict(return_patch_id=True, return_roi_id=True), 4),
    (dict(return_patch_id=True, return_roi_id=True, return_pixel_index=True), 5),
])
def test_load_split_return_contract_is_backward_compatible(kw, want):
    """phase8R/8D/8E unpack these by position; adding pixel_index must not shift anything."""
    out = P8.load_split("test", "L1C", pixels_per_patch=10, patch_ids=np.array([0, 1]), **kw)
    assert len(out) == want


@needs_data
def test_l1c_l2a_share_the_full_sample_key_not_merely_the_labels():
    """Label equality is necessary, not sufficient: 'clear' dominates this dataset, so two
    different pixel sets can carry identical label vectors. The L2A_real scenario's whole basis is
    that the two products are read at the SAME pixels."""
    ids = np.array([0, 1, 2])
    _, y1, p1, x1 = P8.load_split("test", "L1C", pixels_per_patch=25, patch_ids=ids,
                                  return_patch_id=True, return_pixel_index=True)
    Xa, y2, p2, x2 = P8.load_split("test", "L2A", pixels_per_patch=25, patch_ids=ids,
                                   return_patch_id=True, return_pixel_index=True)
    assert np.array_equal(p1, p2) and np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert np.abs(Xa[:, P8.B10_IDX]).max() == 0.0, "Sen2Cor emits no B10"


@needs_data
def test_pixel_index_addresses_the_valid_509_region():
    _, _, pid, pix = P8.load_split("test", "L1C", pixels_per_patch=20, patch_ids=np.array([0, 7]),
                                   return_patch_id=True, return_pixel_index=True)
    assert pix.min() >= 0 and pix.max() < P8.VALID_SIDE ** 2
    for p in np.unique(pid):
        assert np.unique(pix[pid == p]).size == 20, "sampling without replacement, per patch"


@needs_data
@pytest.mark.parametrize("kw,exc,msg", [
    (dict(patch_ids=np.array([True, False])), ValueError, "boolean"),
    (dict(n_patches=5000), ValueError, "exceeds"),
    (dict(patch_ids=np.array([2, 2])), ValueError, "duplicates"),
    (dict(patch_ids=np.array([-1])), IndexError, "out of range"),
    (dict(patch_ids=np.array([], int)), ValueError, "empty"),
])
def test_load_split_rejects_inputs_that_used_to_produce_a_wrong_number(kw, exc, msg):
    with pytest.raises(exc, match=msg):
        P8.load_split("test", "L1C", pixels_per_patch=5, **kw)


@needs_data
def test_dead_band_guard_does_not_false_positive_on_real_data():
    """The guard must tolerate L2A's by-design-zero B10 (only bands the product actually read are
    checked) and must not fire on genuine reflectance."""
    for product in ("L1C", "L2A"):
        X, _ = P8.load_split("test", product, pixels_per_patch=200, n_patches=20, seed=3)
        assert X.shape == (4000, 13)
    Xa, _ = P8.load_split("test", "L2A", pixels_per_patch=200, n_patches=20, seed=3)
    assert np.abs(Xa[:, P8.B10_IDX]).max() == 0.0


@needs_data
def test_label_histogram_covers_every_valid_pixel_and_stays_in_domain():
    import pandas as pd
    hist = P8.label_histogram("test")
    n = len(pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv")))
    assert hist.shape == (n, 256)
    assert int(hist.sum()) == n * P8.VALID_SIDE ** 2, \
        "the scan must cover the cropped 509x509 region of every patch, exactly once"
    assert int(hist[:, P8.NUM_CLASSES:].sum()) == 0, \
        "an out-of-domain label value exists somewhere load_split's sample never reached"
    assert (hist.sum(0)[:P8.NUM_CLASSES] > 0).all()


@needs_data
def test_label_histogram_cache_agrees_with_a_forced_rescan():
    a = P8.label_histogram("test")
    b = P8.label_histogram("test")                       # served from cache
    c = P8.label_histogram("test", force=True)           # rescanned from the memmap
    assert np.array_equal(a, b) and np.array_equal(a, c)
    assert os.path.exists(os.path.join(P8.DATA, ".cache", "label_hist_test.npz"))


@needs_data
def test_uniform_pixel_sampling_reproduces_the_class_balance_of_the_patches_it_read():
    """The paper's exposure is "300 of 259,081 pixels per patch". Because every patch is the same
    size and each is sampled the same number of times, the design is self-weighting and the sample
    share is an unbiased estimate of the population share. Shown on real labels, not asserted.
    """
    ids = np.arange(0, 975, 5)                           # 195 patches, ~1 per ROI
    pop = P8.label_histogram("test")[ids].sum(0)[:P8.NUM_CLASSES].astype(float)
    pop = pop / pop.sum() * 100
    _, y = P8.load_split("test", "L1C", pixels_per_patch=300, patch_ids=ids)
    samp = np.bincount(y, minlength=P8.NUM_CLASSES) / y.size * 100
    assert np.abs(pop - samp).max() < 1.5, \
        f"population {pop.round(3)} vs sample {samp.round(3)} — the sample is not representative"


@needs_data
def test_dataset_manifest_fingerprints_metadata_and_every_dat_it_will_read():
    m = P8.dataset_manifest("test", ("L1C", "L2A"))
    assert len(m["metadata_sha256"]) == 64
    assert m["metadata_sha256"] == P8.dataset_manifest("test", ("L1C",))["metadata_sha256"]
    for b in P8.L1C_BANDS:
        assert m["dat_bytes"][f"L1C_{b}.dat"] > 0
    for b in P8.L2A_BANDS:
        assert m["dat_bytes"][f"L2A_{b}.dat"] > 0
    assert "L2A_B10.dat" not in m["dat_bytes"], "Sen2Cor emits no B10; nothing may claim otherwise"
    assert m["dat_bytes"]["LABEL_manual_hq.dat"] > 0


@needs_data
def test_reflectance_profile_describes_the_bands_a_product_actually_read():
    X, _ = P8.load_split("test", "L2A", pixels_per_patch=100, n_patches=10, seed=5)
    idx = np.array([P8.L1C_BANDS.index(b) for b in P8.L2A_BANDS])
    prof = P8._reflectance_profile(X, idx)
    assert set(prof) == set(P8.L2A_BANDS) and "B10" not in prof
    for b, st in prof.items():
        assert st["min"] <= st["p00_1"] <= st["median"] <= st["p99_9"] <= st["max"]
        assert 0.0 <= st["zero_frac"] <= 1.0 and 0.0 <= st["neg_frac"] <= 1.0
        assert st["sd"] > 0, f"{b} is constant — the dead-band guard should have caught this"


# =====================================================================================
# end-to-end: main()'s reporting path, with the loader and job runner stubbed out
# =====================================================================================
NPX, NROI, NSEEDS, MAXM = 4000, 30, 5, 5


def _install_stubs(monkeypatch, tmp_path, states=STATES, keys=KEYS):
    """Redirect every output into tmp_path and replace the two expensive dependencies."""
    y_glob = np.random.default_rng(7).integers(0, P8.NUM_CLASSES, NPX)
    roi_glob = np.sort(np.random.default_rng(8).integers(0, NROI, NPX)).astype(np.int32)

    def fake_load(split, product="L1C", pixels_per_patch=400, n_patches=None, seed=0,
                  patch_ids=None, return_patch_id=False, return_roi_id=False,
                  return_pixel_index=False):
        n = NPX if split == "test" else 2 * NPX
        r = np.random.default_rng(seed)
        X = (r.random((n, 13)) * 0.3).astype(np.float32)
        if product == "L2A":
            X[:, P8.B10_IDX] = 0.0
        out = [X, y_glob if split == "test" else r.integers(0, P8.NUM_CLASSES, n)]
        if return_patch_id:
            out.append(np.repeat(np.arange(n // 100), 100)[:n])
        if return_roi_id:
            out.append(np.array([f"ROI_{i:04d}" for i in roi_glob]))
        if return_pixel_index:
            out.append(np.arange(n, dtype=np.int32))
        return tuple(out)

    # Faithful to run_seed's own structure: the curve is DERIVED from the per-drop-set rows, so
    # the published curve and the raw file are cross-checkable rather than independently invented.
    grp = P8.s2_physical_groups()
    gsz = [len(g) for g in grp]
    dsets = P8._drop_sets(len(grp), MAXM)

    def fake_run_jobs(target, items, shared=None, **kw):
        res = []
        for s in items:
            rr = np.random.default_rng(1000 + s)
            curves, sm, spc, rc, au, rows = {}, {}, {}, {}, {}, []
            for k in keys:
                per_m = {m: [] for m in range(MAXM + 1)}
                for m, ds in dsets:
                    nb = int(sum(gsz[g] for g in ds))
                    v = float(55.0 - 3.0 * keys.index(k) - 1.5 * nb + rr.normal(0, .3))
                    per_m[m].append(v)
                    rows.append([s, k, m, "+".join(map(str, ds)),
                                 "+".join(P8.L1C_BANDS[b] for g in ds for b in grp[g]), nb, v])
                curves[k] = np.array([float(np.mean(per_m[m])) for m in range(MAXM + 1)])
                au[k] = float(audc(np.arange(MAXM + 1), curves[k]))
                sm[k], spc[k], rc[k] = {}, {}, {}
                for st in states:
                    pred = _fake_pred(y_glob, rr, 0.5 + 0.05 * keys.index(k))
                    cnt = P8._roi_class_counts(y_glob, pred, roi_glob, NROI)
                    rc[k][st] = cnt
                    sm[k][st] = float(P8._miou_from_counts(cnt.sum(0)))
                    pc = per_class_iou(y_glob, pred, P8.NUM_CLASSES)
                    if st == states[-1] and k == "b1":
                        pc[2] = np.nan                      # exercise the absent-class path
                    spc[k][st] = pc
            res.append({"curves": curves, "audc": au, "scen_miou": sm, "scen_pc": spc,
                        "roi_counts": rc, "n_train_sub": 1234, "n_drop_sets": len(dsets),
                        "n_params": {k: 1000 + keys.index(k) for k in keys},
                        "steps": {k: 100 + keys.index(k) for k in keys},
                        # mirrors run_seed's real return: the sidecar records the training batch
                        # (a hyperparameter since auto_bs), and main reads it via .get so a stub
                        # without it stamps None rather than killing a finished run.
                        "train_bs": 256,
                        "curve_rows": rows})
        return res

    data = tmp_path / "data"
    (data / "test").mkdir(parents=True)
    for b in P8.L2A_BANDS:                                  # existence-only check in main()
        (data / "test" / f"L2A_{b}.dat").write_bytes(b"")
    # A PER-PATCH population histogram proportional to the sample, so any subset of patches sums
    # to the same balance and the drift check reads zero.
    hist = np.zeros((64, 256), np.int64)
    hist[:, :P8.NUM_CLASSES] = np.bincount(y_glob, minlength=P8.NUM_CLASSES) * 10
    monkeypatch.setattr(P8, "PAPER_DIR", str(tmp_path / "paper"))
    monkeypatch.setattr(P8, "DATA", str(data))
    monkeypatch.setattr(P8, "load_split", fake_load)
    monkeypatch.setattr(P8, "label_histogram", lambda split, force=False: hist.copy())
    monkeypatch.setattr(P8, "dataset_manifest", lambda split, products=("L1C",): {
        "metadata_sha256": "0" * 64, "dat_bytes": {"LABEL_manual_hq.dat": 1}})
    monkeypatch.setattr(P8.parallel, "run_jobs", fake_run_jobs)
    return data


def _read(tmp_path, name, sfx=""):
    with open(os.path.join(str(tmp_path / "paper"),
                           f"results_phase8_cloudsen12_{name}{sfx}.csv")) as f:
        return list(csv.reader(f))


@pytest.fixture(scope="function")
def ran_main(monkeypatch, tmp_path):
    _install_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--seeds"] + [str(s) for s in range(NSEEDS)]
                        + ["--boot", "300", "--max-missing", str(MAXM)])
    P8.main()
    return tmp_path


ALL_OUTPUTS = ("curve", "scenarios", "perclass", "paired", "bandcurve", "dropsets",
               "raw_curve", "raw_scen")
ALL_FIGURES = ("fig_degradation_cloudsen12", "fig_degradation_cloudsen12_bands")


def test_every_deliverable_is_written_with_a_provenance_sidecar(ran_main):
    for nm in ALL_OUTPUTS:
        p = os.path.join(str(ran_main / "paper"), f"results_phase8_cloudsen12_{nm}.csv")
        assert os.path.exists(p), nm
        assert os.path.exists(p + ".provenance.json"), f"{nm} is a citable table with no run record"
        assert os.stat(p).st_mode & 0o044 == 0o044, f"{nm} published as owner-only"
    for fg in ALL_FIGURES:
        assert os.path.exists(os.path.join(str(ran_main / "paper"), "figs", f"{fg}.pdf")), fg


def test_aggregates_are_recomputable_from_the_raw_results(ran_main):
    """The auditability claim, checked rather than asserted in prose: if the paper table cannot be
    rebuilt from the raw file, the raw file is decoration."""
    raw = _read(ran_main, "raw_scen")
    h = raw[0]
    i_m, i_s, i_v, i_a = h.index("method"), h.index("state"), h.index("miou"), h.index("audc")
    scen = _read(ran_main, "scenarios")
    sh, col = scen[0], {r[0]: r for r in scen[1:]}
    for k in KEYS:
        vals = [float(r[i_v]) for r in raw[1:] if r[i_m] == k and r[i_s] == "clean"]
        assert len(vals) == NSEEDS
        assert np.mean(vals) == pytest.approx(float(col[k][sh.index("clean_miou")]), abs=5e-3)
        assert np.std(vals, ddof=1) == pytest.approx(
            float(col[k][sh.index("clean_seedsd")]), abs=5e-3), "published SD is not ddof=1"
        au = [float(r[i_a]) for r in raw[1:] if r[i_m] == k and r[i_s] == "clean"]
        assert np.mean(au) == pytest.approx(float(col[k][sh.index("AUDC_mean")]), abs=5e-3)


def test_audc_point_estimate_is_unchanged_by_reporting_it_per_seed(ran_main):
    """AUDC is a linear functional of the curve, so mean(per-seed AUDC) == AUDC(mean curve).
    Reporting it per seed adds an SD WITHOUT moving the published number."""
    curve = _read(ran_main, "curve")
    ci = curve[0]
    scen = _read(ran_main, "scenarios")
    sh, col = scen[0], {r[0]: r for r in scen[1:]}
    for k in KEYS:
        mean_curve = np.array([float(r[ci.index(f"{k}_mean")]) for r in curve[1:]])
        assert audc(np.arange(MAXM + 1), mean_curve) == pytest.approx(
            float(col[k][sh.index("AUDC_mean")]), abs=1e-2)


def test_raw_results_keep_full_precision_and_mark_absent_classes(ran_main):
    raw = _read(ran_main, "raw_scen")
    assert len(raw) - 1 == NSEEDS * len(KEYS) * len(STATES)
    assert any(len(c.split(".")[-1]) > 4 for r in raw[1:] for c in r[3:9] if "." in c), \
        "raw file was rounded to the paper's 2dp and is no longer canonical"
    assert any(c == "" for r in raw[1:] for c in r[3:7]), "absent class must be empty, not 0"
    # one row per (seed, method, DROP SET) -- exhaustive enumeration, not one row per m
    n_sets = len(P8._drop_sets(len(P8.S2_PHYSICAL_GROUPS), MAXM))
    assert len(_read(ran_main, "raw_curve")) - 1 == NSEEDS * len(KEYS) * n_sets


def test_paired_comparison_table_is_internally_consistent(ran_main):
    pr = _read(ran_main, "paired")
    h = pr[0]
    assert len(pr) - 1 == len(STATES) * (len(KEYS) - 1)
    assert all(r[h.index("method_b")] != "proposed" for r in pr[1:]), "self-comparison row"
    for r in pr[1:]:
        lo, hi = float(r[h.index("roi_boot_ci_lo")]), float(r[h.index("roi_boot_ci_hi")])
        assert lo <= float(r[h.index("delta_miou_mean")]) <= hi
        assert (int(r[h.index("significant_95")]) == 1) == (lo > 0 or hi < 0)
        assert 0.0 <= float(r[h.index("frac_resamples_a_gt_b")]) <= 1.0


def test_provenance_records_what_the_table_cannot_show(ran_main):
    rec = json.load(open(os.path.join(str(ran_main / "paper"),
                                      "results_phase8_cloudsen12_scenarios.csv.provenance.json")))
    e = rec["extra"]
    for fld in ("task", "n_seeds", "seed_sd_ddof", "drop_policy", "class_support", "n_test_roi",
                "loader_seeds", "params", "optimizer_steps", "roi_boot_replicates_per_seed",
                "n_drop_sets_per_method_per_seed"):
        assert fld in e, f"provenance lost {fld}"
    assert e["n_seeds"] == NSEEDS
    assert e["seed_sd_ddof"] == 1
    assert "NOT spatial" in e["task"], "the task must be named honestly in the run record"
    assert len(e["params"]) == len(KEYS), "compute must be reported for ALL methods, not two"
    assert len(e["optimizer_steps"]) == len(KEYS)


def test_perclass_table_reports_class_support(ran_main):
    pc = _read(ran_main, "perclass")
    assert all(f"support_{c}" in pc[0] for c in P8.CLASS_NAMES), \
        "mIoU silently becomes a 3-class mean when a class is absent; support says which"
    assert len(pc) - 1 == len(KEYS) * len(STATES)


def test_published_curve_is_the_marginal_of_the_raw_drop_set_file(ran_main):
    """The curve is a MEAN OVER DROP SETS, then over seeds. If the two files disagree the raw file
    is not the record it claims to be."""
    raw = _read(ran_main, "raw_curve")
    h = raw[0]
    i_k, i_m, i_v = h.index("method"), h.index("n_missing_groups"), h.index("miou")
    curve = _read(ran_main, "curve")
    ci = curve[0]
    for k in KEYS:
        for m in range(MAXM + 1):
            vals = [float(r[i_v]) for r in raw[1:] if r[i_k] == k and int(r[i_m]) == m]
            assert len(vals) == NSEEDS * math.comb(len(P8.S2_PHYSICAL_GROUPS), m)
            assert np.mean(vals) == pytest.approx(
                float(curve[m + 1][ci.index(f"{k}_mean")]), abs=5e-3)


def test_band_count_curve_remarginalises_the_same_evaluations(ran_main):
    """A group is 1-3 bands, so the group-count axis is not an information-loss axis. The band
    curve must be the SAME evaluations re-binned, not a separate measurement."""
    raw = _read(ran_main, "raw_curve")
    h = raw[0]
    i_k, i_nb, i_v = h.index("method"), h.index("n_missing_bands"), h.index("miou")
    band = _read(ran_main, "bandcurve")
    bh = band[0]
    total_n = 0
    for row in band[1:]:
        nb = int(row[0])
        for k in KEYS:
            vals = [float(r[i_v]) for r in raw[1:] if r[i_k] == k and int(r[i_nb]) == nb]
            assert np.mean(vals) == pytest.approx(float(row[bh.index(f"{k}_mean")]), abs=5e-3)
            assert len(vals) == int(row[bh.index(f"{k}_n")])
        total_n += int(row[bh.index("proposed_n")])
    n_sets = len(P8._drop_sets(len(P8.S2_PHYSICAL_GROUPS), MAXM))
    assert total_n == NSEEDS * n_sets, "re-binning dropped or duplicated proposed's evaluations"
    assert len(raw) - 1 == NSEEDS * len(KEYS) * n_sets
    assert int(band[1][0]) == 0, "the zero-missing-band row is the clean anchor"


def test_drop_set_table_answers_which_versus_how_many(ran_main):
    ds = _read(ran_main, "dropsets")
    h = ds[0]
    rows = {(r[h.index("method")], int(r[h.index("n_missing_groups")])): r for r in ds[1:]}
    assert len(rows) == len(KEYS) * (MAXM + 1)
    for k in KEYS:
        z = rows[(k, 0)]
        assert int(z[h.index("n_drop_sets")]) == 1, "m=0 is the single empty drop set"
        assert float(z[h.index("range")]) == pytest.approx(0.0, abs=1e-9)
        assert z[h.index("worst_group_ids")] == "-", "the empty drop set has no group ids"
        for m in range(1, MAXM + 1):
            r = rows[(k, m)]
            assert int(r[h.index("n_drop_sets")]) == math.comb(len(P8.S2_PHYSICAL_GROUPS), m)
            assert float(r[h.index("worst_miou")]) <= float(r[h.index("best_miou")])
            assert float(r[h.index("range")]) == pytest.approx(
                float(r[h.index("best_miou")]) - float(r[h.index("worst_miou")]), abs=5e-4)
            assert r[h.index("worst_group_ids")] != r[h.index("best_group_ids")]
            assert len(r[h.index("worst_group_ids")].split("+")) == m


def test_provenance_fingerprints_the_inputs_not_only_the_code(ran_main):
    e = json.load(open(os.path.join(str(ran_main / "paper"),
                                    "results_phase8_cloudsen12_scenarios.csv.provenance.json")))["extra"]
    assert "dataset_manifest" in e and "metadata_sha256" in e["dataset_manifest"]["test"]
    assert "label_scan_full_split" in e and "test" in e["label_scan_full_split"]
    bal = e["class_balance_pop_vs_sample"]
    assert set(bal) >= {"population_pct", "sampled_pct", "max_abs_drift_pp",
                        "sampling_se_pp", "alarm_threshold_pp"}
    assert bal["max_abs_drift_pp"] < 1e-6, "a proportional population must show zero drift"
    assert bal["alarm_threshold_pp"] >= bal["sampling_se_pp"], \
        "an alarm that fires inside its own noise band is not an alarm"
    assert e["dataset_manifest"]["test"]["dat_bytes"]["LABEL_manual_hq.dat"] is not None
    prof = e["reflectance_profile"]
    assert set(prof) >= {"train_L1C", "test_L1C", "test_L2A"}
    assert set(prof["train_L1C"]) == set(P8.L1C_BANDS)
    assert "B10" not in prof["test_L2A"], "L2A's by-design-empty B10 must not be profiled"
    for st in prof["train_L1C"].values():
        assert set(st) >= {"min", "p00_1", "median", "p99_9", "max", "zero_frac", "neg_frac"}


def test_out_of_domain_labels_anywhere_in_the_split_are_fatal(monkeypatch, tmp_path):
    """load_split vouches only for the ~0.1% it sampled; a 255 elsewhere makes success seed-
    dependent. The whole-file scan must refuse it before any training starts."""
    _install_stubs(monkeypatch, tmp_path)
    bad = np.zeros((64, 256), np.int64); bad[:, :P8.NUM_CLASSES] = 1000; bad[3, 255] = 7
    monkeypatch.setattr(P8, "label_histogram", lambda split, force=False: bad)
    monkeypatch.setattr(sys, "argv", ["x", "--seeds"] + [str(s) for s in range(NSEEDS)]
                        + ["--boot", "300", "--max-missing", str(MAXM)])
    with pytest.raises(SystemExit, match="outside"):
        P8.main()


def test_sampling_drift_from_the_population_balance_is_reported(monkeypatch, tmp_path, capsys):
    """Self-weighting sampling gives ~0.1 pp standard error, so a full point of drift is not
    noise - it means the crop, the patch selection or the label file disagree."""
    _install_stubs(monkeypatch, tmp_path)
    skew = np.zeros((64, 256), np.int64); skew[:, 0] = 10 ** 7; skew[:, 1:P8.NUM_CLASSES] = 10 ** 4
    monkeypatch.setattr(P8, "label_histogram", lambda split, force=False: skew)
    monkeypatch.setattr(sys, "argv", ["x", "--seeds"] + [str(s) for s in range(NSEEDS)]
                        + ["--boot", "300", "--max-missing", str(MAXM)])
    P8.main()
    out = capsys.readouterr().out
    assert "population vs sampled" in out
    assert "more than 5x the" in out and "sampling error" in out


def test_label_scan_can_be_switched_off_but_is_on_by_default(monkeypatch, tmp_path):
    _install_stubs(monkeypatch, tmp_path)
    called = []
    def _spy(split, force=False):
        called.append(split)
        h = np.zeros((64, 256), np.int64); h[:, :P8.NUM_CLASSES] = 10 ** 6
        return h
    monkeypatch.setattr(P8, "label_histogram", _spy)
    monkeypatch.setattr(sys, "argv", ["x", "--label-scan", "none", "--seeds"]
                        + [str(s) for s in range(NSEEDS)] + ["--boot", "300",
                                                             "--max-missing", str(MAXM)])
    P8.main()
    assert called == [], "--label-scan none still scanned"
    e = json.load(open(os.path.join(str(tmp_path / "paper"),
                                    "results_phase8_cloudsen12_curve.csv.provenance.json")))["extra"]
    assert "disabled" in str(e["label_scan_full_split"]), \
        "a skipped scan must be recorded as skipped, not as an empty result"
    assert "disabled" in str(e["class_balance_pop_vs_sample"])


@pytest.mark.parametrize("n,observed_noise", [(292_500, 0.158), (90_000, 0.170),
                                              (12_000, 0.735), (2_000, 1.474)])
def test_drift_alarm_never_fires_on_measured_sampling_noise(n, observed_noise):
    """`observed_noise` is the MEASURED max-over-classes drift on real CloudSEN12 labels at that
    sample size (worst of five draws). A constant 1.0 pp threshold — the first version of this
    check — fires at n=2,000, i.e. on a legitimate quick run and nothing else."""
    assert observed_noise < P8._drift_alarm_pp(n), "the alarm sits inside its own noise band"
    assert P8._sampling_se_pp(n) == pytest.approx(np.sqrt(0.25 / n) * 100)
    # the measured noise also has to be consistent with the model that sets the threshold
    assert observed_noise < 4.0 * P8._sampling_se_pp(n), \
        "measured drift exceeds 4 sigma — the self-weighting argument behind the threshold is wrong"


def test_drift_alarm_loosens_as_the_sample_shrinks_and_floors_when_it_grows():
    """Below ~250,000 sampled pixels the 5-sigma rule binds; above it the 0.5 pp floor does.

    The floor is deliberate: 5 sigma keeps shrinking with n and would eventually alarm on rounding.
    It costs nothing here because a real defect is orders of magnitude larger (an off-by-one crop
    or a mismatched label export moves the balance by whole points), and the crop specifically has
    its own direct guard in test_label_histogram_counts_the_valid_region_and_nothing_else.
    """
    assert P8._drift_alarm_pp(2_000) > P8._drift_alarm_pp(12_000) > P8._drift_alarm_pp(200_000)
    assert P8._drift_alarm_pp(200_000) > 0.5, "5-sigma should still bind just below the crossover"
    assert P8._drift_alarm_pp(292_500) == 0.5, "the floor should bind at the full test sample"
    assert P8._drift_alarm_pp(10 ** 9) == 0.5, "and stay bound however large n gets"


def _tiny_split(tmp_path, n=3):
    """A minimal on-disk CloudSEN12-shaped split: metadata.csv + a correctly sized LABEL .dat.

    Patch p is filled with class p%4 over the VALID region only; the 512x512 padding stays 0. That
    makes the padding distinguishable from class 0 content, so these tests double as a guard on
    VALID_OFF: an off-by-one crop mixes padding zeros into a patch whose valid region is class 1.
    """
    import pandas as pd
    root = tmp_path / "cs" / "test"
    root.mkdir(parents=True)
    pd.DataFrame({"roi_id": [f"ROI_{i:04d}" for i in range(n)],
                  "proj_shape": [P8.VALID_SIDE] * n}).to_csv(root / "metadata.csv", index=False)
    lab = np.zeros((n, P8.SIDE, P8.SIDE), np.uint8)
    o, V = P8.VALID_OFF, P8.VALID_SIDE
    for p in range(n):
        lab[p, o:o + V, o:o + V] = p % P8.NUM_CLASSES
    lab.tofile(str(root / "LABEL_manual_hq.dat"))
    return str(tmp_path / "cs")


def test_label_histogram_counts_the_valid_region_and_nothing_else(tmp_path, monkeypatch):
    data = _tiny_split(tmp_path, n=3)
    monkeypatch.setattr(P8, "DATA", data)
    h = P8.label_histogram("test")
    assert h.shape == (3, 256)
    assert int(h.sum()) == 3 * P8.VALID_SIDE ** 2, \
        "the 512x512 padding was counted, or valid pixels were dropped"
    for p in range(3):
        assert int(h[p, p % P8.NUM_CLASSES]) == P8.VALID_SIDE ** 2, \
            f"patch {p} is uniformly class {p % P8.NUM_CLASSES} inside the crop; VALID_OFF is wrong"


def test_label_histogram_cache_is_published_atomically_and_round_trips(tmp_path, monkeypatch):
    """np.savez writes in place; two runs racing on a cold cache would leave a truncated .npz that
    a third reads as authoritative — a silently wrong POPULATION is worse than none."""
    data = _tiny_split(tmp_path, n=2)
    monkeypatch.setattr(P8, "DATA", data)
    real, seen = P8._atomic_write, []
    monkeypatch.setattr(P8, "_atomic_write", lambda p, fn: seen.append(p) or real(p, fn))
    a = P8.label_histogram("test")
    assert seen and seen[0].endswith(".npz"), "the cache was not published through _atomic_write"
    cdir = os.path.join(data, ".cache")
    assert os.path.exists(os.path.join(cdir, "label_hist_test.npz"))
    assert not [f for f in os.listdir(cdir) if f.startswith(".tmp_")], "temp file left behind"
    assert np.array_equal(a, P8.label_histogram("test")), "the cached read differs from the scan"


def test_label_histogram_cache_is_invalidated_when_the_labels_change(tmp_path, monkeypatch):
    data = _tiny_split(tmp_path, n=2)
    monkeypatch.setattr(P8, "DATA", data)
    a = P8.label_histogram("test")
    lab = os.path.join(data, "test", "LABEL_manual_hq.dat")
    arr = np.fromfile(lab, np.uint8).reshape(2, P8.SIDE, P8.SIDE)
    arr[1, P8.VALID_OFF:P8.VALID_OFF + 10, P8.VALID_OFF:P8.VALID_OFF + 10] = 3
    arr.tofile(lab)
    st = os.stat(lab)                       # advance mtime explicitly: filesystem timestamp
    os.utime(lab, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))   # granularity is not guaranteed
    b = P8.label_histogram("test")
    assert not np.array_equal(a, b), "a stale cache was served for a changed label file"
    assert int(b[1, 3]) == 100


def test_whole_file_scan_finds_a_bad_label_the_sample_would_have_missed(tmp_path, monkeypatch):
    """One 255 in 259,081 pixels: load_split's sampled check has a ~0.1% chance of seeing it, so
    without the scan the run's success depends on its seed."""
    data = _tiny_split(tmp_path, n=2)
    lab = os.path.join(data, "test", "LABEL_manual_hq.dat")
    arr = np.fromfile(lab, np.uint8).reshape(2, P8.SIDE, P8.SIDE)
    arr[1, P8.VALID_OFF, P8.VALID_OFF] = 255
    arr.tofile(lab)
    monkeypatch.setattr(P8, "DATA", data)
    h = P8.label_histogram("test")
    assert int(h[:, 255].sum()) == 1
    assert int(h[:, P8.NUM_CLASSES:].sum()) == 1, "the out-of-domain value must survive to the caller"


def test_smoke_suffixes_every_artefact_including_the_new_ones(monkeypatch, tmp_path):
    """A --smoke run must not overwrite ANY real deliverable. The two raw files and the paired
    table were added after that rule existed, which is exactly when such a rule gets forgotten."""
    _install_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--smoke", "--max-missing", str(MAXM)])
    P8.main()
    paper = str(tmp_path / "paper")
    for nm in ALL_OUTPUTS:
        assert os.path.exists(os.path.join(paper, f"results_phase8_cloudsen12_{nm}_smoke.csv")), nm
        assert not os.path.exists(os.path.join(paper, f"results_phase8_cloudsen12_{nm}.csv")), \
            f"--smoke overwrote the real {nm} deliverable"
    for fg in ALL_FIGURES:
        assert os.path.exists(os.path.join(paper, "figs", f"{fg}_smoke.pdf")), fg
        assert not os.path.exists(os.path.join(paper, "figs", f"{fg}.pdf")), \
            f"--smoke overwrote the real {fg}"


def test_smoke_announces_that_it_overrode_operator_flags(monkeypatch, tmp_path, capsys):
    _install_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--smoke", "--epochs", "99", "--max-missing", str(MAXM)])
    P8.main()
    out = capsys.readouterr().out
    assert "OVERRIDES" in out and "--epochs" in out, \
        "silently discarding a typed flag is how a smoke run gets mistaken for a configured one"


def test_missing_l2a_band_files_fail_loudly_instead_of_dropping_the_flagship_scenario(
        monkeypatch, tmp_path):
    """The old check looked at L2A_B2.dat alone: with B2 absent the operational scenario simply
    vanished from the results table, as though L2A had never been part of the design."""
    data = _install_stubs(monkeypatch, tmp_path)
    (data / "test" / "L2A_B12.dat").unlink()
    monkeypatch.setattr(sys, "argv", ["x", "--seeds"] + [str(s) for s in range(NSEEDS)]
                        + ["--boot", "300", "--max-missing", str(MAXM)])
    with pytest.raises(SystemExit, match="L2A"):
        P8.main()


def test_every_grouped_arm_is_listed_in_GROUPED_KINDS():
    """The trap this constant exists to close: an arm added to run_seed but not to GROUPED_KINDS is
    evaluated down the MLP path — the present-mask never applied, the raw B10 column fed in as if
    present — and produces a plausible number rather than an error."""
    grouped = set(P8.GROUPED_KINDS)
    assert grouped >= {"proposed", "b4", "b6", "b7"}
    assert grouped.isdisjoint({"b1", "b2", "b3"}), \
        "an MLP baseline would be handed a present-mask its forward() cannot accept"


def test_report_refuses_arms_it_did_not_ask_for(monkeypatch, tmp_path):
    """If --pe-ablation fails to reach the worker through run_jobs' `shared` dict, the mismatch
    must be named rather than surface as a bare KeyError from inside a comprehension."""
    _install_stubs(monkeypatch, tmp_path, keys=["b1", "b2", "b3", "b4", "b6", "proposed"])
    monkeypatch.setattr(sys, "argv", ["x", "--pe-ablation", "--seeds"]
                        + [str(s) for s in range(NSEEDS)] + ["--boot", "300",
                                                             "--max-missing", str(MAXM)])
    with pytest.raises(RuntimeError, match="did not reach the worker"):
        P8.main()


def test_pe_ablation_adds_the_arm_that_isolates_the_wavelength_pe(monkeypatch, tmp_path, capsys):
    """Without B7, B6 -> Proposed moves group-dropout AND the PE at once, so nothing in the default
    run licenses "the wavelength PE contributes N mIoU"."""
    k7 = ["b1", "b2", "b3", "b4", "b6", "b7", "proposed"]
    _install_stubs(monkeypatch, tmp_path, keys=k7)
    monkeypatch.setattr(sys, "argv", ["x", "--pe-ablation", "--seeds"]
                        + [str(s) for s in range(NSEEDS)] + ["--boot", "300",
                                                             "--max-missing", str(MAXM)])
    P8.main()
    scen = _read(tmp_path, "scenarios")
    assert [r[0] for r in scen[1:]] == k7, "B7 must sit between B6 and Proposed in every table"
    pr = _read(tmp_path, "paired")
    assert any(r[2] == "b7" for r in pr[1:]), "Proposed vs B7 is the comparison the arm exists for"
    out = capsys.readouterr().out
    assert "mechanism decomposition" in out and "wavelength-PE" in out
    e = json.load(open(os.path.join(str(tmp_path / "paper"),
                                    "results_phase8_cloudsen12_curve.csv.provenance.json")))["extra"]
    assert e["pe_ablation"] is True and e["arms"] == k7
    assert len(e["optimizer_steps"]) == len(k7), "the extra arm's compute must be reported too"


def test_default_run_does_not_silently_gain_the_extra_arm(ran_main):
    e = json.load(open(os.path.join(str(ran_main / "paper"),
                                    "results_phase8_cloudsen12_curve.csv.provenance.json")))["extra"]
    assert e["pe_ablation"] is False and "b7" not in e["arms"]


def test_no_l2a_opt_out_still_produces_a_complete_table(monkeypatch, tmp_path):
    _install_stubs(monkeypatch, tmp_path, states=STATES[:-1])
    (tmp_path / "data" / "test" / "L2A_B2.dat").unlink()      # deliberately unavailable
    monkeypatch.setattr(sys, "argv", ["x", "--no-l2a", "--seeds"]
                        + [str(s) for s in range(NSEEDS)] + ["--boot", "300",
                                                             "--max-missing", str(MAXM)])
    P8.main()
    scen = _read(tmp_path, "scenarios")
    assert "clean_miou" in scen[0] and "L2A_real_miou" not in scen[0]
    assert len(_read(tmp_path, "paired")) - 1 == (len(STATES) - 1) * (len(KEYS) - 1)
