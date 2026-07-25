"""Standing assertions for phase4R's matched-information baseline and its two breakdowns.

Each pins a property that a pooled average can hide. All three exist because a global number looked
fine while a specific one did not:

  * b2m -- B2's exact network and budget PLUS the group-presence mask. Without it, proposed-vs-b2
    confounds the architecture with merely being told what is missing, and the paper would attribute
    the whole gap to cross-band attention.
  * per-mask -- the pooled metrics average over the mask MIXTURE ("a random band loss"). An
    instrument failure is a PARTICULAR loss, so a survivable mean can contain a catastrophic mask.
  * per-class -- Indian Pines is severely imbalanced, so a flattering global coverage can be bought
    by abstaining almost entirely on the small classes, where the pixels are too few to move it.

Pure numpy plus two 2-epoch CPU trainings; no GPU, no dataset, seconds.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("BANDSIM_DEVICE", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import phase4R_reliability as P4R          # noqa: E402
import phase2_degradation as P2            # noqa: E402


def _toy(seed=0, C=40, G=5, N=400):
    rng = np.random.default_rng(seed)
    groups = [np.arange(i * (C // G), (i + 1) * (C // G)) for i in range(G)]
    X = rng.normal(0, 1, (N, C)).astype(np.float32)
    y = rng.integers(0, P2.NUM_CLASSES, N)
    return X, y, groups


# ------------------------------------------------------------------ matched-information baseline
def test_b2m_is_exactly_b2_plus_the_presence_mask_inputs():
    """The ablation is only clean if the two differ by the mask and NOTHING else. One extra input
    unit per group, same hidden width, same depth -- any other delta reintroduces the confound."""
    X, y, groups = _toy()
    b2 = P2.train_mlp(X, y, groups, seed=0, group_dropout=True, epochs=1)
    b2m = P4R.train_mlp_maskaware(X, y, groups, seed=0, epochs=1)
    n2 = sum(p.numel() for p in b2.parameters())
    n2m = sum(p.numel() for p in b2m.parameters())
    assert n2m - n2 == len(groups) * 256, (
        f"b2m should add exactly G*hidden weights, added {n2m - n2}")


def test_b2m_sees_the_identical_dropout_stream_as_b2():
    """Same seed offset, same generator, same call. If these streams ever diverge the comparison
    carries a different corruption schedule on top of the mask it is supposed to isolate -- the
    exact defect train_mlp's own two-stream comment records for B1-vs-B2."""
    G = 5
    a = P2._vec_group_subset(np.random.default_rng(7 + 1009), 9, G, 0, 4, leave_one=True)
    b = P2._vec_group_subset(np.random.default_rng(7 + 1009), 9, G, 0, 4, leave_one=True)
    assert np.array_equal(a, b)
    src = open(P4R.__file__, encoding="utf-8").read()
    assert "seed + 1009" in src, "b2m no longer uses train_mlp's mask-RNG offset"
    assert "leave_one=True" in src, "b2m no longer matches train_mlp's dropout distribution"


def test_every_method_has_a_logits_function_and_the_seam_stays_patchable():
    """Dispatch is by NAME through globals(), not by bound function object: binding the objects
    silently defeats monkeypatching, and test_reliability_guards patches logits_mlp to drive
    pooled_logits without a real model."""
    assert set(P4R.METHODS) == set(P4R._LOGITS_FN)
    assert all(isinstance(v, str) for v in P4R._LOGITS_FN.values()), \
        "dispatch must hold NAMES so the patch seam survives"
    assert all(hasattr(P4R, v) for v in P4R._LOGITS_FN.values())


def test_unknown_method_is_rejected_rather_than_falling_through():
    X, _, groups = _toy()
    with pytest.raises(ValueError, match="kind must be one of"):
        P4R.pooled_logits("propoesd", None, X, np.zeros(len(X), int), np.zeros(len(X)),
                          groups, m=1, trials=2, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------- per-mask worst case
def test_per_mask_finds_a_catastrophic_mask_the_mixture_mean_hides():
    n_pix, T = 200, 8
    msk = np.repeat(np.arange(T), n_pix)
    corr = np.ones(n_pix * T)
    corr[msk == 3] = 0.0                      # one mask is completely wrong
    conf = np.full(n_pix * T, 0.9)
    acc, cov, sel = P4R._per_mask_stats(corr, conf, msk, 0.5)
    assert corr.mean() == pytest.approx(1 - 1 / T), "the pooled mean stays high by construction"
    assert acc.min() == 0.0, "the worst mask must be visible per-mask"
    assert np.nanmax(sel) == pytest.approx(1.0)
    assert acc.size == T, "one row per mask draw"


def test_per_mask_worst_never_exceeds_its_own_median():
    rng = np.random.default_rng(1)
    n_pix, T = 150, 16
    msk = np.repeat(np.arange(T), n_pix)
    corr = (rng.random(n_pix * T) < 0.8).astype(float)
    conf = np.clip(rng.random(n_pix * T), 0.01, 0.99)
    acc, cov, sel = P4R._per_mask_stats(corr, conf, msk, 0.5)
    assert acc.min() <= np.median(acc) <= acc.max()
    assert np.all((cov >= 0) & (cov <= 1))


# ------------------------------------------------------------------------ per-class reliability
def test_per_class_exposes_a_small_class_abstained_out_of_existence():
    """The pathology: 95% global coverage while one class is never predicted at all. No global
    statistic in the file reports this; cls_cov_min and cls_zero_cov_n do."""
    lab = np.concatenate([np.zeros(950, int), np.ones(50, int)])
    conf = np.concatenate([np.full(950, 0.9), np.full(50, 0.1)])
    corr = np.ones(1000)
    cov, sel = P4R._per_class_stats(corr, conf, lab, 0.5, 2)
    assert (conf >= 0.5).mean() == pytest.approx(0.95), "global coverage looks excellent"
    assert cov.min() == 0.0, "the rejected class must show zero coverage"
    assert int((cov <= 0).sum()) == 1


def test_per_class_skips_absent_classes_rather_than_scoring_them_zero():
    """A class with no pixels in this eval split was not abstained on -- it was not there. Counting
    it as zero coverage would manufacture a pathology out of a split artefact."""
    lab = np.zeros(100, int)                       # only class 0 present, of 3 possible
    cov, sel = P4R._per_class_stats(np.ones(100), np.full(100, 0.9), lab, 0.5, 3)
    assert cov.size == 1, "absent classes must not appear as zero-coverage rows"
    assert cov[0] == pytest.approx(1.0)


def test_min_class_coverage_never_exceeds_the_macro_average():
    rng = np.random.default_rng(2)
    lab = rng.integers(0, 6, 900)
    conf = rng.random(900)
    cov, _ = P4R._per_class_stats(np.ones(900), conf, lab, 0.5, 6)
    assert cov.min() <= cov.mean() + 1e-12


# ------------------------------------------------------------------------------- wiring contract
def test_the_new_metrics_are_declared_and_the_contrasts_decompose_the_headline():
    src = open(P4R.__file__, encoding="utf-8").read()
    for name in ("mask_acc_worst", "mask_cov_worst", "mask_sel_risk_worst",
                 "cls_cov_min", "cls_zero_cov_n", "cls_sel_risk_max"):
        assert name in P4R.METRIC_NAMES, f"{name} is computed but never aggregated"
    # proposed-b2 must be reported as the SUM of the two isolating contrasts, not on its own
    for a, b in (("proposed", "b2m"), ("b2m", "b2"), ("proposed", "b2")):
        assert f'("{a}", "{b}")' in src, f"the {a}-minus-{b} contrast is missing"
