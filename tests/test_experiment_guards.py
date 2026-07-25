"""Guards for the experiment scripts and their shared grouping/metric helpers.

Every test here pins a path that previously produced a WRONG NUMBER SILENTLY — no crash, no
warning, just a plausible-looking value written to a paper CSV. They are grouped by the failure
mode rather than by file, because the same mode recurs: a degenerate configuration that trains
nothing, an axis convention numpy does not enforce, a normalization applied twice, and a
comparison reported without the spread that decides whether it means anything.

Sibling guards live in tests/test_review_fixes.py (library-level) and tests/test_pipeline.py.
"""
import os
import sys
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

from bandsim.grouping import contiguous_groups, group_center_wavelengths   # noqa: E402
from bandsim.metrics import miou, per_class_iou, audc, retention           # noqa: E402
import phase2_degradation as P2                                            # noqa: E402
import phase2_group_ablation as GA                                         # noqa: E402
import phase2_cross_sensor as CS                                           # noqa: E402
import phase4_ablation as P4                                               # noqa: E402
import phase6_second_dataset as P6                                         # noqa: E402
import phase1_indian_pines as P1                                           # noqa: E402


# ---------------------------------------------------------------------------------------------
# Item 1 — a single spectral group makes SGMAE pretraining a silent no-op
# ---------------------------------------------------------------------------------------------
def test_group_masking_rejects_a_single_group():
    """At G=1 the leave_one clamp is min(count, 0) = 0, so NOTHING is ever masked: measured 100% of
    rows with zero masked groups, SGMAE loss weight sum 0, loss identically 0.0, and 5 epochs of
    pretraining moving the weights by exactly 0.0. Unsatisfiable, so it must raise."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least 2 groups"):
        P2._vec_group_subset(rng, 16, 1, 1, 4, leave_one=True)      # SGMAE mask caller
    with pytest.raises(ValueError, match="at least 2 groups"):
        P2._vec_group_subset(rng, 16, 1, 0, 4, leave_one=True)      # band-group dropout caller


def test_group_masking_leaves_a_usable_context_for_every_valid_group_count():
    """The invariant the G=1 case silently violated: every training row must have >=1 group masked
    AND >=1 group left as context. Checked over the whole small-G regime the clamp exists for."""
    rng = np.random.default_rng(1)
    for G in range(2, 13):
        m = P2._vec_group_subset(rng, 4000, G, 1, 4, leave_one=True)
        assert m.sum(1).min() >= 1, f"G={G}: some row masks nothing -> zero SGMAE loss on that row"
        assert m.sum(1).max() <= G - 1, f"G={G}: some row masks every group -> no context left"


def test_hcs_sampling_still_allows_selecting_all_groups():
    """The HCS caller passes leave_one=False on purpose (keeping all G groups is a valid sample);
    the new G>=2 rule must not leak into it."""
    rng = np.random.default_rng(2)
    m = P2._vec_group_subset(rng, 500, 1, 1, 2, leave_one=False)
    assert m.all(), "HCS keep-all sampling must remain available at G=1"


def test_phase2_run_seed_rejects_a_single_group():
    """Gate the degenerate configuration before ~2 minutes of per-seed training, not after."""
    with pytest.raises(ValueError, match="n_groups must be >= 2"):
        P2.run_seed(0, cube=None, gt=None, n_groups=1, max_missing=0, trials=1, epochs=1)


def test_group_ablation_rejects_the_grid_a_single_group_collapses():
    """G=1 reaches missing fraction 0, and frac_max_common is a MIN over G — so one G=1 entry
    collapses the shared grid to zero width for EVERY G. That grid (21 points all at 0.0) used to
    divide by zero and write `inf` into the CSV for every method at every group count. main()
    rejects G<2 up front; this pins the downstream consequence so neither guard can be dropped."""
    with pytest.raises(ValueError, match="positive range"):
        GA.frac_auc(np.linspace(0.0, 0.0, 21), np.array([0.0]), np.array([55.0]))
    src = open(os.path.join(_ROOT, "experiments", "phase2_group_ablation.py")).read()
    assert "every G >= 2" in src, "main() must reject G=1 before it collapses the shared grid"


# ---------------------------------------------------------------------------------------------
# Item 1b — trials <= 0 must be rejected, not silently downgraded to one draw
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1, -5])
def test_degradation_curve_rejects_nonpositive_trials(bad):
    """max(1, trials) turned `--trials 0` into a single unaveraged draw that still looked like a
    Monte-Carlo mean over `trials` draws."""
    with pytest.raises(ValueError, match="trials must be >= 1"):
        P2.degradation_curve("b1", None, np.zeros((2, 4), np.float32), np.zeros(2, int),
                             [np.arange(2), np.arange(2, 4)], np.array([1.0, 2.0, 3.0, 4.0]),
                             max_missing=1, trials=bad, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------------------------
# Item 2 — np.interp does not enforce an increasing x-axis
# ---------------------------------------------------------------------------------------------
_G3 = [np.array([0]), np.array([1]), np.array([2])]


def test_interp_missing_is_correct_on_a_valid_axis():
    """Anchor the right answer first: dropping the middle of [1, 10, 100] on an evenly spaced
    ascending axis must impute the mean of its neighbours, 50.5."""
    out = P2.interp_missing(np.array([[1.0, 10.0, 100.0]]), _G3, [1],
                            np.array([400.0, 450.0, 500.0]))
    assert out[0, 1] == pytest.approx(50.5)


@pytest.mark.parametrize("wl", [
    np.array([500.0, 450.0, 400.0]),      # descending — returned 1.0 instead of 50.5 (98% error)
    np.array([450.0, 400.0, 500.0]),      # shuffled
    np.array([400.0, 400.0, 500.0]),      # duplicate -> tie broken by array order, not physics
])
def test_interp_missing_rejects_non_increasing_wavelengths(wl):
    with pytest.raises(ValueError, match="STRICTLY INCREASING"):
        P2.interp_missing(np.array([[100.0, 10.0, 1.0]]), _G3, [1], wl)


def test_b3_impute_inherits_the_axis_check():
    """The B3 eval path must not be a way around the guard."""
    with pytest.raises(ValueError, match="STRICTLY INCREASING"):
        P2.b3_impute(np.array([[100.0, 10.0, 1.0]]), _G3, [1],
                     np.array([500.0, 450.0, 400.0]), np.zeros(3), np.ones(3))


def test_interp_missing_rejects_a_mismatched_axis_length():
    """A longer axis interpolates against the wrong band centres without raising."""
    with pytest.raises(ValueError, match="one entry per band"):
        P2.interp_missing(np.array([[1.0, 10.0, 100.0]]), _G3, [1],
                          np.array([400.0, 450.0, 500.0, 550.0]))


def test_interp_missing_rejects_a_non_finite_axis():
    with pytest.raises(ValueError, match="finite"):
        P2.interp_missing(np.array([[1.0, 10.0, 100.0]]), _G3, [1],
                          np.array([400.0, np.nan, 500.0]))


# ---------------------------------------------------------------------------------------------
# Item 3 — phase 4: the brightness-norm guard, the D1 degeneracy, and paired statistics
# ---------------------------------------------------------------------------------------------
def test_mean_brightness_norm_is_bounded_on_dead_column_pixels():
    """Design D2 zeroes whole detector columns, so ~7% of test pixels arrive as all-zero spectra
    (mean == 0 AND ||x||_2 == 0). They must come back finite, not 0/0 = NaN."""
    out = P4.mean_brightness_norm(np.zeros((3, 200)))
    assert np.isfinite(out).all() and np.all(out == 0.0)


def test_mean_brightness_norm_is_bounded_on_a_zero_mean_spectrum():
    out = P4.mean_brightness_norm(np.array([[-1.0, 0.0, 1.0]]))
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.0


def test_mean_brightness_norm_documented_bound_for_positive_spectra_is_one_over_nbands():
    """The docstring used to promise a ~0.07 (= 1/sqrt(n)) margin before the L2 fallback engages.
    The true worst case for a non-negative spectrum is one-hot, giving 1/n_bands = 0.005 — still
    above the 1e-3 trigger, but a 5x margin rather than a 71x one. Pin the real bound."""
    n = 200
    onehot = np.zeros((1, n)); onehot[0, 0] = 1.0
    ratio = onehot.mean() / np.sqrt((onehot ** 2).sum())
    assert ratio == pytest.approx(1.0 / n)
    assert ratio > 1e-3, "the L2 fallback must not engage for a strictly positive spectrum"
    assert np.isfinite(P4.mean_brightness_norm(onehot)).all()


def test_design_d1_stripe_gain_is_cancelled_by_mean_normalization():
    """Design D1's flat 100%-retention line is an ALGEBRAIC IDENTITY: a per-column scalar gain
    cancels under per-spectrum mean normalization, (g*x)/mean(g*x) == x/mean(x). Measured residual
    is 6.7e-16 (float rounding). Locked so the panel can never be re-read as robustness."""
    assert P4.selfcheck_d1_identity()[0] < 1e-12


def test_paired_retention_reports_a_dispersion_and_pairs_by_seed():
    """Retention was a ratio of seed-MEANS: one number per cell, no spread, pairing discarded."""
    vals = [40.0, 60.0, 50.0]
    base = [80.0, 80.0, 100.0]
    mean, std = P4.paired_retention(vals, base)
    assert mean == pytest.approx(np.mean(np.array(vals) / np.array(base)) * 100.0)
    assert std > 0.0, "a paired comparison must come with the spread that qualifies it"
    assert mean != pytest.approx(np.mean(vals) / np.mean(base) * 100.0), \
        "ratio-of-means and mean-of-ratios must be distinguishable on an unbalanced example"


def test_paired_retention_is_nan_off_a_zero_baseline():
    mean, std = P4.paired_retention([1.0, 2.0], [0.0, 4.0])
    assert np.isnan(mean) and np.isnan(std)


# ---------------------------------------------------------------------------------------------
# Item 4 — the cross-G AUC must be a MEAN in [0, 100], normalized exactly once
# ---------------------------------------------------------------------------------------------
def test_cross_g_auc_is_a_per_point_mean_not_a_rescaled_area():
    """`audc` already divides the trapezoid by the x-range. main() divided by the range width a
    SECOND time, inflating every reported value by 1/frac_max_common — 2.00x on the default
    {5,10,20} sweep — with an inflation factor that depended on which G were swept."""
    common = np.linspace(0.0, 0.5, 21)
    fracs = np.arange(0, 11) / 20.0
    flat = np.full(11, 42.0)
    assert GA.frac_auc(common, fracs, flat) == pytest.approx(42.0)


def test_cross_g_auc_matches_across_group_counts_for_identical_physics():
    """The whole point of the fraction axis: the same physical response sampled at two group
    resolutions must summarise to the same number on a shared grid."""
    common = np.linspace(0.0, 0.5, 21)
    f5 = np.arange(0, 5) / 5.0
    f20 = np.arange(0, 11) / 20.0
    a5 = GA.frac_auc(common, f5, 100.0 * (1.0 - f5))
    a20 = GA.frac_auc(common, f20, 100.0 * (1.0 - f20))
    assert a5 == pytest.approx(a20)
    assert 0.0 <= a5 <= 100.0


def test_cross_g_auc_refuses_to_extrapolate_past_the_measured_range():
    """np.interp EXTRAPOLATES FLAT rather than raising, which would pad a curve with its last
    measured mIoU and report robustness that was never evaluated."""
    fracs = np.arange(0, 5) / 20.0                      # measured only out to 0.20
    with pytest.raises(ValueError, match="extrapolate flat"):
        GA.frac_auc(np.linspace(0.0, 0.5, 21), fracs, np.linspace(70, 40, 5))


def test_cross_g_auc_refuses_a_zero_width_grid():
    with pytest.raises(ValueError, match="positive range"):
        GA.frac_auc(np.array([0.3]), np.arange(0, 5) / 5.0, np.linspace(70, 40, 5))


def test_selfcheck_fraction_axis_exercises_the_production_path():
    """The old guard called audc() directly while main() divided again, so it passed while every
    published number was 2x too large. It must now go through the same helper main() uses."""
    (raw5, raw20), (auc5, auc20) = GA.selfcheck_fraction_axis()
    assert auc5 == pytest.approx(auc20)
    assert abs(raw5 - raw20) > 1.0
    assert 0.0 <= auc5 <= 100.0


# ---------------------------------------------------------------------------------------------
# Item 5 — phase 1: leakage lock, and per-class support must be reported
# ---------------------------------------------------------------------------------------------
def test_per_class_support_counts_test_pixels_per_class():
    y = np.array([0, 0, 0, 1, 1, 5])
    sup = P1.per_class_support(y)
    assert sup.shape == (P1.NUM_CLASSES,)
    assert sup[0] == 3 and sup[1] == 2 and sup[5] == 1 and sup[2] == 0


def test_low_support_classes_flags_uninterpretable_per_class_numbers():
    """Four Indian Pines classes sit under 30 test px (Grass-pasture-mowed 3.8, Oats 4.0,
    Alfalfa 22.2, Stone-Steel-Towers 24.4); their per-class IoU cannot be read as performance."""
    sup = np.full(P1.NUM_CLASSES, 500)
    sup[6] = 4; sup[8] = 4; sup[0] = 22
    low = P1.low_support_classes(sup)
    assert set(low) == {0, 6, 8}
    assert P1.low_support_classes(np.full(P1.NUM_CLASSES, 500)) == []


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_ROOT, "data", "indian_pines", "Indian_pines_gt.mat")),
    reason="Indian Pines ground truth not present")
def test_phase1_split_has_no_pixel_or_adjacency_leakage():
    """REGRESSION LOCK (this path was already clean — verified, not fixed): train/test pixel sets
    are disjoint AND, at the default guard=1, no test pixel is 8-connected (diagonals included) to
    a train pixel or to the train REGION. Without the guard 34% of test pixels touch train, so the
    lock is what keeps guard>=1 from being dropped as an optimisation."""
    from scipy.ndimage import binary_dilation
    from bandsim.io import load_mat_cube, disjoint_block_split
    gt = load_mat_cube(os.path.join(_ROOT, "data", "indian_pines", "Indian_pines_gt.mat"),
                       key="indian_pines_gt").astype(int)
    eight = np.ones((3, 3), bool)
    for seed in range(5):
        tr, te = disjoint_block_split(gt, block=10, guard=1, offset=seed)
        assert not (tr & te).any(), f"seed {seed}: a pixel is in both splits"
        assert not (te & binary_dilation(tr, eight) & ~tr).any(), \
            f"seed {seed}: a test pixel is 8-adjacent to a train pixel"
        bi = (np.arange(gt.shape[0])[:, None] + seed) // 10
        bj = (np.arange(gt.shape[1])[None, :] + seed) // 10
        region = ((bi + bj) % 2 == 0)
        assert not (te & binary_dilation(region, eight) & ~region).any(), \
            f"seed {seed}: a test pixel is 8-adjacent to the train REGION"


# ---------------------------------------------------------------------------------------------
# Item 6 — phase 6 must not call a source-tuned transfer result "generalization"
# ---------------------------------------------------------------------------------------------
def test_phase6_does_not_claim_generalization():
    """Nothing in phase 6 is re-tuned: --groups/--epochs and every model hyper-parameter are the
    Indian-Pines values, and there is no tuning split. The docstring claimed the panel "prov[es]
    the conclusion is not an Indian-Pines artefact" and the summary printed "generalizes"."""
    doc = P6.__doc__
    assert "proving the conclusion is not an Indian-Pines artefact" not in doc
    assert "does NOT establish dataset generalization" in doc
    assert "no tuning split" in doc.lower() or "no tuning/validation split" in doc.lower()
    src = open(os.path.join(_ROOT, "experiments", "phase6_second_dataset.py")).read()
    assert "'generalizes' if best" not in src, "the summary line must not print 'generalizes'"
    assert "NOT dataset generalization" in src


def test_phase4_docstring_reports_the_negative_result_and_the_d1_degeneracy():
    """Measured at the defaults (5 seeds, 60 epochs): proposed loses to B2 by 5.3-6.7 mIoU at every
    non-trivial cirrus depth (0/5 seeds won), and D1 measures nothing. The docstring used to open
    with "Shows the proposed model's robustness holds ACROSS corruption physics"."""
    doc = " ".join(P4.__doc__.split())          # collapse the line wrapping before matching
    assert "Shows the proposed model's robustness holds ACROSS corruption physics" not in doc
    assert "IT DOES NOT" in doc, "the measured direction of the result must be stated"
    assert "NEGATIVE result and must be reported as one" in doc
    assert "DESIGN D1 IS SPLIT IN TWO" in doc


# ---------------------------------------------------------------------------------------------
# Item 7 — grouping / metrics: silent-wrong-answer paths
# ---------------------------------------------------------------------------------------------
def test_group_center_wavelengths_rejects_a_longer_axis():
    """The 220-band nominal AVIRIS axis handed to a 200-band 'corrected' grouping indexes fine and
    returns plausible-looking, WRONG group centres — and both axes live in bandsim/io.py."""
    groups = contiguous_groups(200, 4)
    with pytest.raises(ValueError, match="grouping spans"):
        group_center_wavelengths(np.linspace(400, 2500, 220), groups)
    ok = group_center_wavelengths(np.linspace(400, 2500, 200), groups)
    assert ok.shape == (4,) and np.isfinite(ok).all()


def test_group_center_wavelengths_rejects_a_non_finite_axis():
    """A NaN centre propagates into the wavelength PE and yields NaN logits with no traceback."""
    with pytest.raises(ValueError, match="finite"):
        group_center_wavelengths(np.array([400.0, np.nan, 500.0, 600.0]), contiguous_groups(4, 2))


@pytest.mark.parametrize("n_groups", [10.9, 2.5, True])
def test_contiguous_groups_rejects_non_integer_counts(n_groups):
    """int(10.9) silently built 10 groups; contiguous_groups(200, True) silently built 1 — which is
    exactly the degenerate single-group case Phase 2 must never run in."""
    with pytest.raises(ValueError, match="must be an integer"):
        contiguous_groups(200, n_groups)


def test_contiguous_groups_still_tiles_the_band_axis_exactly():
    for G in (1, 3, 7, 10, 200):
        cov = np.concatenate(contiguous_groups(200, G))
        assert np.array_equal(np.sort(cov), np.arange(200))


@pytest.mark.parametrize("fn", [miou, per_class_iou])
def test_metrics_reject_labels_outside_num_classes(fn):
    """`for c in range(num_classes)` makes an out-of-range label INVISIBLE: miou([0,5],[0,5],3)
    returned a perfect 100.0 while ignoring half the pixels — and the same pixels are dropped by
    confusion (AA/kappa) but counted by overall_accuracy. Three metrics, three populations."""
    with pytest.raises(ValueError, match="outside"):
        fn(np.array([0, 5]), np.array([0, 5]), 3)
    with pytest.raises(ValueError, match="outside"):
        fn(np.array([0, 1]), np.array([0, -1]), 3)


def test_miou_still_averages_over_present_classes_only():
    """No-op check: the new label guard must not disturb the absent-class convention."""
    yt = np.array([0, 0, 1, 1]); yp = np.array([0, 1, 1, 1])
    assert miou(yt, yp, 2) == pytest.approx(miou(yt, yp, 2))
    assert np.isnan(per_class_iou(yt, yp, 3)[2]), "an absent class must stay NaN, not become 100"


def test_audc_rejects_a_non_finite_curve():
    """One broken evaluation point silently turned the whole robustness summary into nan/inf, which
    then went into the CSV/TeX as the string 'nan'."""
    with pytest.raises(ValueError, match="finite"):
        audc([0, 1, 2], [100.0, np.nan, 60.0])
    with pytest.raises(ValueError, match="finite"):
        audc([0, 1, 2], [100.0, np.inf, 60.0])


def test_audc_edges_are_sane():
    """REGRESSION LOCK (verified clean): order-independence, degenerate x, duplicate x, empty."""
    assert audc([0, 1, 2], [100, 80, 60]) == pytest.approx(audc([2, 1, 0], [60, 80, 100]))
    assert audc([3], [42.0]) == pytest.approx(42.0)          # single point -> that value
    assert audc([2, 2, 2], [10.0, 20.0, 30.0]) == pytest.approx(20.0)   # zero-width -> plain mean
    assert 0.0 <= audc([0, 1, 1, 2], [100.0, 80.0, 20.0, 60.0]) <= 100.0
    with pytest.raises(ValueError):
        audc([], [])


def test_retention_edges_are_sane():
    """REGRESSION LOCK (verified clean): a non-positive or NaN clean score gives NaN, never inf."""
    assert retention(80.0, 40.0) == pytest.approx(0.5)
    for clean in (0.0, -1.0, np.nan):
        assert np.isnan(retention(clean, 5.0))


# ---------------------------------------------------------------------------------------------
# Item 8 — the cross-sensor panel changes TWO things at once (spectral content AND capacity)
# ---------------------------------------------------------------------------------------------
def test_mlp_param_count_matches_the_model_it_predicts():
    """`mlp_param_count` is arithmetic on the layer shapes, so it can drift from MLPBaseline
    without anything noticing. Pin it against the real module for every band count the panel uses."""
    from bandsim.model import MLPBaseline, count_params
    for nb in (200, 12, 9, 7, 1):
        for hidden in (256, 253, 180, 64):
            assert CS.mlp_param_count(nb, hidden=hidden) == \
                count_params(MLPBaseline(nb, CS.NUM_CLASSES, hidden=hidden)), \
                f"predicted param count wrong at bands={nb}, hidden={hidden}"


def test_cross_sensor_models_do_not_have_equal_capacity_by_default():
    """THE CONFOUND, pinned as a number, on the CANONICAL band sets (S2 L2A = 12 bands, OLI L2 = 7).

    Swapping the band-set swaps the MLP's INPUT DIMENSION, so the first Linear is (bands x hidden)
    and the parameter count moves with the band count: 121,360 for full-HSI (200 bands) vs 71,952 for
    OLI (7). A gap between those two is NOT attributable to spectral content alone.

    Note what is deliberately NOT asserted any more: that S2-vs-OLI is "effectively matched already"
    because they differ by only ~1.8% of parameters. A percentage of parameters and a number of mIoU
    points are different units, so their relative size supports no inference — that argument was
    removed from the docstring and must not come back as a test."""
    full, s2, oli = (CS.mlp_param_count(n) for n in (200, 12, 7))
    assert (full, s2, oli) == (121_360, 73_232, 71_952)
    assert full / oli > 1.6, "the full-HSI vs multispectral comparison is capacity-confounded"


def test_capacity_matching_makes_the_richer_band_set_the_SMALLER_model():
    """`--match-capacity` sizes each band-set's hidden width to a shared budget. Equalising the count
    is the point, but the fact that carries the CLAIM is stronger and is what this pins: after
    matching, the Sentinel-2 model is strictly SMALLER than the Landsat OLI model. That is what makes
    an S2 win unbuyable with capacity — an argument that survives without needing to compare
    percentages of parameters against mIoU points."""
    bands = [200, 12, 7]
    budget = min(CS.mlp_param_count(n) for n in bands)
    hid = {n: CS.hidden_for_budget(n, budget) for n in bands}
    counts = {n: CS.mlp_param_count(n, hidden=hid[n]) for n in bands}
    assert all(c <= budget for c in counts.values()), f"a matched model exceeded the budget: {counts}"
    assert max(counts.values()) / min(counts.values()) < 1.01, f"capacity not equalised: {counts}"
    assert hid[200] < hid[7], "the 200-band model must give up width to pay for its input layer"
    assert counts[12] < counts[7], \
        f"matched S2 must be SMALLER than matched OLI for the claim to hold: {counts}"


def test_capacity_matching_never_returns_a_degenerate_width():
    """A budget too small for the band count must raise rather than hand back hidden=0 (or a
    negative width), which would build a model with no hidden layer at all and still 'train'."""
    with pytest.raises(ValueError, match="too small"):
        CS.hidden_for_budget(200, 100)
    for bad in ((0, 10_000), (10, 0), (-1, 10_000)):
        with pytest.raises(ValueError):
            CS.hidden_for_budget(*bad)


def test_cross_sensor_records_capacity_where_the_result_is_reported():
    """Stating the confound in a docstring nobody opens is not stating it. The per-condition
    parameter count must travel in the CSV next to the mIoU it qualifies, and a PER-SEED table must
    exist at all: a summary of means cannot support the paired comparison the capacity control needs
    (the previous control compared a 2-seed run against a 5-seed one and could not tell the capacity
    effect from split sampling)."""
    src = open(os.path.join(_ROOT, "experiments", "phase2_cross_sensor.py")).read()
    assert '"params"' in src, "the CSV header must carry a per-condition parameter count"
    assert "_raw.csv" in src, "a per-seed long-form CSV must be written, not just a summary"
    assert '"split_seed"' in src and '"model_seed"' in src, \
        "the split seed and the model seed must be recorded separately"
    assert '"n_test_classes"' in src, \
        "mIoU averages over GT-present classes, so the per-seed class support must be recorded"
    assert "capacity" in CS.__doc__.lower(), "the confound must be named in the module docstring"


def test_every_number_quoted_in_the_docstring_is_recomputable_from_the_code():
    """The docstring quotes arithmetic, which is exactly the pattern that goes stale (a TAU-
    sensitivity conclusion once sat in phase 3's docstring that phase 3 never computed). Recompute
    every quoted figure here so the prose can never drift from the code.

    The previous version of this test also pinned a MEASURED mIoU table. That table was invalidated
    when the band-set contract was fixed (it had been measured on the RSR store's 13/9-band lists,
    including a cirrus band synthesized across a data gap), which is the argument for not keeping
    measured results in a docstring at all: arithmetic can be re-derived here, a measurement cannot."""
    doc = " ".join(CS.__doc__.split())
    bands = {"full": 200, "s2": 12, "oli": 7}
    unmatched = {k: CS.mlp_param_count(n) for k, n in bands.items()}
    assert unmatched == {"full": 121_360, "s2": 73_232, "oli": 71_952}
    for k, v in unmatched.items():
        assert f"{v:,} params" in doc, f"docstring does not quote its own {k} figure {v:,}"
    budget = min(unmatched.values())
    widths = {k: CS.hidden_for_budget(n, budget) for k, n in bands.items()}
    assert widths == {"full": 180, "s2": 253, "oli": 256}, f"quoted matched widths are stale: {widths}"
    assert f"h=256->{widths['full']}" in doc and f"256->{widths['s2']}" in doc
    matched = {k: CS.mlp_param_count(bands[k], hidden=widths[k]) for k in bands}
    spread = max(matched.values()) / min(matched.values())
    assert f"{spread:.3f}" == "1.005", f"quoted 1.005x spread is stale: {spread:.4f}"
    assert "1.005x" in doc
    # The load-bearing sentence: matched S2 is SMALLER than matched OLI.
    assert matched["s2"] < matched["oli"]
    assert f"{matched['s2']:,} vs {matched['oli']:,} params" in doc
    assert "--paired-capacity" in doc, "name the control that identifies the capacity delta"
    assert "--seeds 0 1 --epochs 60 --match-capacity" in doc, "name the matched-arm command"
    assert "_raw.csv" in doc, "name the per-seed artefact a paired analysis needs"
    assert "different units" in doc, \
        "the docstring must keep warning that %-of-parameters vs mIoU-points is not an argument"


# ---------------------------------------------------------------------------------------------
# Item 9 — B3's standardized-space interpolation fallback must not be reachable
# ---------------------------------------------------------------------------------------------
def test_eval_mlp_refuses_to_impute_without_raw_reflectance():
    """`eval_mlp` had a back-compat branch that interpolated the ALREADY-STANDARDIZED features when
    a caller omitted Xte_raw/mu/sd — the exact order selfcheck_b3 exists to prove WRONG. On the
    selfcheck's own worked example the two branches return 5.05 (correct) and 1.0 (fallback): a
    silently different B3 baseline chosen by which keyword arguments a caller happened to pass."""
    args = (np.zeros((1, 3), np.float32), np.zeros(1, int), _G3, [1],
            np.array([400.0, 450.0, 500.0]))
    for missing in ({}, {"Xte_raw": np.ones((1, 3))},
                    {"Xte_raw": np.ones((1, 3)), "mu": np.zeros(3)}):
        with pytest.raises(ValueError, match="RAW reflectance"):
            P2.eval_mlp(None, *args, impute=True, **missing)


def test_eval_mlp_still_zero_fills_without_raw_reflectance():
    """The guard is specific to B3: the zero-fill kinds legitimately need no raw reflectance and
    must keep working (verify_guardband.py calls degradation_curve with b1/b2/proposed and no raw)."""
    out = P2.zero_missing(np.array([[1.0, 2.0, 3.0]], np.float32), _G3, [1])
    assert out.tolist() == [[1.0, 0.0, 3.0]]


def test_the_two_b3_orders_still_provably_disagree():
    """Keep the counterexample executable: if raw-space and standardized-space interpolation ever
    agreed, the guard above would be protecting nothing and could be quietly dropped."""
    wrong, right = P2.selfcheck_b3()
    assert abs(right - wrong) > 1.0


# ---------------------------------------------------------------------------------------------
# Item 10 — np.interp does not enforce an increasing x-axis HERE either (missing-fraction grid)
# ---------------------------------------------------------------------------------------------
_FR_OK = np.array([0.0, 0.2, 0.4, 0.5])
_CV_OK = np.array([70.0, 55.0, 45.0, 40.0])


def test_frac_auc_is_correct_on_a_valid_axis():
    """Anchor the right answer before pinning the rejections."""
    assert GA.frac_auc(np.linspace(0.0, 0.5, 21), _FR_OK, _CV_OK) == pytest.approx(53.5)


@pytest.mark.parametrize("fracs, curve", [
    (np.array([0.0, 0.4, 0.2, 0.5]), np.array([70.0, 45.0, 55.0, 40.0])),   # shuffled -> 54.5
    (np.array([0.0, 0.5, 0.2, 0.4]), np.array([70.0, 40.0, 55.0, 45.0])),   # shuffled differently
    (np.array([0.5, 0.4, 0.2, 0.0]), np.array([40.0, 45.0, 55.0, 70.0])),   # fully descending
])
def test_frac_auc_rejects_a_non_monotonic_measured_axis(fracs, curve):
    """numpy's docs: "The x-coordinate sequence is expected to be increasing, but this is not
    explicitly enforced". The SAME four (fraction, mIoU) pairs merely reordered returned 54.5
    instead of 53.5 — a normal-looking cross-G summary, no exception, straight into the CSV."""
    with pytest.raises(ValueError, match="STRICTLY INCREASING"):
        GA.frac_auc(np.linspace(0.0, 0.5, 21), fracs, curve)


def test_frac_auc_rejects_duplicate_fractions():
    """A tie makes np.interp pick a side by binary-search order, so the summary depends on array
    order rather than on the measurement (the same rule interp_missing already enforces on wl)."""
    with pytest.raises(ValueError, match="STRICTLY INCREASING"):
        GA.frac_auc(np.linspace(0.0, 0.4, 21), np.array([0.0, 0.2, 0.2, 0.4]),
                    np.array([70.0, 55.0, 50.0, 40.0]))


def test_frac_auc_rejects_a_non_monotonic_common_grid():
    """An unsorted grid also DEFEATS the existing extrapolation guard, which only inspects
    common[0] and common[-1]: [0.0, 0.9, 0.20] passes those two bounds while 0.9 sits far outside
    the measured range and gets flat-extrapolated — the precise failure that guard exists to stop."""
    fracs = np.arange(0, 5) / 20.0                       # measured only out to 0.20
    curve = np.linspace(70.0, 40.0, 5)
    with pytest.raises(ValueError, match="STRICTLY INCREASING"):
        GA.frac_auc(np.array([0.0, 0.9, 0.20]), fracs, curve)


def test_frac_auc_axis_guard_does_not_disturb_the_valid_paths():
    """REGRESSION LOCK: the production grid (np.linspace) and the selfcheck must be unaffected."""
    assert GA.frac_auc(np.linspace(0.0, 0.5, 21), np.arange(0, 11) / 20.0,
                       np.full(11, 42.0)) == pytest.approx(42.0)
    GA.selfcheck_fraction_axis()


# ---------------------------------------------------------------------------------------------
# Item 11 — Design D1 cannot move, so it must not be drawn as a robustness curve
# ---------------------------------------------------------------------------------------------
def test_design_d1_is_not_a_panel_in_the_main_robustness_figure():
    """A curve whose flatness is an algebraic identity ((g*x)/mean(g*x) == x/mean(x)) reads as
    "perfectly robust" to anyone who does not read the 6.5pt caption. It is excluded from the
    figure; the finding survives in the docstring, the selfcheck, and the CSV."""
    designs = [p["design"] for p in P4.FIGURE_PANELS]
    assert "D1b_noise_then_gain" not in designs, "a degenerate curve must not be plotted as robustness"
    # D1a IS plotted: with the noise added after the gain the per-column scalar no longer factors
    # out of the per-spectrum mean, so that axis measures something and belongs in the figure. The
    # pair is the finding, and only the half that cannot move is withheld.
    assert "D1a_gain_then_noise" in designs, \
        "the informative half of D1 (y = g*x + n) must be plotted"
    assert designs == ["C_cirrus", "D1a_gain_then_noise", "D2_dead_cols"]


def test_design_d1_is_still_recorded_in_the_csv():
    """Removing the panel must not remove the RESULT: D1 is a real negative finding about the
    normalization and stays in the CSV, flagged, so it can be cited as what it is."""
    assert "D1b_noise_then_gain" in [d["design"] for d in P4.CSV_DESIGNS]
    d1 = next(d for d in P4.CSV_DESIGNS if d["design"] == "D1b_noise_then_gain")
    assert d1["degenerate"], "the D1 rows must stay flagged in the CSV"


def test_the_figure_caption_names_exactly_the_axes_it_omits():
    """The "measured but not plotted" caption is DERIVED from the two specs, not hand-written, so
    re-adding a panel cannot leave a caption behind claiming an axis is missing while it is on the
    page. Every plotted design must also be a measured one, or the figure would plot a curve with
    no CSV row behind it."""
    assert P4.omitted_designs() == ["D1b_noise_then_gain"]
    measured = {d["design"] for d in P4.CSV_DESIGNS}
    assert {p["design"] for p in P4.FIGURE_PANELS} <= measured, \
        "a plotted panel has no corresponding CSV design"


def test_phase4_d2_sweep_realises_its_fraction_exactly_instead_of_in_expectation():
    """D2's x is a SWEEP AXIS, so it must apply the severity its label claims. Under add_striping's
    default Bernoulli draw it did not even order its own conditions on this cube: measured over the
    5 default seeds as the fraction of TEST pixels zeroed, nominal 3% gave 2.55%-6.71% and nominal
    5% gave 3.32%-8.34% -- overlapping -- while seed 0 realised ZERO dead columns at nominal 1% (an
    uncorrupted point plotted at x=0.01) and destroyed the identical 6.71% at both 3% and 5%."""
    # The BEHAVIOUR is asserted in tests/test_phase4_end_to_end.py, off the CSV a real run writes.
    # This used to grep run_seed's source for 'dead_col_mode="exact"', which passes if the string
    # sits in a comment, in a dead branch, or on the wrong call -- the guard-theatre the phase 3
    # session's mutation campaign documented in shared_layer.md. What is left here is a property of
    # the CONSTANTS, which no run-level assertion covers because a bad grid raises before it writes.
    #
    # Every level must be representable as a DISTINCT, nonzero column count at the Indian Pines
    # width, or two nominal levels are secretly the same experiment / one is the clean baseline.
    # add_striping raises on the latter, but only once a run has already started training.
    counts = [int(np.floor(f * 145 + 0.5)) for f in P4.DEAD_FRACS]
    assert counts[0] == 0, f"the first level must be the zero point, got {counts}"
    assert all(c > 0 for c in counts[1:]), f"a nonzero level applies no corruption: {counts}"
    assert len(set(counts)) == len(counts), f"two levels are the same experiment: {counts}"


def test_phase4_docstring_still_documents_the_d1_negative_result():
    doc = " ".join(P4.__doc__.split())
    assert "DESIGN D1 IS SPLIT IN TWO" in doc
    assert "not plotted" in doc.lower() or "excluded from the figure" in doc.lower(), \
        "the docstring must say WHERE the D1 finding now lives"


# ---------------------------------------------------------------------------------------------
# Item 12 — phase 6 must not reconfigure phase 2 by mutating its module global
# ---------------------------------------------------------------------------------------------
def test_phase2_training_takes_an_explicit_class_count():
    """`P2.NUM_CLASSES = n` rewrote another module's state to pass one argument. It is not
    reentrant, and `bandsim.parallel.run_jobs` runs the SERIAL path in the CALLING process, so the
    mutation escapes into whatever runs next."""
    groups = [np.arange(0, 5), np.arange(5, 10)]
    X = np.zeros((8, 10), np.float32)
    y = np.zeros(8, int)
    m = P2.train_mlp(X, y, groups, 0, group_dropout=False, epochs=1, num_classes=7)
    assert m.net[-1].out_features == 7
    assert P2.NUM_CLASSES == 16, "asking for 7 classes must not rewrite the module default"


def test_every_phase2_helper_that_reads_num_classes_accepts_it_explicitly():
    """The metric side of the same leak: eval_* read NUM_CLASSES at CALL time too, so a leaked
    value silently changed which classes mIoU averaged over. Every entry point that consults the
    module global must offer an argument instead, or a caller is forced back into assigning it."""
    import inspect
    for fn in (P2.train_mlp, P2.eval_mlp, P2.eval_proposed, P2.degradation_curve):
        assert "num_classes" in inspect.signature(fn).parameters, \
            f"{fn.__name__} reads NUM_CLASSES but takes no explicit class count"
        assert inspect.signature(fn).parameters["num_classes"].default is None, \
            f"{fn.__name__}: the default must be None (= 'use this module's NUM_CLASSES')"


def test_phase6_does_not_mutate_phase2_module_state():
    """The concrete hazard: after phase 6 ran a 9-class dataset, phase 4 — which imported
    NUM_CLASSES BY VALUE (16) but calls P2.train_mlp, which read the global AT CALL TIME — built a
    9-output head and then scored it with miou(..., 16). Classes 9..15 became unpredictable and
    each contributed IoU 0: a real-looking, silently deflated mIoU with no exception.

    Asserted BEHAVIOURALLY (run a real 9-class seed and check P2 is untouched) rather than by
    grepping the source, which would also match the comment explaining the old bug."""
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(24, 24, 20))
    gt = rng.integers(1, 10, size=(24, 24))                  # 9 classes, labels 1..9
    before = P2.NUM_CLASSES
    P6.run_seed(0, cube=cube, gt=gt, wl=np.linspace(430, 860, 20), n_classes=9,
                n_groups=2, max_missing=1, trials=1, epochs=1)
    assert P2.NUM_CLASSES == before == 16, \
        f"phase 6 rewrote phase 2's class count to {P2.NUM_CLASSES}; it must thread num_classes="


def test_phase6_requires_an_explicit_dataset():
    """`synthetic` was the DEFAULT, so a bare `python experiments/phase6_second_dataset.py`
    produced results_phase6_synthetic.csv — a real-looking "second dataset" deliverable built from
    fabricated data. Choosing the dataset must be a deliberate act."""
    ap = P6.build_argparser()
    action = next(a for a in ap._actions if a.dest == "dataset")
    assert action.required, "--dataset must be explicit; fabricated data must not be the default"
    assert action.default is None


def test_phase6_still_marks_synthetic_as_fabricated():
    """REGRESSION LOCK: the provenance markers that stop 'synthetic' being quoted as a dataset."""
    src = open(os.path.join(_ROOT, "experiments", "phase6_second_dataset.py")).read()
    assert "FABRICATED" in src and "not a dataset" in src.lower()


# ---------------------------------------------------------------------------------------------
# Item 13 — docstrings must not quote a bound or a measurement the script does not produce
# ---------------------------------------------------------------------------------------------
def test_phase2_does_not_claim_a_parameter_bound_that_is_false():
    """"Methods compared (all <100k params)" was false for the very first baseline it listed:
    MLPBaseline's input layer scales with the FULL band count (200 x 256), giving 121,360 params
    against the grouped attention model's 70,692. The attention model is the SMALLER one, which is
    the point worth making — with the numbers, not with a bound that does not hold."""
    from bandsim.model import MLPBaseline, GroupedCrossBandAttention, count_params
    groups = contiguous_groups(200, 10)
    cwl = group_center_wavelengths(np.linspace(400, 2500, 200), groups)
    mlp = count_params(MLPBaseline(200, 16, hidden=256))
    prop = count_params(GroupedCrossBandAttention(groups, cwl, 16))
    assert (mlp, prop) == (121_360, 70_692)
    assert mlp > 100_000, "the MLP baseline is over 100k — the old blanket bound was false"
    assert prop < mlp, "the grouped attention model is the smaller of the two"
    # Match the CLAIM, not the string: the docstring still quotes the old wording in order to
    # explain why it was withdrawn, and a bare substring test would forbid documenting the fix.
    doc = " ".join(P2.__doc__.split())
    assert "Methods compared (all <100k params" not in doc, "the false bound must not be the claim"
    assert "121,360" in doc and "70,692" in doc, "state the measured counts instead of a bound"


def test_group_ablation_docstring_numbers_are_reproducible():
    """Both quoted sensitivity figures are recomputed here, because neither is produced by a run:
    the coarse-sampling bias (+0.77 mIoU, G=5 vs G=20) and the --grid inflation (+11%, G=5)."""
    f = lambda x: 70.0 * np.exp(-2.5 * np.asarray(x))          # noqa: E731
    common = np.linspace(0.0, 0.5, 21)
    f5, f20 = np.arange(0, 5) / 5.0, np.arange(0, 11) / 20.0
    bias = GA.frac_auc(common, f5, f(f5)) - GA.frac_auc(common, f20, f(f20))
    assert bias == pytest.approx(0.77, abs=0.01)
    coarse = GA.frac_auc(np.linspace(0.0, 0.5, 2), f5, f(f5))
    fine = GA.frac_auc(common, f5, f(f5))
    assert (coarse, fine) == (pytest.approx(45.3, abs=0.05), pytest.approx(40.8, abs=0.05))
    doc = " ".join(GA.__doc__.split())
    assert "+0.77 mIoU advantage over G=20" in doc


def test_vec_group_subset_docstring_rates_match_the_sampler():
    """The "~75%/50%/25% of rows" figure is the BAND-GROUP DROPOUT caller (counts ~ U{0..3}); the
    SGMAE caller (U{1..3}) degenerates harder at 100%/67%/33%. The docstring quoted one pair of
    numbers for a helper both callers share, so pin which is which."""
    rng = np.random.default_rng(0)
    drop = rng.integers(0, 4, size=200_000)         # band-group dropout caller
    sgmae = rng.integers(1, 4, size=200_000)        # SGMAE mask caller
    for G, want in ((1, 75.0), (2, 50.0), (3, 25.0)):
        assert np.mean(drop >= G) * 100 == pytest.approx(want, abs=0.5)
    for G, want in ((1, 100.0), (2, 66.7), (3, 33.3)):
        assert np.mean(sgmae >= G) * 100 == pytest.approx(want, abs=0.5)
    doc = " ".join(P2._vec_group_subset.__doc__.split()).lower()
    assert "dropout caller" in doc, "say which caller the 75/50/25 rates belong to"
    assert "sgmae mask caller" in doc, "and which caller the harsher 100/67/33 rates belong to"
