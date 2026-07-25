"""Regression guards for the 2026-07 adversarial-review fix campaign.

Each test pins a SPECIFIC counterexample from the external reviews so a fix can never silently
regress. Grouped by module; every assertion documents the bug it locks down.
Run: pytest tests/test_review_fixes.py -v
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== model.py : SGMAE mask-token PE (M1) + no-raw-leak (M2) =====================
import torch  # noqa: E402
from bandsim.grouping import contiguous_groups, group_center_wavelengths  # noqa: E402
from bandsim.model import GroupedCrossBandAttention  # noqa: E402


def _attn_model(G=10, C=200, K=16, seed=0):
    groups = contiguous_groups(C, G)
    cwl = group_center_wavelengths(np.linspace(400, 2500, C), groups)
    torch.manual_seed(seed)
    return GroupedCrossBandAttention(groups, cwl, K), groups


def test_sgmae_simultaneously_masked_groups_are_distinguishable():
    # M1: the wavelength/group PE is added to ALL tokens (incl. masked), so two groups masked at the
    # same time do NOT reconstruct identically. Before the fix both masked tokens were the identical
    # mask_token (no PE) -> the decoder produced the SAME reconstruction for them (maxdiff ~0).
    m, groups = _attn_model()
    m.eval()
    x = torch.randn(4, 200)
    masked = np.zeros((4, len(groups)), bool)
    masked[:, [2, 5]] = True
    with torch.no_grad():
        recon = m.reconstruct(x, torch.from_numpy(masked))            # (4, G, S)
    d = (recon[:, 2] - recon[:, 5]).abs().max().item()
    assert d > 1e-3, f"masked groups 2 and 5 reconstructed identically (maxdiff {d:.2e}) -- PE collapse"


def test_fully_missing_row_does_not_leak_raw_values():
    # M2: a fully-missing row attends only to its own mask tokens; changing the raw (masked) spectrum
    # must NOT change its logits. The old guard set the row all-present -> it read the raw spectrum it
    # was supposed to be missing.
    m, _groups = _attn_model()
    m.eval()
    x = torch.randn(4, 200)
    pm = np.ones((4, 10), bool)
    pm[1] = False                                                     # row 1 fully missing
    with torch.no_grad():
        o1 = m(x, torch.from_numpy(pm))
        x2 = x.clone()
        x2[1] = torch.randn(200) * 10 + 50                           # perturb the missing row's raw values
        o2 = m(x2, torch.from_numpy(pm))
    assert torch.allclose(o1[1], o2[1], atol=1e-5), "fully-missing row leaked raw x into its output"
    assert torch.isfinite(o1).all()


# ============== reliability.py : conformal_risk_control (R3) + RC ties (R2) + AURC (R1) ==============
from bandsim.reliability import (conformal_risk_control, risk_coverage_curve,  # noqa: E402
                                 aurc, augrc)


def test_crc_all_wrong_keeps_at_most_alpha_joint_mass():
    # R3/CRC: the loss is the JOINT "confidently-wrong" mass  L = 1{conf >= lambda AND wrong}.  So when
    # EVERYTHING is wrong, CRC keeps only the most-confident ~alpha fraction (so E[kept AND wrong] <=
    # alpha) -- it does NOT keep everything, and it does NOT need to abstain entirely. Calib and eval
    # are the SAME sample here, so both the certificate and its realization must clear alpha; both
    # are asserted, because a regression that returns one under the other's name would otherwise
    # pass unnoticed in exactly this in-distribution case.
    cc = np.zeros(200, bool)
    conf = np.random.default_rng(0).uniform(0, 1, 200)
    out = conformal_risk_control(cc, conf, cc, conf, alpha=0.10)
    assert out["feasible"] is True
    assert out["calib_crc_bound"] <= 0.10 + 1e-9      # the CRC certificate P(kept AND wrong) <= alpha
    assert out["eval_group_joint_risk"] <= 0.10 + 1e-9      # and the realized joint mass clears it too
    assert out["eval_group_coverage"] <= 0.10 + 0.02  # keeps ~alpha of the data, not all of it
    assert out["eval_group_selective_risk"] == 1.0    # descriptive: everything KEPT is wrong


def test_crc_infeasible_when_too_few_calibration_units():
    # R3: with n=3 and alpha=0.1, even the abstain-all floor B/(n+1)=0.25 exceeds alpha -> the risk is
    # NOT certifiable, and `feasible` must report that instead of silently returning a bogus threshold.
    cc = np.array([True, False, True])
    conf = np.array([0.9, 0.8, 0.7])
    out = conformal_risk_control(cc, conf, cc, conf, alpha=0.10)
    assert out["feasible"] is False


def test_crc_certifies_in_distribution():
    rng = np.random.default_rng(5)
    n = 4000
    conf = rng.uniform(0, 1, n)
    correct = rng.uniform(0, 1, n) < conf                            # higher conf -> more likely correct
    h = n // 2
    out = conformal_risk_control(correct[:h], conf[:h], correct[h:], conf[h:], alpha=0.10)
    assert out["feasible"] is True
    assert out["calib_crc_bound"] <= 0.10                            # the certificate: exact, no slack
    assert out["eval_group_joint_risk"] <= 0.10 + 0.03                     # realized on a DISJOINT eval half
    #                                                                  (sampling slack: a realization
    #                                                                  is not itself bounded per-run)
    assert out["eval_group_coverage"] > 0.0


def test_risk_coverage_is_permutation_invariant_under_ties():
    # R2: identical confidences -> evaluate at UNIQUE thresholds, so the curve cannot depend on the
    # arbitrary order among tied samples (the old argsort broke this).
    correct = np.array([1, 0, 1, 0, 1.0])
    conf = np.full(5, 0.5)
    c1, r1 = risk_coverage_curve(correct, conf)
    p = [3, 1, 4, 0, 2]
    c2, r2 = risk_coverage_curve(correct[p], conf[p])
    assert np.array_equal(c1, c2) and np.allclose(r1, r2)


def test_aurc_singleton_equals_its_own_risk():
    # R1: a single sample -> AURC is that sample's selective risk (1.0 wrong / 0.0 right), not the
    # degenerate single-point trapezoid that returned 0 for a wrong sample.
    assert abs(aurc(np.array([0.0]), np.array([0.9])) - 1.0) < 1e-9
    assert abs(aurc(np.array([1.0]), np.array([0.9])) - 0.0) < 1e-9
    assert augrc(np.array([0.0]), np.array([0.9])) <= aurc(np.array([0.0]), np.array([0.9])) + 1e-9


# ===================== metrics.py : shape guard + out-of-range drop + kappa degeneracy =====================
from bandsim.metrics import overall_accuracy, confusion, cohen_kappa  # noqa: E402


def test_metric_rejects_broadcastable_shape_mismatch():
    # _check_pair must reject (2,1) vs (2,) BEFORE ravel -- otherwise numpy broadcasts to a 2x2
    # comparison and reports a nonsense accuracy. (This caught a self-inflicted incomplete fix.)
    with pytest.raises((ValueError, AssertionError)):
        overall_accuracy(np.zeros((2, 1), int), np.zeros((2,), int))


def test_confusion_drops_out_of_range_labels():
    # labels outside [0, K) (e.g. an ignore index) are DROPPED, not wrapped onto the last class.
    yt = np.array([0, 1, 2, 7])
    yp = np.array([0, 1, 2, 1])
    C = confusion(yt, yp, num_classes=3)
    assert C.shape == (3, 3) and C.sum() == 3                        # the label-7 sample dropped (total 3, not 4)


def test_cohen_kappa_single_class_is_nan():
    # all truth+pred one class -> expected agreement pe=1 -> 0/0 -> NaN (not a spurious 0 or 1).
    yt = np.zeros(10, int)
    yp = np.zeros(10, int)
    assert np.isnan(cohen_kappa(yt, yp, num_classes=3))


# ===================== grouping.py : degenerate-grouping guards =====================
def test_contiguous_groups_rejects_degenerate_counts():
    with pytest.raises(ValueError):
        contiguous_groups(200, 0)                                    # was ZeroDivision
    with pytest.raises(ValueError):
        contiguous_groups(10, 20)                                    # more groups than bands


def test_group_center_rejects_empty_group():
    with pytest.raises(ValueError):
        group_center_wavelengths(np.linspace(400, 2500, 200), [np.array([], int)])


# ===================== noise.py : striping on non-3D cubes (broadcast fix) =====================
from bandsim.noise import add_striping  # noqa: E402


def test_add_striping_handles_2d_and_4d_cubes():
    # the striping gain was broadcast with a hard-coded 3-D shape; [1]*ndim makes 2-D/4-D work.
    rng = np.random.default_rng(0)
    for shape, cax in [((30, 12), 0), ((3, 4, 5, 6), 2)]:
        out = add_striping(np.ones(shape), rng, stripe_eps=0.02, dead_col_frac=0.1, col_axis=cax)
        assert out.shape == shape and np.isfinite(out).all()


# ===================== srf.py : trapezoid quadrature on nonuniform grid + name preservation =====================
from bandsim.srf import gaussian_srf, build_resampling_matrix, apply_srf  # noqa: E402


def test_srf_preserves_flat_spectrum_on_nonuniform_grid():
    # trapezoid dλ weighting -> each SRF row integrates to 1 even on a NON-uniform axis, so a flat
    # spectrum maps to itself. A plain r/r.sum() over unequal Δλ would bias this.
    wl = np.sort(np.concatenate([np.linspace(400, 1000, 60), np.linspace(1010, 2500, 40)]))
    srf = gaussian_srf(wl, [665, 1610], fwhm_nm=40.0)
    W, _ = build_resampling_matrix(wl, srf)
    assert np.allclose(W.sum(1), 1.0, atol=1e-6)
    assert np.allclose(apply_srf(np.full((3, wl.size), 0.42), W), 0.42, atol=1e-6)


def test_gaussian_srf_preserves_given_band_names():
    # real sensor identity (B8A/B9) must survive, not be auto-renamed to B1..BN.
    wl = np.linspace(400, 2500, 200)
    srf = gaussian_srf(wl, [842, 865, 945], fwhm_nm=20.0, names=["B8", "B8A", "B9"])
    _, names = build_resampling_matrix(wl, srf)
    assert names == ["B8", "B8A", "B9"]


# ===================== io.py : AVIRIS axis has the real water-vapour gaps =====================
def test_aviris_axis_has_real_water_vapour_gaps():
    from bandsim.io import AVIRIS_WL_NM
    d = np.diff(AVIRIS_WL_NM)
    assert AVIRIS_WL_NM.size == 200
    assert not np.allclose(d, d[0])                                  # NOT a gapless linspace
    big = d[d > 3 * np.median(d)]
    assert big.size >= 2 and big.max() > 100                        # two removed-band holes (~1378-1436, ~1819-1963 nm)


# ===================== config_runner.py : Sentinel-2 band identity (P0-3) + scope validation =====================
def test_config_srf_preserves_sentinel2_band_identity():
    from bandsim.config_runner import build_cfg
    from bandsim.io import AVIRIS_WL_NM
    spec = {"input": {"wavelength_axis": "aviris"},
            "designs": {"A": {"enable": True, "target_sensor": "sentinel2"}}}
    cfg = build_cfg(spec, AVIRIS_WL_NM)
    _, names = build_resampling_matrix(AVIRIS_WL_NM, cfg["A"]["srf"])
    assert "B8A" in names and "B9" in names                          # not collapsed to B1..B12


def test_config_runner_warns_on_unconsumed_fields():
    import warnings
    from bandsim.config_runner import validate_scope
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_scope({"input": {"wavelength_axis": "aviris", "path": "x.mat"}, "eval": {"seeds": 5}})
    msgs = " ".join(str(x.message) for x in w)
    assert "does NOT load data" in msgs and "does NOT run evaluation" in msgs
