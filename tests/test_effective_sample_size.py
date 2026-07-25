"""Standing assertions for the design-effect margin in `conformal_at_risk`.

Every test here pins a property that an adversarial review either proved or measured, so a future
edit that quietly breaks one fails instead of shipping a mis-sized margin. Pure numpy, deterministic,
no torch, no GPU, sub-second.

The defect class this guards: the margin is `z*sqrt(p(1-p)/n)`, a BERNOULLI formula, so the whole
correction hangs on what `n` is. Counting rows assumes rho=0 and under-corrects; counting clusters
assumes rho=1 and over-corrects (measured: it collapsed the operating point's coverage 0.62 -> 0.24).
Nothing about a wrong `n` crashes -- it just silently reports an operating point that misses.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from bandsim.reliability import effective_sample_size, conformal_at_risk  # noqa: E402


def _clustered(G, k, rho, seed=0, p=0.85):
    """(outcome, group) with an intra-cluster correlation of roughly `rho`."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), k)
    shift = rng.normal(0, np.sqrt(rho) * 2.0, G)
    pr = np.clip(p + 0.12 * shift[g], 0.02, 0.98)
    return (rng.random(g.size) < pr).astype(float), g


# --------------------------------------------------------------------- effective_sample_size
@pytest.mark.parametrize("G,k,rho", [(27, 320, 0.0), (27, 320, 0.05), (27, 320, 0.5),
                                     (5, 4, 0.2), (50, 7, 0.9)])
def test_n_eff_is_bracketed_by_the_two_wrong_extremes(G, k, rho):
    """n_clusters <= n_eff <= n_rows. Provable for integer sizes with n > G, and pinned because the
    two bounds ARE the rho=1 and rho=0 answers: leaving the bracket means the design effect is not a
    design effect any more."""
    x, g = _clustered(G, k, rho)
    n_eff, r, k0 = effective_sample_size(x, g)
    assert G - 1e-9 <= n_eff <= x.size + 1e-9
    assert 0.0 <= r <= 1.0
    assert k0 >= 1.0


def test_icc_is_invariant_under_relabelling_correct_as_wrong():
    """rho is estimated on the correctness indicator, but the loss actually controlled is its
    complement. The ANOVA rho is invariant under x -> 1-x, so that substitution costs nothing --
    pinned so nobody 'fixes' it by flipping the variable and changing the margin."""
    x, g = _clustered(27, 200, 0.3)
    a = effective_sample_size(x, g)
    b = effective_sample_size(1.0 - x, g)
    assert np.allclose(a, b)


def test_zero_information_calibration_assumes_the_worst_not_the_best():
    """An all-correct (or all-wrong) calibration set carries NO information about clustering. The
    ANOVA denominator is 0 there, and returning rho=0 would pick n_eff = n_rows -- the maximally
    anti-conservative answer in exactly the case where nothing is known. It must collapse to one
    effective unit per cluster instead."""
    g = np.repeat(np.arange(27), 100)
    for const in (np.ones(g.size), np.zeros(g.size)):
        n_eff, rho, _ = effective_sample_size(const, g)
        assert n_eff == pytest.approx(27), "zero-information calibration must not return n_rows"
        assert rho == 1.0


def test_kish_k0_collapses_under_size_imbalance_and_is_detectable():
    """The failure the phase4R guard checks for. Kish's k0 is dominated by one huge cluster, so a
    skewed split reports a large n_eff while the data holds ~G units. The bracket invariant above
    PASSES here -- which is why it is not the guard."""
    sizes = [8614] + [1] * 26
    g = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    x = np.repeat(np.array([0] + [1] * 26), sizes).astype(float)   # constant within block -> rho=1
    n_eff, rho, k0 = effective_sample_size(x, g)
    mean_size = g.size / len(sizes)
    assert rho == pytest.approx(1.0, abs=1e-6)
    assert k0 < 0.5 * mean_size, "k0 must visibly collapse, so the imbalance guard can see it"
    assert 27 <= n_eff <= g.size, "the bracket still passes -- a tautology, not a guard"


# ------------------------------------------------------------------------- conformal_at_risk
def test_omitting_calib_group_is_byte_identical_to_the_ungrouped_path():
    """The refinement pass is gated on `calib_group`, NOT on `conservative`. Gated the loose way it
    silently moved every caller that does not group (phase8R, phase8E) by ~1pp of coverage while the
    docstring promised byte-identical behaviour."""
    rng = np.random.default_rng(3)
    n = 4000
    corr = (rng.random(n) < 0.8).astype(float)
    conf = np.clip(rng.random(n), 1e-3, 1 - 1e-3)
    out = conformal_at_risk(corr, conf, corr, conf, target_risk=0.10)
    # the ungrouped margin is the plain row-count one, with no refinement applied
    assert out["n_calib_units"] == n
    assert out["margin"] == pytest.approx(np.sqrt(0.10 * 0.90 / n))
    assert out["icc"] == 0.0


def test_reported_n_reconstructs_the_reported_margin():
    """The column exists to be audited. Before this was pinned, `n_calib_units` reported the
    full-set n while `margin` had been resized by the accepted fraction -- they disagreed by ~2-3x,
    which is precisely the quantity the column is for."""
    x, g = _clustered(27, 320, 0.05, seed=7)
    conf = np.clip(0.5 + 0.45 * x + np.random.default_rng(8).normal(0, 0.05, x.size), 0.01, 0.99)
    out = conformal_at_risk(x, conf, x, conf, target_risk=0.10, calib_group=g)
    if out["margin"] > 0:
        implied = 0.10 * 0.90 / out["margin"] ** 2
        assert implied == pytest.approx(out["n_calib_units"], rel=0.02)


@pytest.mark.parametrize("z,n", [(1.0, 27), (1.645, 27), (2.0, 100), (0.5, 27)])
def test_degeneracy_boundary_is_z_squared_over_n_plus_z_squared(z, n):
    """margin >= target  <=>  target <= z^2/(n + z^2). NOT 1/(n+1), which is only the z=1 case --
    and z is a public parameter, so a reviewer tightening it to 1.645 lands inside the degenerate
    region at alpha=0.10 with 27 units."""
    crit = z ** 2 / (n + z ** 2)
    margin = lambda t: z * np.sqrt(t * (1 - t) / n)
    assert margin(crit) == pytest.approx(crit, rel=1e-9)
    assert margin(crit * 0.9) > crit * 0.9          # below the boundary -> degenerate
    assert margin(crit * 1.1) < crit * 1.1          # above it -> a real target survives


def test_degenerate_cell_abstains_and_reports_a_target_it_actually_used():
    """With eff_target <= 0 the selection stops targeting the risk and starts fitting the longest
    error-free calibration prefix -- risk 0.0 at an arbitrary coverage, insensitive to alpha. It must
    abstain, and must not report an eff_target that was never applied."""
    x, g = _clustered(27, 40, 0.6, seed=11)
    conf = np.clip(np.linspace(0.99, 0.5, x.size), 0.01, 0.99)
    out = conformal_at_risk(x, conf, x, conf, target_risk=0.002, calib_group=g)
    assert out["degenerate"] is True
    assert not np.isfinite(out["threshold"])
    assert out["coverage"] == 0.0
    assert np.isnan(out["risk"]), "an empty accepted set has no risk, and 0.0 would read as perfect"
    assert out["eff_target"] == 0.0


def test_refinement_only_ever_raises_the_threshold():
    """Monotone by construction (n_acc <= n_cal so the margin only grows), which is what makes the
    single pass non-oscillating. Grouping must never buy MORE coverage than not grouping."""
    for seed in range(6):
        x, g = _clustered(27, 200, 0.15, seed=seed)
        conf = np.clip(0.55 + 0.4 * x + np.random.default_rng(seed + 50).normal(0, 0.06, x.size),
                       0.01, 0.99)
        plain = conformal_at_risk(x, conf, x, conf, target_risk=0.10)
        grouped = conformal_at_risk(x, conf, x, conf, target_risk=0.10, calib_group=g)
        assert grouped["threshold"] >= plain["threshold"] - 1e-12
        assert grouped["coverage"] <= plain["coverage"] + 1e-12
