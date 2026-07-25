"""Design D (instrument noise/striping) physics-sanity unit tests (roadmap docs/guide/03 §3).

Guards: per-band SNR curve in [50,500] with VNIR>SWIR; empirical per-band noise std matches
sigma_b = signal/SNR_b; striping produces dead (=0) columns and near-unity live-column gain.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandsim.noise import hyperion_like_snr, add_band_noise, add_striping
from bandsim.io import AVIRIS_WL_NM


def test_snr_curve_range_and_vnir_higher_than_swir():
    wl = AVIRIS_WL_NM
    snr = hyperion_like_snr(wl)
    assert (snr >= 50).all() and (snr <= 500).all()
    assert snr[0] > snr[-1]                     # VNIR ~500 > SWIR ~50


def test_empirical_noise_matches_set_snr():
    wl = AVIRIS_WL_NM
    signal = 1000.0
    cube = np.full((5000, wl.size), signal)     # constant signal -> clean per-band SNR read-out
    snr = hyperion_like_snr(wl)
    noisy = add_band_noise(cube, snr, np.random.default_rng(0))
    emp_std = (noisy - cube).std(axis=0)         # per-band empirical noise std
    expected = signal / snr                       # sigma_b = signal / SNR_b
    ratio = emp_std / expected
    assert np.all((ratio > 0.85) & (ratio < 1.15))   # empirical SNR ~ configured SNR


def test_striping_makes_dead_columns_and_unit_gain():
    cube = np.ones((20, 40, 5))                  # (H, W=40 columns, B)
    out = add_striping(cube, np.random.default_rng(0),
                       stripe_eps=0.02, dead_col_frac=0.25, col_axis=1)
    col_mean = out.mean(axis=(0, 2))             # (W,) per-column mean gain
    dead = col_mean == 0
    assert dead.sum() > 0                         # forced 25% dead -> some columns zeroed
    live = col_mean[~dead]
    assert np.all(np.abs(live - 1.0) < 0.15)      # live columns ~ N(1, eps)


# ===================== parameter validation + auditability (Design D) =====================
import pytest  # noqa: E402


def test_negative_stripe_eps_is_rejected_not_silently_abs():
    """eps is the STD of the column gain; a negative one is meaningless and used to be abs()'d.

    stripe_eps=-0.2 previously returned results bit-identical to +0.2, so a sign error in a config
    produced a plausible number rather than a complaint."""
    cube = np.ones((4, 20, 5))
    with pytest.raises(ValueError, match="stripe_eps"):
        add_striping(cube, np.random.default_rng(0), stripe_eps=-0.2)


def test_non_finite_parameters_are_rejected():
    # NaN/Inf used to flow straight into rng.normal and return a non-finite cube with NO error.
    cube = np.ones((4, 20, 5))
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="stripe_eps"):
            add_striping(cube, np.random.default_rng(0), stripe_eps=bad)
        with pytest.raises(ValueError, match="dead_col_frac"):
            add_striping(cube, np.random.default_rng(0), dead_col_frac=bad)
    for bad in (-0.01, 1.5):
        with pytest.raises(ValueError, match="dead_col_frac"):
            add_striping(cube, np.random.default_rng(0), dead_col_frac=bad)


def test_zero_eps_is_still_allowed():
    # eps=0 is a legitimate "no striping gain" control and must not be swept up by the guard
    out = add_striping(np.ones((3, 8, 2)), np.random.default_rng(0), stripe_eps=0.0,
                       dead_col_frac=0.0)
    assert np.allclose(out, 1.0)


def test_add_band_noise_rejects_non_positive_or_non_finite_snr():
    cube = np.ones((10, 4))
    for bad in ([500.0, 0.0, 100.0, 50.0],            # /0 -> inf sigma
                [500.0, -100.0, 100.0, 50.0],         # sign-flipped noise
                [500.0, np.nan, 100.0, 50.0]):
        with pytest.raises(ValueError, match="snr_per_band"):
            add_band_noise(cube, np.array(bad), np.random.default_rng(0))


def test_dead_columns_are_reported_not_assumed():
    """dead_col_frac is an EXPECTED fraction (independent Bernoulli per column), so the realised
    count varies per seed and must be recorded for a run to be auditable."""
    out, info = add_striping(np.ones((4, 12, 3)), np.random.default_rng(0), stripe_eps=0.02,
                             dead_col_frac=1.0, return_info=True)
    assert info["dead_col_count"] == 12 and info["dead_col_indices"] == list(range(12))
    assert info["dead_col_frac_requested"] == 1.0 and info["dead_col_frac_realised"] == 1.0
    assert info["n_cols"] == 12 and info["stripe_eps"] == 0.02 and np.allclose(out, 0.0)

    _, none = add_striping(np.ones((4, 12, 3)), np.random.default_rng(0), dead_col_frac=0.0,
                           return_info=True)
    assert none["dead_col_count"] == 0 and none["dead_col_indices"] == []

    # the reported indices must be exactly the columns that are actually zero
    out2, mid = add_striping(np.ones((4, 40, 3)), np.random.default_rng(7), stripe_eps=0.02,
                             dead_col_frac=0.3, return_info=True)
    zeroed = np.flatnonzero(np.all(out2 == 0.0, axis=(0, 2)))
    assert mid["dead_col_indices"] == zeroed.tolist()
    assert mid["dead_col_count"] == zeroed.size


def test_add_striping_does_not_write_through_its_input():
    """Load-bearing since phase 4 started sharing ONE noised cube across D1a, D1b and all four D2
    levels: an in-place write would make each level corrupt the next, and the sweep would still
    look monotone while measuring something else entirely."""
    cube = np.random.default_rng(0).uniform(1.0, 9.0, (4, 40, 6))
    before = cube.copy()
    for kw in ({"stripe_eps": 0.2, "dead_col_frac": 0.3},
               {"stripe_eps": 0.0, "dead_col_frac": 0.1, "dead_col_mode": "exact"},
               {"stripe_eps": 0.05, "dead_col_frac": 1.0, "return_info": True}):
        add_striping(cube, np.random.default_rng(3), **kw)
        assert np.array_equal(cube, before), f"add_striping wrote through its input with {kw}"


def test_dead_col_mode_defaults_to_bernoulli_and_says_so_in_info():
    """The default must stay Bernoulli: bandsim.pipeline and every existing result depend on it,
    so 'exact' is opt-in only. `info` names the mode because (requested, realised) means two
    different things depending on it."""
    _, info = add_striping(np.ones((2, 145, 3)), np.random.default_rng(0), stripe_eps=0.0,
                           dead_col_frac=0.03, return_info=True)
    assert info["dead_col_mode"] == "bernoulli"
    counts = {add_striping(np.ones((2, 145, 3)), np.random.default_rng(s), stripe_eps=0.0,
                           dead_col_frac=0.03, return_info=True)[1]["dead_col_count"]
              for s in range(20)}
    assert len(counts) > 1, "Bernoulli must still vary across seeds"


def test_exact_mode_realises_the_nominal_fraction_to_the_nearest_column():
    """In 'exact' mode the realised count is a function of (frac, ncols) alone -- the same for every
    seed -- so a sweep axis is ordered by the severity it actually applies. Under Bernoulli it is
    not: on 145 columns the nominal-3% and nominal-5% realisations overlap (1.38%-6.21% vs
    2.76%-7.59% over 5 seeds), so a nominally-worse point can be strictly less corrupted."""
    for frac, expected in ((0.01, 1), (0.03, 4), (0.05, 7), (0.5, 73), (1.0, 145)):
        counts = {add_striping(np.ones((2, 145, 3)), np.random.default_rng(s), stripe_eps=0.0,
                               dead_col_frac=frac, return_info=True,
                               dead_col_mode="exact")[1]["dead_col_count"] for s in range(20)}
        assert counts == {expected}, f"frac={frac}: {counts} != {{{expected}}}"

    out, info = add_striping(np.ones((4, 40, 3)), np.random.default_rng(7), stripe_eps=0.0,
                             dead_col_frac=0.1, return_info=True, dead_col_mode="exact")
    zeroed = np.flatnonzero(np.all(out == 0.0, axis=(0, 2)))
    assert info["dead_col_indices"] == zeroed.tolist() == sorted(zeroed.tolist())
    assert info["dead_col_frac_realised"] == 4 / 40 == info["dead_col_frac_requested"]


def test_exact_mode_sweeps_are_nested_so_a_level_only_adds_dead_columns():
    """Implemented as permutation()[:n], so increasing `frac` ADDS columns instead of resampling
    them and per-seed severity is monotone along the axis. rng.choice(replace=False) would NOT give
    this: its size-4 draw is not a prefix of its size-7 draw from the same seed."""
    for s in range(10):
        sets = [set(add_striping(np.ones((2, 145, 3)), np.random.default_rng(s), stripe_eps=0.0,
                                 dead_col_frac=f, return_info=True,
                                 dead_col_mode="exact")[1]["dead_col_indices"])
                for f in (0.01, 0.03, 0.05)]
        assert sets[0] <= sets[1] <= sets[2], f"seed {s} resampled instead of adding: {sets}"
        assert [len(x) for x in sets] == [1, 4, 7]


def test_exact_mode_refuses_a_nonzero_level_that_applies_no_corruption():
    """A sweep point that rounds to zero dead columns IS the clean baseline, and would be plotted as
    perfect robustness at a nonzero x. Bernoulli mode must NOT raise -- there a zero draw is the
    model, and it is reported rather than rejected."""
    with pytest.raises(ValueError, match="rounds to 0"):
        add_striping(np.ones((2, 145, 3)), np.random.default_rng(0), stripe_eps=0.0,
                     dead_col_frac=0.001, dead_col_mode="exact")
    _, info = add_striping(np.ones((2, 145, 3)), np.random.default_rng(0), stripe_eps=0.0,
                           dead_col_frac=0.001, return_info=True)          # bernoulli: allowed
    assert info["dead_col_count"] in (0, 1)
    # a zero-width axis has nothing to kill; that is not the error above
    add_striping(np.ones((2, 0, 3)), np.random.default_rng(0), dead_col_frac=0.5,
                 dead_col_mode="exact")


def test_unknown_dead_col_mode_raises_instead_of_silently_using_bernoulli():
    """A typo'd or renamed mode falling back to the default is the exact failure this parameter
    exists to prevent, and it would leave no trace in the output."""
    for bad in ("exakt", "Exact", "", None, 0):
        with pytest.raises(ValueError, match="dead_col_mode"):
            add_striping(np.ones((2, 145, 3)), np.random.default_rng(0), dead_col_frac=0.03,
                         dead_col_mode=bad)


def test_expected_fraction_semantics_frequently_yields_no_dead_column():
    """The documented consequence, pinned as a test: at Indian Pines' 145 columns with the default
    frac=0.01, (1-0.01)^145 = 23.3% of seeds contain NO dead column at all. A reader must not assume
    'dead_col_frac=0.01' means 'a dead column was present'."""
    ncols, frac, trials = 145, 0.01, 2000
    empty = sum(add_striping(np.ones((2, ncols, 2)), np.random.default_rng(s), stripe_eps=0.02,
                             dead_col_frac=frac, return_info=True)[1]["dead_col_count"] == 0
                for s in range(trials))
    assert abs(empty / trials - (1 - frac) ** ncols) < 0.05
    assert 0.15 < empty / trials < 0.32          # ~23%, i.e. roughly one seed in four
