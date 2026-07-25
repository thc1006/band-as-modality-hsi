"""Guard P2.auto_bs — the batch-size scaler the campaign's correctness argument rests on.

auto_bs lets a training function pick a batch proportional to the dataset instead of P2's fixed
256. Two properties it MUST keep, because other things silently depend on them:

  1. THE FLOOR. Every real dataset in this repo below ~76k pixels must map to exactly 256. That is
     the whole reason phases 1/2/3/4/4R/6/7/9 stayed byte-identical when phase8/8R/8D switched to
     auto_bs: those phases never call it, but the ones that share P2.train_mlp with them would
     change the shared default's behaviour if the floor ever moved. A regression here re-optimises
     every historical result without a single test going red.

  2. DETERMINISM. It reads only n_train; no RNG, no clock. Same size -> same batch, so a resumed or
     re-run experiment trains identically.

The scaling and cap are guarded too, but the floor is the load-bearing one.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)

import phase2_degradation as P2  # noqa: E402


# The actual per-seed training sizes every non-auto_bs phase hands P2.train_mlp. All of them floor
# to 256, so auto_bs would be a no-op on them even if they adopted it -- which is the belt to the
# suspenders of "they don't call it at all". Writing this test is what forced the true floor
# boundary to be computed rather than hand-waved: nearest-power-of-two rounding keeps n at 256 up
# to ~72,400 (n/200 < 2**8.5), NOT up to 51,200 as a first guess assumed. Every real size here is
# well below that; a made-up "76,000 upper bound" that a draft of this test carried actually maps
# to 512, which is the bug the test caught -- in the test's premise, not in auto_bs.
REPO_SMALL_SIZES = [
    9_600,     # phase8 --smoke subsample
    10_249,    # Indian Pines, one checkerboard offset (~half of 21k labelled)
    21_025,    # Indian Pines, full labelled
    42_776,    # Pavia University, one block-split train fold
    49_549,    # EMIT land-cover (phase8G)
    54_129,    # Salinas, full labelled -- the largest single-scene train fold in the repo
]


@pytest.mark.parametrize("n", REPO_SMALL_SIZES)
def test_floor_holds_for_every_small_repo_dataset(n):
    # THE invariant: below the floor's reach, auto_bs is indistinguishable from the old bs=256, so
    # phases that never adopted it are provably unchanged.
    assert P2.auto_bs(n) == 256, (
        f"auto_bs({n})={P2.auto_bs(n)} != 256 -- a small-data phase would re-optimise silently")


def test_floor_boundary_is_where_the_rounding_puts_it():
    # The exact 256->512 edge, found by binary search rather than by hand -- two hand estimates
    # (51,200, then 72,408) were both wrong because the boundary is set by INTEGER floor division
    # (n // 200) feeding nearest-power-of-two rounding: the flip is at raw==363, i.e. n==72,600
    # (363*200), where log2(363)=8.504 rounds to 9. Pinned so any change to target_steps or the
    # rounding rule fails here instead of silently moving which datasets floor.
    assert P2.auto_bs(72_599) == 256
    assert P2.auto_bs(72_600) == 512


def test_large_datasets_scale_to_the_campaign_values():
    # The two sizes the campaign actually ran, and the values its provenance recorded.
    assert P2.auto_bs(720_000) == 4096      # phase8R subsample
    assert P2.auto_bs(600_000) == 4096      # phase8D train split
    assert P2.auto_bs(2_037_600) == 8192    # phase8 per-seed subsample


def test_targets_roughly_200_steps_per_epoch():
    for n in (300_000, 1_000_000, 3_000_000):
        steps = -(-n // P2.auto_bs(n))      # ceil division
        assert 100 <= steps <= 400, f"n={n}: {steps} steps/epoch is outside the ~200 target band"


def test_always_a_power_of_two():
    for n in (1, 100, 51_200, 600_000, 2_000_000, 10 ** 9):
        bs = P2.auto_bs(n)
        assert bs & (bs - 1) == 0, f"auto_bs({n})={bs} is not a power of two"


def test_cap_bounds_the_largest_datasets():
    # Even an absurd dataset cannot ask for an unbounded batch that would not fit or would starve
    # the optimiser of steps.
    assert P2.auto_bs(10 ** 12) == 32768


def test_deterministic_no_hidden_state():
    # Same size, many calls, identical answer -- no RNG, no clock, so a re-run trains identically.
    assert len({P2.auto_bs(123_456) for _ in range(50)}) == 1


@pytest.mark.parametrize("bad", [0, -1, -1000])
def test_rejects_nonpositive_sizes(bad):
    # An empty or negative training set is a caller bug; auto_bs must name it, not silently return
    # a batch for a set that cannot be trained on.
    with pytest.raises(ValueError):
        P2.auto_bs(bad)


def test_floor_and_cap_are_configurable_but_default_sane():
    # The knobs exist for callers with a different step budget, but the defaults are what the
    # campaign relied on.
    assert P2.auto_bs(10 ** 9, floor=1024) >= 1024
    assert P2.auto_bs(10 ** 9, cap=4096) == 4096
    assert P2.auto_bs(1, floor=256) == 256
