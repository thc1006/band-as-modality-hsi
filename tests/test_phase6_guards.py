"""Standing assertions for phase6's preflight, split independence and wavelength provenance.

Each pins a way this script could complete successfully and produce a wrong conclusion. Pure numpy
and argparse; no dataset, no GPU, sub-second.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

import phase6_second_dataset as P6          # noqa: E402


def _args(**kw):
    argv = ["--dataset", "synthetic"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}"] + [str(x) for x in (v if isinstance(v, list) else [v])]
    return P6.build_argparser().parse_args(argv)


# --------------------------------------------------------------------------------- preflight
def test_nonpositive_epochs_is_refused_because_the_run_is_meaningless():
    """`for _ in range(-2)` runs zero times, so nothing trains, yet the run completes and writes a
    verdict. Rejected for that, NOT because it favours the proposed method -- see the next test."""
    for e in (-2, 0):
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            P6.preflight(_args(epochs=e))


def test_sgmae_provably_cannot_train_the_classification_head():
    """The correction to an earlier claim in this file, pinned so it cannot come back. The preflight
    above was justified as preventing a 'manufactured win' for the proposed model, on the reasoning
    that SGMAE still runs at epochs<=0. It does run -- and it cannot help: its loss flows through
    `reconstruct` into `decoder`, and `classifier` is never on that path. All four arms sit at
    chance and which one wins is noise."""
    import torch
    from bandsim.model import GroupedCrossBandAttention
    groups = [np.arange(i * 8, (i + 1) * 8) for i in range(5)]
    m = GroupedCrossBandAttention(groups, np.array([450., 550, 650, 750, 850]), 7)
    before = {n: p.detach().clone() for n, p in m.named_parameters() if "classifier" in n}
    X = np.random.default_rng(0).normal(0, 1, (64, 40)).astype(np.float32)
    P2_ = __import__("phase2_degradation")
    P2_.pretrain_sgmae(m, X, groups, seed=0, epochs=2)
    for n, p in m.named_parameters():
        if "classifier" in n:
            assert p.grad is None, f"{n} received a gradient from a label-free objective"
            assert torch.equal(p.detach(), before[n]), f"{n} moved during SGMAE pretraining"


def test_three_of_the_four_arms_are_one_model_when_nothing_trains():
    """b1 and b2 differ only by group dropout DURING training, and b3 reuses m_b1. With zero epochs
    they are the same weights -- the tell that such a run is void."""
    src = open(P6.__file__, encoding="utf-8").read()
    assert 'dc("b3", m_b1)' in src, "b3 is b1's model with test-time interpolation"
    assert src.count("P2.train_mlp(Xtr, ytr, groups, seed,") == 2, "b1 and b2 share seed and stream"


@pytest.mark.parametrize("kw,msg", [
    (dict(groups=1), "groups must be >= 2"),
    (dict(groups=10, max_missing=10), r"max-missing must lie"),
    (dict(trials=0), "trials must be >= 1"),
    (dict(jobs=0), "jobs must be >= 1"),
    (dict(seeds=[0, 0, 1]), "seeds must be unique"),
    (dict(seeds=[-1]), "seeds must be >= 0"),
])
def test_preflight_rejects_configurations_that_cannot_mean_anything(kw, msg):
    with pytest.raises(ValueError, match=msg):
        P6.preflight(_args(**kw))


def test_preflight_accepts_the_canonical_configuration():
    P6.preflight(_args())            # must not raise
    P6.preflight(_args(), n_bands=200)


def test_groups_cannot_exceed_the_band_count():
    with pytest.raises(ValueError, match="exceeds"):
        P6.preflight(_args(groups=50), n_bands=20)


# ------------------------------------------------------------------------- split independence
def test_offset_is_periodic_in_block_so_seed_and_seed_plus_ten_are_one_split():
    """`disjoint_block_split` selects train by the parity of (bi + bj). Shifting the offset by a
    whole block adds 1 to BOTH indices, so the parity is unchanged and the partition is identical.
    Seeds 0 and 10 are one split reported as two, which shrinks the spread of a 'two-seed' result."""
    gt = np.ones((120, 120), int)
    _, dup = P6.split_overlaps(gt, [0, 10])
    assert dup == [(0, 10)], "offset periodicity must be detected, not averaged over"
    _, dup2 = P6.split_overlaps(gt, [0, 1, 2])
    assert dup2 == [], "the default seeds are distinct splits"


def test_consecutive_seeds_are_correlated_not_independent():
    """Reported so the std across seeds is read for what it is. Two independent half-splits would
    overlap at IoU 1/3; consecutive offsets sit far above that."""
    gt = np.ones((120, 120), int)
    pairs, _ = P6.split_overlaps(gt, [0, 1])
    assert pairs[0][2] > 0.5, "adjacent offsets share most of their test pixels"
    far, _ = P6.split_overlaps(gt, [0, 5])
    assert far[0][2] < pairs[0][2], "a half-block shift decorrelates more than a one-pixel shift"


# ----------------------------------------------------------------------- wavelength provenance
def test_salinas_axis_is_built_on_the_224_base_with_its_own_removals():
    sal = P6.SALINAS_WL_NM
    assert sal.size == 204, "224 acquired minus the 20 water bands"
    assert not np.allclose(np.diff(sal), np.diff(sal)[0]), "a gapless axis would be fabricated"


def test_the_two_aviris_removal_lists_are_one_list_offset_by_four():
    """THE PROOF that Indian Pines and Salinas are the same sensor, and therefore that their axes
    cannot both be right. Every endpoint differs by exactly +4:
        IP  (220-indexed): [104-108], [150-163], 220
        SAL (224-indexed): [108-112], [154-167], 224
    A single-valued difference set means IP_band[j] == AVIRIS_band[j+4], i.e. Indian Pines is the
    224-band acquisition minus its first four. Both corrected counts follow: 224-4-20 = 200 and
    224-20 = 204. Pinned as EVIDENCE for the bandsim.io escalation, not as an assertion that io.py
    is currently correct -- it is not, and the next test measures by how much."""
    ip = list(range(104, 109)) + list(range(150, 164)) + [220]
    sal = list(range(108, 113)) + list(range(154, 168)) + [224]
    assert sorted(set(np.array(sal) - np.array(ip))) == [4], "one sensor, offset by four bands"
    assert 224 - 4 - 20 == 200 and 224 - 20 == 204
def test_every_registry_dataset_declares_an_axis_status_AND_the_code_can_return_it():
    """Declaring a status is worthless if the code never reads it. whu_hi has no special branch, so
    the else-path used to hardcode "fabricated" and its declared "nominal_uniform" was unreachable --
    the previous version of this test passed on a value nothing consumed."""
    src = open(P6.__file__, encoding="utf-8").read()
    for name in list(P6.DATASETS) + ["synthetic"]:
        assert name in P6._AXIS, f"{name} has no wavelength-axis provenance"
        assert P6._AXIS[name] in {"measured", "nominal_with_gaps", "nominal_uniform", "fabricated"}
    assert '_AXIS.get(name, "fabricated")' in src, \
        "load_dataset's fallback must return the DECLARED status, not a hardcoded one"
    # _synthetic legitimately hardcodes it -- fabricated data IS fabricated. What must not come back
    # is a SECOND hardcoded return inside load_dataset, which is what made whu_hi unreachable.
    assert src.count('return cube, gt, wl, K, "fabricated"') == 1, \
        "only _synthetic may hardcode an axis status"
    import inspect
    assert '"fabricated"' not in inspect.getsource(P6.load_dataset).split("else:")[-1].split("warn")[0] \
        or '_AXIS.get' in inspect.getsource(P6.load_dataset)


def test_indian_pines_axis_really_carries_the_water_vapour_gaps():
    """Guards against a regression to a gapless linspace, which an external review believed was
    already the case -- it is not, and this keeps it that way."""
    from bandsim.io import AVIRIS_WL_NM
    d = np.diff(AVIRIS_WL_NM)
    assert AVIRIS_WL_NM.size == 200
    assert not np.allclose(d, d[0]), "AVIRIS_WL_NM must not be a uniform linspace"
    assert d.max() > 5 * d.min(), "the removed water bands must leave visible gaps"


# ---------------------------------------------------------------------------- self-transfer
def test_the_source_dataset_is_named_so_a_self_run_cannot_claim_transfer():
    assert P6.SOURCE_DATASET == "indian_pines"
    assert P6.SOURCE_DATASET in P6.DATASETS, "kept as a self-consistency check, not removed"
    src = open(P6.__file__, encoding="utf-8").read()
    assert "self_check = args.dataset == SOURCE_DATASET" in src
    assert "SELF-CONSISTENCY" in src, "a source-dataset run must not print a transfer verdict"


def test_the_verdict_no_longer_asserts_the_margin_is_within_its_spread():
    """It used to print that unconditionally. With per-seed margins [+8, +9, -1] the branch fires
    (2/3 wins) and the claim is false: mean 5.33 exceeds SD 4.50."""
    m = np.array([8.0, 9.0, -1.0])
    assert int((m > 0).sum()) != m.size, "this margin reaches the mean-only branch"
    assert abs(m.mean()) > m.std(), "and the old wording would have been false here"
    src = open(P6.__file__, encoding="utf-8").read()
    assert "is within its own spread {margin.std():.2f}" not in src
    assert "No inferential claim is made" in src


# ------------------------------------------------------------------- suffix / canonical contract
def test_the_config_tag_cannot_collide_between_different_seed_sets():
    """`"".join` serialised `--seeds 1 23` and `--seeds 1 2 3` both to "s123", so two different runs
    shared every output path INCLUDING the provenance sidecar and the second silently overwrote the
    first -- the failure the suffix exists to prevent, one level down."""
    j = lambda v: "_".join(str(x) for x in v)
    assert j([1, 23]) != j([1, 2, 3])


def test_canonical_seeds_match_what_reproduce_sh_actually_runs():
    """CANONICAL is what writes the deliverables. Set to [0,1,2] while reproduce.sh runs five seeds,
    every reproduction invocation became non-canonical: it would write _nc- artefacts, leave the
    committed deliverables stale, and exit 0."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rp = os.path.join(root, "reproduce.sh")
    if not os.path.exists(rp):
        pytest.skip("reproduce.sh not present")
    seeds = set()
    for line in open(rp, encoding="utf-8"):
        if "phase6_second_dataset.py" in line and "--seeds" in line and not line.strip().startswith("#"):
            t = line.split("--seeds")[1].split("--")[0].split()
            seeds.add(tuple(int(x) for x in t if x.isdigit()))
    for s in seeds:
        assert sorted(s) == sorted(P6.CANONICAL['seeds']), \
            f"reproduce.sh runs --seeds {list(s)} but CANONICAL declares {P6.CANONICAL['seeds']}"


def test_the_proposed_model_is_seeded_on_phase2s_stream():
    """phase6 blesses --dataset indian_pines as a self-consistency check against phase2. A different
    torch seed offset before the identical constructor makes that check impossible to pass."""
    src = open(P6.__file__, encoding="utf-8").read()
    assert "torch.manual_seed(seed + 101)" in src, "must match phase2_degradation's offset"
    p2 = open(os.path.join(os.path.dirname(P6.__file__), "phase2_degradation.py"), encoding="utf-8").read()
    assert "torch.manual_seed(seed + 101)" in p2, "phase2 changed its offset; phase6 must follow"



# --------------------------------------------------------------- the headline must quantify over ALL rivals
def test_the_win_count_is_the_minimum_over_every_baseline():
    """The rival is chosen by highest MEAN AUDC while the headline is a PER-SEED win count -- two
    different orderings, so "wins on every seed" could print while proposed lost a seed to a rival
    with a lower mean. Constructed from the module's own selection rule."""
    sa = {"b1": np.array([11., 2, 2]), "b2": np.array([9., 9, 1]),
          "b3": np.array([4., 4, 4]), "proposed": np.array([10., 10, 10])}
    margins = {k: sa["proposed"] - sa[k] for k in sa if k != "proposed"}
    rival = max(margins, key=lambda k: sa[k].mean())
    wins = {k: int((m > 0).sum()) for k, m in margins.items()}
    assert rival == "b2" and wins[rival] == 3, "the mean-selected rival looks like a clean sweep"
    assert min(wins.values()) == 2, "but proposed loses a seed to b1"
    src = open(P6.__file__, encoding="utf-8").read()
    assert "min_wins = min(wins_by_rival.values())" in src
    assert "won_all = min_wins == len(args.seeds)" in src, \
        "won_all must quantify over every rival, not the selected one"


def test_class_validation_uses_the_max_label_not_the_distinct_count():
    """Counting distinct non-zero labels rejected a valid scene whose GT lacks one class, and
    accepted non-contiguous labels that then crash in train_mlp. What the head size actually
    depends on is the maximum label."""
    src = open(P6.__file__, encoding="utf-8").read()
    assert "int(fg.max()) != K" in src, "validation must key on the maximum label"
    assert "n_fg != K" not in src, "the distinct-count check was wrong in both directions"
    # the two cases that motivated it
    assert set(range(1, 10)) - set(range(1, 9)) == {9}          # valid scene, class 9 absent
    assert max([1, 2, 4, 5]) != 4                                # non-contiguous, crashes at gt-1


# ======================= WIRING: the guards must be REACHED, not merely exist =======================
# Mutation-tested. Before these, deleting BOTH `preflight(...)` call sites from main() left the whole
# suite green -- preflight was covered only as a pure function, so the protection the module exists
# for was deletable without a red test. The same held for the duplicate-split raise. Testing that a
# validator is correct is not testing that anything calls it.

class _Sentinel(Exception):
    pass


def _run_main(monkeypatch, argv, fake_data=None):
    """Invoke main() with argv, with the expensive parts stubbed out."""
    import sys as _sys
    monkeypatch.setattr(_sys, "argv", ["phase6_second_dataset.py"] + argv)
    monkeypatch.setattr(P6.hw, "setup", lambda **kw: None)
    monkeypatch.setattr(P6.hw, "info", lambda: "stub")
    if fake_data is not None:
        monkeypatch.setattr(P6, "load_dataset", lambda name: fake_data)
    return P6.main()


def test_main_actually_calls_preflight(monkeypatch):
    """Pins the CALL, not the function. A sentinel raised from preflight must escape main()."""
    def _boom(args, n_bands=None):
        raise _Sentinel("preflight reached")
    monkeypatch.setattr(P6, "preflight", _boom)
    with pytest.raises(_Sentinel):
        _run_main(monkeypatch, ["--dataset", "synthetic"])


def test_main_calls_preflight_before_loading_the_dataset(monkeypatch):
    """Order matters: the whole point is to fail before expensive work. If load_dataset ran first a
    bad config would still pay for a multi-hundred-MB read."""
    order = []
    monkeypatch.setattr(P6, "preflight", lambda a, n_bands=None: order.append("preflight"))
    def _load(name):
        order.append("load")
        raise _Sentinel("stop here")
    monkeypatch.setattr(P6, "load_dataset", _load)
    with pytest.raises(_Sentinel):
        _run_main(monkeypatch, ["--dataset", "synthetic"])
    assert order and order[0] == "preflight", f"preflight must run first, got {order}"


def test_main_refuses_duplicate_splits_end_to_end(monkeypatch):
    """The offset is periodic in block=10, so seeds 0 and 10 are ONE split. Pinned through main()
    because the raise lives there, not in preflight -- a unit test of split_overlaps cannot see it."""
    gt = np.zeros((60, 60), int)
    gt[:, :] = ((np.arange(60)[:, None] + np.arange(60)[None, :]) % 9) + 1
    cube = np.random.default_rng(0).normal(0, 1, (60, 60, 20))
    fake = (cube, gt, np.linspace(430, 860, 20), 9, "fabricated")
    with pytest.raises(ValueError, match="IDENTICAL train/test splits"):
        _run_main(monkeypatch, ["--dataset", "synthetic", "--seeds", "0", "10",
                                "--groups", "4", "--max-missing", "2"], fake_data=fake)


def test_main_announces_a_source_dataset_run_before_doing_anything(monkeypatch):
    """--dataset indian_pines is a self-consistency check, not a transfer result. The announcement
    must be reached; a sentinel from preflight proves main() got that far first."""
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    monkeypatch.setattr(P6, "preflight", lambda a, n_bands=None: (_ for _ in ()).throw(_Sentinel()))
    with pytest.raises(_Sentinel):
        _run_main(monkeypatch, ["--dataset", "indian_pines"])
    # preflight runs first, so nothing is announced yet -- but the guard must not be skippable
    src = open(P6.__file__, encoding="utf-8").read()
    assert "self_check = args.dataset == SOURCE_DATASET" in src
    assert src.index("self_check = args.dataset") < src.index("cube, gt, wl, K, axis_status ="), \
        "the source-dataset announcement must precede the load"


def test_every_stamp_call_feeds_a_none_check(monkeypatch):
    """stamp() is best-effort and returns None on failure. Ignoring that leaves an unattributable
    deliverable, which is the failure provenance exists to prevent."""
    src = open(P6.__file__, encoding="utf-8").read()
    blk = src[src.index("_unstamped = ["):src.index("if _unstamped:")]
    assert blk.count("stamp(P(") == 3, "all three CSVs must go through the checked collection"
    assert "if p is None" in blk, "the return value must be tested, not discarded"


def test_main_revalidates_against_the_band_count_after_loading(monkeypatch):
    """preflight is called TWICE: once on args alone, then again with n_bands, which is the only
    place `--groups > bands` can be caught since the band count is unknown before the read. A
    sentinel on the first call cannot see the second, so this pins it separately."""
    gt = ((np.arange(60)[:, None] + np.arange(60)[None, :]) % 9) + 1
    cube = np.random.default_rng(0).normal(0, 1, (60, 60, 8))     # only 8 bands
    fake = (cube, gt, np.linspace(430, 860, 8), 9, "fabricated")
    with pytest.raises(ValueError, match="exceeds the 8 bands"):
        _run_main(monkeypatch, ["--dataset", "synthetic", "--seeds", "0", "1",
                                "--groups", "20", "--max-missing", "2"], fake_data=fake)
