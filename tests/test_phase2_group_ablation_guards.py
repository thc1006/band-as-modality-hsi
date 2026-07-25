"""Standing assertions for the Phase 2 group-count sweep.

Every test here pins a defect that produced a WRONG NUMBER WITHOUT CRASHING, so each one is written
to fail if the defect is reintroduced — not merely to exercise the happy path:

  * B4 and B6 were trained at every G and every seed and then excluded from every comparison,
    because the method list was hard-coded as a SUBSET of what run_seed returns.
  * "proposed wins on every seed" was decided against ONE rival chosen by highest mean AUC, which
    does not imply winning against the others.
  * The cross-G summary resampled onto a linspace guarded by "at least as many points as knots",
    which does not make a trapezoid exact — the grid has to CONTAIN the knots.
  * A one-seed spread was published as 0.00 rather than "no spread measurable".
  * The integrity harness overwrote the deliverable with a 1-seed smoke run.

`argparse`-level mistakes are also pinned, because the alternative to rejecting them in milliseconds
is discovering them after a GPU-hour of training.
"""
import argparse
import ast
import inspect
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)

import phase2_group_ablation as GA                                            # noqa: E402
import phase2_degradation as P2                                               # noqa: E402


def _f(x):
    """The convex synthetic response the module docstring quotes numbers against."""
    return 70.0 * np.exp(-2.5 * np.asarray(x, float))


def _reference_mean(frac_max, fracs, y):
    """Independent segment-walk reference for the mean of a piecewise-linear curve on [0, frac_max].

    Deliberately a different SHAPE of implementation from GA.exact_knots (a Python loop over
    segments, not a vectorised knot union), so agreement is evidence rather than a restatement."""
    fracs = np.asarray(fracs, float)
    total, a = 0.0, float(fracs[0])
    while a < frac_max - 1e-15:
        i = int(np.searchsorted(fracs, a, side="right")) - 1
        b = min(float(fracs[i + 1]), frac_max)
        total += 0.5 * (np.interp(a, fracs, y) + np.interp(b, fracs, y)) * (b - a)
        a = b
    return total / (frac_max - float(fracs[0]))


# ---------------------------------------------------------------------------------------------
# 1. every method that gets TRAINED must get COMPARED
# ---------------------------------------------------------------------------------------------
def _run_seed_curve_keys():
    """The literal keys of the `curves = {...}` dict run_seed returns, read out of its source.

    Read statically rather than by calling run_seed, because calling it costs six model trainings.
    That is the whole point: the drift this catches is invisible until someone pays that cost."""
    tree = ast.parse(inspect.getsource(P2.run_seed))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == "curves" for t in node.targets)):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("could not locate run_seed's `curves = {...}` literal")


def test_the_sweep_compares_every_method_run_seed_trains():
    """The list was ["b1","b2","b3","proposed"] while run_seed returned six curves, so the sweep
    paid for B4 and B6 at every G and every seed and then answered the question without them."""
    assert set(GA.EXPECTED_METHODS) == _run_seed_curve_keys()


def test_b4_and_b6_are_in_the_comparison():
    """Named explicitly: they are the CLOSEST architecture ablations to the proposed model, so
    excluding them turned "proposed is best" into "proposed beats the three MLP baselines"."""
    assert {"b4", "b6"} <= set(GA.EXPECTED_METHODS)


# ---------------------------------------------------------------------------------------------
# 2. the verdict must survive a high-variance baseline the mean-rival rule hides
# ---------------------------------------------------------------------------------------------
def test_the_verdict_is_not_decided_by_a_single_data_chosen_rival():
    """Counterexample from the review, reproduced end to end: b2 beats proposed on seed 0, yet the
    retired rule picked b1 as "the strongest baseline" and reported a clean 3/3 sweep."""
    aucs = {"proposed": np.array([10.0, 10.0, 10.0]),
            "b1": np.array([9.0, 9.0, 9.0]),          # mean 9.00  -> chosen as "best rival"
            "b2": np.array([11.0, 0.0, 0.0])}         # mean 3.67  -> ignored, but wins seed 0
    baselines = ["b1", "b2"]

    rival = max(baselines, key=lambda k: aucs[k].mean())                    # the retired rule
    assert rival == "b1", "the mean picks the wrong rival — that is the defect"
    assert int((aucs["proposed"] - aucs[rival] > 0).sum()) == 3, "which reported a clean sweep"

    worst, beats = GA.paired_verdict(aucs, baselines)                       # the rule in force
    assert worst == pytest.approx(-1.0), "worst margin is over ALL baselines and ALL seeds"
    assert beats is False


def test_the_verdict_refuses_to_broadcast_a_short_baseline():
    """numpy broadcasts (1,) against (3,) instead of raising, so a baseline with one seed's AUC
    would silently produce three paired differences against the same number."""
    with pytest.raises(ValueError, match="one value per seed"):
        GA.paired_verdict({"proposed": np.array([1.0, 2.0, 3.0]), "b1": np.array([1.0])}, ["b1"])


def test_the_verdict_refuses_an_empty_baseline_set():
    """min() over no baselines is vacuously a win. A robustness claim against nothing is not one."""
    with pytest.raises(ValueError, match="no baselines"):
        GA.paired_verdict({"proposed": np.array([1.0])}, [])


# ---------------------------------------------------------------------------------------------
# 3. integration must be EXACT, not "resampled onto a grid with enough points"
# ---------------------------------------------------------------------------------------------
def test_the_old_grid_guard_accepts_a_grid_that_misses_every_knot():
    """--groups 6 7 is the counterexample: max_knots is 6, so `--grid 6` passed the retired check,
    and that linspace lands on G=6's knots and misses all of G=7's.

    The consequence is not a rounding error. G=7 summarises 0.38 mIoU too high, and the G6-vs-G7
    gap comes out -0.270 when it is in truth +0.111 — the misalignment can INVERT a cross-G
    statement, which is the whole output of this experiment."""
    s6, s7 = np.arange(0, 6) / 6, np.arange(0, 7) / 7
    fmax = 5.0 / 6.0
    max_knots = max(int(np.sum(fr <= fmax + 1e-12)) for fr in (s6, s7))
    assert max_knots == 6, "the retired guard's own threshold"
    bad = np.linspace(0.0, fmax, max_knots)                     # ...which it therefore accepted

    g6, g7 = GA.frac_auc(bad, s6, _f(s6)), GA.frac_auc(bad, s7, _f(s7))
    e6, e7 = GA.frac_auc_exact(fmax, s6, _f(s6)), GA.frac_auc_exact(fmax, s7, _f(s7))

    assert g7 == pytest.approx(30.11112061, abs=1e-6), "the number the grid reported"
    assert e7 == pytest.approx(29.72961970, abs=1e-6), "the number the curve actually implies"
    assert g7 - e7 == pytest.approx(0.38150092, abs=1e-6)
    assert (g6 - g7) < 0 < (e6 - e7), "the cross-G gap changes SIGN, not just magnitude"

    for fr, exact in ((s6, e6), (s7, e7)):
        assert exact == pytest.approx(_reference_mean(fmax, fr, _f(fr)), abs=1e-12)


def test_the_integrator_change_moves_no_previously_published_default_number():
    """The default {5,10,20} sweep's linspace happens to contain every knot (every fraction is a
    multiple of 0.025), so grid and exact agree bit for bit. Without this, "we fixed the integrator"
    would be indistinguishable from "we changed the results"."""
    grid = np.linspace(0.0, 0.5, 21)
    for G, fr in ((5, np.arange(0, 5) / 5), (10, np.arange(0, 7) / 10), (20, np.arange(0, 11) / 20)):
        g = GA.frac_auc(grid, fr, _f(fr))
        e = GA.frac_auc_exact(0.5, fr, _f(fr))
        assert g == pytest.approx(e, abs=1e-12), f"G={G} default path must not move"
        assert e == pytest.approx(_reference_mean(0.5, fr, _f(fr)), abs=1e-12)


def test_a_flat_curve_summarises_to_its_own_level():
    """audc already divides by the x-range, so this is a MEAN in [0,100]. main() divided by the
    range a second time and a flat mIoU=42 curve came back as 84."""
    fr = np.arange(0, 11) / 20
    assert GA.frac_auc_exact(0.5, fr, np.full(11, 42.0)) == pytest.approx(42.0, abs=1e-12)


def test_retention_auc_is_the_integral_of_the_retention_curve():
    """Production computes retention as 100*absolute/clean. That is an identity, not an
    approximation (clean is a per-seed constant and integration is linear) — pin it, because a
    future "let me integrate it properly" rewrite must land on the same number."""
    fr = np.arange(0, 11) / 20
    x, y = GA.exact_knots(0.5, fr, _f(fr))
    direct = GA.audc(x, y / y[0]) * 100.0
    ratio = 100.0 * GA.frac_auc_exact(0.5, fr, _f(fr)) / _f(fr)[0]
    assert direct == pytest.approx(ratio, abs=1e-9)


def test_seed_summary_matches_the_direct_retention_integral():
    """The production path computes retention as a ratio; this pins it against the integral it
    claims to equal. A swapped ratio (clean/absolute) reads as a plausible 175% instead of 57%."""
    fr = np.arange(0, 11) / 20
    y = _f(fr)
    abs_auc, clean, ret = GA.seed_summary(0.5, fr, y)
    x, yy = GA.exact_knots(0.5, fr, y)
    assert abs_auc == pytest.approx(GA.audc(x, yy), abs=1e-12)
    assert clean == pytest.approx(float(y[0]), abs=1e-12)
    assert ret == pytest.approx(GA.audc(x, yy / yy[0]) * 100.0, abs=1e-9)
    assert 0.0 < ret < 100.0, "a decreasing curve retains SOME but not all of its clean score"


def test_seed_summary_rejects_a_model_that_scores_zero_with_nothing_missing():
    """retention divides by the clean score, so a collapsed run would otherwise emit inf/NaN."""
    fr = np.arange(0, 11) / 20
    with pytest.raises(ValueError, match="has not trained"):
        GA.seed_summary(0.5, fr, np.zeros(11), who="G=10 seed=0 proposed")


@pytest.mark.parametrize("who,kw", [
    ("zero width", dict(frac_max=0.0)),
    ("past the measured range", dict(frac_max=0.9)),
])
def test_a_degenerate_range_raises_instead_of_returning_a_plausible_number(who, kw):
    """A zero-width range used to divide by zero and write `inf` for every method at every G; a
    range past the last measured point makes np.interp extrapolate FLAT, reporting robustness at
    fractions that were never evaluated."""
    fr = np.arange(0, 11) / 20
    with pytest.raises(ValueError):
        GA.frac_auc_exact(fracs=fr, curve=_f(fr), **kw)


def test_a_reordered_measured_axis_raises_instead_of_returning_54_5():
    """np.interp does not enforce an increasing xp. The same four measured pairs, merely reordered,
    summarised to 54.5 instead of 53.5 — a normal-looking number, straight into the CSV."""
    with pytest.raises(ValueError, match="STRICTLY INCREASING"):
        GA.frac_auc_exact(0.5, np.array([0.0, 0.4, 0.2, 0.5]), np.array([70.0, 45.0, 55.0, 40.0]))


def test_a_curve_of_the_wrong_length_raises():
    fr = np.arange(0, 11) / 20
    with pytest.raises(ValueError, match="shape mismatch"):
        GA.frac_auc_exact(0.4, fr, _f(fr)[:-1])


def test_a_non_finite_curve_point_inside_the_range_raises():
    """Belt and braces: `bandsim.metrics.audc` also refuses non-finite input, so this one is caught
    twice. The test below is the case only _check_curve can catch."""
    fr = np.arange(0, 11) / 20
    y = _f(fr).copy(); y[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        GA.frac_auc_exact(0.4, fr, y)


def test_a_non_finite_point_ABOVE_the_common_range_still_raises():
    """The case the downstream guard structurally cannot see, found by mutation testing.

    A knot above the common maximum is dropped from the integration set, so its NaN never reaches
    audc: the summary comes out clean and an evaluation that blew up at m=6 is reported as a healthy
    curve. That matters here specifically because the common maximum is a MIN over G — every G
    except the most restrictive one is integrated over less than it measured, so this is the normal
    case, not a corner. _check_curve looks at the whole measured curve instead."""
    fr = np.arange(0, 7) / 10                      # G=10 measures out to 0.6
    y = _f(fr).copy(); y[6] = np.nan               # ...and the point at 0.6 is broken
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])   # the knots a 0..0.5 common range integrates
    assert np.isfinite(np.interp(x, fr, y)).all(), "the NaN provably never reaches audc"
    with pytest.raises(ValueError, match="finite"):
        GA.frac_auc_exact(0.5, fr, y)


# ---------------------------------------------------------------------------------------------
# 4. argument mistakes must cost milliseconds, not a GPU-hour
# ---------------------------------------------------------------------------------------------
def _args(**kw):
    base = dict(seeds=[0, 1, 2], groups=[5, 10, 20], epochs=60, trials=8, jobs=None, grid=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("kw,match", [
    (dict(seeds=[]), "at least one seed"),
    (dict(groups=[]), "at least one group"),
    (dict(seeds=[0, 0, 1]), "must be unique"),
    (dict(groups=[5, 10, 10]), "must be unique"),
    (dict(groups=[1, 5]), "every G >= 2"),
    (dict(groups=[0]), "every G >= 2"),
    (dict(epochs=0), "--epochs must be >= 1"),
    (dict(epochs=-3), "--epochs must be >= 1"),
    (dict(trials=0), "--trials must be >= 1"),
    (dict(jobs=0), "--jobs must be >= 1"),
    (dict(grid=21), "--grid was removed"),
])
def test_bad_arguments_are_rejected_up_front(kw, match):
    with pytest.raises(ValueError, match=match):
        GA._validate(_args(**kw))


def test_a_valid_argument_set_passes():
    """The negative control: without it every parametrisation above could be passing for the wrong
    reason (e.g. _validate raising unconditionally)."""
    GA._validate(_args())


def test_validation_runs_before_anything_expensive():
    """`--trials 0` used to be caught inside degradation_curve — correctly, but only after the first
    seed's six models had trained. Ordering is the property; assert it on main()'s own source."""
    tree = ast.parse(inspect.getsource(GA.main))
    first = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", None))
            if name in ("_validate", "setup", "load_data", "run_jobs",
                        "require_class_set_contract"):
                first.setdefault(name, node.lineno)
    assert "_validate" in first, "main() must validate its arguments"
    for expensive in ("setup", "load_data", "run_jobs"):
        assert first["_validate"] < first[expensive], (
            f"_validate must run before {expensive}() — a typo should not cost a training run")
    assert first["require_class_set_contract"] < first["load_data"]


# ---------------------------------------------------------------------------------------------
# 5. a smoke run, or a different sweep, must not be able to overwrite a deliverable
# ---------------------------------------------------------------------------------------------
def _paper_write_paths(module):
    """The f-string handed to every P(...) that reaches a write call, as source text.

    Resolves ONE hop through a local variable (`summary = P(f"...")`, then
    `_atomic_write_csv(summary, ...)`). Without that hop this helper collected only the write sites
    that happen to inline P(...) — one of the three here — so the assertion below passed while two
    thirds of the deliverables went unchecked. A guard that inspects less than it appears to is the
    failure mode this whole file is about."""
    tree = ast.parse(inspect.getsource(module))
    assigned = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "P" and len(node.value.args) == 1):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id] = ast.unparse(node.value.args[0])
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.attr if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", None))
        if name not in ("open", "stamp", "_atomic_write_csv", "savefig", "to_csv"):
            continue
        for arg in list(node.args) + [k.value for k in node.keywords]:
            if (isinstance(arg, ast.Call) and getattr(arg.func, "id", None) == "P"
                    and len(arg.args) == 1):
                out.append(ast.unparse(arg.args[0]))
            elif isinstance(arg, ast.Name) and arg.id in assigned:
                out.append(assigned[arg.id])
    return out


def test_every_paper_artefact_is_named_after_the_sweep_and_the_smoke_flag():
    """Two distinct clobber vectors, one assertion each.

    `{out_tag}` is the one the repo already knows about: without it a --smoke run replaces a real
    deliverable. `{tag}` is the one specific to this script: `--groups 5 10 20` and `--groups 5 10`
    integrate over different common ranges (0..0.5 vs 0..0.6), i.e. they are DIFFERENT ESTIMANDS,
    and both used to be written to one filename."""
    paths = _paper_write_paths(GA)
    assert len(paths) >= 3, (
        f"expected the summary CSV, the raw CSV and the provenance stamp; found {len(paths)}: "
        f"{paths}. A count this low means the collector missed write sites, not that they are safe")
    for p in paths:
        assert "{tag}" in p, f"output name must encode which G were swept: {p}"
        assert "{out_tag}" in p, f"output name must encode --smoke: {p}"


def test_the_integrity_harness_no_longer_targets_the_canonical_deliverable():
    """The harness ran `--seeds 0 --groups 5 10 --epochs 12` and expected the UNSUFFIXED file, so
    running it overwrote the sweep it exists to protect with a 1-seed two-group run."""
    src = open(os.path.join(_ROOT, "experiments", "integrity_check.py")).read()
    entry = [ln for ln in src.splitlines() if "phase2_group_ablation.py" in ln]
    assert entry, "the harness must still cover this phase"
    assert all("--smoke" in ln for ln in entry), "it must invoke the suffix-protected path"
    assert "\"results_phase2_group_ablation.csv\"" not in src, (
        "expecting the canonical name makes the harness write to it")


# ---------------------------------------------------------------------------------------------
# 6. the metric must be the same one, for every method and every seed
# ---------------------------------------------------------------------------------------------
def test_the_contract_rejects_an_eval_proposed_that_ignores_class_set(monkeypatch):
    """Passing class_set into a build where eval_proposed drops it is WORSE than not passing it:
    b1/b2/b3 would be scored on the fixed macro class set and b4/b6/proposed on a drifting one, so
    the within-G paired comparison stops being like-for-like."""
    def eval_proposed_ignoring_it(model, Xte, yte, groups, drop_ids, num_classes=None,
                                  class_set=None):
        return P2.miou(yte, model, num_classes)          # the bare metric — no fixed class set
    monkeypatch.setattr(P2, "eval_proposed", eval_proposed_ignoring_it)
    with pytest.raises(RuntimeError, match="never reads it"):
        GA.require_class_set_contract()


def test_the_contract_accepts_an_eval_proposed_that_uses_miou_over(monkeypatch):
    """Negative control: the check must be satisfiable, or it is just an unconditional failure.

    Every input the contract inspects is stubbed, so this tests the CHECK rather than whatever state
    phase2_degradation happens to be in. It passed for the wrong reason while the worktree carried
    another session's uncommitted class_set work, and would have gone red the moment the branch was
    rebased onto a base without it."""
    def eval_proposed_honouring_it(model, Xte, yte, groups, drop_ids, num_classes=None,
                                   class_set=None):
        return P2.miou_over(yte, model, num_classes, class_set)

    def run_seed_taking_it(seed, cube, gt, n_groups, max_missing, trials, epochs, block=10,
                           class_set=None):
        raise AssertionError("not called")

    monkeypatch.setattr(P2, "eval_proposed", eval_proposed_honouring_it)
    monkeypatch.setattr(P2, "run_seed", run_seed_taking_it)
    monkeypatch.setattr(P2, "common_class_set", lambda *a, **k: ([0, 1], None), raising=False)
    monkeypatch.setattr(P2, "SPLIT_BLOCK", 10, raising=False)
    GA.require_class_set_contract()


def test_the_contract_is_not_satisfied_by_prose_that_merely_mentions_miou_over(monkeypatch):
    """The fix carries a comment explaining itself, and that comment names `miou_over`. A revert
    that removed the call but left the prose would keep passing a naive substring search, so the
    contract strips comment lines before looking."""
    def eval_proposed_with_only_a_comment(model, Xte, yte, groups, drop_ids, num_classes=None,
                                          class_set=None):
        # miou_over, not miou -- prose that outlives the line it describes
        return P2.miou(yte, model, num_classes)

    monkeypatch.setattr(P2, "eval_proposed", eval_proposed_with_only_a_comment)
    with pytest.raises(RuntimeError, match="never reads it"):
        GA.require_class_set_contract()


def test_the_contract_rejects_a_run_seed_that_cannot_take_a_class_set(monkeypatch):
    def run_seed_without_it(seed, cube, gt, n_groups, max_missing, trials, epochs):
        raise AssertionError("not called")
    monkeypatch.setattr(P2, "run_seed", run_seed_without_it)
    with pytest.raises(RuntimeError, match="does not accept class_set"):
        GA.require_class_set_contract()


class _FixedLogits(torch.nn.Module):
    """A model that returns a fixed prediction, so the two eval paths can be compared without
    training anything. Takes `pm` optionally, because eval_mlp calls model(x) and eval_proposed
    calls model(x, present_mask)."""

    def __init__(self, pred, num_classes):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))          # gives .parameters() a device
        self.logits = torch.nn.functional.one_hot(torch.tensor(pred), num_classes).float()

    def forward(self, x, pm=None):
        return self.logits


def test_eval_proposed_and_eval_mlp_score_on_the_same_class_set():
    """The two arms of Phase 2 must be scored with the SAME metric — checked by RUNNING both.

    `eval_proposed` used to take `class_set=` and ignore it, calling the bare `miou()` while
    `eval_mlp` routed through `miou_over(..., class_set)`. B1/B2/B3 were therefore averaged over the
    fixed macro class set and Proposed/B4/B6 over whichever classes each seed's checkerboard split
    happened to contain: two different estimands for the two arms, plus a per-seed drift that
    `common_class_set` exists to remove.

    The worked case below is the mechanism in miniature. Ground truth [0,0,1,1,2] predicted as
    [0,0,1,1,1]: class 0 is perfect (IoU 100), class 1 picks up one false positive (IoU 66.7) and
    class 2 is missed entirely (IoU 0). Restricted to the fixed set {0,1} that is 83.3; averaged
    over every present class it is 55.6. A tiny, hopeless class drags the mean down by 27.8 points
    — which is the direction the old code took for the attention models and not for the MLPs."""
    yte = np.array([0, 0, 1, 1, 2])
    model = _FixedLogits([0, 0, 1, 1, 1], 3)
    Xte = np.zeros((5, 2), np.float32)
    groups = [np.array([0]), np.array([1])]
    wl = np.array([400.0, 500.0])
    common = dict(num_classes=3, class_set=[0, 1])

    over = P2.eval_proposed(model, Xte, yte, groups, [], **common)
    legacy = P2.eval_proposed(model, Xte, yte, groups, [], num_classes=3, class_set=None)
    assert over == pytest.approx(83.333, abs=0.01), "the fixed macro set must be honoured"
    assert legacy == pytest.approx(55.556, abs=0.01), "and must differ from averaging over present"

    mlp_arm = P2.eval_mlp(model, Xte, yte, groups, [], wl, impute=False, **common)
    assert over == pytest.approx(mlp_arm, abs=1e-9), (
        "the proposed arm and the baseline arm must return the same metric for the same class_set")


# ---------------------------------------------------------------------------------------------
# 7. reporting: a one-seed run has no spread, and a killed run leaves no half-written CSV
# ---------------------------------------------------------------------------------------------
def test_seed_spread_is_the_sample_sd_and_is_undefined_for_one_seed():
    """The published CSV read `*_auc_std = 0.0` for a 1-seed run, which is how a smoke run passes
    for a reproducible result."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert GA._sd1(a) == pytest.approx(np.std(a, ddof=1))
    assert GA._sd1(a) != pytest.approx(np.std(a, ddof=0)), "ddof=0 understates the bar"
    assert np.isnan(GA._sd1([7.0])), "one seed has no measurable spread — not 0.0"
    assert GA._fmt(float("nan")) == "", "and it reaches the CSV as an empty cell, not 'nan'"
    assert GA._fmt(1.5) == "1.5000"


def test_a_failed_write_cannot_leave_a_half_written_deliverable(tmp_path):
    """Temp file + os.replace: the destination is either the old file or the whole new one."""
    dst = tmp_path / "out.csv"
    dst.write_text("groups,value\n5,1\n")
    with pytest.raises(ValueError):
        GA._atomic_write_csv(str(dst), ["groups"], [{"groups": 5}, {"groups": 6, "extra": 7}])
    assert dst.read_text() == "groups,value\n5,1\n", "the previous deliverable must survive"


def test_the_atomic_write_lands_the_whole_file(tmp_path):
    dst = tmp_path / "out.csv"
    GA._atomic_write_csv(str(dst), ["g", "v"], [{"g": 5, "v": 1}, {"g": 10, "v": 2}])
    assert dst.read_text().splitlines() == ["g,v", "5,1", "10,2"]
    assert not (tmp_path / "out.csv.tmp").exists(), "the temp file must be renamed, not left behind"
