"""Standing assertions for the Phase 0 synthetic mechanism check.

Each test pins a defect that produced a wrong number, or a wrong CLAIM, without crashing:

  * the third curve was labelled "Proposed (SGMAE imputation + attn)" in the figure, the CSV column,
    the LaTeX macros and STATUS_REPORT — while only two MLPs are trained and it shares its model
    with the second curve;
  * the three configs were scored on three DIFFERENT random mask sets, so the comparison was not
    paired, on a problem where the choice of missing feature alone moves mIoU by 64 points;
  * 12 masks were sampled with replacement, covering ~65% of the 12 single-feature cases;
  * the baseline and dropout arms differed in initial weights AND minibatch order, so the ablation
    was not controlled;
  * `\\audcProp` was defined by this phase AND phase 2 with different meanings;
  * `\\baseSixMS` said "Six" while indexing `max_missing`;
  * the integrity harness pointed at the canonical deliverable.

Names are asserted against the module's DATA (`CONFIGS`, `tex_macros`), never by searching the
source text: a substring guard cannot tell a claim from its own retraction, and this file's
docstrings necessarily quote the labels they retract.
"""
import argparse
import ast
import inspect
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)

import experiment_synthetic_multiseed as P0                                   # noqa: E402
import bandsim.metrics as M                                                   # noqa: E402


# ---------------------------------------------------------------------------------------------
# 1. the third curve must not claim to be something this file never builds
# ---------------------------------------------------------------------------------------------
def test_no_config_claims_sgmae_or_attention():
    """Only two MLPs are trained here. Asserted against CONFIGS — the data that drives the CSV
    header, the LaTeX macro tags and the figure legend — so it cannot be satisfied by prose."""
    banned = ("sgmae", "attn", "attention", "autoencoder", "transformer", "proposed")
    for key, label, _, _, tag in P0.CONFIGS:
        blob = f"{key} {label} {tag}".lower()
        for word in banned:
            assert word not in blob, f"config {key!r} claims {word!r}: {label!r}"


def test_the_third_config_names_the_same_model_as_the_second():
    """It IS the same model — the dropout MLP, evaluated with an imputer. Both labels must say so,
    or the figure implies three trained systems where there are two."""
    labels = {k: lab for k, lab, _, _, _ in P0.CONFIGS}
    assert labels["drop"].startswith("Group-dropout MLP")
    assert labels["impute"].startswith("Group-dropout MLP")
    assert "imputation" in labels["impute"].lower()


def test_only_two_models_are_trained():
    """Structural: run_once must build exactly the two models the labels promise."""
    tree = ast.parse(inspect.getsource(P0.run_once))
    n_train = sum(1 for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "train_mlp")
    assert n_train == 2, f"run_once trains {n_train} models; the labels describe 2"


# ---------------------------------------------------------------------------------------------
# 2. every config must see the same masks
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cap", [P0.ENUMERATION_CAP, 0])
def test_all_configs_are_evaluated_on_identical_masks(cap, monkeypatch):
    """The defect, end to end: three sequential `eval_missing` calls drew from one generator in
    turn, so each config faced a different random mask set. Checked on the raw record, which is the
    artefact a reader would use to verify pairing.

    Run BOTH with enumeration (where pairing is guaranteed twice over, since every config would
    enumerate the same masks even if it drew them separately) and with `ENUMERATION_CAP=0`, which
    forces the sampling branch — the only regime where sharing the mask list is what does the work.
    Without the second case a mutation that re-draws masks per config survives the whole suite."""
    monkeypatch.setattr(P0, "ENUMERATION_CAP", cap)
    world = P0.build_world(20260630)
    _, raw = P0.run_once(0, world, world_seed=20260630, max_missing=1, trials=4, epochs=1)
    by_mask = {}
    for _, key, m, mask, _ in raw:
        by_mask.setdefault((m, mask), set()).add(key)
    assert by_mask, "no raw rows recorded"
    for (m, mask), keys in by_mask.items():
        assert keys == set(P0.KEYS), f"mask {mask!r} at m={m} was only scored for {sorted(keys)}"
    # ...and every config saw the SAME set of masks, not merely the same count.
    per_key = {k: {(m, mask) for _, kk, m, mask, _ in raw if kk == k} for k in P0.KEYS}
    assert len(set(map(frozenset, per_key.values()))) == 1


def test_zero_missing_is_identical_for_the_two_configs_sharing_a_model():
    """With nothing masked the imputer is a no-op, so `drop` and `impute` must coincide exactly.
    The shipped deliverable showed this too (94.88 +/- 3.16 in both columns) and it was read as
    three independent systems agreeing rather than as two curves from one model."""
    world = P0.build_world(20260630)
    curves, _ = P0.run_once(0, world, world_seed=20260630, max_missing=0, trials=1, epochs=1)
    assert curves["drop"][0] == pytest.approx(curves["impute"][0], abs=1e-12)
    assert curves["base"][0] != pytest.approx(curves["drop"][0], abs=1e-12), (
        "base and drop are different models and must not coincide")


# ---------------------------------------------------------------------------------------------
# 3. masks: exhaustive where affordable, unique where sampled
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("m,expected", [(0, 1), (1, 12), (2, 66), (3, 220), (6, 924)])
def test_masks_are_enumerated_not_sampled(m, expected):
    """Sampling 12 masks with replacement covered ~7.8 of the 12 single-feature cases. Every count
    at C=12 fits under ENUMERATION_CAP, so the curve is the exact mean over all masks of that size."""
    masks = P0.masks_for(P0.C, m, trials=12, rng=np.random.default_rng(0))
    assert len(masks) == expected
    assert len(set(masks)) == expected, "enumerated masks must be distinct"
    assert all(len(mask) == m for mask in masks)
    assert all(tuple(sorted(mask)) == mask for mask in masks), "masks must be canonical (sorted)"


def test_sampled_masks_are_distinct_when_the_space_is_too_large_to_enumerate(monkeypatch):
    """The fallback path. `rng.choice(replace=False)` only forbids repeats WITHIN one mask.

    Exercised in a regime where collisions are LIKELY (8 draws from 12 possibilities — a birthday
    problem), not from a space so large that independent draws are distinct by luck. An earlier
    version of this test asked for 40 masks out of C(64,8) and passed with the deduplication
    deleted, which is a guard that guards nothing."""
    monkeypatch.setattr(P0, "ENUMERATION_CAP", 0)          # force the sampling branch
    for seed in range(25):
        masks = P0.masks_for(12, 1, trials=8, rng=np.random.default_rng(seed))
        assert len(masks) == 8, "must return the requested count"
        assert len(set(masks)) == 8, f"seed {seed} returned duplicate masks: {masks}"


def test_sampling_cannot_ask_for_more_masks_than_exist():
    """`min(trials, n_sets)` — otherwise the unique-sampling loop cannot terminate."""
    P0_cap = P0.ENUMERATION_CAP
    try:
        P0.ENUMERATION_CAP = 0
        masks = P0.masks_for(5, 1, trials=99, rng=np.random.default_rng(0))
        assert len(masks) == 5 and len(set(masks)) == 5
    finally:
        P0.ENUMERATION_CAP = P0_cap


def test_the_whole_default_sweep_is_exhaustive():
    total = sum(len(P0.masks_for(P0.C, m, 12, np.random.default_rng(0))) for m in range(7))
    assert total == 2510, "m=0..6 at C=12 is 2510 masks, all under ENUMERATION_CAP"


# ---------------------------------------------------------------------------------------------
# 4. the ablation must be controlled: same init, same minibatch order, augmentation the only change
# ---------------------------------------------------------------------------------------------
class _RecordingShuffle:
    """Wraps a Generator and records every permutation it hands out."""

    def __init__(self, rng, log):
        self._rng, self._log = rng, log

    def permutation(self, n):
        p = self._rng.permutation(n)
        self._log.append(p.copy())
        return p


def _tiny_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (256, P0.C)); y = rng.integers(0, P0.K, 256)
    return X, y


def test_the_two_arms_start_from_identical_weights():
    """`seed+101` vs `seed+202` gave them different initial weights, so the full-band gap carried
    initialisation variance on top of the augmentation being studied."""
    X, y = _tiny_problem()
    init_ss, shuffle_ss, aug_ss = np.random.SeedSequence([20260630, 0]).spawn(3)
    a = P0.train_mlp(X, y, np.random.default_rng(init_ss), np.random.default_rng(shuffle_ss),
                     None, epochs=0)
    b = P0.train_mlp(X, y, np.random.default_rng(init_ss), np.random.default_rng(shuffle_ss),
                     np.random.default_rng(aug_ss), epochs=0)
    for arr_a, arr_b in zip(a, b):
        assert np.array_equal(arr_a, arr_b)


def test_the_augmentation_does_not_perturb_minibatch_order():
    """The subtler half of the same defect: one generator served init, shuffling AND the mask
    draws, so the augmented arm fell out of step with the other's minibatch order from the second
    epoch on — a difference no amount of seeding the START could remove."""
    X, y = _tiny_problem()
    init_ss, shuffle_ss, aug_ss = np.random.SeedSequence([20260630, 0]).spawn(3)
    log_plain, log_aug = [], []
    P0.train_mlp(X, y, np.random.default_rng(init_ss),
                 _RecordingShuffle(np.random.default_rng(shuffle_ss), log_plain), None, epochs=3)
    P0.train_mlp(X, y, np.random.default_rng(init_ss),
                 _RecordingShuffle(np.random.default_rng(shuffle_ss), log_aug),
                 np.random.default_rng(aug_ss), epochs=3)
    assert len(log_plain) == len(log_aug) == 3
    for p, q in zip(log_plain, log_aug):
        assert np.array_equal(p, q), "the augmented arm saw a different minibatch order"


def test_run_once_hands_both_arms_the_same_init_and_shuffle_streams(monkeypatch):
    """The CALL SITE, not the callee. `train_mlp` honouring the streams it is given proves nothing
    if run_once hands the two arms different ones — which is exactly what `seed+101` / `seed+202`
    did, and what a one-word edit here would restore. A generator's `bit_generator.state` before
    any draw identifies the stream, so the spy can compare without consuming anything."""
    seen = []
    real = P0.train_mlp

    def spy(Xs, y, init_rng, shuffle_rng, aug_rng=None, **kw):
        seen.append((repr(init_rng.bit_generator.state), repr(shuffle_rng.bit_generator.state),
                     aug_rng is not None))
        return real(Xs, y, init_rng, shuffle_rng, aug_rng, **kw)

    monkeypatch.setattr(P0, "train_mlp", spy)
    P0.run_once(0, P0.build_world(20260630), world_seed=20260630, max_missing=0, trials=1, epochs=1)
    assert len(seen) == 2, "run_once must train exactly the two arms"
    (init_a, shuf_a, aug_a), (init_b, shuf_b, aug_b) = seen
    assert init_a == init_b, "the two arms start from DIFFERENT initial weights"
    assert shuf_a == shuf_b, "the two arms see DIFFERENT minibatch orders"
    assert (aug_a, aug_b) == (False, True), "exactly one arm may receive the augmentation stream"


def test_two_generators_from_one_spawned_seedsequence_agree():
    """The property the pairing above rests on. If it ever stopped holding, the two arms would
    silently stop being paired while every other test still passed."""
    ss, = np.random.SeedSequence([1, 2]).spawn(1)
    assert np.array_equal(np.random.default_rng(ss).normal(size=32),
                          np.random.default_rng(ss).normal(size=32))


def test_seed_offsets_no_longer_collide_across_runs():
    """`--seeds 0 101` gave seed 0's dropout model and seed 101's baseline the same stream (202)."""
    streams = []
    for seed in (0, 101):
        streams += [tuple(ss.generate_state(4))
                    for ss in np.random.SeedSequence([20260630, seed]).spawn(5)]
    assert len(set(streams)) == len(streams), "two 'independent' streams share a state"


# ---------------------------------------------------------------------------------------------
# 5. one metric in the repo, not two that agree by luck
# ---------------------------------------------------------------------------------------------
def test_the_module_uses_the_audited_metric():
    """The file said "Reuse the project's audited metric instead of a local copy" and then defined
    its own, which handed a free IoU of 1.0 to every class absent from the ground truth."""
    assert P0.miou is M.miou


def test_the_local_metrics_difference_that_made_this_matter():
    """The discriminating case. (A reviewer's example, y_true=[0,0]/y_pred=[0,0], returns 100 from
    BOTH implementations and proves nothing — the free 1.0s have to be outvoted to show up.)"""
    yt, yp = np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])

    def local_miou_as_it_was(y_true, y_pred, k=4):
        ious = []
        for c in range(k):
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            denom = tp + fp + fn
            ious.append(tp / denom if denom > 0 else 1.0)
        return float(np.mean(ious)) * 100.0

    assert local_miou_as_it_was(yt, yp) == pytest.approx(62.50)
    assert P0.miou(yt, yp, 4) == pytest.approx(25.00)


# ---------------------------------------------------------------------------------------------
# 6. LaTeX macro names
# ---------------------------------------------------------------------------------------------
def _macros():
    xs = np.arange(0, 7)
    stats = {k: (np.linspace(90, 40, len(xs)), np.full(len(xs), 1.5)) for k in P0.KEYS}
    audcs = {k: 70.0 for k in P0.KEYS}
    return P0.tex_macros(stats, audcs, n_seeds=5, world_seed=20260630, max_missing=6, n_masks=2510)


def test_every_macro_is_namespaced():
    assert all(name.startswith("ms") for name in _macros()), "an un-namespaced macro can collide"


def test_no_macro_collides_with_phase_2():
    """`\\audcProp` was emitted by BOTH phases with different meanings (91.4 for the Gaussian
    imputer, 56.5 for the real proposed model). `\\newcommand` twice is a LaTeX error; a
    `\\providecommand` workaround prints one phase's number under the other's name."""
    other = os.path.join(_ROOT, "paper", "results_phase2_summary.tex")
    if not os.path.exists(other):
        pytest.skip("phase 2 summary not present in this tree")
    theirs = {ln.split("{\\", 1)[1].split("}", 1)[0]
              for ln in open(other).read().splitlines() if ln.startswith("\\newcommand{\\")}
    assert theirs, "could not parse phase 2's macro names"
    assert not (set(_macros()) & theirs)


def test_no_macro_name_hardcodes_a_count_it_does_not_control():
    """`\\baseSixMS` said Six while indexing `args.max_missing`, so `--max-missing 3` wrote the
    3-missing value into a macro the paper reads as six."""
    names = _macros()
    assert not any("Six" in n for n in names)
    assert "msMaxMissing" in names, "the count must be published rather than implied by a name"
    at_three = P0.tex_macros({k: (np.linspace(90, 40, 7), np.full(7, 1.5)) for k in P0.KEYS},
                             {k: 70.0 for k in P0.KEYS}, 5, 20260630, 3, 100)
    assert at_three["msMaxMissing"] == "3"
    assert at_three["msBaseAtMax"] != names["msBaseAtMax"], "AtMax must follow max_missing"


def test_no_macro_name_says_proposed():
    assert not any("Prop" in n for n in _macros())


# ---------------------------------------------------------------------------------------------
# 7. argument errors cost milliseconds, not a training run
# ---------------------------------------------------------------------------------------------
def _args(**kw):
    base = dict(seeds=[0, 1, 2], world_seed=20260630, max_missing=6, trials=12, epochs=60,
                jobs=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("kw,match", [
    (dict(seeds=[]), "at least one seed"),
    (dict(seeds=[0, 0, 1]), "must be unique"),
    (dict(seeds=[0, -1]), "non-negative"),
    (dict(max_missing=12), r"must be in \[0, 11\]"),
    (dict(max_missing=13), r"must be in \[0, 11\]"),
    (dict(max_missing=-1), r"must be in \[0, 11\]"),
    (dict(trials=0), "--trials must be >= 1"),
    (dict(epochs=0), "--epochs must be >= 1"),
    (dict(jobs=0), "--jobs must be >= 1"),
])
def test_bad_arguments_are_rejected_up_front(kw, match):
    with pytest.raises(ValueError, match=match):
        P0._validate(_args(**kw))


def test_a_valid_argument_set_passes():
    """Negative control: without it every case above could pass for the wrong reason."""
    P0._validate(_args())


def test_validation_runs_before_anything_expensive():
    tree = ast.parse(inspect.getsource(P0.main))
    first = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", None))
            if name in ("_validate", "build_world", "run_jobs"):
                first.setdefault(name, node.lineno)
    assert first["_validate"] < first["build_world"] < first["run_jobs"]


# ---------------------------------------------------------------------------------------------
# 8. deliverables: suffixed, atomic, and with an honest spread
# ---------------------------------------------------------------------------------------------
def _paper_write_paths(module):
    """The f-string handed to every P(...) that reaches a write call, resolving one hop through a
    local variable so the check does not silently cover only the inlined sites."""
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
        if name not in ("open", "stamp", "_atomic_write", "savefig", "to_csv"):
            continue
        for arg in list(node.args) + [k.value for k in node.keywords]:
            if (isinstance(arg, ast.Call) and getattr(arg.func, "id", None) == "P"
                    and len(arg.args) == 1):
                out.append(ast.unparse(arg.args[0]))
            elif isinstance(arg, ast.Name) and arg.id in assigned:
                out.append(assigned[arg.id])
    return out


def test_every_paper_artefact_is_smoke_suffixed():
    paths = _paper_write_paths(P0)
    assert len(paths) >= 4, f"expected summary, raw, tex and figure; found {len(paths)}: {paths}"
    for p in paths:
        assert "{out_tag}" in p, f"a smoke run could overwrite this deliverable: {p}"


def test_the_integrity_harness_no_longer_targets_the_canonical_deliverable():
    """Asserted against integrity_check.PHASES — the list the harness actually runs — not against
    the source text, which also mentions this script inside the comment explaining the fix."""
    import integrity_check as IC
    entries = [e for e in IC.PHASES if "experiment_synthetic_multiseed.py" in e[1]]
    assert len(entries) == 1, "the harness must still cover this phase exactly once"
    _, argv, outputs = entries[0]
    assert "--smoke" in argv, f"harness invokes the unprotected path: {argv!r}"
    assert outputs and all("_smoke" in o for o in outputs), (
        f"harness expects a canonical deliverable and would therefore write one: {outputs}")


def test_seed_spread_is_the_sample_sd_and_is_undefined_for_one_seed():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert P0._sd1(a) == pytest.approx(np.std(a, ddof=1))
    assert P0._sd1(a) != pytest.approx(np.std(a, ddof=0)), "ddof=0 understates the bar"
    assert np.isnan(P0._sd1([7.0])), "one seed has no measurable spread — not 0.0"
    assert P0._fmt(float("nan")) == ""


def test_a_failed_write_cannot_leave_a_half_written_deliverable(tmp_path):
    dst = tmp_path / "out.csv"
    dst.write_text("original\n")

    def boom(f):
        f.write("partial")
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError):
        P0._atomic_write(str(dst), boom)
    assert dst.read_text() == "original\n"


# ---------------------------------------------------------------------------------------------
# 8b. end to end: run the entrypoint and assert on what it WRITES
# ---------------------------------------------------------------------------------------------
def test_the_entrypoint_writes_artefacts_the_integrity_harness_accepts(tmp_path, monkeypatch):
    """Testing the pieces is not enough — the defects that reached `paper/` were about which file
    got written and what was in it. Runs the real main() into a temp directory (smoke config, a
    couple of seconds) and checks the artefacts against the harness's own rules.

    This is the test that caught `world_seed` as a CSV column: integrity_check rejects any cell
    above 1e4 as an out-of-range metric, and 20260630 is one."""
    import integrity_check as IC
    monkeypatch.setenv("BANDSIM_WORKERS", "1")
    monkeypatch.setattr(P0, "P", lambda rel: str(tmp_path / rel))
    (tmp_path / "figs").mkdir()
    monkeypatch.setattr(sys, "argv", ["experiment_synthetic_multiseed.py", "--smoke", "--jobs", "1"])
    assert P0.main() == 0

    summary = tmp_path / "results_multiseed_smoke.csv"
    raw = tmp_path / "results_multiseed_smoke_raw.csv"
    for path in (summary, raw):
        assert path.exists(), f"{path.name} was not written"
        ok, msg = IC.csv_finite_and_sane(str(path))
        assert ok, f"{path.name} would fail the integrity harness: {msg}"
        assert (tmp_path / f"{path.name}.provenance.json").exists(), (
            f"{path.name} is unattributed; the raw evidence carries no world identity without it")
    assert (tmp_path / "figs" / "fig_degradation_multiseed_smoke.pdf").exists()

    # a smoke run must not have created any canonical name
    for canonical in ("results_multiseed.csv", "results_multiseed_raw.csv",
                      "results_multiseed.tex"):
        assert not (tmp_path / canonical).exists(), f"smoke wrote the deliverable {canonical}"

    tex = (tmp_path / "results_multiseed_smoke.tex").read_text()
    names = [ln.split("{\\", 1)[1].split("}", 1)[0]
             for ln in tex.splitlines() if ln.startswith("\\newcommand{\\")]
    assert names and all(n.startswith("ms") for n in names)
    assert "msNSeeds" in names and tex.count("\\newcommand{\\msNSeeds}") == 1
    # n=1 -> the spread is undefined, and must not be printed as a confident 0.0
    assert "$\\pm$\\,n/a" in tex and "$\\pm$\\,0.0" not in tex


# ---------------------------------------------------------------------------------------------
# 9. the imputer
# ---------------------------------------------------------------------------------------------
def test_solve_matches_the_explicit_inverse_it_replaced():
    """Documenting that this changed no number: at the worst condition number this world produces
    the two forms agree to ~1e-13. `solve` is the form that stays right if that stops holding."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (400, P0.C))
    Sigma = np.cov((A - A.mean(0)) / A.std(0), rowvar=False) + 1e-4 * np.eye(P0.C)
    gmean = np.zeros(P0.C)
    xs = rng.normal(0, 1, (16, P0.C))
    for mask in [(0,), (1, 5), (2, 3, 7, 11)]:
        obs = np.array([i for i in range(P0.C) if i not in mask])
        miss = np.array(mask)
        legacy = xs.copy()
        Wc = Sigma[np.ix_(miss, obs)] @ np.linalg.inv(Sigma[np.ix_(obs, obs)])
        legacy[:, miss] = gmean[miss] + (xs[:, obs] - gmean[obs]) @ Wc.T
        assert np.allclose(P0.impute(xs, mask, Sigma, gmean), legacy, atol=1e-10)


def test_imputing_with_nothing_observed_raises():
    Sigma = np.eye(P0.C)
    with pytest.raises(ValueError, match="nothing left to condition on"):
        P0.impute(np.zeros((2, P0.C)), tuple(range(P0.C)), Sigma, np.zeros(P0.C))


def test_non_finite_parameters_raise():
    """argmax over NaN logits returns 0, so every pixel becomes class 0 and the run reports a
    plausible-looking low mIoU. Triggered here through non-finite input, which is what actually
    produces NaN weights."""
    X, y = _tiny_problem()
    X = X.copy(); X[0, 0] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite"):
        P0.train_mlp(X, y, np.random.default_rng(0), np.random.default_rng(1), None, epochs=1)


def test_a_collapsed_but_FINITE_model_is_caught_too():
    """The failure a finiteness check cannot see, and the one this architecture actually has.

    Measured: at lr=1e12 the ReLU units die, every parameter stays finite, the network collapses to
    three of four classes and nothing raises. A guard that only checked `np.isfinite` would have
    looked like coverage of "training failed silently" while catching none of it."""
    dead = (np.zeros((P0.C, P0.H)), np.zeros(P0.H),
            np.zeros((P0.H, P0.K)), np.array([1.0, 0.0, 0.0, 0.0]))
    assert all(np.all(np.isfinite(a)) for a in dead), "the point is that it IS finite"
    X, y = _tiny_problem()
    with pytest.raises(RuntimeError, match="collapsed"):
        P0._assert_learned(dead, X, y, "unit test")


def test_run_once_actually_applies_the_collapse_check_to_every_model():
    """A test of a callee cannot see a deleted CALL SITE. `_assert_learned` being correct is worth
    nothing if run_once stops calling it, and no behavioural test would notice — the curves would
    still be produced, just from a broken model."""
    tree = ast.parse(inspect.getsource(P0.run_once))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_assert_learned"]
    assert calls, "run_once does not check that its models learned anything"
    # ...and it must cover BOTH arms, not just the first one built.
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.comprehension))]
    covered_by_loop = any(
        any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_assert_learned"
            for c in ast.walk(node)) for node in loops)
    assert covered_by_loop or len(calls) >= 2, (
        "only one model is checked; the other could collapse unnoticed")


def test_a_model_that_did_learn_passes():
    """Negative control for the collapse floor: it must be clearable, and a one-epoch run on the
    real world clears it (73.6% against a 35% floor)."""
    base, B_load = P0.build_world(20260630)
    rng = np.random.default_rng(0)
    y = rng.integers(0, P0.K, 2000)
    X = base[y] + rng.normal(0, 1, (2000, P0.R)) @ B_load.T + rng.normal(0, 0.02, (2000, P0.C))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    model = P0.train_mlp(X, y, np.random.default_rng(1), np.random.default_rng(2), None, epochs=3)
    assert P0._assert_learned(model, X, y, "unit test") > 100.0 / P0.K + 10.0
