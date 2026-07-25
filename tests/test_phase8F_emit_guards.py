"""Guards for phase8F_emit: the SUPERSEDED single-granule EMIT experiment, and the function the
LIVE experiment still imports from it.

Two different jobs here, and the split matters:

  1. `recon_error_matrix` is a LIVE dependency — `phase8F_multi.py` imports it for the result the
     paper actually uses. Defects in it change published numbers, so it is guarded hardest. The one
     that mattered most was silent: the output width was sized as `max(max(group)) + 1` rather than
     from X, so a grouping that did not reach the last band returned a NARROWER matrix than the
     per-band uncertainty vector it gets correlated against — every band compared to a different
     band's uncertainty, no error raised.

  2. Everything else in the file is history, and the risk is that a reader takes it for evidence.
     Its docstring claimed to be the "STRONGEST anti-simulation-artifact validation" against an
     "INDEPENDENT physical retrieval uncertainty" and printed a verdict off a hardcoded `mp > 0.2`
     — while its successor measured the sign FLIPPING under a proper spatial split (sahara
     +0.089 -> -0.007). Those claims are guarded as text because that is what they are.

Nothing here trains anything except one tiny model on synthetic data; the suite runs in seconds.
"""
import ast
import os
import sys

import numpy as np
import pytest


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

import phase8F_emit as F                                                  # noqa: E402
from bandsim.grouping import contiguous_groups                            # noqa: E402
from bandsim.model import GroupedCrossBandAttention                       # noqa: E402

_SRC = open(os.path.join(_ROOT, "experiments", "phase8F_emit.py"), encoding="utf-8").read()
_HAS_NPZ = os.path.exists(F.EMIT_NPZ)


def _tiny_model(n_band=12, n_groups=3):
    groups = contiguous_groups(n_band, n_groups)
    cwl = [400.0 + 100 * i for i in range(len(groups))]
    m = GroupedCrossBandAttention(groups, cwl, 2)
    m.eval()
    return m, groups


# ---------------------------------------------------------------------------------------------
# 1 — the LIVE dependency
# ---------------------------------------------------------------------------------------------
def test_phase8F_multi_still_imports_only_recon_error_matrix():
    """Pins the coupling this file is retained for. If phase8F_multi starts using more of this
    module, the 'everything else is history' framing stops being true and the guards below are
    guarding the wrong surface."""
    multi = open(os.path.join(_ROOT, "experiments", "phase8F_multi.py"), encoding="utf-8").read()
    used = set(__import__("re").findall(r"\bF\.(\w+)", multi))
    assert used == {"recon_error_matrix"}, f"phase8F_multi now uses {sorted(used)} from phase8F_emit"


@pytest.mark.parametrize("groups,why", [
    ([np.array([0, 1]), np.array([2])], "misses the last band"),
    ([np.array([0, 1]), np.array([1, 2, 3])], "overlaps"),
    ([np.array([0]), np.array([2, 3])], "leaves a hole"),
    ([], "empty"),
])
def test_recon_error_matrix_refuses_a_grouping_that_is_not_a_partition(groups, why):
    """A partial grouping used to return a matrix NARROWER than X and a holed one left an all-NaN
    column; both then aligned wrongly against a full-width per-band uncertainty vector, silently."""
    X = np.zeros((4, 4), np.float32)
    m, _ = _tiny_model(4, 2)
    with pytest.raises(ValueError, match="cover every band|empty"):
        F.recon_error_matrix(m, X, groups)


def test_recon_error_matrix_output_is_as_wide_as_X():
    """The width must come from X, not from the largest band index any group happens to mention."""
    m, groups = _tiny_model(12, 3)
    E = F.recon_error_matrix(m, np.zeros((5, 12), np.float32), groups, bs=2)
    assert E.shape == (5, 12)
    assert np.isfinite(E).all()


def test_recon_error_matrix_rejects_non_finite_input():
    """EMIT uses -9999 for nodata and -0.01 where reflectance was not estimated; standardising such
    a cube propagates NaN into every error and then into a correlation that still prints a number."""
    m, groups = _tiny_model(12, 3)
    X = np.zeros((4, 12), np.float32); X[1, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        F.recon_error_matrix(m, X, groups)


def test_recon_error_matrix_rejects_a_nonpositive_batch_size():
    m, groups = _tiny_model(12, 3)
    with pytest.raises(ValueError, match="bs must be positive"):
        F.recon_error_matrix(m, np.zeros((4, 12), np.float32), groups, bs=0)


def test_recon_error_matrix_restores_the_models_training_mode():
    """It evaluates, so it must switch to eval mode — and it must not leave a model the caller was
    training stuck in eval. `@torch.no_grad()` does neither."""
    m, groups = _tiny_model(12, 3)
    m.train()
    F.recon_error_matrix(m, np.zeros((4, 12), np.float32), groups)
    assert m.training, "left the model in eval mode"
    m.eval()
    F.recon_error_matrix(m, np.zeros((4, 12), np.float32), groups)
    assert not m.training


def test_recon_error_matrix_rejects_a_wrongly_shaped_reconstruction():
    """Guards the mapping from `pred[:, g, li]` onto `groups[g][li]`.

    If reconstruct() ever returns a different layout — a change to the group padding, or to the
    reconstruct API — predictions would be read onto the WRONG BANDS with no error, and the per-band
    correlation would be computed against a permuted error vector. This is the failure this
    function's output can least afford, since phase8F_multi publishes from it.
    """
    import torch as _t

    class _BadShape(_t.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = _t.nn.Parameter(_t.zeros(1))

        def reconstruct(self, x, masked):
            return _t.zeros(x.shape[0], 1, 1)          # not (b, G, S)

    _, groups = _tiny_model(12, 3)
    with pytest.raises(ValueError, match="reconstruct\\(\\) returned"):
        F.recon_error_matrix(_BadShape(), np.zeros((4, 12), np.float32), groups)


def test_recon_error_matrix_is_batch_size_invariant():
    """Batching is an implementation detail; if the numbers move with `bs`, something is stateful."""
    m, groups = _tiny_model(12, 3)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 12)).astype(np.float32)
    a = F.recon_error_matrix(m, X, groups, bs=2)
    b = F.recon_error_matrix(m, X, groups, bs=64)
    assert np.allclose(a, b, atol=1e-6)


def test_recon_error_matrix_measures_error_while_the_band_is_hidden():
    """The whole point: a band's error is recorded while its group is masked, so the model cannot
    have copied that band from its own input. Perturbing a band must not leave its error unchanged
    at zero — and the reconstruction must not simply echo the input."""
    m, groups = _tiny_model(12, 3)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 12)).astype(np.float32)
    E = F.recon_error_matrix(m, X, groups)
    assert (E > 0).any(), "every error is exactly zero; the target may be leaking into the input"


# ---------------------------------------------------------------------------------------------
# 2 — correlation hygiene
# ---------------------------------------------------------------------------------------------
def test_spearman_drops_non_finite_pairs_and_reports_how_many_survived():
    """scipy's default nan_policy is 'propagate', so one NaN returned NaN — and the caller's
    np.nanmean then SKIPPED that seed, reporting a mean over fewer runs than it claimed."""
    r, n = F._spearman([1.0, np.nan, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert n == 3 and r == pytest.approx(1.0)


def test_spearman_does_not_let_infinity_rank_as_a_value():
    """inf is orderable, so it used to rank as the largest observation and produce a plausible
    finite rho from corrupt data."""
    r_inf, n_inf = F._spearman([1.0, np.inf, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    r_ref, n_ref = F._spearman([1.0, 3.0, 4.0], [1.0, 3.0, 4.0])
    assert n_inf == 3 == n_ref and r_inf == pytest.approx(r_ref)


def test_spearman_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        F._spearman([1, 2, 3], [1, 2])


def test_spearman_is_nan_on_a_constant_or_too_short_vector():
    assert np.isnan(F._spearman([1, 1, 1], [1, 2, 3])[0])
    assert np.isnan(F._spearman([1, 2], [1, 2])[0])


def test_spearman_handles_ties_by_average_rank():
    """The argsort-of-argsort this replaced ranked ties arbitrarily, which can bias the correlation
    in EITHER direction (the old comment said it inflates it)."""
    r, _ = F._spearman([1, 1, 2, 2], [1, 1, 2, 2])
    assert r == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# 3 — data contract
# ---------------------------------------------------------------------------------------------
def _write_npz(tmp_path, **over):
    n, c = 6, 5
    d = dict(reflectance=np.full((n, c), 0.3, np.float32),
             uncertainty=np.full((n, c), 0.02, np.float32),
             wavelengths=np.linspace(400, 2400, c))
    d.update(over)
    p = tmp_path / "e.npz"
    np.savez(p, **d)
    return str(p)


@pytest.mark.parametrize("over,match", [
    (dict(reflectance=np.full((6, 5), -9999.0, np.float32)), "fill values"),
    (dict(uncertainty=np.full((6, 5), -0.1, np.float32)), "non-negative"),
    (dict(wavelengths=np.array([400.0, 300.0, 500.0, 600.0, 700.0])), "strictly increasing"),
    (dict(uncertainty=np.full((6, 4), 0.02, np.float32)), "same 2-D shape"),
    (dict(reflectance=np.full((6, 5), np.nan, np.float32)), "non-finite"),
])
def test_load_emit_refuses_a_broken_extract(tmp_path, over, match):
    """The npz is a REGENERABLE extract, so 'the extraction script cleans it' is a property of a
    script this file does not control. A single -9999 poisons the per-band mean and sd, and the
    correlation that follows still looks like a number."""
    with pytest.raises((ValueError, KeyError), match=match):
        F.load_emit(_write_npz(tmp_path, **over))


def test_load_emit_reports_missing_keys(tmp_path):
    p = tmp_path / "e.npz"
    np.savez(p, reflectance=np.zeros((3, 2), np.float32))
    with pytest.raises(KeyError, match="missing"):
        F.load_emit(str(p))


@pytest.mark.skipif(not _HAS_NPZ, reason="EMIT extract not present")
def test_the_shipped_extract_satisfies_its_own_contract():
    R, U, wl = F.load_emit(F.EMIT_NPZ)
    assert R.shape == U.shape and wl.size == R.shape[1]
    assert np.isfinite(R).all() and (U >= 0).all()


# ---------------------------------------------------------------------------------------------
# 4 — claims. This file is superseded; the risk is that someone reads it as evidence.
# ---------------------------------------------------------------------------------------------
def _runtime_strings():
    """String literals that REACH A READER — printed lines, CSV headers, provenance text, --help.

    Excludes docstrings (and comments, which ast drops anyway). This repository documents its own
    withdrawn claims in prose, so a plain `"claim" not in source` guard fires on the RETRACTION and
    pressures the next author to delete the explanation in order to go green. That mistake was made
    three times before this helper was written; the rule is to police what gets EMITTED, and to
    assert the correction POSITIVELY where the claim lives in prose.
    """
    tree = ast.parse(_SRC)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docs.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_the_file_declares_itself_superseded_and_names_the_successor():
    assert "SUPERSEDED" in _SRC and "phase8F_multi" in _SRC
    assert "DO NOT CITE" in _SRC, "a reader must be told not to quote these numbers"


def test_no_independent_physical_ground_truth_claim_reaches_a_reader():
    """EMIT reflectance and its uncertainty come from the SAME ISOFIT optimal-estimation retrieval,
    so the uncertainty is external to our model but NOT independent of the reflectance it trains on,
    and it is not measured error against a reference.

    Checked on emitted strings plus a POSITIVE assertion that the correction is stated — the
    withdrawn wording legitimately survives inside the paragraph that withdraws it.
    """
    bad = [s for s in _runtime_strings()
           if "physics-based ground-truth" in s or "INDEPENDENT physical" in s]
    assert not bad, f"withdrawn claim reaches the output: {bad}"
    assert "SAME ISOFIT" in _SRC, "state WHY the claim was withdrawn, do not merely delete it"
    assert "retrieval-uncertainty PROXY" in _SRC or "RETRIEVAL-UNCERTAINTY PROXY" in _SRC


def test_the_data_licence_claim_does_not_reach_a_reader():
    """NASA EMIT L2A is not MIT-licensed; the MIT licence in this repo covers this repo's code."""
    assert not [s for s in _runtime_strings() if "MIT-licensed" in s]
    assert "EMITL2ARFL" in _SRC, "cite the product DOI instead of asserting a licence"
    assert "That is wrong" in _SRC, "the correction must be explicit"


def test_no_hardcoded_verdict_threshold_decides_a_scientific_conclusion():
    """`'IS GROUNDED in' if mp > 0.2` turned 0.199 and 0.201 into opposite conclusions, asserted a
    causal grounding claim from an in-sample single-granule correlation, and decided a PER-BAND
    experiment on a PER-PIXEL statistic.

    Checked structurally: no emitted string may announce a grounding verdict, and `main` must
    contain no IfExp (the ternary that produced it) comparing a correlation against a constant.
    """
    assert not [s for s in _runtime_strings() if "IS GROUNDED" in s]
    tree = ast.parse(_SRC)
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    for node in ast.walk(main):
        if isinstance(node, ast.IfExp) and isinstance(node.test, ast.Compare):
            rendered = ast.unparse(node)
            assert "mp" not in rendered and "mb" not in rendered, \
                f"a correlation is still being thresholded into a verdict: {rendered}"
    assert any("No verdict is issued" in s for s in _runtime_strings())


def test_the_in_sample_and_single_granule_limits_are_stated():
    low = _SRC.lower()
    assert "training residual" in low, "the error is in-sample; say so"
    assert "autocorrelated" in low, "50k pixels from one granule are not 50k independent samples"


# ---------------------------------------------------------------------------------------------
# 5 — reproducibility and artefact routing
# ---------------------------------------------------------------------------------------------
def test_the_seed_is_set_before_the_model_is_constructed():
    """P2.pretrain_sgmae seeds torch INSIDE itself, i.e. after the parameters have been drawn, so
    the seed never controlled initialisation. Measured: two models built from the same RNG state
    differ, so an init depended on how many models had been built earlier in the loop."""
    tree = ast.parse(_SRC)
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main)
                if isinstance(n, ast.For) and "seed" in ast.unparse(n.target))
    body = ast.unparse(loop)
    seed_at = body.find("manual_seed")
    build_at = body.find("GroupedCrossBandAttention")
    assert seed_at != -1 and build_at != -1 and seed_at < build_at, \
        "torch.manual_seed must precede model construction, or the seed label does not fix the init"


def test_the_model_is_placed_on_the_resolved_device_by_this_script():
    """`dev` used to be resolved and never used: whether --device cuda took effect depended on
    another module happening to move the model.

    Checked on the AST, not with `".to(dev)" in source` — that substring also appears in the comment
    explaining the fix, so removing the actual call still passed. (Same trap as guarding prose by
    substring; it keeps reappearing because the explanation quotes the code.)
    """
    tree = ast.parse(_SRC)
    placed = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                and isinstance(node.func.value, ast.Call)
                and getattr(node.func.value.func, "id", "") == "GroupedCrossBandAttention"):
            placed = True
    assert placed, "the model is constructed without .to(dev); device placement is left to another module"


def test_a_non_canonical_configuration_cannot_overwrite_the_deliverable():
    """--smoke was guarded because it changes --groups, which re-partitions the bands and changes
    every per-band error. `--groups 5` does exactly the same and was NOT guarded."""
    assert "CANONICAL" in _SRC and "_nonCanonical" in _SRC
    tree = ast.parse(_SRC)
    assert any(isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "CANONICAL" for t in n.targets)
               for n in ast.walk(tree)), "the canonical configuration must be declared explicitly"


def test_per_seed_correlations_reach_an_artefact():
    """The per-pixel correlation decided the old verdict and existed only in a print, so no reader
    could recompute or stratify it."""
    assert "results_phase8F_emit_perseed" in _SRC
    assert "spearman_perpixel_per_seed" in _SRC, "per-seed values must be stamped too"


def test_the_csv_carries_both_unit_conventions():
    """The error is measured in per-band z units while EMIT uncertainty is a posterior sd in
    reflectance units. Dividing each band by its OWN sd is not one monotone transform, so it
    reorders the per-band ranks (measured: all 244 bands change rank)."""
    for col in ("recon_error_standardised", "recon_error_reflectance",
                "emit_uncertainty_reflectance"):
        assert col in _SRC, f"the CSV must name its units: missing {col}"
