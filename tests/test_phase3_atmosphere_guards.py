"""Guards for phase3_atmosphere: the atmosphere -> band-loss experiment.

Every test here pins a claim the SCRIPT MAKES IN PROSE to something executable, because the review
that prompted this file found that phase 3's real exposure was never a crash — it was construct
validity. The code computed reproducible numbers whose stated physical meaning was not what the
code did:

  * the module said surviving bands were "multiplicatively suppressed by rho*T" while the primary
    arm passes T_eff=1 and suppresses nothing (test_primary_ignores_T_values_given_the_same_mask);
  * it called rho*T an "L1C-style product" when L1C is TOA reflectance and this is one gaseous
    factor (test_no_l1c_or_l2a_product_claims_reach_an_artefact);
  * it asserted Indian Pines "is itself already a surface-reflectance product" on the strength of a
    filename, while the cube holds integers in [955, 9604]
    (test_provenance_payload_records_identity_and_the_caveats_as_DATA);
  * it reported "clean_full = 0 bands removed" on a cube that is already missing 20 of 220 AVIRIS
    bands (test_summary_rows_label_the_ceiling_for_the_benchmark_not_the_spectrum);
  * and it quoted a TAU sweep that has no artefact anywhere in the repository
    (test_the_tau_sweep_is_runnable_and_its_claim_is_not_asserted_unbacked).

The information-parity tests are the load-bearing ones. The obvious reviewer attack on this
experiment is that the proposed model gets a group-present mask B2 cannot have, so the margin is an
oracle rather than a method. That is answerable on the shipped LUT and it is answered here, not in
prose -- but see test_the_missingness_channel_is_mechanically_inert_under_an_all_present_mask for
what that does and does NOT establish.

WRITE GUARDS THAT CALL THE PRODUCTION CODE. A mutation campaign against an earlier version of this
file found 16 of 17 defects surviving, because tests for logic buried in closures had to
REIMPLEMENT what they were checking, which makes them tautologies. Where a claim lives in code,
that code is now module-level (product_multiplier, build_eval_input, control_rng, validate_args,
provenance_extra) and the test calls it. Where a claim lives in prose, prefer asserting on the
machine-readable artefact (provenance payload, CSV labels) over searching the source for words: a
source search cannot tell a claim from its own retraction, and this file made that mistake twice.

These tests train NOTHING. Everything asserted is a property of the masks, the thresholds, the LUT,
the provenance payload and the source text, all decided before a single epoch runs.
"""
import ast
import os
import re
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

from bandsim.io import AVIRIS_WL_NM                                        # noqa: E402
from bandsim.grouping import contiguous_groups                             # noqa: E402
from bandsim.atmosphere import hard_mask_absorption_cores                  # noqa: E402
import phase3_atmosphere as P3                                             # noqa: E402

_SRC = open(os.path.join(_ROOT, "experiments", "phase3_atmosphere.py"), encoding="utf-8").read()
_HAS_LUT = os.path.exists(P3.TABLE)
_needs_lut = pytest.mark.skipif(not _HAS_LUT, reason="6S table not present")


def _keep():
    return hard_mask_absorption_cores(AVIRIS_WL_NM)


def _masks():
    """The physics missing-mask per CWV at the shipped defaults."""
    keep = _keep()
    return {c: P3.physics_missing_bands(P3.cwv_transmittance(P3.TABLE, c), keep,
                                        tau_missing=P3.TAU_MISSING) for c in P3.CWVS}


# ---------------------------------------------------------------------------------------------
# 1 — what the PRIMARY arm actually does with T
# ---------------------------------------------------------------------------------------------
def test_primary_ignores_T_values_given_the_same_mask():
    """In PRIMARY, T enters ONLY through the threshold. Two different transmittance columns that
    threshold to the SAME mask must give bit-identical model inputs.

    This is the claim the old comment contradicted ("every OTHER band is kept but MULTIPLICATIVELY
    suppressed"). If a future edit reintroduces suppression into the primary path, the CWV axis
    silently stops being a pure band-loss ladder and starts mixing in a radiometric shift.

    Goes through the PRODUCTION functions (`product_multiplier` + `build_eval_input`). An earlier
    version of this test reimplemented the arm inline because the logic was trapped in a closure
    inside run_seed; a mutation reintroducing suppression into PRIMARY then passed the entire
    suite, and the inline version was tautological besides (with T_eff forced to ones by the test
    itself, the result depends only on the mask by arithmetic). A guard for a claim must call the
    code that makes the claim.
    """
    C = 20
    keep = np.ones(C, bool)
    Ta = np.full(C, 0.9); Ta[3] = 0.1          # one band below tau
    Tb = np.full(C, 0.6); Tb[3] = 0.4          # same band below tau, everything else different
    ma = P3.physics_missing_bands(Ta, keep, tau_missing=0.5)
    mb = P3.physics_missing_bands(Tb, keep, tau_missing=0.5)
    assert np.array_equal(ma, mb), "precondition: the two columns must threshold identically"

    cube = np.arange(5 * C, dtype=float).reshape(5, C) + 1.0
    mu, sd = cube.mean(0), cube.std(0) + 1e-8
    outs = [P3.build_eval_input(cube, P3.product_multiplier(T, True),
                                P3.physics_missing_bands(T, keep, tau_missing=0.5), mu, sd)
            for T in (Ta, Tb)]
    assert np.array_equal(outs[0], outs[1]), \
        "PRIMARY must not depend on the T VALUES, only on the thresholded mask"
    # and it must differ from the SUPPLEMENTARY arm, or 'primary ignores T' is true only because
    # nothing anywhere uses T (which would make the whole supplementary contrast vacuous)
    supp = P3.build_eval_input(cube, P3.product_multiplier(Ta, False), ma, mu, sd)
    assert not np.array_equal(outs[0], supp), \
        "primary and supplementary produce identical inputs; the product distinction is not live"


def test_product_multiplier_is_ones_only_for_the_corrected_arm():
    """The single branch the primary/supplementary distinction reduces to."""
    T = np.array([0.2, 0.7, 1.0])
    assert np.array_equal(P3.product_multiplier(T, True), np.ones(3))
    assert np.array_equal(P3.product_multiplier(T, False), T)


def test_missing_bands_are_imputed_at_standardized_zero_not_the_raw_sentinel():
    """Order-of-operations guard: mask -> standardise -> SET MISSING TO 0.

    If the final imputation is dropped, a missing band lands at (0-mu)/sd, an out-of-distribution
    sentinel neither model was trained on, and the reported drop mixes band loss with a spurious
    shift. Uses a cube with a large non-zero mean so the two differ unmistakably.
    """
    cube = np.full((4, 6), 1000.0) + np.arange(6)
    mu, sd = cube.mean(0), cube.std(0) + 1e-8
    missing = np.array([False, True, False, False, True, False])
    X = P3.build_eval_input(cube, np.ones(6), missing, mu, sd)
    assert np.allclose(X[:, missing], 0.0), "missing bands must sit at standardized 0 (train mean)"
    assert np.all(np.abs((0.0 - mu[missing]) / sd[missing]) > 1.0), \
        "precondition: the raw-0 sentinel must be far from 0 here, or the test proves nothing"


def test_supplementary_suppresses_through_phase3s_own_path():
    """The SUPPLEMENTARY arm multiplies surviving bands by T; the PRIMARY arm does not.

    Routed through phase3's `product_multiplier` + `build_eval_input` rather than through
    bandsim.apply_atmosphere. The earlier version asserted `cube*T <= cube` for T in [0,1], which is
    a property of multiplication in another module -- true regardless of anything phase 3 does.
    Here the comparison is in STANDARDISED space, which is where the models actually see it, and
    the surviving bands must move DOWN relative to the primary arm.
    """
    C = 12
    rng = np.random.default_rng(0)
    cube = rng.uniform(1e3, 1e4, size=(64, C))
    mu, sd = cube.mean(0), cube.std(0) + 1e-8
    T = rng.uniform(0.2, 0.9, size=C)
    keep_all = np.zeros(C, bool)                       # nothing masked: isolate the multiplier
    prim = P3.build_eval_input(cube, P3.product_multiplier(T, True), keep_all, mu, sd)
    supp = P3.build_eval_input(cube, P3.product_multiplier(T, False), keep_all, mu, sd)
    assert np.all(supp <= prim + 1e-5), "rho*T must not exceed rho for a non-negative cube"
    assert np.any(supp < prim - 1e-3), "the supplementary arm changed nothing; T is not applied"
    # and the shift is what the docstring blames the collapse on: bands leave the training
    # distribution, by many sigma on the low-variance ones
    assert np.abs(supp - prim).max() > 3.0, "the suppression is not reaching multiple sigma"


# ---------------------------------------------------------------------------------------------
# 2 — information parity: is the present-mask an oracle B2 lacks?
# ---------------------------------------------------------------------------------------------
# NOTE: a test asserting `group_present_mask(n, groups, []).all()` used to sit here. It was a
# tautology -- that function is `np.ones(...)` followed by a loop over an empty list, so it asserted
# that np.ones returns ones -- and it carried a spurious @_needs_lut that would have made it vanish
# on a machine without the 6S table. The claim it was meant to support (that an all-present mask
# conveys nothing) is measured properly, against a mask-free reference computation, in
# test_the_missingness_channel_is_mechanically_inert_under_an_all_present_mask below.


def test_the_missingness_channel_is_mechanically_inert_under_an_all_present_mask():
    """The oracle objection, closed MECHANICALLY rather than by argument.

    Reasoning from `absent == []` to "no information is conveyed" is an inference about the model's
    internals. This measures it instead, and it measures it against a REFERENCE forward pass that
    never sees a mask at all -- NOT by perturbing `mask_token`. Perturbing the token only proves
    `1e6 * (1 - 1) == 0`, which is true by construction and is therefore blind to a NEW presence
    channel (a bias, a second embedding, an attention prior) -- exactly the failure an earlier
    version of this test claimed to catch and did not. Verified: a subclass adding
    `present.float() * bias` to the encoder output survives the token perturbation and IS caught
    here, because the reference has no presence term to reproduce. If this fails, phase 3's answer
    to the fairness objection needs rewriting before publication, which is when we want to know.
    """
    import torch
    from phase2_degradation import group_present_mask
    from bandsim.model import GroupedCrossBandAttention

    groups = contiguous_groups(20, 4)
    model = GroupedCrossBandAttention(groups, [400.0, 500.0, 600.0, 700.0], 3)
    model.eval()
    x = torch.randn(5, 20)
    pm = torch.from_numpy(group_present_mask(5, groups, []))          # nothing absent

    with torch.no_grad():
        got = model(x, pm)
        # Mask-free reference: tokens = embed(group values) + PE, encode with NO padding mask,
        # mean-pool, classify. Presence appears nowhere.
        xg = x[:, model.group_idx] * model.group_valid[None]
        tok = model.embed(xg) + model.pe[None]
        h = model.encoder(tok, src_key_padding_mask=None)
        ref = model.classifier(h.mean(1))
    assert torch.allclose(got, ref, atol=1e-6), (
        "with NO group absent the forward pass does not reduce to the mask-free computation, so "
        "presence information reaches the model through some channel -- the parity claim is void")

    # and the token really is inert here, which is what makes `absent == []` equivalent to parity
    with torch.no_grad():
        model.mask_token.data.fill_(1e6)
        assert torch.equal(got, model(x, pm))


@_needs_lut
def test_absent_group_count_on_the_shipped_lut_matches_the_documented_parity_claim():
    """The module docstring claims 0 absent groups at clean/CWV0.5/CWV2.0 and 1 at CWV4.0.

    That claim is the whole answer to the oracle objection, so it must track the shipped artefact.
    If the LUT, TAU_MISSING, GROUP_ABSENT_FRAC or the grouping changes, this fails and the prose
    must be rewritten -- rather than the prose quietly describing a table nobody re-checked.
    """
    groups = contiguous_groups(len(AVIRIS_WL_NM), 10)
    masks = _masks()
    got = {c: len(P3.absent_groups_for(m, groups, P3.GROUP_ABSENT_FRAC)[0])
           for c, m in masks.items()}
    # All-zero on the corrected axis: the mask mechanism is inert in every condition, so the
    # information-parity story is now unconditional -- and the paper's oracle answer simplifies.
    assert got == {0.5: 0, 2.0: 0, 4.0: 0}, (
        f"absent-group counts changed to {got}; the docstring's information-parity paragraph and "
        f"the paper's answer to the oracle objection both depend on these numbers")
    # clean (T=1) has no T-driven losses at all, only the fixed cores
    clean_missing = P3.physics_missing_bands(np.ones(len(AVIRIS_WL_NM)), _keep(),
                                             tau_missing=P3.TAU_MISSING)
    assert len(P3.absent_groups_for(clean_missing, groups, P3.GROUP_ABSENT_FRAC)[0]) == 0


@_needs_lut
def test_the_cwv4_absent_group_sits_exactly_on_the_threshold():
    """HISTORY: on the stretched (pre-2026-07-20) axis, CWV=4.0's one absent group sat at a
    missing fraction of EXACTLY 0.5, so `>=` vs `>` decided whether the group-mask mechanism
    engaged at all in the headline condition -- a knife-edge fragility this test used to pin.

    On the corrected axis that phenomenon is GONE, and this test now pins its replacement: group
    fractions top out at 0.40 at every CWV, so at the default threshold NO group is absent, the
    closest approach is a full 0.10 (>= 2 eps, no knife edge), and the group-mask mechanism is
    inert in ALL FOUR conditions -- which is what phase 3's margin must be read against."""
    groups = contiguous_groups(len(AVIRIS_WL_NM), 10)
    absent, fracs = P3.absent_groups_for(_masks()[4.0], groups, P3.GROUP_ABSENT_FRAC)
    assert absent == [], "no group crosses the default threshold on the corrected axis"
    assert max(fracs) == pytest.approx(0.4, abs=1e-12), (
        f"the worst CWV=4.0 group sits at {max(fracs)}; if this moved, re-derive the whole "
        f"knife-edge story rather than patching one number")
    margin = float(np.min(np.abs(np.asarray(fracs) - P3.GROUP_ABSENT_FRAC)))
    assert margin >= 2 * P3.GROUP_FRAC_EPS - 1e-12, \
        "the default threshold sits >= 2 eps clear of every group; a knife edge here would be noise"


@_needs_lut
def test_no_condition_ever_removes_a_whole_group():
    """The corruption-shift confound, pinned.

    Both models are trained on WHOLE-group corruption (train_mlp zeroes whole groups,
    finetune_proposed masks whole groups), but no phase-3 condition ever removes one: the per-group
    missing fraction tops out at 0.5. So every test point is a partial, scattered pattern NEITHER
    model was trained on, and part of what the margin measures is tolerance of an out-of-distribution
    corruption SHAPE. If a future LUT or grouping ever does empty a group, that changes what the
    experiment measures and the confound paragraph must be rewritten.
    """
    groups = contiguous_groups(len(AVIRIS_WL_NM), 10)
    for cwv, m in _masks().items():
        _, fracs = P3.absent_groups_for(m, groups, P3.GROUP_ABSENT_FRAC)
        assert fracs.max() <= 0.5 + 1e-12, f"CWV={cwv} now removes {fracs.max():.0%} of a group"


def test_core_bands_are_not_the_least_informative_in_the_cube():
    """Guards a claim this file got WRONG in an earlier revision, in a way that contradicted its own
    headline number.

    The docstring justified the random control's expected direction by calling the 8 hard-masked
    core bands "among the least informative in the cube" -- while ALSO reporting that removing those
    same 8 bands costs B2 19.4 mIoU. Both cannot be true. Measured, half the core set is mid-pack by
    variance (sd-ranks 60, 34, 32, 14 out of 200), so the "uninformative" half of the contradiction
    is the false one. Skipped when the cube is absent so the suite still runs without the dataset.
    """
    data = os.path.join(_ROOT, "data", "indian_pines", "Indian_pines_corrected.mat")
    if not os.path.exists(data):
        pytest.skip("Indian Pines cube not present")
    from phase2_degradation import load_data
    cube, _ = load_data()
    sd = cube.reshape(-1, cube.shape[-1]).std(0)
    rank = np.argsort(np.argsort(sd))                      # 0 = lowest variance
    core = np.where(~_keep())[0]
    assert len(core) == 10   # corrected axis: 97-102 + 140-143
    mid = [int(i) for i in core if rank[i] > len(sd) // 10]
    assert len(mid) >= 3, (
        f"only {len(mid)} of the 8 core bands are above the 10th variance percentile; the "
        f"docstring's 'NOT uniformly so (ranks 32, 34, 60)' correction would need revisiting")
    # NO negative substring check here. The phrase "among the least informative" still appears in
    # the source -- inside the paragraph that RETRACTS it -- and prose cannot be distinguished from
    # its own retraction by substring search. (That mistake was made twice in this file already.)
    # The empirical assertion above is the real guard; this one requires the retraction's reasoning
    # to still be present, since dropping a wrong claim without saying why invites its return.
    assert "variance is not information" in _SRC, \
        "the file must say why a variance argument cannot carry the claim, not just drop the claim"
    assert "19.4" in _SRC, "the direct measurement that refutes the variance argument must be cited"


def test_low_variance_bands_are_not_all_water_shoulders():
    """Also guards a corrected claim: bands 199/198 sit at ~2480-2490 nm (the SWIR terminus) and 144
    at 1963 nm, so 'the residual shoulders of the truncated water windows' was wrong for half of
    the six lowest-variance bands."""
    assert AVIRIS_WL_NM[199] > 2400 and AVIRIS_WL_NM[198] > 2400
    # Core indices re-pinned at the 2026-07-20 axis correction: the masked set is now
    # bands 97-102 (1.38 um window) and 140-143 (1.8 um window); 103 left the core set.
    assert not _keep()[97] and not _keep()[143] and not _keep()[102]    # these three ARE cores
    assert _keep()[199] and _keep()[198] and _keep()[144]               # these three are NOT
    # The correction must still be stated. (An earlier line here read
    # `assert "residual SHOULDERS..." not in _SRC or "false for half" in _SRC` -- the quoted phrase
    # was never in the source in that casing, so the left disjunct was unconditionally true and the
    # assertion could not fail in either direction.)
    assert "false for half of them" in _SRC, \
        "the retraction of the water-shoulder identification must stay in the docstring"


def test_the_mask_isolating_baseline_is_not_misattributed_to_B4():
    """B4 (train_hcs) PASSES a present-mask to the model, so proposed-vs-B4 holds the mask constant
    and varies PE type + SGMAE + training mask distribution. An earlier revision cited B4 as the
    baseline that isolates the mask, which is backwards."""
    import inspect
    from phase2_degradation import train_hcs
    src = inspect.getsource(train_hcs)
    assert "pm" in src and "model(xb, " in src, "train_hcs no longer passes a present-mask; re-check"
    assert "DOES NOT EXIST in" in _SRC or "does not exist" in _SRC.lower(), \
        "the file must say plainly that no mask-isolating baseline exists in this repo"


def test_absent_threshold_is_inclusive_at_the_boundary():
    """Pin the tie-break itself: a group at EXACTLY the fraction counts as absent."""
    groups = [np.array([0, 1]), np.array([2, 3])]
    missing = np.array([True, False, False, False])       # group 0 is exactly 50% missing
    absent, fracs = P3.absent_groups_for(missing, groups, 0.5)
    assert absent == [0] and fracs[0] == 0.5


# ---------------------------------------------------------------------------------------------
# 3 — the mask-construction contract
# ---------------------------------------------------------------------------------------------
def test_tau_missing_is_keyword_only_and_required():
    """A positional/defaulted tau would bind the module global ONCE at def time, so --tau-sweep
    would report swept values while thresholding at the original 0.5 forever."""
    with pytest.raises(TypeError):
        P3.physics_missing_bands(np.ones(4) * 0.9, np.ones(4, bool))          # no tau
    with pytest.raises(TypeError):
        P3.physics_missing_bands(np.ones(4) * 0.9, np.ones(4, bool), 0.5)     # positional tau


@pytest.mark.parametrize("T,keep,tau", [
    (np.ones((2, 3)), np.ones(3, bool), 0.5),            # 2-D T
    (np.ones(3), np.ones(4, bool), 0.5),                 # length mismatch (would broadcast)
    (np.array([0.5, np.nan, 0.5]), np.ones(3, bool), 0.5),   # non-finite
    (np.array([0.5, 1.5, 0.5]), np.ones(3, bool), 0.5),      # T > 1 is not a transmittance
    (np.array([0.5, -0.2, 0.5]), np.ones(3, bool), 0.5),     # T < 0
    (np.ones(3) * 0.5, np.ones(3, bool), 1.5),               # tau outside [0,1]
])
def test_physics_missing_bands_rejects_unphysical_input(T, keep, tau):
    """The function is importable and is called directly by the sweep and the tests, so it must not
    rely on the LUT loader upstream having validated its arguments."""
    with pytest.raises(ValueError):
        P3.physics_missing_bands(T, keep, tau_missing=tau)


def test_missing_mask_is_the_union_of_low_transmittance_and_the_hard_cores():
    T = np.array([0.9, 0.1, 0.9, 0.9])
    keep = np.array([True, True, False, True])
    m = P3.physics_missing_bands(T, keep, tau_missing=0.5)
    assert list(m) == [False, True, True, False]


@_needs_lut
def test_missing_masks_are_nested_in_cwv():
    """More water vapour must never RESTORE a band. If this fails, the CWV axis is not a monotone
    band-loss ladder and the figure's left-to-right reading is wrong."""
    m = _masks()
    assert not (m[0.5] & ~m[2.0]).any(), "a band missing at CWV=0.5 reappeared at CWV=2.0"
    assert not (m[2.0] & ~m[4.0]).any(), "a band missing at CWV=2.0 reappeared at CWV=4.0"
    assert m[0.5].sum() < m[2.0].sum() < m[4.0].sum()


# ---------------------------------------------------------------------------------------------
# 4 — LUT identity and key resolution
# ---------------------------------------------------------------------------------------------
def test_ambiguous_lut_keys_are_refused_when_they_disagree(tmp_path):
    """Two columns matching one CWV used to resolve by numpy archive order, silently."""
    p = tmp_path / "T.npz"
    np.savez(p, wl_nm=AVIRIS_WL_NM, generation_mode="direct_6s",
             **{"cwv2.0_a": np.full(200, 0.9), "cwv2.0_b": np.full(200, 0.2)})
    with pytest.raises(ValueError, match="NOT identical"):
        P3.resolve_cwv_key(str(p), 2.0)


def test_ambiguous_lut_keys_are_accepted_when_bitwise_equal(tmp_path):
    """The legacy layout tagged each column with an inert atmosphere suffix; identical duplicates
    are the case that layout produces and must keep working."""
    p = tmp_path / "T.npz"
    col = np.full(200, 0.77)
    np.savez(p, wl_nm=AVIRIS_WL_NM, generation_mode="direct_6s",
             **{"cwv2.0_maritime": col, "cwv2.0_continental": col.copy()})
    assert P3.resolve_cwv_key(str(p), 2.0) == "cwv2.0_continental"   # deterministic: sorted order


def test_missing_cwv_key_raises_keyerror(tmp_path):
    p = tmp_path / "T.npz"
    np.savez(p, wl_nm=AVIRIS_WL_NM, generation_mode="direct_6s", **{"cwv2.0": np.full(200, 0.5)})
    with pytest.raises(KeyError):
        P3.resolve_cwv_key(str(p), 9.0)


def test_a_resampled_table_is_refused(tmp_path):
    """A table INTERPOLATED onto this axis reproduces the axis and its hash exactly and is still
    not what 6S returns at the narrow absorption edges the whole CWV ladder is thresholded on."""
    p = tmp_path / "T.npz"
    np.savez(p, wl_nm=AVIRIS_WL_NM, generation_mode="resampled",
             **{"cwv2.0": np.full(200, 0.5)})
    with pytest.raises(ValueError, match="generation_mode"):
        P3.cwv_transmittance(str(p), 2.0)


@_needs_lut
def test_shipped_table_transmittance_decreases_with_column_water_vapour():
    """A PHYSICS check the loader does not perform.

    The earlier version asserted shape/finiteness/[0,1] -- every one of which
    `load_cached_transmittance` already raises on, so none of those asserts could ever execute.
    More water vapour cannot make the atmosphere MORE transmitting, so T must be monotonically
    non-increasing in CWV pointwise; if it is not, the LUT columns are mislabelled or swapped, and
    the whole CWV ladder would run in an order the labels deny.
    """
    T = {c: P3.cwv_transmittance(P3.TABLE, c) for c in P3.CWVS}
    assert np.all(T[2.0] <= T[0.5] + 1e-9), "T rose from CWV=0.5 to 2.0; columns are mislabelled"
    assert np.all(T[4.0] <= T[2.0] + 1e-9), "T rose from CWV=2.0 to 4.0; columns are mislabelled"
    # and it must actually MOVE, or the ladder has no driver
    assert (T[4.0] < T[0.5] - 1e-3).sum() >= 10, "almost no band responds to CWV at all"


# ---------------------------------------------------------------------------------------------
# 5 — evaluation mode
# ---------------------------------------------------------------------------------------------
def test_eval_mode_sets_eval_and_restores_the_previous_mode():
    """`torch.no_grad()` is not eval mode: it stops autograd, it does not switch Dropout/BatchNorm.
    Currently a no-op on the numbers (dropout=0.0, no BatchNorm) — pinned so it stays one.
    """
    import torch
    m = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.5))
    m.train()
    with P3.eval_mode(m):
        assert not m.training
    assert m.training, "the previous mode must be restored"
    m.eval()
    with P3.eval_mode(m):
        assert not m.training
    assert not m.training


# ---------------------------------------------------------------------------------------------
# 6 — construct-validity claims in the source text
# ---------------------------------------------------------------------------------------------
def _runtime_strings():
    """String literals that REACH A READER: CSV labels, printed lines, figure text, --help.

    Deliberately excludes docstrings (and, since ast drops them, comments). This repository
    documents its own past defects in prose -- the phase3 docstring explains at length which
    product claims were withdrawn and why -- so a naive "this phrase must not appear anywhere"
    guard would fire on the retraction notice itself and pressure a future author to DELETE the
    explanation in order to go green. The claim that matters is the one written into an artefact,
    so that is the surface these tests police.
    """
    tree = ast.parse(_SRC)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    out = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
    # Also emit each string-building EXPRESSION with its quotes and `+` folded away, so a claim
    # split across literals cannot hide. Measured evasions of the literal-only version:
    # `"L1C-" + "style"`, `f"{tag}-style"`, `"%s-style" % "L1C"`, `"{}-style".format("L1C")` --
    # the first two are folded here; the format-style ones remain out of reach of any static check
    # and are the reason the important guards assert on emitted ROWS and payloads instead.
    for n in ast.walk(tree):
        if isinstance(n, (ast.BinOp, ast.JoinedStr)) and id(n) not in docstrings:
            try:
                txt = ast.unparse(n)
            except Exception:                                    # pragma: no cover - defensive
                continue
            out.append(re.sub(r"['\"]\s*\+\s*['\"]|['\"]", "", txt))
    return out


def test_no_l1c_or_l2a_product_claims_reach_an_artefact():
    """rho*T is one gaseous factor; L1C is TOA reflectance (path radiance, spherical albedo,
    Rayleigh/aerosol scattering, geometry). Naming it L1C claims a product we do not simulate, and
    the same goes for calling the primary arm an L2A product. Checked on runtime strings: the
    docstring is allowed -- required, in fact -- to explain that these labels were withdrawn.

    Matches the PRODUCT NAMES, not four hyphenated spellings of them. The earlier version banned
    exactly the four hyphenated forms, so a plain print of "PRIMARY: L1C product simulation, TOA
    reflectance" sailed through -- and that is the phrasing someone reintroducing the claim would
    actually write. Strings that NEGATE the label are allowed, because the file has to be able to
    say it is not an L1C product; that exemption is what the negation check encodes.
    """
    negations = ("not ", "NOT ", "never", "no longer", "cannot", "is not")
    bad = [s for s in _runtime_strings()
           if ("L1C" in s or "L2A" in s) and not any(n in s for n in negations)]
    assert not bad, f"product claims this script cannot support reach the output: {bad}"


def test_the_module_docstring_is_not_piped_into_help_output():
    """`_runtime_strings()` deliberately excludes docstrings so that prose DOCUMENTING a withdrawn
    claim does not trip the product-claim guard. That exemption is only sound while the docstring
    stays out of the artefacts.

    `ArgumentParser(description=__doc__)` is a one-line refactor and extremely common, and it would
    put every withdrawn-claim quotation into --help -- an artefact by this file's own definition --
    while the guard stayed blind to all of it.
    """
    tree = ast.parse(_SRC)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "ArgumentParser":
            for kw in n.keywords:
                assert not (kw.arg == "description" and isinstance(kw.value, ast.Name)
                            and kw.value.id == "__doc__"), (
                    "the module docstring would become --help output, so the retracted claims it "
                    "quotes would reach a reader while _runtime_strings() ignores them")


def test_cube_physical_quantity_is_recorded_as_unverified():
    """The filename says 'corrected', which refers to the BAND SUBSET, not to atmospheric
    correction; the cube holds integers in [955, 9604]. The caveat has to be MACHINE-READABLE next
    to the numbers, not only prose in a file no CSV reader opens — so it is asserted on the
    provenance payload."""
    assert "DATA PROVENANCE" in _SRC, "the physical-quantity caveat must be stated in the docstring"
    assert "cube_physical_quantity" in _SRC, "the caveat must be stamped beside the results"
    assert "cube_sha256" in _SRC, "the cube's identity must be recorded, not inferred from a filename"
    assert any("UNVERIFIED" in s for s in _runtime_strings()), \
        "the stamped physical quantity must say plainly that it is unverified"


def _fake_agg():
    """Two seeds of plausible aggregates, shaped exactly as main() builds them."""
    def block(p, b, nm=None):
        return {"proposed": p, "b2": b, "proposed_nomask": nm or p,
                "proposed_uncorrected": [x / 10 for x in p], "b2_uncorrected": [x / 10 for x in b]}
    agg = {"clean": block([64.0, 66.0], [47.0, 49.0]), 0.5: block([54.0, 56.0], [40.0, 42.0]),
           2.0: block([39.0, 41.0], [32.0, 34.0]),
           4.0: block([34.0, 36.0], [22.0, 24.0], nm=[30.0, 32.0])}
    agg["clean"]["proposed_full"] = [71.0, 73.0]
    agg["clean"]["b2_full"] = [66.0, 68.0]
    return agg


def test_summary_rows_label_the_ceiling_for_the_benchmark_not_the_spectrum():
    """20 of the nominal 220 AVIRIS bands are already absent, so a 'full spectrum' reference on this
    cube is a 200-band benchmark ceiling.

    Asserted on the ROWS THE WRITER EMITS, not on a source search. The previous guard checked the
    source for the string "clean_full", which `"clean" + "_full"` evades outright and which says
    nothing about what actually lands in the CSV.
    """
    rows = P3.summary_rows(_fake_agg(), {"clean": 8, 4.0: 37}, {"clean": 8, 4.0: 37},
                           ["clean"] + list(P3.CWVS), 2)
    labels = [r[0] for r in rows[1:]]
    assert labels[0] == "benchmark200_full", "the ceiling must be the reader's first row"
    assert not any("clean_full" in str(l) for l in labels), \
        "the old label overstates what the reference contains (20 bands are already gone)"
    assert labels == ["benchmark200_full", "clean", "cwv0.5", "cwv2.0", "cwv4.0"]


def test_summary_rows_do_not_report_a_corrected_number_under_an_uncorrected_header():
    """The ceiling row has no uncorrected variant (T=1 makes correction a no-op). It used to write
    the CORRECTED value there, handing a corrected figure to anyone aggregating that column."""
    rows = P3.summary_rows(_fake_agg(), {}, {}, ["clean"] + list(P3.CWVS), 2)
    hdr, ceiling = rows[0], rows[1]
    for col in ("proposed_gasonly_uncorrected_mean", "b2_gasonly_uncorrected_mean"):
        assert ceiling[hdr.index(col)] == "-", f"{col} must be '-' on the ceiling row"
    # and the real conditions DO carry numbers there, or the column is pointless
    assert ceiling[hdr.index("proposed_miou_mean")] == "72.00"
    assert float(rows[2][hdr.index("proposed_gasonly_uncorrected_mean")]) > 0


def test_summary_rows_paired_diff_is_the_mean_of_per_seed_differences():
    """paired_diff is the statistic the 'proposed > B2' claim rests on; it must be the mean of the
    per-seed differences, not the difference of the means dressed up as one."""
    rows = P3.summary_rows(_fake_agg(), {}, {}, ["clean"] + list(P3.CWVS), 2)
    hdr = rows[0]
    clean = rows[2]
    # clean: proposed [64,66], b2 [47,49] -> diffs [17,17] -> 17.00
    assert clean[hdr.index("paired_diff_mean")] == "17.00"
    assert clean[hdr.index("n_seeds")] == 2


def test_summary_rows_expose_the_mask_ablation_column():
    """proposed_nomask must equal proposed wherever no group is absent, and differ where one is --
    that equality IS the finding that the mask mechanism is inert in most conditions."""
    rows = P3.summary_rows(_fake_agg(), {}, {}, ["clean"] + list(P3.CWVS), 2)
    hdr = rows[0]
    by = {r[0]: r for r in rows[1:]}
    assert by["clean"][hdr.index("proposed_nomask_mean")] == by["clean"][hdr.index("proposed_miou_mean")]
    assert by["cwv4.0"][hdr.index("proposed_nomask_mean")] != by["cwv4.0"][hdr.index("proposed_miou_mean")]


def test_the_tau_sweep_is_runnable_and_its_claim_is_not_asserted_unbacked():
    """The docstring used to claim the ranking survived TAU in {0.3, 0.5, 0.7} with no artefact in
    the repository backing it. The claim must stay withdrawn until --tau-sweep has been run and
    stamped, and the sweep must actually be runnable rather than merely un-claimed."""
    assert "--tau-sweep" in _SRC, "the sweep must be runnable"
    assert "withdrawn" in _SRC, "the retraction must be explicit, not a silent deletion"
    assert any("results_phase3_tau_sweep" in s for s in _runtime_strings()), \
        "the sweep must write its own artefact so the claim can be re-made from evidence"
    # The sweep must REACH THE WORKERS. run_jobs' parallel path spawns processes that re-import
    # this module, so a threshold left in the parent's namespace never arrives: the worker would
    # threshold at the default while the CSV faithfully reported the swept value. Checked on the
    # actual run_jobs call rather than on a substring, because "tau_sweep=" appears in the
    # function signature too and would make this assertion pass without the plumbing existing.
    tree = ast.parse(_SRC)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "run_jobs")
    shared = next(k for k in call.keywords if k.arg == "shared")
    passed = {k.arg for k in shared.value.keywords} if isinstance(shared.value, ast.Call) else set()
    for needed in ("tau_missing", "group_absent_frac", "tau_sweep", "random_control"):
        assert needed in passed, f"run_jobs(shared=...) must carry {needed} across the spawn boundary"


def test_docstring_numbers_still_match_the_shipped_csv():
    """Documentation-artefact drift: the docstring quotes measured mIoU values, and nothing made
    them track the CSV they claim to come from.

    The repo's convention is to quote measured numbers inline (they are the evidence for the
    surrounding argument), so the fix is not to strip them but to FAIL when they stop being true.
    Tolerances are loose because the docstring rounds; the point is to catch a re-run that moves a
    number, not to re-derive it.
    """
    csv_path = os.path.join(_ROOT, "paper", "results_phase3_atmosphere.csv")
    if not os.path.exists(csv_path):
        pytest.skip("no phase3 CSV shipped")
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = {r["condition"]: r for r in _csv.DictReader(f)}

    def col(row, *names):
        """Read the first column name that exists (the schema gained clearer names)."""
        for n in names:
            if n in row and row[n] not in ("", "-"):
                return float(row[n])
        raise AssertionError(f"none of {names} in {sorted(row)}")

    # the ceiling row was renamed clean_full -> benchmark200_full; accept whichever shipped
    ceiling = rows.get("benchmark200_full") or rows.get("clean_full")
    assert ceiling is not None, "the no-further-removal ceiling row is missing"
    clean = rows["clean"]

    # Each figure is tied to the CSV in BOTH directions: the value is computed from the artefact,
    # then the docstring is required to quote that value. The earlier version compared the CSV
    # against numeric literals written in this test and never read the docstring at all -- so
    # rewriting 54.4 -> 62.1 and +17.7 -> +24.9 in the module docstring left it green, which is the
    # exact drift the docstring says "a TEST" replaces its own discipline with.
    def doc_num(pattern, what):
        """The number the DOCSTRING states, pulled out so it can be compared numerically.

        Compared as a value, not as a formatted string: the fixed-core cost is 19.46, which renders
        as "19.5" while the docstring reasonably rounds it to 19.4. A string comparison would fail
        on presentation and teach the next author to delete the check.
        """
        m = re.search(pattern, _SRC)
        assert m, f"the docstring no longer states {what} (pattern {pattern!r})"
        return float(m.group(1))

    # WHAT THIS TEST PINS, AND WHAT IT DELIBERATELY DOES NOT (rewritten 2026-07-20).
    # It pins the docstring AGAINST THE ARTEFACT and nothing else. The previous version also carried
    # the experiment's own values as literals (54.4, 3.2, 19.4, 6.5, 17.7) -- the exact design its
    # own comment above criticises -- so the corrected-axis rerun turned it red for the one reason
    # that is NOT drift: the experiment legitimately moved. A test whose repair procedure is "edit
    # the number" teaches the next author to edit the number.
    # Gross-breakage protection stays, as RANGES rather than pins: a halved or sign-flipped metric
    # still fails, while a real re-run does not.
    c05 = rows["cwv0.5"]
    v = col(c05, "proposed_miou_mean")
    assert 0.0 < v <= 100.0, f"CWV=0.5 proposed mIoU out of range: {v}"
    assert doc_num(r"scores ([\d.]+) under band loss", "the CWV=0.5 band-loss score") == \
        pytest.approx(v, abs=0.15)

    # the fixed-core removal must COST both models something, and the docstring must quote what
    b2_cost = col(ceiling, "b2_miou_mean") - col(clean, "b2_miou_mean")
    prop_cost = col(ceiling, "proposed_miou_mean") - col(clean, "proposed_miou_mean")
    assert b2_cost > 0 and prop_cost > 0, \
        f"removing the absorption cores must cost accuracy; got B2 {b2_cost:.2f}, prop {prop_cost:.2f}"
    assert doc_num(r"costs B2 ([\d.]+) mIoU", "B2's fixed-core cost") == \
        pytest.approx(b2_cost, abs=0.15)
    assert doc_num(r"proposed only ([\d.]+)", "proposed's fixed-core cost") == \
        pytest.approx(prop_cost, abs=0.15)

    # the clean margin: whatever it is, the docstring must quote THAT
    margin = col(clean, "proposed_miou_mean") - col(clean, "b2_miou_mean")
    assert doc_num(r"\+([\d.]+) clean", "the headline clean margin") == \
        pytest.approx(margin, abs=0.15)

    # the parity claim is a COLUMN, so check the artefact actually carries it. On the corrected
    # 2026-07-20 axis no group crosses the absence threshold at ANY CWV -- the mask mechanism is
    # inert in all four conditions, which is the claim the margin above must be read against.
    for cond in ("clean", "cwv0.5", "cwv2.0", "cwv4.0"):
        assert int(float(rows[cond]["n_absent_groups"])) == 0, \
            f"{cond} reports an absent group; the corrected axis has none at any CWV"


def _args(**over):
    """A minimal args namespace with valid defaults; override one field per test."""
    import argparse as _ap
    d = dict(seeds=[0, 1, 2], groups=10, epochs=60, jobs=None, tau_missing=0.5,
             group_absent_frac=0.5, tau_sweep=None, random_control=False, smoke=False)
    d.update(over)
    return _ap.Namespace(**d)


@pytest.mark.parametrize("over,frag", [
    (dict(seeds=[0, 0, 1]), "distinct"),          # a repeated seed is not a second replicate
    (dict(groups=1), ">= 2"),                     # SGMAE + group dropout become no-ops
    (dict(groups=10_000), "band count"),          # more groups than bands
    (dict(epochs=0), "epochs"),
    (dict(jobs=0), "jobs"),
    (dict(tau_missing=1.5), "tau-missing"),       # not a transmittance
    (dict(tau_missing=-0.1), "tau-missing"),
    (dict(group_absent_frac=2.0), "group-absent-frac"),
    (dict(tau_sweep=[0.3, 1.7]), "tau-sweep"),
])
def test_validate_args_rejects_configurations_that_would_silently_mislead(over, frag):
    """Every one of these produces a PLAUSIBLE WRONG NUMBER rather than a crash if it slips through.

    These checks lived inline in main(), where no test could reach them: a mutation deleting all of
    them left the suite green. Reachability is the whole point of extracting validate_args.
    """
    with pytest.raises(ValueError, match=frag):
        P3.validate_args(_args(**over))


def test_validate_args_accepts_the_defaults_and_normalises_the_sweep():
    a = P3.validate_args(_args())
    assert a.tau_sweep is None                                  # absent flag stays absent
    a = P3.validate_args(_args(tau_sweep=[]))                   # flag given with no values
    assert a.tau_sweep == [0.3, 0.4, 0.5, 0.6, 0.7], "the default grid must be filled in"
    a = P3.validate_args(_args(tau_sweep=[0.7, 0.3, 0.7]))
    assert a.tau_sweep == [0.3, 0.7], "the sweep must be deduped and sorted"


@_needs_lut
def test_provenance_payload_records_identity_and_the_caveats_as_DATA():
    """Asserts on the stamped PAYLOAD, not on the source text that builds it.

    The earlier guard checked for magic words in the source. Mutations satisfied it by (a) deleting
    the whole stamp and leaving the words in a comment and (b) keeping the token 'UNVERIFIED' while
    inverting the claim. Only the dict is evidence of what gets written.
    """
    cube = np.array([[955.0, 9604.0], [1000.0, 2000.0]])
    extra = P3.provenance_extra(_args(), cube, {"clean": 0, 4.0: 1}, {"clean": 8, 4.0: 37})

    for k in ("lut_sha256", "cube_sha256", "gt_sha256", "wavelength_axis_sha256",
              "cube_physical_quantity", "cube_value_range", "lut_keys_used",
              "n_absent_groups_per_condition", "information_parity_note",
              "benchmark_bands_removed_upstream", "wavelength_axis_provenance"):
        assert k in extra, f"provenance payload lost {k}"

    # The claim must be MACHINE-READABLE. Prose alone is not checkable: a mutation rewrote the
    # sentence to "Actually a VERIFIED BOA surface-reflectance product" while keeping the token
    # UNVERIFIED, and passed a guard that searched for that token.
    assert extra["cube_physical_quantity_verified"] is False, \
        "the payload now claims the cube's physical quantity IS verified; nothing establishes that"
    q = extra["cube_physical_quantity"]
    assert q.startswith("UNVERIFIED."), \
        f"the caveat must OPEN with the verdict, got {q[:40]!r} (a token prefix is not a verdict)"
    assert "does NOT establish" in q and "not labelled L2A" in q
    assert "VERIFIED BOA" not in q, "the prose contradicts the flag"
    # and it must not have over-corrected into the opposite unsupported claim
    assert "consistent with scaled DN/radiance AND with reflectance" in q

    assert extra["cube_value_range"] == [955.0, 9604.0]
    # keys stringified (JSON), values kept as ints so a reader can compare them numerically
    assert extra["n_absent_groups_per_condition"] == {"clean": 0, "4.0": 1}
    assert extra["n_missing_bands_per_condition"] == {"clean": 8, "4.0": 37}
    assert "NOMINAL, NOT MEASURED" in extra["wavelength_axis_provenance"]
    # the parity note must carry its own limit, or it reads as a closed defence
    assert "does NOT make the margin non-oracular" in extra["information_parity_note"]


def test_transmittance_tie_break_is_strict():
    """T exactly AT tau counts as PRESENT (`T < tau`), not missing.

    Pinned for the same reason the group-fraction tie is pinned: it is a one-character decision that
    changes which bands the experiment removes, and no shipped T value sits exactly on it today --
    so nothing else would notice if it flipped."""
    T = np.array([0.5, 0.4999, 0.5001])
    m = P3.physics_missing_bands(T, np.ones(3, bool), tau_missing=0.5)
    assert list(m) == [False, True, False]


def test_eval_functions_actually_enter_eval_mode():
    """`eval_mode` is tested in isolation elsewhere; this checks it is WIRED IN.

    Removing `with eval_mode(model):` from both eval functions left the suite green, because the
    context manager's own unit test still passed. A helper nothing calls is not a guard.
    """
    tree = ast.parse(_SRC)
    for name in ("eval_proposed", "eval_mlp"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        used = any(isinstance(n, ast.With) and any(
            isinstance(i.context_expr, ast.Call)
            and getattr(i.context_expr.func, "id", None) == "eval_mode"
            for i in n.items) for n in ast.walk(fn))
        assert used, f"{name} does not enter eval_mode; the model's mode is then the caller's luck"


def test_random_control_rng_is_deterministic_and_differs_per_condition():
    """The control's reproducibility claim, measured rather than proxied.

    The existing guard only asserts `hash()` is absent from run_seed, which is a proxy for a property
    it never checks. Dropping `cwv` from the seed vector -- making all three CWV cells draw the
    IDENTICAL band set, so the control silently stops being three independent draws -- passes that
    proxy. This reproduces the production derivation and checks both properties directly.
    """
    def draw(seed, cwv, n, C=200):
        rng = P3.control_rng(seed, cwv)              # the PRODUCTION derivation, not a copy of it
        m = np.zeros(C, bool)
        m[rng.choice(C, size=n, replace=False)] = True
        return m

    assert np.array_equal(draw(0, 2.0, 27), draw(0, 2.0, 27)), "not reproducible"
    assert not np.array_equal(draw(0, 0.5, 27), draw(0, 2.0, 27)), \
        "the CWV cells draw the same band set; the control is one draw reported three times"
    assert not np.array_equal(draw(0, 2.0, 27), draw(1, 2.0, 27)), "seeds do not vary the draw"
    assert draw(0, 2.0, 27).sum() == 27


def test_check_results_complete_refuses_a_short_or_holed_result_list():
    """zip() TRUNCATES SILENTLY, so a short results list would produce a table averaged over fewer
    seeds than its own provenance claims -- undetectable by any reader. Deleting this guard used to
    leave the suite green because it lived inline in main()."""
    with pytest.raises(RuntimeError, match="short-averaged"):
        P3.check_results_complete([{"a": 1}], [0, 1, 2])
    with pytest.raises(RuntimeError, match="None result"):
        P3.check_results_complete([{"a": 1}, None], [0, 1])
    ok = [{"a": 1}, {"a": 2}]
    assert P3.check_results_complete(ok, [0, 1]) is ok


def test_check_mask_invariant_catches_a_seed_dependent_mask():
    """n_absent / n_missing_bands are functions of (LUT, tau, keep, groups) and NEVER of the seed.
    They used to be overwritten per seed, reporting the last seed's value as everyone's."""
    absent, missing = {}, {}
    first = {"n_absent": 1, "n_missing_bands": 37}
    P3.check_mask_invariant(4.0, 0, first, absent, missing)          # nothing recorded yet: no-op
    absent[4.0], missing[4.0] = 1, 37
    P3.check_mask_invariant(4.0, 1, first, absent, missing)          # agrees: still fine
    with pytest.raises(RuntimeError, match="absent groups"):
        P3.check_mask_invariant(4.0, 2, {"n_absent": 2, "n_missing_bands": 37}, absent, missing)
    with pytest.raises(RuntimeError, match="missing bands"):
        P3.check_mask_invariant(4.0, 2, {"n_absent": 1, "n_missing_bands": 36}, absent, missing)


@_needs_lut
def test_band_loss_geometry_reports_the_counts_and_fires_the_knife_edge_warning():
    """The knife-edge disclosure must actually reach the log, and the quoted band counts must be
    the ones the run produces.

    Deleting the warning branch left the suite green: a warning nothing asserts on can stop firing
    silently, and this one carries the fact that CWV=4.0 flips on `>=` vs `>`. The counts
    8/15/27/37 are quoted in the module docstring and were asserted nowhere.
    """
    lines = P3.band_loss_geometry_report(P3.TAU_MISSING, P3.GROUP_ABSENT_FRAC, 10)
    text = "\n".join(lines)
    for cwv, n in [(0.5, 19), (2.0, 32), (4.0, 40)]:   # corrected-axis geometry
        assert f"CWV={cwv}: {n}/200 bands missing" in text, f"CWV={cwv} is no longer {n} bands"
    # Corrected axis: no whole group crosses the default threshold at ANY CWV (fractions top out
    # at 0.40), so the report must say so at every level -- absence of absence is the finding.
    assert "absent groups []" in text and text.count("absent groups []") == 3, \
        "no group is absent at the default threshold on the corrected axis"
    # At the DEFAULT threshold nothing sits near the boundary on this axis, so no knife edge may
    # fire there (silence is the disclosure being truthful). The warning path itself is exercised
    # where real groups DO sit within one eps: threshold 0.3.
    assert "[knife edge]" not in text, "a knife edge fired with every group >= 2 eps from 0.5"
    fired = P3.band_loss_geometry_report(P3.TAU_MISSING, 0.3, 10)
    assert any("[knife edge]" in l for l in fired), \
        "the knife-edge disclosure never reaches the reader at a threshold real groups sit near"


@_needs_lut
def test_knife_edge_warning_is_deterministic_at_the_epsilon_boundary():
    """Re-derived at the 2026-07-20 axis correction: group fractions are a function of the axis.
    On the corrected-axis LUT the group fractions top out at 0.40, so the DEFAULT threshold (0.5)
    sits >= 2 eps from every group and must be QUIET -- silence is the disclosure working. At
    threshold 0.3 real groups sit at 0.30 (margin 0) and 0.25/0.35 (margin one eps), so the warning
    must fire deterministically -- the tolerance-based comparison, not floating-point luck."""
    fired = [l for l in P3.band_loss_geometry_report(P3.TAU_MISSING, 0.3, 10)
             if "[knife edge]" in l]
    assert len(fired) >= 2, "groups at 0.25/0.30/0.35 sit within eps of a 0.3 threshold"
    # the default threshold is far from every group fraction on this axis and must NOT warn
    quiet = P3.band_loss_geometry_report(P3.TAU_MISSING, P3.GROUP_ABSENT_FRAC, 10)
    assert not any("[knife edge]" in l for l in quiet), \
        "no group is within eps of the default threshold on the corrected axis; firing would be noise"
    # and a threshold far from everything stays quiet too
    quiet9 = P3.band_loss_geometry_report(P3.TAU_MISSING, 0.9, 10)
    assert not any("[knife edge]" in l for l in quiet9)


@_needs_lut
def test_an_absent_group_costs_the_proposed_model_its_surviving_bands():
    """The docstring claims that where a group IS absent, proposed sees strictly LESS band content
    than B2, because the mask token replaces the whole group including its survivors.

    Measured on the real CWV=4.0 mask ON THE CORRECTED AXIS: at the default threshold no group is
    absent at all (fractions top out at 0.40), so the absence regime is exercised at
    group_absent_frac=0.4, where group 5 (8 of 20 bands missing) crosses the inclusive threshold
    and its 12 SURVIVING bands are discarded from proposed's input while B2 keeps them. Same claim,
    re-derived geometry.
    """
    import torch
    from bandsim.model import GroupedCrossBandAttention
    from phase2_degradation import group_present_mask

    groups = contiguous_groups(len(AVIRIS_WL_NM), 10)
    missing = _masks()[4.0]
    absent, fracs = P3.absent_groups_for(missing, groups, 0.4)
    assert absent == [5]
    survivors = [int(b) for b in groups[5] if not missing[b]]
    assert len(survivors) == 12, f"group 5 should have 12 surviving bands, has {len(survivors)}"

    cwl = [float(np.mean(AVIRIS_WL_NM[np.asarray(g)])) for g in groups]
    model = GroupedCrossBandAttention(groups, cwl, 16)
    model.eval()
    x = torch.randn(3, len(AVIRIS_WL_NM))
    pm = torch.from_numpy(group_present_mask(3, groups, absent))
    with torch.no_grad():
        base = model(x, pm)
        # perturb ONLY the surviving bands of the absent group; a model that could see them
        # would change its output. It cannot, because the group is replaced by the mask token.
        x2 = x.clone(); x2[:, survivors] += 1000.0
        assert torch.equal(base, model(x2, pm)), \
            "proposed responded to the absent group's surviving bands; it is not discarding them"
        # and with the group PRESENT those same bands do matter -- otherwise the test is vacuous
        pm_all = torch.from_numpy(group_present_mask(3, groups, []))
        assert not torch.equal(model(x, pm_all), model(x2, pm_all))


def test_cwv_transmittance_still_demands_the_wavelength_axis_identity(tmp_path):
    """Reinstating the r3/r4 P0 must fail LOUDLY.

    A LUT computed on a gapless linspace(400,2500,200) has the right LENGTH and loads cleanly, while
    183 of 200 bands land on the wrong channel (worst 98 nm off) -- near-transparent bands get
    multiplied by deep-absorption transmittance and every number in this phase changes silently.
    `load_cached_transmittance` implements the guard, but phase3's CALL is what has to request it:
    dropping `expected_wavelengths_nm=` there survives every test of the loader.
    """
    p = tmp_path / "wrongaxis.npz"
    np.savez(p, wl_nm=np.linspace(400.0, 2500.0, len(AVIRIS_WL_NM)),   # same length, wrong axis
             generation_mode="direct_6s", **{"cwv2.0": np.full(len(AVIRIS_WL_NM), 0.8)})
    with pytest.raises(ValueError, match="axis"):
        P3.cwv_transmittance(str(p), 2.0)


def test_run_seed_keeps_the_spatial_guard_band(monkeypatch):
    """guard=1 drops test pixels adjacent to train across a block seam. Setting guard=0 reintroduces
    adjacency leakage, inflates every score, and changes nothing a helper test can see.

    Captured from the actual call rather than asserted on source text, so a renamed keyword or a
    positional argument cannot slip past.
    """
    seen = {}
    real = P3.disjoint_block_split
    monkeypatch.setattr(P3, "disjoint_block_split",
                        lambda gt, **kw: seen.update(kw) or real(gt, **kw))
    monkeypatch.setattr(P3, "train_mlp", lambda *a, **k: None)
    monkeypatch.setattr(P3, "pretrain_sgmae", lambda *a, **k: None)
    monkeypatch.setattr(P3, "finetune_proposed", lambda *a, **k: None)
    monkeypatch.setattr(P3, "GroupedCrossBandAttention", lambda *a, **k: None)
    gt = np.tile(np.arange(1, 17).reshape(4, 4), (10, 10))
    cube = np.zeros((40, 40, len(AVIRIS_WL_NM)))
    with pytest.raises(Exception):          # the stubbed trainers abort it right after the split
        P3.run_seed(0, cube, gt, n_groups=10, epochs=1)
    assert seen.get("guard") == 1, f"run_seed split with guard={seen.get('guard')}, not 1 (leakage)"


def test_resolve_cwv_key_does_not_let_one_cwv_swallow_another(tmp_path):
    """A bare `startswith(pref)` makes `cwv2.0` match `cwv2.05`, so the run would silently take a
    different atmosphere than the one it reports. The separator is what makes the prefix a key."""
    p = tmp_path / "T.npz"
    np.savez(p, wl_nm=AVIRIS_WL_NM, generation_mode="direct_6s",
             **{"cwv2.0": np.full(200, 0.8), "cwv2.05": np.full(200, 0.2)})
    assert P3.resolve_cwv_key(str(p), 2.0) == "cwv2.0"


def test_eval_mode_restores_the_previous_mode_even_when_the_body_raises():
    """Without try/finally an exception inside the block leaves the model in eval mode. The caller
    would then keep 'training' a model whose dropout/BatchNorm never switch back on -- silently, and
    only on the error path, which is where nobody looks."""
    import torch
    m = torch.nn.Linear(2, 2)
    m.train()
    with pytest.raises(RuntimeError):
        with P3.eval_mode(m):
            raise RuntimeError("boom")
    assert m.training, "eval_mode leaked eval mode out of a failed block (missing try/finally)"


def test_control_rng_is_not_derived_from_pythons_randomised_hash():
    """AST-checked on control_rng ITSELF.

    The previous guard inspected `run_seed`, which is where the derivation used to live; when it
    moved to control_rng the guard did not follow, and putting hash() back inside control_rng passed
    both that check and the behavioural reproducibility test -- because hash() is stable WITHIN a
    process, so a same-process test can never detect the per-process salting that breaks
    reproducibility across run_jobs' spawned workers.
    """
    tree = ast.parse(_SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "control_rng")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "hash" not in called, (
        "control_rng derives its seed from hash(), which Python salts per process unless "
        "PYTHONHASHSEED is pinned -- run_jobs SPAWNS workers, so the 'reproducible' control would "
        "differ every run and per worker")


@_needs_lut
def test_knife_edge_threshold_is_inclusive_of_exactly_epsilon():
    """The determinism the tolerance exists to provide, tested at the boundary itself.

    Asserting only that the warning fires on the shipped LUT does NOT test this: CWV=2.0's group
    sits at 0.45 and abs(0.45-0.5) evaluates to 0.049999999999999996, so a bare `< eps` fires there
    too -- by the very floating-point accident the tolerance removes. A fraction exactly eps away
    must warn, deterministically.
    """
    # 20-band groups, so fractions step by 0.05 and a group can land exactly one eps from the
    # threshold. 11/20 = 0.55, and abs(0.55 - 0.5) evaluates to 0.05000000000000004 -- a hair ABOVE
    # eps. This is the discriminating case: a bare `< eps` refuses to warn, the tolerant form warns.
    groups = [np.arange(0, 20), np.arange(20, 40)]
    frac = P3.GROUP_ABSENT_FRAC
    missing = np.zeros(40, bool); missing[:11] = True
    _, fracs = P3.absent_groups_for(missing, groups, frac)
    margin = float(np.min(np.abs(fracs - frac)))
    assert margin == pytest.approx(P3.GROUP_FRAC_EPS, abs=1e-9), \
        f"precondition: this group must sit one eps from the threshold, got margin={margin!r}"
    assert margin > P3.GROUP_FRAC_EPS, (
        "precondition: float representation must put this margin just ABOVE eps, which is what "
        "makes a bare `<` comparison silently refuse to warn")
    assert margin <= P3.GROUP_FRAC_EPS + 1e-12, \
        "a group exactly one eps from the threshold must still count as a knife edge"

    # And exercise the PRODUCTION comparison, not just the arithmetic above. Re-derived at the
    # 2026-07-20 axis correction (the geometry is a function of the axis): on the CORRECTED-axis
    # LUT a CWV=0.5 group sits at 0.15, so at group_absent_frac=0.2 the margin is
    # abs(0.2-0.15) == 0.05000000000000001 -- a hair above eps. A bare `< eps` refuses to warn
    # there; the tolerant form warns. Without this call the whole test passes with the comparison
    # reverted, which is exactly what happened.
    fired = [l for l in P3.band_loss_geometry_report(P3.TAU_MISSING, 0.2, 10) if "[knife edge]" in l]
    assert fired, (
        "band_loss_geometry_report did not warn for a group one eps from the threshold; the "
        "comparison is a bare `<` and the disclosure depends on floating-point luck")


def test_seed_sd_is_the_sample_sd_and_undefined_for_one_seed():
    """Cross-seed dispersion estimates the run-to-run variability we generalise over, so it needs
    ddof=1. np.std's default ddof=0 treats the seeds as the whole population and is too small by
    sqrt((n-1)/n) -- 18% at the 3 seeds this phase ships. An earlier revision named the column
    `_std_pop` and documented ddof=0, which is honest labelling of the wrong estimator.

    One seed must give NaN, not 0.0: a --smoke run has no dispersion information, and printing a
    spread of exactly zero reads as perfect agreement rather than as nothing measured.
    """
    assert P3.seed_sd([10.0, 20.0]) == pytest.approx(7.0710678, abs=1e-6)   # ddof=1, not 5.0
    assert np.isnan(P3.seed_sd([3.0]))
    assert np.isnan(P3.seed_sd([]))


def test_summary_std_columns_are_sample_sd(tmp_path):
    """The published +/- must be the sample sd. A silent revert to the population formula narrows
    every error bar in the paper's figure while the header keeps promising otherwise."""
    agg = {"clean": {"proposed": [10.0, 20.0], "b2": [10.0, 20.0], "proposed_nomask": [10.0, 20.0],
                     "proposed_uncorrected": [1.0, 2.0], "b2_uncorrected": [1.0, 2.0],
                     "proposed_full": [10.0, 20.0], "b2_full": [10.0, 20.0]}}
    rows = P3.summary_rows(agg, {}, {}, ["clean"], 2)
    hdr = rows[0]
    # EVERY std column on EVERY row -- the ceiling row builds its own, from a different array, so
    # checking only the clean row leaves half the writer unguarded (and a mutation there survived).
    for row in rows[1:]:
        for col_name in ("proposed_miou_std_sample", "b2_miou_std_sample"):
            got = float(row[hdr.index(col_name)])
            # abs=0.01 because the CSV writes 2 decimals; the population formula would give 5.00,
            # which is far outside this tolerance, so the check still separates the two estimators.
            assert got == pytest.approx(7.07, abs=0.01), (
                f"{row[0]}.{col_name}: sample sd of [10,20] is 7.07 (the population formula gives "
                f"5.0, understating run-to-run variability by 18% at n=3); got {got}")


def test_the_macro_estimand_is_a_fixed_class_set_shared_with_phases_1_and_2():
    """`miou` averages over whatever classes a split happens to contain, so each seed reported a
    DIFFERENT estimand and the mean over seeds averaged different quantities.

    Measured on this config: seeds 0/1/2 each see 15 of 16 classes but not the same 15, while seeds
    3/4 see all 16. Phases 1 and 2 fixed this by scoring on the classes present in EVERY split
    (phase2_degradation.common_class_set); until phase 3 did the same, its mIoU was not comparable
    with theirs in one table. Bare `miou` must not be reachable from this module, or a stray call
    silently reintroduces the drift.
    """
    tree = ast.parse(_SRC)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "miou_over" in called, "the fixed-class-set metric is not used"
    assert "miou" not in called, "a bare miou() call reintroduces the drifting estimand"
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "miou" not in imported, "bandsim.metrics.miou must not be importable here by accident"
    assert "common_class_set" in imported and "miou_over" in imported


def test_eval_functions_score_on_the_fixed_class_set_not_whatever_is_present():
    """BEHAVIOURAL, not structural: the two estimands must give DIFFERENT numbers here, and phase 3
    must report the fixed one.

    Structural checks could not catch this. `miou_over(..., None)` still calls miou_over, and the
    end-to-end fixture cannot see the difference at all because its synthetic ground truth carries
    every class in every split — and with a single seed the common set IS the present set, so a
    --smoke run can never distinguish them. This constructs the case where they diverge.
    """
    import torch
    from bandsim.metrics import miou as _bare_miou
    from phase2_degradation import miou_over as _over

    n_cls = P3.NUM_CLASSES
    # The real shape of the fix: the common set is a SUBSET of the classes a given split contains,
    # because rare classes drop out of some splits. (A set containing an ABSENT class would give
    # NaN — per_class_iou returns NaN there — which is precisely why common_class_set only ever
    # keeps classes present in every split.)
    yte = np.array([1] * 8 + [2] * 8)
    pred = np.array([1] * 8 + [1] * 8)         # perfect on class 1, wrong on class 2
    fixed = [1]                                # the class that survives in every split
    drift = _bare_miou(yte, pred, n_cls)       # averages classes 1 AND 2
    fix = _over(yte, pred, n_cls, fixed)       # averages class 1 only
    assert fix > drift, "precondition: the two estimands must differ for this test to prove anything"

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, present_mask=None):
            out = torch.full((x.shape[0], n_cls), -9.0)
            out[torch.arange(x.shape[0]), torch.from_numpy(pred).long()] = 9.0
            return out

    X = np.zeros((16, 4), np.float32)
    groups = [np.arange(0, 2), np.arange(2, 4)]
    assert P3.eval_mlp(_Stub(), X, yte, fixed) == pytest.approx(fix), \
        "eval_mlp ignored the class set and reported the drifting estimand"
    assert P3.eval_proposed(_Stub(), X, yte, groups, [], fixed) == pytest.approx(fix), \
        "eval_proposed ignored the class set and reported the drifting estimand"
    # and with class_set=None they fall back to the drifting one, which is the legacy contract
    assert P3.eval_mlp(_Stub(), X, yte, None) == pytest.approx(drift)


@_needs_lut
def test_the_common_class_set_is_stable_across_the_seed_lists_phase3_ships():
    """The class set depends on WHICH seeds run, which would be a new (if smaller) drift. Measured:
    it is 14 of 16 for both the 3-seed and the 5-seed list, so the estimand does not move between
    the two configurations this phase actually uses."""
    from phase2_degradation import load_data as _ld, common_class_set
    _, gt = _ld()
    a, _ = common_class_set(gt, 10, [0, 1, 2], guard=1, num_classes=16)
    b, _ = common_class_set(gt, 10, [0, 1, 2, 3, 4], guard=1, num_classes=16)
    assert list(a) == list(b), f"the estimand changes with the seed list: {a} vs {b}"
    assert len(a) == 14, f"the common class set is now {len(a)}/16; the docstring says 14"


def test_the_class_set_reaches_the_workers_and_the_metric():
    """Two wiring checks, because a computed-but-unused class set is the drift with extra steps:
    it must cross the spawn boundary via run_jobs(shared=...), and run_seed must hand it to the
    eval functions."""
    tree = ast.parse(_SRC)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "run_jobs")
    shared = next(k for k in call.keywords if k.arg == "shared")
    kw = {k.arg: k.value for k in shared.value.keywords}
    assert "class_set" in kw, \
        "class_set does not cross the spawn boundary; workers would score the drifting estimand"
    # The VALUE, not just the keyword name. `class_set=None` keeps the keyword and silently
    # restores the drifting estimand in every worker — which is exactly what a mutation did while
    # this test still passed.
    assert isinstance(kw["class_set"], ast.Name) and kw["class_set"].id == "class_set", \
        f"class_set is passed as {ast.unparse(kw['class_set'])}, not the computed set"
    run_seed = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "run_seed")
    body = ast.unparse(run_seed)
    assert "eval_proposed(m_prop, Xte_corr, yte, groups, absent, class_set)" in body
    assert "eval_mlp(m_b2, Xte_corr, yte, class_set)" in body


def test_this_files_own_docstring_does_not_cite_tests_that_no_longer_exist():
    """The drift guard, applied to the drift guard.

    This module's docstring names the test that pins each withdrawn claim. Four of those names were
    stale after a rename -- documentation drift inside the file whose whole purpose is catching
    documentation drift. A reader following a dead name concludes the claim is unguarded.
    """
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    doc = ast.get_docstring(tree) or ""
    cited = {w.strip("(),.;*") for w in doc.split() if w.strip("(),.;*").startswith("test_")}
    missing = sorted(c for c in cited if c not in defined)
    assert not missing, f"module docstring cites tests that do not exist: {missing}"


def test_this_file_is_not_silently_skipping_its_lut_backed_guards():
    """Six tests carry @_needs_lut and one self-skips on a missing CSV, so a green run can mean
    'nothing ran'. The shipped repo HAS the table, so on this repo the skips must not engage --
    the same countermeasure tests/test_smoke_isolation.py uses for its discovery."""
    assert os.path.exists(os.path.join(_ROOT, "data", "srf_cache", "T_6s_grid.npz")), \
        "the 6S table is missing, so the LUT-backed guards in this file are all skipping"
    assert _HAS_LUT


def test_the_documented_upstream_band_removal_matches_the_actual_axis_construction():
    """The docstring states which bands the benchmark already lacks; check it against the SOURCE OF
    TRUTH, not against itself.

    The earlier version asserted the string "104-108, 150-163, 220" appeared in the very file that
    string was quoted from -- the docstring checking the docstring. bandsim.io is where the removal
    actually happens, and that is what the claim has to match.
    """
    from bandsim import io as bio
    removed = list(bio._IP_REMOVED_BANDS_1BASED)
    assert removed == list(range(104, 109)) + list(range(150, 164)) + [220]
    assert len(removed) == 20
    assert len(AVIRIS_WL_NM) == 220 - 20 == 200
    # the removal must leave REAL GAPS on the axis, which is what makes the hard-mask alignment and
    # the wavelength PE meaningful -- a gapless linspace(400,2500,200) would pass a length check
    steps = np.diff(AVIRIS_WL_NM)
    assert steps.max() > 3 * np.median(steps), "the axis has no gaps; it is not the AVIRIS axis"
    assert "104-108, 150-163, 220" in _SRC, "the docstring must quote the removal it describes"


# ---------------------------------------------------------------------------------------------
# 7 — reproducibility of the optional arms
# ---------------------------------------------------------------------------------------------
def test_random_control_seed_is_not_derived_from_pythons_randomised_hash():
    """`hash()` on str/tuple is salted per process unless PYTHONHASHSEED is pinned, and run_jobs
    SPAWNS its workers — a hash-derived seed would give a different 'reproducible' control on every
    run and a different one per worker."""
    tree = ast.parse(_SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_seed")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "hash" not in called, "derive the control's RNG seed arithmetically, not via hash()"


def test_run_seed_takes_the_thresholds_as_parameters_not_globals():
    """run_jobs' parallel path re-imports this module in a spawned worker, so a threshold assigned
    in the parent's namespace would never reach it: the worker would threshold at the default while
    the CSV reported the swept value."""
    import inspect
    sig = inspect.signature(P3.run_seed)
    for p in ("tau_missing", "group_absent_frac", "tau_sweep", "random_control"):
        assert p in sig.parameters, f"run_seed must receive {p} explicitly (spawn-safe)"
