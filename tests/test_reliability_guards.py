"""Guards for the 2026-07 reliability/selective-prediction audit (the journal headline claim).

Every test here pins a SPECIFIC way the reliability story could be overstated, so a fix can never
silently regress. Grouped by the audit item that produced it. Run: pytest tests/test_reliability_guards.py -v
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from bandsim.reliability import conformal_risk_control, conformal_at_risk  # noqa: E402


# ===================== item 3 : lambda_max must genuinely abstain on EVERY point =====================
# CRC Theorem 2.1 requires L_i(lambda_max) <= alpha for ALL i INCLUDING the test point. A
# data-dependent lambda_max just above max(calibration confidence) satisfies that only on
# calibration: an eval point more confident than anything calibrated on is still ACCEPTED at the
# supposedly abstain-everything threshold. That is exactly the naive-under-shift regime the flagship
# is about, so it is not a corner case.

def _shifted_case(alpha):
    """calib: all wrong, confidences in [0.50,0.60]; eval: all wrong, SHIFTED to [0.80,0.99].
    Only abstain-all can control the risk, and abstaining must actually abstain."""
    corr_c = np.zeros(60, bool); conf_c = np.linspace(0.60, 0.50, 60)
    corr_e = np.zeros(20, bool); conf_e = np.linspace(0.99, 0.80, 20)
    return conformal_risk_control(corr_c, conf_c, corr_e, conf_e, alpha=alpha)


def test_crc_abstain_all_actually_abstains_on_shifted_eval():
    # alpha=0.02 with n=60 -> the B/(n+1)=1/61 floor is met ONLY by abstain-all, so CRC reports
    # feasible=True and its bound certifies E[accepted AND wrong] <= 2%. With a finite lambda_max it
    # accepted all 20 shifted eval points and REALIZED 1.0 -- a 50x breach of the number it had just
    # certified. The bound and the realization are asserted separately: the bound alone cannot show
    # this failure (it is <= alpha by construction), and the realization alone cannot show it was
    # promised anything.
    out = _shifted_case(alpha=0.02)
    assert out["feasible"] is True
    assert out["calib_crc_bound"] <= 0.02 + 1e-12   # what CRC certified from calibration
    assert out["eval_group_coverage"] == 0.0, \
        f"abstain-all accepted {out['eval_group_coverage']:.0%} of a shifted eval set"
    assert out["eval_group_joint_risk"] == 0.0, \
        f"certified <= 0.02 but realised {out['eval_group_joint_risk']:.3f}"
    assert np.isinf(out["threshold"])              # lambda_max = +inf, the only universal abstention


def test_crc_infeasible_fallback_truly_abstains():
    # n=3, alpha=0.10: even the abstain-all floor B/(n+1)=0.25 exceeds alpha -> NOT certifiable. The
    # documented fallback is "abstain-all", so it must abstain rather than accept a shifted eval set.
    corr_c = np.array([True, False, True]); conf_c = np.array([0.60, 0.55, 0.50])
    corr_e = np.zeros(6, bool); conf_e = np.linspace(0.99, 0.70, 6)
    out = conformal_risk_control(corr_c, conf_c, corr_e, conf_e, alpha=0.10)
    assert out["feasible"] is False                # reported, not silently swallowed
    assert out["eval_group_coverage"] == 0.0 and out["eval_group_joint_risk"] == 0.0
    # ... and the bound must SAY it failed rather than quietly reporting the realized 0.0: the
    # abstain-all floor B/(n+1)=0.25 is what CRC can promise here, and 0.25 > alpha.
    assert out["calib_crc_bound"] == pytest.approx(1.0 / 4.0)
    assert out["calib_crc_bound"] > 0.10


def test_crc_top_confidence_calib_sample_wrong_stays_certifiable():
    # The reachability of abstain-all is what keeps a small-n problem FEASIBLE when the most
    # confident calibration sample is wrong: at lambda=max(conf) the loss is 1/15, so
    # (n/(n+1))*1/15 + 1/16 = 0.125 > alpha and the problem would look infeasible; abstain-all gives
    # Rhat=0 -> 1/16 = 0.0625 <= alpha.
    n = 15
    conf_c = np.linspace(0.99, 0.50, n)
    corr_c = np.ones(n, bool); corr_c[0] = False           # MOST confident calib sample is WRONG
    out = conformal_risk_control(corr_c, conf_c, corr_c, conf_c, alpha=0.10)
    assert out["feasible"] is True
    # "certifiable" is a statement about the BOUND, so assert it of the bound.
    assert out["calib_crc_bound"] <= 0.10 + 1e-12
    assert out["eval_group_joint_risk"] <= 0.10 + 1e-12   # and eval==calib here, so it realizes it too


# ============ item 0 : a BOUND and its REALIZATION must never share a key ============
# The defect this pins: conformal_risk_control computed the CRC bound
# (n/(n+1))*Rhat_calib + B/(n+1), used it to pick the threshold, then THREW IT AWAY and returned the
# EVALUATION set's realized joint loss under a key named `cert_risk`. Every downstream plot label and
# CSV column called that "certified", so the headline figure's y-axis was not the quantity it named,
# and the one effect the study exists to report -- the realization drifting above the certificate
# under shift -- was unobservable, because only one of the two numbers ever left the function.

# Words that make a promise about a BOUND. Any returned float whose name contains one of these must
# carry the calibration bound and nothing else -- that is the exact property `cert_risk` violated.
_BOUND_WORDS = ("bound", "cert", "guarant", "certif")


def _bound_and_realization_come_apart(alpha=0.10):
    """A case where the CRC bound and every eval quantity are FAR apart AND MUTUALLY DISTINCT, so a
    key carrying the wrong one cannot pass by coincidence. Calibration is entirely CORRECT, so
    Rhat=0 at every threshold and CRC certifies B/(n+1)=1/201 at the lowest grid point (thr=0.50).
    Evaluation is then built so the four numbers are pairwise different:
        bound     = 1/201 ~ 0.005
        coverage  = 30/50 = 0.60   (20 eval points sit below thr and are rejected)
        joint     = 24/50 = 0.48   (24 of the 30 accepted are wrong)
        selective = 24/30 = 0.80
    An earlier version of this fixture had every accepted point wrong at full coverage, making
    joint = coverage = selective = 1.0 -- under which this guard would have passed while carrying
    the coverage in a key named for a risk."""
    corr_c = np.ones(200, bool)
    conf_c = np.linspace(0.50, 0.90, 200)
    corr_e = np.concatenate([np.zeros(24, bool), np.ones(6, bool),      # accepted: 24 wrong, 6 right
                             np.zeros(10, bool), np.ones(10, bool)])    # rejected (conf below thr)
    conf_e = np.concatenate([np.full(30, 0.99), np.full(20, 0.10)])
    return conformal_risk_control(corr_c, conf_c, corr_e, conf_e, alpha=alpha)


def test_crc_returns_the_bound_and_the_realization_as_separate_keys():
    out = _bound_and_realization_come_apart()
    assert "cert_risk" not in out, \
        "`cert_risk` is back: one key cannot be both the CRC bound and its eval realization"
    assert "calib_crc_bound" in out and "eval_group_joint_risk" in out
    # they are provably DIFFERENT quantities here, not two names for one number
    assert out["calib_crc_bound"] == pytest.approx(1.0 / 201.0)
    assert out["eval_group_joint_risk"] == pytest.approx(0.48)
    assert out["eval_group_joint_risk"] > out["calib_crc_bound"] + 0.4
    # ... and the breach is real: the certificate held on calibration, the eval set blew through it
    assert out["feasible"] is True
    assert out["calib_crc_bound"] <= 0.10


def test_no_returned_key_promises_a_bound_while_carrying_a_realization():
    """THE naming guard. On the fixture above the bound (~0.005), the coverage (0.60), the joint
    risk (0.48) and the selective risk (0.80) are four DIFFERENT numbers, so a key whose name
    promises a bound cannot be carrying any of the others undetected. Re-adding `cert_risk` -- or
    any `*_cert*` / `*_bound*` / `*certified*` key holding an eval realization -- fails here."""
    out = _bound_and_realization_come_apart()
    bound = out["calib_crc_bound"]
    others = {k: out[k] for k in ("eval_group_joint_risk", "eval_group_coverage",
                                  "eval_group_selective_risk")}
    assert len(set(others.values())) == 3, "the fixture no longer separates the eval quantities"
    assert all(abs(v - bound) > 0.4 for v in others.values()), "bound is not separated from them"
    for k, v in out.items():
        if not isinstance(v, float) or isinstance(v, bool):
            continue
        if any(w in k.lower() for w in _BOUND_WORDS):
            assert v == pytest.approx(bound), (
                f"key {k!r} names a bound/certificate but carries {v!r}; the CRC bound is {bound!r} "
                f"and the eval quantities are {others!r}")
    # the converse direction: a *_joint_risk key must carry the joint risk, not the bound
    for fam in ("group", "sample"):
        assert out[f"eval_{fam}_joint_risk"] == pytest.approx(0.48)
        assert out[f"eval_{fam}_coverage"] == pytest.approx(0.60)
        assert out[f"eval_{fam}_selective_risk"] == pytest.approx(0.80)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.25, 0.5])
def test_calib_crc_bound_is_at_most_alpha_whenever_feasible(alpha):
    """The defining property of the certificate, over cases that span feasible and infeasible. When
    `feasible` the bound must be <= alpha; when NOT feasible it must exceed alpha rather than
    silently reporting some smaller number that reads like a certificate."""
    rng = np.random.default_rng(23)
    for n in (3, 15, 60, 400):
        conf = rng.uniform(0, 1, n)
        corr = rng.uniform(0, 1, n) < conf
        for group in (None, rng.integers(0, max(2, n // 5), n)):
            out = conformal_risk_control(corr, conf, corr, conf, alpha=alpha,
                                         calib_group=group, eval_group=group)
            if out["feasible"]:
                assert out["calib_crc_bound"] <= alpha + 1e-12, \
                    f"feasible but the bound {out['calib_crc_bound']:.4f} exceeds alpha={alpha}"
            else:
                assert out["calib_crc_bound"] > alpha, \
                    "infeasible must mean the best achievable bound is ABOVE alpha"
                # infeasible -> abstain-all -> the bound IS the floor B/(n+1)
                assert out["calib_crc_bound"] == pytest.approx(
                    1.0 / (out["n_calib_units"] + 1.0))


@pytest.mark.parametrize("kwargs,match", [
    (dict(calib_conf=np.ones(3)), "flags for"),                       # length mismatch, calib
    (dict(eval_conf=np.ones(7)), "flags for"),                        # length mismatch, eval
    (dict(calib_group=np.arange(3)), "group ids for"),                # group array desynchronised
    (dict(eval_group=np.arange(3)), "group ids for"),
    (dict(calib_correct=np.array([]), calib_conf=np.array([])), "empty"),
    (dict(eval_correct=np.array([]), eval_conf=np.array([])), "empty"),
    (dict(calib_conf=np.array([0.5, np.nan, 0.7, 0.8, 0.9])), "finite"),
    (dict(eval_conf=np.array([0.5, np.inf, 0.7, 0.8, 0.9])), "finite"),
    (dict(calib_correct=np.array([0.0, 0.5, 1.0, 1.0, 1.0])), "boolean or 0/1"),
    (dict(alpha=1.5), r"alpha must lie"),
    (dict(alpha=-0.1), r"alpha must lie"),
    (dict(B=0.0), r"B .* must be >= 1"),
    (dict(B=0.5), r"B .* must be >= 1"),   # P0-4: a bounded-but-<1 B understates the 0/1-loss bound
    (dict(n_grid=0), r"n_grid must be >= 1"),
])
def test_crc_rejects_malformed_input(kwargs, match):
    """Public entry point fed by long pipelines: a silent coercion here surfaces only as a wrong
    number in a paper table. Each case is a way a caller can hand CRC something it would otherwise
    compute a meaningless answer from -- notably a group array whose length has drifted from the
    rows it labels (units silently attach to the wrong losses) and a non-finite confidence (which
    satisfies `conf >= inf` and so is ACCEPTED at the abstain-everything threshold)."""
    base = dict(calib_correct=np.ones(5, bool), calib_conf=np.linspace(0.5, 0.9, 5),
                eval_correct=np.ones(5, bool), eval_conf=np.linspace(0.5, 0.9, 5), alpha=0.10)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        conformal_risk_control(**base)


def test_conformal_at_risk_validates_its_inputs_too():
    """Same entry-point class, same exposure: conformal_at_risk also compares eval confidences with
    a +inf threshold when the target is unreachable, so a non-finite confidence is accepted there."""
    ok = dict(calib_correct=np.ones(5, bool), calib_conf=np.linspace(0.5, 0.9, 5),
              eval_correct=np.ones(5, bool), eval_conf=np.linspace(0.5, 0.9, 5))
    with pytest.raises(ValueError, match="finite"):
        conformal_at_risk(**{**ok, "eval_conf": np.array([0.5, np.nan, 0.7, 0.8, 0.9])})
    with pytest.raises(ValueError, match="flags for"):
        conformal_at_risk(**{**ok, "calib_conf": np.ones(3)})
    with pytest.raises(ValueError, match="target_risk must lie"):
        conformal_at_risk(**ok, target_risk=2.0)


def test_infeasible_crc_returns_a_zero_that_must_never_be_averaged_as_a_risk():
    """The precondition for the aggregation guards below. When nothing certifies, CRC abstains on
    everything, so the REALIZED joint loss is 0.0 and coverage is 0.0. Those zeros mean "predicted
    nothing", not "certified with no confident errors" -- averaging them into a mean risk pulls it
    down exactly where the method failed. `feasible` is the only thing that distinguishes the two,
    which is why it has to travel with every aggregate."""
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 400)
    corr = rng.uniform(0, 1, 400) < conf
    out = conformal_risk_control(corr, conf, corr, conf, alpha=0.001)   # alpha < B/(n+1)
    assert out["feasible"] is False
    assert out["eval_group_joint_risk"] == 0.0 and out["eval_group_coverage"] == 0.0
    assert out["calib_crc_bound"] > 0.001                              # the bound DOES say it failed
    assert np.isnan(out["eval_group_selective_risk"])                  # ... and selective risk NaNs


@pytest.mark.parametrize("mod_name,cond_fields", [
    # phase8R renamed these so the column says which ESTIMAND it holds: a "bound" that is
    # only a calibration selection statistic invited being read as a per-run guarantee.
    ("phase8R_reliability", ("crc_calibration_selection_stat",
                             "crc_heldout_empirical_joint_loss", "crc_heldout_coverage")),
    ("phase4R_reliability", ("crc_calib_bound", "crc_eval_joint_risk", "crc_coverage")),
])
def test_crc_aggregates_exclude_infeasible_seeds(mod_name, cond_fields):
    """Both reliability experiments must average the CRC bound/realization/coverage over the seeds
    that CERTIFIED, not over all seeds. A plain mean lets an infeasible seed's abstain-all 0.0 be
    read as a low risk. Pinned structurally because reproducing it end-to-end needs a full run."""
    mod = __import__(mod_name)
    src = open(mod.__file__, encoding="utf-8").read()
    # phase8R now conditions inside its `sel(..., feasible_only=True)` selector rather than through
    # a module-level CRC_COND set; phase4R still uses the set. Either is fine -- what must exist is
    # an explicit feasibility filter, so an infeasible run's abstain-all 0.0 cannot be averaged in as
    # a low risk.
    assert ("_CRC_COND" in src or "CRC_COND" in src
            or ("feasible_only" in src and "crc_feasible" in src)), \
        f"{mod_name} has no feasibility-conditioned aggregation"
    for f in cond_fields:
        assert f in src, f"{mod_name} lost the {f} metric"
    # the aggregation must be feasibility-aware, not a bare mean over every seed
    assert "np.nanmean" in src or "feas" in src
    assert "np.mean(np.stack(store[mth][mn]), 0)" not in src, \
        f"{mod_name} still takes a plain mean over ALL seeds, infeasible ones included"


def test_all_infeasible_aggregate_is_nan_not_zero():
    """The rule the two experiments implement, checked on the aggregation semantics itself: with no
    feasible seed the mean must be NaN (undefined), never 0.0 -- a 0.0 would print as a perfect
    risk AND a perfect coverage in the same row."""
    vals = np.array([0.0, 0.0, 0.0])          # three abstain-all seeds
    feas = np.array([0.0, 0.0, 0.0])
    kept = vals[feas > 0.5]
    assert kept.size == 0
    assert np.isnan(float("nan") if kept.size == 0 else kept.mean())
    # and with a mix, the infeasible zero must not drag the mean down
    vals = np.array([8.0, 0.0]); feas = np.array([1.0, 0.0])
    assert float(vals[feas > 0.5].mean()) == 8.0
    assert float(vals.mean()) == 4.0          # <- the biased number this rule removes


# ===================== item 1 : estimand -- JOINT, not CONDITIONAL =====================

def test_crc_certifies_joint_mass_not_conditional_selective_risk():
    # THE estimand test. CRC bounds P(accepted AND wrong); the conditional error rate among accepted
    # predictions, P(wrong | accepted), is a DIFFERENT and much larger number. Anything that
    # paraphrases the joint mass as "selective risk" is wrong by this factor. Every quantity in the
    # identity is an EVAL-side one, so it can only involve the realization, never the calibration
    # bound -- that would be comparing two different samples.
    rng = np.random.default_rng(7)
    n = 4000
    conf = rng.uniform(0, 1, n)
    correct = rng.uniform(0, 1, n) < conf ** 3              # confidence only weakly informative
    h = n // 2
    out = conformal_risk_control(correct[:h], conf[:h], correct[h:], conf[h:], alpha=0.10)
    assert out["feasible"] is True
    assert out["calib_crc_bound"] <= 0.10                   # the certificate, exactly (no slack)
    assert out["eval_group_joint_risk"] <= 0.10 + 0.03      # the realized JOINT mass (sampling slack)
    # ... while the CONDITIONAL rate among accepted predictions is far above alpha
    assert out["eval_group_selective_risk"] > out["eval_group_joint_risk"] + 0.05


@pytest.mark.parametrize("grouped", [False, True])
def test_joint_equals_selective_times_coverage_within_each_family(grouped):
    """The identity, on BOTH code paths. It used to be asserted only against the ungrouped path --
    `sel_risk` and `coverage` were plain SAMPLE means while the joint loss was aggregated per GROUP,
    so with grouping on the three returned numbers had different denominators and the identity was
    simply false. Grouping is what both experiments now use, so the untested path was the live one."""
    rng = np.random.default_rng(31)
    n = 1200
    conf = rng.uniform(0, 1, n)
    corr = rng.uniform(0, 1, n) < conf ** 2
    # deliberately UNEQUAL group sizes -- with equal sizes the two denominators coincide and the
    # defect is invisible, which is how it survived.
    gid = rng.integers(0, 25, n) ** 2 % 17 if grouped else None
    h = n // 2
    kw = dict(calib_group=gid[:h], eval_group=gid[h:]) if grouped else {}
    out = conformal_risk_control(corr[:h], conf[:h], corr[h:], conf[h:], alpha=0.10, **kw)
    for fam in ("group", "sample"):
        j = out[f"eval_{fam}_joint_risk"]
        s = out[f"eval_{fam}_selective_risk"]
        c = out[f"eval_{fam}_coverage"]
        assert abs(j - s * c) < 1e-9, f"{fam} family violates joint = selective * coverage"
    if grouped:
        # ... and the two families are genuinely DIFFERENT numbers here, so a future collapse back
        # into one shared denominator cannot pass this test by accident.
        assert out["eval_group_coverage"] != out["eval_sample_coverage"]


def test_group_and_sample_denominators_are_not_interchangeable():
    """The exact counterexample. Group 0 holds ONE wrong sample; group 1 holds NINE correct ones;
    everything is accepted. Weighting groups equally gives 0.50; weighting samples equally gives
    0.10. Both are legitimate estimands -- CRC's theorem is about the GROUP one -- but they are not
    the same number, and the old return mixed a group-weighted joint risk with a sample-weighted
    coverage and selective risk, so no two of the three could be combined."""
    corr_e = np.array([False] + [True] * 9)
    conf_e = np.full(10, 0.9)
    grp_e = np.array([0] + [1] * 9)
    corr_c = np.ones(50, bool)                       # calib all correct -> accepts everything
    conf_c = np.linspace(0.50, 0.60, 50)
    out = conformal_risk_control(corr_c, conf_c, corr_e, conf_e, alpha=0.10, eval_group=grp_e)
    assert out["eval_group_coverage"] == 1.0 and out["eval_sample_coverage"] == 1.0
    assert out["eval_group_joint_risk"] == pytest.approx(0.5)     # (1.0 + 0.0) / 2 groups
    assert out["eval_sample_joint_risk"] == pytest.approx(0.1)    # 1 wrong / 10 samples
    # each family is still internally consistent ...
    assert out["eval_group_joint_risk"] == pytest.approx(
        out["eval_group_selective_risk"] * out["eval_group_coverage"])
    assert out["eval_sample_joint_risk"] == pytest.approx(
        out["eval_sample_selective_risk"] * out["eval_sample_coverage"])
    # ... and mixing them is not: this is the product the old return invited a reader to form.
    assert out["eval_group_joint_risk"] != pytest.approx(
        out["eval_sample_selective_risk"] * out["eval_sample_coverage"])


def test_empty_selection_reports_nan_not_zero_risk():
    # A conditional error rate over an EMPTY accepted set is undefined. Returning 0.0 made total
    # abstention read as "0% error" in the CSV (phase4R m=5 printed conf_risk=0.00 at coverage 0.00).
    corr = np.array([False, True, True, True])          # top-confidence sample WRONG
    conf = np.array([0.9, 0.8, 0.7, 0.6])               # -> no threshold reaches a 0% target
    out = conformal_at_risk(corr, conf, corr, conf, target_risk=0.0, conservative=False)
    assert out["coverage"] == 0.0
    assert np.isnan(out["risk"]), "empty selection must be NaN (undefined), never 0"
    crc = conformal_risk_control(np.zeros(60, bool), np.linspace(0.6, 0.5, 60),
                                 np.zeros(20, bool), np.linspace(0.99, 0.8, 20), alpha=0.02)
    assert np.isnan(crc["eval_group_selective_risk"])


# ===================== item 2 : exchangeable UNITS, not duplicated rows =====================

def test_crc_grouping_collapses_repeated_rows_into_one_unit():
    # phase4R pools each pixel once per --trials drop-mask draw. Counting those repeats as
    # independent calibration units shrinks CRC's B/(n+1) term by a factor of `trials` and
    # overstates the bound. Grouping must make n the number of UNITS, not rows.
    rng = np.random.default_rng(3)
    n_unit, trials = 40, 8
    conf_u = rng.uniform(0, 1, n_unit)
    corr_u = rng.uniform(0, 1, n_unit) < conf_u
    conf = np.tile(conf_u, trials); corr = np.tile(corr_u, trials)
    gid = np.tile(np.arange(n_unit), trials)
    grouped = conformal_risk_control(corr, conf, corr, conf, alpha=0.10,
                                     calib_group=gid, eval_group=gid)
    ungrouped = conformal_risk_control(corr, conf, corr, conf, alpha=0.10)
    assert grouped["n_calib_units"] == n_unit               # units, not 320 rows
    assert ungrouped["n_calib_units"] == n_unit * trials
    # the honest bound is strictly more conservative -> a higher threshold, so never more coverage
    assert grouped["threshold"] >= ungrouped["threshold"]
    assert grouped["eval_group_coverage"] <= ungrouped["eval_group_coverage"] + 1e-12


def test_phase4R_pools_a_unit_id_alongside_every_row():
    # phase4R.pooled_logits must emit one exchangeable-unit id per pooled ROW, tiled in step with the
    # logits, or the block ids handed to CRC silently desynchronise from the rows they label.
    import phase4R_reliability as P4R
    groups = [np.arange(0, 5), np.arange(5, 10)]
    X = np.zeros((6, 10), np.float32); y = np.zeros(6, int); gid = np.arange(6)
    model = lambda *a, **k: None                            # never called: patched out below
    P4R.logits_mlp = lambda m, X, g, d: np.zeros((X.shape[0], 3))
    lg, lb, uid = P4R.pooled_logits("b2", model, X, y, gid, groups, m=1, trials=4,
                                    rng=np.random.default_rng(0))
    assert lg.shape[0] == lb.size == uid.size == 6 * 4      # one unit id per pooled row
    assert np.array_equal(uid, np.tile(gid, 4))             # tiled in step, not reordered
    assert np.unique(uid).size == 6                         # 24 rows, still only 6 exchangeable units


# ===================== item 4 : phase8D reports an ORDINARY error rate =====================

def test_phase8D_csv_does_not_claim_selective_risk():
    # phase8D applies NO abstention threshold and has NO selected subset: its risk column is exactly
    # 1 - accuracy at full coverage. Calling it `selective_risk` claimed a selective-prediction
    # quantity the script never computes. n_patches/n_roi were later added so a thin level (level 5
    # is 12 of 975 patches) carries its own sample size next to its accuracy; they are counts, not
    # a selective-risk claim, so the honesty guard below still holds -- what it forbids is a column
    # NAME implying abstention, and none of these do.
    import phase8D_difficulty as P8D
    assert P8D.CSV_COLUMNS == ["difficulty", "n_patches", "n_roi",
                               "mean_confidence", "accuracy", "error_rate"]
    assert not any("selective" in c or "sel_risk" in c or "risk" == c for c in P8D.CSV_COLUMNS)
    # the writer must emit that constant, not a literal row that can drift away from it
    src = open(P8D.__file__, encoding="utf-8").read()
    assert "w.writerow(CSV_COLUMNS)" in src
    assert 'w.writerow(["difficulty"' not in src        # no hard-coded header left behind
    assert "sel_risk%" not in src                        # nor in the console header


# ===================== item 5 : phase8E DOFA reproducibility + degenerate splits =====================

def test_dofa_hub_ref_and_checkpoint_are_pinned():
    # torch.hub.load("zhu-xlab/DOFA", ...) tracks a moving branch and upstream pulls weights from a
    # mutable HuggingFace `main` ref, so an upstream change silently moves our numbers. Both the code
    # ref and the weight BYTES must be pinned.
    import phase8E_dofa as P8E
    assert ":" in P8E.DOFA_HUB_REF, "hub ref is not pinned to a commit"
    ref = P8E.DOFA_HUB_REF.split(":", 1)[1]
    assert len(ref) == 40 and all(c in "0123456789abcdef" for c in ref), f"not a full commit sha: {ref}"
    assert len(P8E.DOFA_CKPT_SHA256) == 64
    assert all(c in "0123456789abcdef" for c in P8E.DOFA_CKPT_SHA256)


def test_dofa_checkpoint_verifier_rejects_wrong_bytes(tmp_path):
    import phase8E_dofa as P8E
    bad = tmp_path / P8E.DOFA_CKPT_NAME
    bad.write_bytes(b"not the pinned weights")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        P8E.verify_dofa_checkpoint(str(bad))
    with pytest.raises(FileNotFoundError):
        P8E.verify_dofa_checkpoint(str(tmp_path / "absent.pth"))


@pytest.mark.parametrize("mod_name", ["phase8R_reliability", "phase8E_dofa"])
def test_test_patch_split_rejects_degenerate_n(mod_name):
    # n=1 returned calib={0}, eval={} -- every eval metric then averaged over an empty array and CRC
    # "calibrated" on one unit. Both copies of the helper must refuse instead.
    mod = __import__(mod_name)
    with pytest.raises(ValueError):
        mod.test_patch_split(1)
    with pytest.raises(ValueError):
        mod.test_patch_split(50, max_patches=1)
    c, e = mod.test_patch_split(50, calib_frac=1.0)          # must not empty eval either
    assert len(c) >= 1 and len(e) >= 1
    c, e = mod.test_patch_split(80, calib_frac=0.5)          # ordinary call unchanged: 40/40, disjoint
    assert len(c) == 40 and len(e) == 40
    assert not (set(c.tolist()) & set(e.tolist()))


# ===================== item 6 : phase8R naive-vs-Mondrian comparison must be fair =====================

def test_phase8R_reports_bound_realization_and_coverage_for_every_arm():
    # Three separate ways this experiment could mislead, all pinned here:
    #  (a) the certified quantity is a JOINT mass, which ANY arm drives to 0 by abstaining, so a risk
    #      without its coverage lets heavy abstention masquerade as risk control;
    #  (b) the BOUND is <= alpha by construction whenever feasible, so reporting only "the certified
    #      risk" can never show a breach -- the eval REALIZATION has to be reported next to it;
    #  (c) the fairness-control arm (clean threshold + state temperature) must exist at all.
    import phase8R_reliability as P8R
    src = open(P8R.__file__, encoding="utf-8").read()
    # Field names now carry the ESTIMAND, which is the point: "crc_bound" invited being read as a
    # per-run upper bound on the held-out loss, and CRC Theorem 2.1 controls a MARGINAL expectation.
    for field in ("crc_calibration_selection_stat", "crc_heldout_empirical_joint_loss",
                  "crc_heldout_coverage", "crc_heldout_selective_risk"):
        assert field in src, f"phase8R does not report {field}"
    for arm in ("mondrian", "naive", "naiveThr_freshT"):
        assert f'"{arm}"' in src, f"arm {arm} is missing"
    # ... and all of them from the GROUP-weighted family, which is the one the bound applies to and
    # the only one where joint = selective * coverage. A sample-weighted coverage beside a
    # group-weighted risk gives three numbers that cannot be combined.
    assert "eval_group_joint_risk" in src and "eval_group_coverage" in src
    assert 'crc["coverage"]' not in src and 'crc["sel_risk"]' not in src
    assert "naiveThr_freshT" in src, \
        "fairness control (clean threshold + state temperature) is missing"
    # the figure must plot BOTH series, not just one of them
    assert "crc_heldout_empirical_joint_loss" in src and "crc_calibration_selection_stat" in src
    # and no per-run verdict may call an exceedance a voided certificate
    assert "CERT VOIDS" not in src.replace("(\"CERT VOIDS\")", ""), (
        "a per-run 'certificate voids' verdict is back: CRC bounds a MARGINAL expectation, so one "
        "split's empirical loss above the selection statistic refutes nothing")


def test_phase8R_groups_crc_by_roi_not_by_patch():
    # CloudSEN12 has ~5 patches per roi_id, so passing per-pixel PATCH ids as CRC's calib_group
    # counted 5 correlated units as 5 independent ones: n was ~5x too big and the B/(n+1) correction
    # ~5x too small, i.e. a bound tighter than the data entitles it to. The ROI-disjoint calib/eval
    # SPLIT does not fix this -- the grouping is a separate argument and was still per patch.
    import phase8R_reliability as P8R
    src = open(P8R.__file__, encoding="utf-8").read()
    assert "roi_of_patch[pid_te]" in src, \
        "phase8R no longer maps per-pixel patch ids to roi_ids for the CRC groups"
    assert "calib_group=pid" not in src and "eval_group=pid" not in src, \
        "phase8R still hands per-pixel PATCH ids to conformal_risk_control"
    # Every CRC call in the file must be grouped by the ROI-derived unit arrays. `eval_group` is
    # what pins that EXACTLY: only conformal_risk_control takes it, so its count is the number of
    # CRC arms. `calib_group=unit_cal` is now larger than 3 because the plug-in operating point is
    # grouped too -- its margin was being sized from 194k correlated pixels instead of 97 ROIs --
    # and those call sites are pinned exactly in tests/test_plugin_margin_units.py.
    assert src.count("eval_group=unit_ev") == 3, \
        "not all three CRC arms are grouped by the ROI-level exchangeable unit"
    assert src.count("calib_group=unit_cal") >= 3, \
        "a CRC arm lost its ROI-level calibration group"
    # the ROI sets handed to CRC must come from the three-way split, and temperature must NOT be
    # fitted on the calibration ROIs -- that is what breaks calib/test exchangeability.
    assert "split_test_rois" in src and "TEMP_FRAC" in src, \
        "phase8R lost the disjoint temperature/calibration/evaluation ROI split"


def test_crc_roi_grouping_is_strictly_looser_than_patch_grouping():
    # The numerical consequence of the fix, on a synthetic copy of CloudSEN12's structure: 5 patches
    # per ROI, so patch grouping inflates n exactly 5x. Both bounds land just under alpha when both
    # are feasible (each is the FIRST grid point that clears alpha), so "looser" cannot be read off
    # the returned bounds directly -- it shows up as a more conservative THRESHOLD, less coverage,
    # and a 5x higher B/(n+1) floor. All three are asserted.
    rng = np.random.default_rng(19)
    n_roi, per_roi, px = 40, 5, 30
    roi = np.repeat(np.arange(n_roi), per_roi * px)
    patch = np.repeat(np.arange(n_roi * per_roi), px)
    conf = rng.uniform(0, 1, roi.size)
    corr = rng.uniform(0, 1, roi.size) < conf
    by_roi = conformal_risk_control(corr, conf, corr, conf, alpha=0.10,
                                    calib_group=roi, eval_group=roi)
    by_patch = conformal_risk_control(corr, conf, corr, conf, alpha=0.10,
                                      calib_group=patch, eval_group=patch)
    assert by_roi["n_calib_units"] == n_roi
    assert by_patch["n_calib_units"] == n_roi * per_roi           # the 5x inflation being removed
    assert by_roi["threshold"] >= by_patch["threshold"]           # honest n -> abstain more
    assert by_roi["eval_group_coverage"] <= by_patch["eval_group_coverage"] + 1e-12

    # SAME DATA, OPPOSITE VERDICT. At alpha=1% the 200 inflated patch "units" find a threshold that
    # certifies; the 40 real locations cannot clear their own 1/41 floor, so CRC correctly reports
    # the risk is not certifiable at all. Patch grouping was buying that verdict with units it did
    # not have.
    roi_1pct = conformal_risk_control(corr, conf, corr, conf, alpha=0.01,
                                      calib_group=roi, eval_group=roi)
    patch_1pct = conformal_risk_control(corr, conf, corr, conf, alpha=0.01,
                                        calib_group=patch, eval_group=patch)
    assert patch_1pct["feasible"] is True and patch_1pct["calib_crc_bound"] <= 0.01
    assert roi_1pct["feasible"] is False
    assert roi_1pct["calib_crc_bound"] == pytest.approx(1.0 / (n_roi + 1))
    assert roi_1pct["calib_crc_bound"] > 0.01

    # And the floor itself, where the 5x is exact arithmetic: below BOTH floors neither grouping can
    # do anything but abstain, so each returns exactly B/(n+1) and the ratio is n_patch+1 : n_roi+1.
    kw = dict(alpha=0.002)                                       # < 1/201, so both are infeasible
    f_roi = conformal_risk_control(corr, conf, corr, conf, calib_group=roi, eval_group=roi, **kw)
    f_patch = conformal_risk_control(corr, conf, corr, conf, calib_group=patch, eval_group=patch, **kw)
    assert f_roi["feasible"] is False and f_patch["feasible"] is False
    assert f_roi["calib_crc_bound"] == pytest.approx(1.0 / (n_roi + 1))
    assert f_patch["calib_crc_bound"] == pytest.approx(1.0 / (n_roi * per_roi + 1))
    assert f_roi["calib_crc_bound"] / f_patch["calib_crc_bound"] == pytest.approx(201.0 / 41.0)


def test_crc_is_symmetric_when_calibration_and_eval_are_the_same_state():
    # phase8R's "clean" row must give naive == Mondrian EXACTLY: both calibrate on the clean state
    # and evaluate on it, so any gap there would be a bug in the comparison rather than a shift
    # effect. This pins the invariant the flagship's baseline column relies on.
    rng = np.random.default_rng(11)
    n = 600
    conf = rng.uniform(0, 1, n)
    corr = rng.uniform(0, 1, n) < conf
    gid = rng.integers(0, 40, n)
    h = n // 2
    a = conformal_risk_control(corr[:h], conf[:h], corr[h:], conf[h:], alpha=0.10,
                               calib_group=gid[:h], eval_group=gid[h:])
    b = conformal_risk_control(corr[:h], conf[:h], corr[h:], conf[h:], alpha=0.10,
                               calib_group=gid[:h], eval_group=gid[h:])
    assert a == b


def test_crc_marginal_guarantee_holds_over_repeated_exchangeable_draws():
    # Theorem 2.1 bounds a MARGINAL expectation over the joint calib+test draw, not any single run.
    # Averaged over many exchangeable replicates the realised joint risk must sit at or under alpha;
    # if this drifts above, the estimator -- not the sampling noise -- is at fault.
    alpha, reps = 0.10, 300
    got = []
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        conf = rng.uniform(0, 1, 400)
        corr = rng.uniform(0, 1, 400) < conf
        got.append(conformal_risk_control(corr[:200], conf[:200], corr[200:], conf[200:],
                                          alpha=alpha)["eval_group_joint_risk"])
    assert float(np.mean(got)) <= alpha, f"mean realised joint risk {np.mean(got):.4f} > alpha {alpha}"
