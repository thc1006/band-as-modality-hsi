"""`conformal_risk_control` against a slow reference written from the theorem, plus its invariants.

WHY. Every reliability claim in this project passes through this one function, and its production
form is vectorised: a boolean broadcast to build the per-sample loss over a threshold grid,
`np.add.at` for the group aggregation, an argmin over a monotone statistic. Vectorised code can be
right and can be subtly wrong in ways that reading it will not reveal, and a test written from the
same mental model as the implementation mostly proves the model self-consistent.

So the reference below is written from the DEFINITION in Angelopoulos, Bates, Fisch, Lei & Schuster
(2022, arXiv:2208.02814), with explicit loops and no clever indexing:

    L_i(lambda) = 1{ conf_i >= lambda  AND  prediction_i is wrong }
    Rhat_n(lambda) = mean over calibration UNITS of L_i(lambda)
    lambda_hat = inf{ lambda : (n/(n+1)) * Rhat_n(lambda) + B/(n+1) <= alpha }

and the two implementations must agree to the bit on random and adversarial inputs.

The invariants matter as much as the agreement, because they pin the ESTIMAND rather than the
arithmetic: which unit is exchangeable, which denominator each number uses, and that a threshold is
never reported as certifying when nothing certifies.
"""
import numpy as np
import pytest

from bandsim.reliability import conformal_risk_control


# --------------------------------------------------------------------------------- the reference
def crc_reference(calib_correct, calib_conf, eval_correct, eval_conf, alpha=0.10, B=1.0,
                  n_grid=256, calib_group=None, eval_group=None):
    """Deliberately slow and obvious. Loops, no broadcasting, no np.add.at."""
    cc = [bool(x) for x in calib_correct]
    ck = [float(x) for x in calib_conf]
    ec = [bool(x) for x in eval_correct]
    ek = [float(x) for x in eval_conf]
    cg = list(calib_group) if calib_group is not None else None
    eg = list(eval_group) if eval_group is not None else None

    grid = list(np.linspace(min(ck), max(ck), n_grid)) + [float("inf")]

    def mean_loss(correct, conf, group, lam):
        """Rhat(lambda): per-unit mean loss, then the mean over units."""
        loss = [(0.0 if correct[i] else 1.0) * (1.0 if conf[i] >= lam else 0.0)
                for i in range(len(conf))]
        if group is None:
            return sum(loss) / len(loss), len(loss)
        buckets = {}
        for g, v in zip(group, loss):
            buckets.setdefault(g, []).append(v)
        per_unit = [sum(v) / len(v) for v in buckets.values()]
        return sum(per_unit) / len(per_unit), len(per_unit)

    n = mean_loss(cc, ck, cg, grid[0])[1]
    stat = [(n / (n + 1.0)) * mean_loss(cc, ck, cg, lam)[0] + B / (n + 1.0) for lam in grid]

    feasible = any(s <= alpha for s in stat)
    j = next((i for i, s in enumerate(stat) if s <= alpha), len(grid) - 1)
    thr = grid[j]

    kept = [1.0 if v >= thr else 0.0 for v in ek]
    wrongkept = [(0.0 if ec[i] else 1.0) * kept[i] for i in range(len(ek))]
    if eg is None:
        m = len(ek)
        g_joint, g_cov, n_eval = sum(wrongkept) / m, sum(kept) / m, m
    else:
        b = {}
        for g, w, k in zip(eg, wrongkept, kept):
            b.setdefault(g, []).append((w, k))
        per = [(sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)) for v in b.values()]
        g_joint = sum(x[0] for x in per) / len(per)
        g_cov = sum(x[1] for x in per) / len(per)
        n_eval = len(per)
    return {"threshold": thr, "calib_crc_bound": stat[j], "eval_group_joint_risk": g_joint,
            "eval_group_coverage": g_cov, "n_calib_units": n, "n_eval_units": n_eval,
            "feasible": feasible}


def _draw(rng, n_cal=400, n_ev=400, n_group=20, acc=0.8):
    cc = rng.random(n_cal) < acc
    ec = rng.random(n_ev) < acc
    # correct predictions get higher confidence on average, as a real model's do
    ck = np.clip(rng.beta(5, 2, n_cal) * np.where(cc, 1.0, 0.75), 0, 1)
    ek = np.clip(rng.beta(5, 2, n_ev) * np.where(ec, 1.0, 0.75), 0, 1)
    cg = rng.integers(0, n_group, n_cal)
    eg = rng.integers(100, 100 + n_group, n_ev)
    return cc, ck, ec, ek, cg, eg


# ------------------------------------------------------------------------ production == reference
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("grouped", [False, True])
def test_production_matches_the_reference(seed, grouped):
    rng = np.random.default_rng(seed)
    cc, ck, ec, ek, cg, eg = _draw(rng)
    kw = dict(calib_group=cg, eval_group=eg) if grouped else {}
    got = conformal_risk_control(cc, ck, ec, ek, alpha=0.10, **kw)
    want = crc_reference(cc, ck, ec, ek, alpha=0.10, **kw)
    for k in ("threshold", "calib_crc_bound", "eval_group_joint_risk", "eval_group_coverage"):
        assert got[k] == pytest.approx(want[k], rel=1e-12, abs=1e-12), k
    for k in ("n_calib_units", "n_eval_units", "feasible"):
        assert got[k] == want[k], k


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.25, 0.50])
def test_agreement_across_alpha_including_the_infeasible_regime(alpha):
    """alpha below B/(n+1) cannot be certified by ANY threshold; both must agree that it is
    infeasible AND fall back to the same abstain-everything threshold."""
    rng = np.random.default_rng(99)
    cc, ck, ec, ek, cg, eg = _draw(rng, n_cal=40, n_ev=60, n_group=8, acc=0.6)
    got = conformal_risk_control(cc, ck, ec, ek, alpha=alpha, calib_group=cg, eval_group=eg)
    want = crc_reference(cc, ck, ec, ek, alpha=alpha, calib_group=cg, eval_group=eg)
    assert got["feasible"] == want["feasible"]
    assert got["threshold"] == pytest.approx(want["threshold"]) or (
        np.isinf(got["threshold"]) and np.isinf(want["threshold"]))
    assert got["calib_crc_bound"] == pytest.approx(want["calib_crc_bound"], rel=1e-12)


# ------------------------------------------------------------------------------------ invariants
def test_the_selection_statistic_is_at_or_below_alpha_whenever_feasible():
    """The one thing the statistic guarantees. If this can exceed alpha while `feasible` is True,
    every "the certificate was tighter than alpha" reading downstream is wrong."""
    rng = np.random.default_rng(3)
    for _ in range(30):
        cc, ck, ec, ek, cg, eg = _draw(rng, n_cal=rng.integers(30, 300), n_ev=200,
                                       n_group=int(rng.integers(4, 30)))
        r = conformal_risk_control(cc, ck, ec, ek, alpha=0.10, calib_group=cg, eval_group=eg)
        if r["feasible"]:
            assert r["calib_crc_bound"] <= 0.10 + 1e-12


def test_joint_equals_coverage_times_selective_risk_within_one_family():
    """joint = selective x coverage holds EXACTLY inside a family and NOT across families. Mixing a
    group-weighted risk with a sample-weighted coverage was a real defect here once."""
    rng = np.random.default_rng(7)
    cc, ck, ec, ek, cg, eg = _draw(rng)
    r = conformal_risk_control(cc, ck, ec, ek, alpha=0.10, calib_group=cg, eval_group=eg)
    for fam in ("group", "sample"):
        j, c, s = (r[f"eval_{fam}_joint_risk"], r[f"eval_{fam}_coverage"],
                   r[f"eval_{fam}_selective_risk"])
        if c > 0:
            assert j == pytest.approx(s * c, rel=1e-10)


def test_permuting_pixels_inside_a_unit_changes_nothing():
    """The exchangeable unit is the group, so the ORDER of rows within it must be irrelevant."""
    rng = np.random.default_rng(11)
    cc, ck, ec, ek, cg, eg = _draw(rng)
    base = conformal_risk_control(cc, ck, ec, ek, alpha=0.10, calib_group=cg, eval_group=eg)
    order = np.argsort(cg, kind="stable")          # regroup rows without changing membership
    perm = conformal_risk_control(cc[order], ck[order], ec, ek, alpha=0.10,
                                  calib_group=cg[order], eval_group=eg)
    for k in ("threshold", "calib_crc_bound", "eval_group_joint_risk", "eval_group_coverage",
              "n_calib_units"):
        assert perm[k] == pytest.approx(base[k], rel=1e-12), k


def test_duplicating_rows_inside_a_unit_changes_nothing():
    """A unit contributes its MEAN, so imaging the same ROI twice as many times must not buy it
    extra weight -- that is the entire reason the unit is the ROI and not the pixel."""
    rng = np.random.default_rng(13)
    cc, ck, ec, ek, cg, eg = _draw(rng, n_cal=200, n_group=10)
    dup = cg == cg[0]
    base = conformal_risk_control(cc, ck, ec, ek, alpha=0.10, calib_group=cg, eval_group=eg)
    grown = conformal_risk_control(np.concatenate([cc, cc[dup]]), np.concatenate([ck, ck[dup]]),
                                   ec, ek, alpha=0.10,
                                   calib_group=np.concatenate([cg, cg[dup]]), eval_group=eg)
    assert grown["n_calib_units"] == base["n_calib_units"]
    assert grown["calib_crc_bound"] == pytest.approx(base["calib_crc_bound"], rel=1e-12)


def test_a_new_unit_does_change_the_bound():
    """The mirror image of the test above: adding a genuinely NEW unit must move n, or the B/(n+1)
    correction is not counting what it claims to."""
    rng = np.random.default_rng(17)
    cc, ck, ec, ek, cg, eg = _draw(rng, n_cal=200, n_group=10)
    grown = conformal_risk_control(np.concatenate([cc, cc[:20]]), np.concatenate([ck, ck[:20]]),
                                   ec, ek, alpha=0.10,
                                   calib_group=np.concatenate([cg, np.full(20, 999)]), eval_group=eg)
    assert grown["n_calib_units"] == conformal_risk_control(
        cc, ck, ec, ek, alpha=0.10, calib_group=cg, eval_group=eg)["n_calib_units"] + 1


def test_coverage_is_non_increasing_in_the_threshold():
    """L(lambda) is non-increasing by construction; if coverage ever rose with the threshold the
    grid search for 'the most coverage that still certifies' would be selecting the wrong end."""
    rng = np.random.default_rng(23)
    cc, ck, ec, ek, cg, eg = _draw(rng)
    covs = []
    for a in (0.50, 0.30, 0.20, 0.10, 0.05):
        r = conformal_risk_control(cc, ck, ec, ek, alpha=a, calib_group=cg, eval_group=eg)
        covs.append(r["eval_group_coverage"])
    assert all(covs[i] >= covs[i + 1] - 1e-12 for i in range(len(covs) - 1)), covs


def test_infeasible_abstains_on_everything_rather_than_reporting_a_low_risk():
    """An infeasible run must not look like a good one. The fallback is lambda_max = +inf, which
    accepts nothing -- including an eval point more confident than anything in calibration."""
    rng = np.random.default_rng(29)
    cc, ck, ec, ek, cg, eg = _draw(rng, n_cal=20, n_ev=50, n_group=4, acc=0.5)
    r = conformal_risk_control(cc, ck, ec, ek, alpha=0.001, calib_group=cg, eval_group=eg)
    assert not r["feasible"]
    assert np.isinf(r["threshold"])
    assert r["eval_group_coverage"] == pytest.approx(0.0)
    assert r["eval_group_joint_risk"] == pytest.approx(0.0)
    assert np.isnan(r["eval_group_selective_risk"])     # undefined, never 0


def test_a_perfect_model_certifies_at_full_coverage():
    """Sanity anchor at the easy end: nothing is ever wrong, so no threshold is needed."""
    n = 100
    cc = np.ones(n, bool); ec = np.ones(n, bool)
    ck = np.linspace(0.5, 1.0, n); ek = np.linspace(0.5, 1.0, n)
    g = np.arange(n) // 5
    r = conformal_risk_control(cc, ck, ec, ek, alpha=0.10, calib_group=g, eval_group=g)
    assert r["feasible"]
    assert r["eval_group_coverage"] == pytest.approx(1.0)
    assert r["eval_group_joint_risk"] == pytest.approx(0.0)


def test_crc_controls_the_mean_loss_over_repeated_draws_not_each_run():
    """The estimand, stated as a test.

    Under exchangeability CRC bounds E[L_{n+1}] -- an average over repeated (calibration, test)
    draws. Individual draws exceed alpha routinely, and this asserts BOTH halves: the mean is
    controlled, and single-run exceedance happens often enough that treating one as a refutation
    would be wrong. That asymmetry is exactly what the phase8R/phase9 wording got backwards."""
    rng = np.random.default_rng(101)
    alpha, losses = 0.10, []
    for _ in range(300):
        cc, ck, ec, ek, cg, eg = _draw(rng, n_cal=150, n_ev=150, n_group=30, acc=0.75)
        r = conformal_risk_control(cc, ck, ec, ek, alpha=alpha, calib_group=cg, eval_group=eg)
        if r["feasible"]:
            losses.append(r["eval_group_joint_risk"])
    losses = np.array(losses)
    assert losses.size > 100
    assert losses.mean() <= alpha + 1e-3, f"mean loss {losses.mean():.4f} exceeds alpha"
    assert (losses > alpha).mean() > 0.02, (
        "single-run exceedance never happened, so this fixture cannot demonstrate why a per-run "
        "breach is not a theorem violation")
