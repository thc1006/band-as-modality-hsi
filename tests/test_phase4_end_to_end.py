"""Run phase 4's entrypoint with stubbed trainers and assert on WHAT IT WRITES.

Adopted from the phase 3 session's finding in shared_layer.md: a mutation campaign killed 16 of 17
of their first-pass guards, and the transferable lesson was that a guard which greps the source (or
the docstring) cannot tell a claim from its own retraction. This file's predecessor asserted
`'dead_col_mode="exact"' in inspect.getsource(run_seed)` -- which passes if the string appears in a
comment, in a dead branch, or on the wrong call. The properties below are read out of the CSV and
the provenance sidecar the run actually produced, so they fail if the behaviour is wrong no matter
how the source reads.

Cheap because only the TRAINING is stubbed. The corruption chain, the severity measurement, the
aggregation, the paired statistics, the CSV writer and the stamp are all the production code.
"""
import os
import sys
import csv
import json
from collections import Counter

import numpy as np
import pytest
import torch
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import phase4_ablation as P4                                                    # noqa: E402

D2 = "D2_dead_cols"
D1B = "D1b_noise_then_gain"
NCOLS = 145                                                       # Indian Pines image width
N_NONTRIVIAL = 10                                                 # nonzero levels of C + D1a + D2


class _StubNet(torch.nn.Module):
    """A fixed random linear map. Must be a real nn.Module (eval_both calls .eval() and reads
    .parameters()), and its output must DEPEND on the input -- otherwise a corruption that changes
    nothing and a corruption that changes everything would both leave the metric flat, and this
    file could not tell them apart."""

    def __init__(self, n_bands, n_classes, seed):
        super().__init__()
        self.lin = torch.nn.Linear(n_bands, n_classes)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(n_classes, n_bands, generator=g) * 0.1)
            self.lin.bias.zero_()

    def forward(self, x, present=None):
        return self.lin(x)


def _run(out, seeds):
    """main() with the training stubbed out and every path redirected into `out`."""
    os.makedirs(out / "figs", exist_ok=True)
    mp = pytest.MonkeyPatch()
    nb = len(P4.AVIRIS_WL_NM)
    try:
        mp.setattr(P4, "P", lambda rel: str(out / rel))       # never touch the real paper/ dir
        mp.setattr(P4.hw, "setup", lambda **kw: None)         # keep global torch state untouched
        mp.setattr(P4, "train_mlp",
                   lambda X, y, groups, seed, **kw: _StubNet(nb, P4.NUM_CLASSES, seed + 7))
        mp.setattr(P4, "GroupedCrossBandAttention",
                   lambda groups, cwl, n_classes: _StubNet(nb, n_classes, 11))
        mp.setattr(P4, "pretrain_sgmae", lambda *a, **kw: None)
        mp.setattr(P4, "finetune_proposed", lambda *a, **kw: None)
        # in-process, so the stubs above are visible: run_jobs spawns, and a spawned worker
        # re-imports the module and would get the real trainers back.
        mp.setattr(P4.parallel, "run_jobs",
                   lambda fn, items, shared=None, **kw: [fn(i, **(shared or {})) for i in items])
        mp.setattr(sys, "argv",
                   ["phase4_ablation.py", "--seeds", *map(str, seeds), "--device", "cpu"])
        P4.main()
    finally:
        mp.undo()
    csv_path = out / "results_phase4_ablation.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    with open(str(csv_path) + ".provenance.json") as f:
        stamp = json.load(f)
    with open(out / "results_phase4_raw.csv") as f:
        raw = list(csv.DictReader(f))
    assert rows and raw, "main() wrote an empty CSV"
    return {"rows": rows, "raw": raw, "stamp": stamp}


@pytest.fixture(scope="module")
def run3(tmp_path_factory):
    """Three seeds: the sample std and the paired-t interval both exist."""
    return _run(tmp_path_factory.mktemp("phase4_n3"), [0, 1, 2])


@pytest.fixture(scope="module")
def run1(tmp_path_factory):
    """One seed: no dispersion and no interval exist at all. A separate case on purpose."""
    return _run(tmp_path_factory.mktemp("phase4_n1"), [0])


def rounding_tol(mult):
    """How far a width recomputed from the CSV may legitimately sit from the one the CSV reports.

    Derived, not guessed. Each bound is written to 2dp, so their difference carries +-0.01. The std
    is ALSO written to 2dp, and the width is `mult * std`, so its +-0.005 arrives amplified by mult
    -- which for the Bonferroni interval at n=3 is ~16x, i.e. +-0.081. A flat 0.03 tolerance passed
    the uncorrected check by luck and failed the corrected one; the amplification is the whole
    point of the correction, so the tolerance has to carry it."""
    return 0.011 + 0.005 * mult


def of(run, design):
    return [r for r in run["rows"] if r["design"] == design]


def test_every_measured_design_reaches_the_csv(run3):
    """Dropping a panel must not drop the result: the figure plots three axes and the CSV carries
    all four, with the withheld one flagged."""
    assert {r["design"] for r in run3["rows"]} == {d["design"] for d in P4.CSV_DESIGNS}
    assert {"realised_dead_cols", "dead_test_px_pct_mean", "dead_test_px_pct_std",
            "paired_ci_lo", "paired_ci_hi", "paired_verdict"} <= set(run3["rows"][0])


def test_d2_realises_its_nominal_fraction_exactly(run3):
    """The sweep parameter must BE the severity applied, not its expectation. Read from the run:
    the realised count is floor(f*ncols + 0.5) at every level, deterministic in (f, ncols)."""
    d2 = of(run3, D2)
    want = [str(int(np.floor(float(r["param"]) * NCOLS + 0.5))) for r in d2]
    assert [r["realised_dead_cols"] for r in d2] == want == ["0", "1", "4", "7"]


def test_d2_severity_is_strictly_increasing_so_the_axis_orders_its_own_conditions(run3):
    """Under the Bernoulli draw this failed on the real seeds: nominal 3% destroyed 2.55-6.71% of
    the test pixels and nominal 5% destroyed 3.32-8.34%, so a nominally worse point could be
    strictly less corrupted than a nominally better one."""
    px = [float(r["dead_test_px_pct_mean"]) for r in of(run3, D2)]
    assert px[0] == 0.0
    assert px == sorted(px) and len(set(px)) == len(px), f"not strictly increasing: {px}"


def test_no_nonzero_d2_level_is_secretly_the_clean_baseline(run3):
    """The failure that motivated all of this: seed 0 realised ZERO dead columns at nominal 1%, so
    that point was the clean baseline plotted at x=0.01 with retention 100% by construction."""
    d2 = of(run3, D2)
    base = d2[0]["proposed_miou_mean"]
    for r in d2[1:]:
        assert r["proposed_miou_mean"] != base, \
            f"level {r['param']} scores exactly the baseline {base} -- it applied no corruption"


def test_the_degenerate_half_of_d1_is_flat_and_flagged_and_the_other_half_is_not(run3):
    """The pair is the finding, so both halves have to behave: g*(x+n) cancels under the
    per-spectrum mean and g*x+n does not. Checked on the metric, and on the flag beside it."""
    d1b, d1a = of(run3, D1B), of(run3, "D1a_gain_then_noise")
    assert len({r["proposed_miou_mean"] for r in d1b}) == 1, "D1b must be pinned by algebra"
    assert all(r["degenerate"] for r in d1b), "a curve fixed by algebra must stay flagged"
    assert len({r["proposed_miou_mean"] for r in d1a}) > 1, "D1a must move with the gain"
    assert not any(r["degenerate"] for r in d1a), "the measuring half must not be flagged"


def test_the_three_d_axes_share_one_snr_only_zero_point(run3):
    """Their retentions are comparable only if the denominators are the same experiment, and the
    three zero points are built from three separate seed literals."""
    zero = {d: next(r for r in of(run3, d) if float(r["param"]) == 0.0)["proposed_miou_mean"]
            for d in ("D1a_gain_then_noise", D1B, D2)}
    assert len(set(zero.values())) == 1, f"D zero points diverged: {zero}"


def test_a_verdict_is_only_given_where_the_interval_supports_it(run3):
    """A level is named only when the paired interval excludes zero -- otherwise the file keeps
    filing a -0.22 mIoU mean margin as a loss beside a -6.72 one."""
    for r in run3["rows"]:
        v = r["paired_verdict"]
        assert v in {"loses", "wins", "indistinguishable", "n<2"}, v
        if v == "n<2":
            continue
        lo, hi = float(r["paired_ci_lo"]), float(r["paired_ci_hi"])
        assert lo <= hi
        assert (hi < 0) if v == "loses" else (lo > 0) if v == "wins" else (lo <= 0 <= hi)


def test_the_interval_is_built_from_the_dispersion_the_row_reports(run3):
    """Ties the reported std to the interval drawn from it so the two cannot drift apart: the width
    must be 2 * t(0.975, n-1) * std / sqrt(n) for the very std in the same row."""
    for r in run3["rows"]:
        if r["paired_verdict"] == "n<2":
            continue
        n, sd = int(r["n_seeds"]), float(r["paired_margin_std"])
        width = float(r["paired_ci_hi"]) - float(r["paired_ci_lo"])
        mult = 2.0 * float(stats.t.ppf(0.975, n - 1)) / np.sqrt(n)
        assert abs(width - mult * sd) < rounding_tol(mult), \
            f"{r['design']} {r['param']}: {width} vs {mult * sd}"


def test_the_stamped_tally_counts_the_same_levels_the_verdict_column_does(run3):
    """The closing summary is a print; the stamp is its machine-readable twin and must count the
    same run. Only the three MEASURING axes at nonzero levels are tallied -- D1b's margin is the
    eps=0 margin repeated by algebra, so counting its four levels would enter one fact four times.
    """
    tallied = Counter(r["paired_verdict"] for r in run3["rows"]
                      if r["design"] != D1B and float(r["param"]) != 0.0)
    assert sum(tallied.values()) == N_NONTRIVIAL
    assert dict(tallied) == run3["stamp"]["extra"]["verdict_counts"]


def test_severity_columns_are_blank_where_nothing_was_measured(run3):
    """A 0 in those columns would claim a measurement the C and D1 axes never took."""
    for r in run3["rows"]:
        if r["design"] != D2:
            assert r["realised_dead_cols"] == "" and r["dead_test_px_pct_mean"] == ""


def test_the_run_stamps_the_scope_the_numbers_may_be_read_in(run3):
    """Both corruption models are schematic and their own modules exclude them from physics claims.
    A CSV that outlives this docstring must still say so."""
    extra = run3["stamp"]["extra"]
    assert extra["physics_scope"] == "schematic_stress_test_not_calibrated_radiometry"
    assert extra["dead_col_mode"] == "exact"
    assert extra["dispersion"] == "sample_std_ddof1"


def test_a_single_seed_is_reported_as_untested_rather_than_as_indistinguishable(run1):
    """One seed yields no dispersion and no interval, so nothing was compared. The first draft of
    the closing summary printed "no level separates the two methods" in exactly this case, which is
    a claim about a comparison that never happened."""
    assert {r["paired_verdict"] for r in run1["rows"]} == {"n<2"}
    assert run1["stamp"]["extra"]["verdict_counts"] == {"n<2": N_NONTRIVIAL}
    for r in run1["rows"]:
        assert r["paired_margin_std"] == "nan", "n=1 must not report a dispersion"
        assert r["paired_ci_lo"] == "+nan" and r["paired_ci_hi"] == "+nan"
    # the severity axis is still exact with one seed -- it does not depend on the seed at all
    assert [r["realised_dead_cols"] for r in of(run1, D2)] == ["0", "1", "4", "7"]


def test_every_aggregate_is_recomputable_from_the_raw_per_seed_rows(run3):
    """The raw rows exist so that a dispersion can be CHECKED. A mutation campaign showed ddof
    could be switched back to the population formula at any of three call sites with the aggregate
    CSV alone and no test would notice: a std of 1.74 looks exactly as plausible as 1.95."""
    discriminating = 0
    for agg in run3["rows"]:
        rows = [r for r in run3["raw"]
                if r["design"] == agg["design"] and r["param"] == agg["param"]]
        assert len(rows) == int(agg["n_seeds"]), f"{agg['design']} {agg['param']}: raw rows missing"
        p = np.array([float(r["proposed_miou"]) for r in rows])
        b = np.array([float(r["b2_miou"]) for r in rows])
        assert float(agg["proposed_miou_mean"]) == pytest.approx(p.mean(), abs=0.006)
        assert float(agg["b2_miou_mean"]) == pytest.approx(b.mean(), abs=0.006)
        assert float(agg["proposed_miou_std"]) == pytest.approx(p.std(ddof=1), abs=0.006)
        assert float(agg["b2_miou_std"]) == pytest.approx(b.std(ddof=1), abs=0.006)
        assert float(agg["paired_margin_mean"]) == pytest.approx((p - b).mean(), abs=0.006)
        assert float(agg["paired_margin_std"]) == pytest.approx((p - b).std(ddof=1), abs=0.006)
        # and the PAIRED RETENTION, which goes through a different helper
        zero = [r for r in run3["raw"]
                if r["design"] == agg["design"] and float(r["param"]) == 0.0]
        p0 = np.array([float(r["proposed_miou"]) for r in zero])
        assert float(agg["proposed_retention_std"]) == pytest.approx((p / p0 * 100).std(ddof=1),
                                                                     abs=0.06)
        # Whether a row can tell the two formulas apart is decided by ROUNDING, not by magnitude.
        # The first version used a magnitude threshold; the phase 2 cross-sensor session took that
        # advice from shared_layer.md and it rejected a perfectly good run -- spreads of 0.0211 and
        # 0.0149 differ by 0.006 yet print as "0.02" and "0.01". Compare what the CSV stores, which
        # is also what the assertions above compare against.
        if f"{p.std(ddof=1):.2f}" != f"{p.std():.2f}":
            discriminating += 1
    # Without this the test could silently lose its power: if every spread rounded the same way
    # under both formulas, the assertions above would pass either way.
    assert discriminating >= 3, "the raw rows no longer distinguish ddof=0 from ddof=1"


def test_the_recorded_severity_is_measured_not_copied_from_the_request(run3):
    """The realised COUNT is deterministic in (f, ncols), but WHICH columns are hit is not, so the
    destroyed-test-pixel fraction varies across seeds. A value copied from the request would not --
    and 0/1/3/5 is just as strictly increasing as the measured 0/1.14/3.40/5.77, so monotonicity
    alone cannot tell them apart. This is the assertion that can."""
    for r in of(run3, D2):
        f = float(r["param"])
        if f == 0.0:
            continue
        assert float(r["dead_test_px_pct_std"]) > 0.0, \
            f"level {f}: identical across seeds -- that is the request, not a measurement"
        assert float(r["dead_test_px_pct_mean"]) != pytest.approx(f * 100.0, abs=1e-6)
    per_seed = [r["dead_test_px_pct"] for r in run3["raw"]
                if r["design"] == D2 and float(r["param"]) == 0.05]
    assert len(set(per_seed)) == len(per_seed), f"per-seed severity is constant: {per_seed}"


def test_the_bonferroni_column_can_only_be_weaker_than_the_uncorrected_one(run3):
    """Multiplicity is answered in the artefact rather than flagged and left: at 10 levels the
    family-wise error of an uncorrected sweep reaches ~40%. A correction may only ever withdraw a
    verdict, never create one."""
    assert run3["stamp"]["extra"]["bonferroni_family_size"] == N_NONTRIVIAL
    assert {"paired_ci_lo_bonf", "paired_ci_hi_bonf"} <= set(run3["rows"][0])
    named = {"loses": 1, "wins": 1, "indistinguishable": 0, "n<2": 0}
    for r in run3["rows"]:
        if r["design"] == D1B:
            assert r["paired_verdict_bonferroni"] == "", "D1b is not in the correction family"
            continue
        b, u = r["paired_verdict_bonferroni"], r["paired_verdict"]
        assert b in named
        assert named[b] <= named[u], f"correction strengthened {u} into {b}"
        if u == "n<2":
            continue
        # The label alone is not checkable -- a Bonferroni column computed at the UNCORRECTED alpha
        # satisfies the monotonicity above, because "identical" is weaker-or-equal. The interval is
        # what pins the alpha, and it must be STRICTLY wider since alpha shrank.
        n_s, sd = int(r["n_seeds"]), float(r["paired_margin_std"])
        wide = float(r["paired_ci_hi_bonf"]) - float(r["paired_ci_lo_bonf"])
        mult = 2.0 * float(stats.t.ppf(1 - (0.05 / N_NONTRIVIAL) / 2, n_s - 1)) / np.sqrt(n_s)
        assert abs(wide - mult * sd) < rounding_tol(mult), \
            f"{r['design']} {r['param']}: {wide} vs {mult * sd}"
        assert wide > (float(r["paired_ci_hi"]) - float(r["paired_ci_lo"])) + 0.01, \
            "the corrected interval is not wider -- it was computed at the uncorrected alpha"


def test_eval_both_puts_the_models_in_eval_mode_before_scoring():
    """Nothing else can detect the removal of that line -- the trainers already return models in
    eval mode and the six nn.Dropout layers are built at p=0.0 -- which is exactly why it is
    asserted directly on the call instead of hoped for downstream."""
    a, b = _StubNet(8, P4.NUM_CLASSES, 1), _StubNet(8, P4.NUM_CLASSES, 2)
    a.train(), b.train()
    assert a.training and b.training
    P4.eval_both(a, b, np.zeros((5, 8), np.float32), np.zeros(5, int),
                 P4.contiguous_groups(8, 2))
    assert not a.training and not b.training, "eval_both scored a model in train mode"


def test_the_d1a_axis_is_validated_on_the_input_not_on_the_metric(run3):
    """Whether the column gain survives the normalisation is a property of what the model is FED.
    Whether either model RESPONDS is a separate question whose answer here is nearly no (~0.1 mIoU
    against a per-seed spread of several points). An earlier runtime check read a flat metric as a
    broken corruption chain -- but a merely insensitive model produces the identical symptom, so
    the check now measures the standardised test matrix and the metric is only reported."""
    extra = run3["stamp"]["extra"]
    assert extra["d1a_min_input_delta"] > 0.0, \
        "the D1a corruption never reached the model, which no metric could have told us"
    assert "d1a_flat_this_run" in extra, "the metric's flatness is reported, not treated as a defect"


def test_collapsed_levels_flags_only_where_both_methods_fell_below_the_floor():
    """The logic is unit-testable now that it is not a closure. The THRESHOLD is not: it is a
    reporting policy with no ground truth, so it is a named constant recorded in the stamp rather
    than something a test can validate -- pinning 25.0 would only restate the constant."""
    acc = {0.0: {"proposed": [50.0, 50.0], "b2": [50.0, 50.0]},
           0.1: {"proposed": [40.0, 40.0], "b2": [5.0, 5.0]},     # one still works -> not flagged
           0.5: {"proposed": [5.0, 5.0], "b2": [4.0, 4.0]},       # both collapsed -> flagged
           1.0: {"proposed": [0.0, 0.0], "b2": [0.0, 0.0]}}       # both dead -> flagged
    axes = [("C", acc, [0.0, 0.1, 0.5, 1.0])]
    assert P4.collapsed_levels(axes, 25.0) == [("C", 0.5), ("C", 1.0)]
    assert P4.collapsed_levels(axes, 0.0) == []
    assert P4.collapsed_levels(axes, 100.0) == [("C", 0.1), ("C", 0.5), ("C", 1.0)]
    # a zero baseline makes retention undefined; undefined is not "low" and must not be flagged
    nan_acc = {0.0: {"proposed": [0.0, 0.0], "b2": [0.0, 0.0]},
               0.5: {"proposed": [1.0, 1.0], "b2": [1.0, 1.0]}}
    assert P4.collapsed_levels([("X", nan_acc, [0.0, 0.5])], 25.0) == []


def test_sample_std_is_the_sample_formula_and_declines_to_guess_at_n_1():
    """ddof=1. np.std's default underestimates a sample's spread by sqrt((n-1)/n) -- 10.6% at the 5
    seeds this file runs by default -- and every '+-' in phase 4 used to be that number."""
    x = [1.0, 2.0, 4.0, 8.0, 16.0]
    assert P4.sample_std(x) == pytest.approx(float(np.std(x, ddof=1)))
    assert P4.sample_std(x) > float(np.std(x))
    assert P4.sample_std(x) == pytest.approx(float(np.std(x)) * np.sqrt(len(x) / (len(x) - 1)))
    assert np.isnan(P4.sample_std([3.0])) and np.isnan(P4.sample_std([]))
