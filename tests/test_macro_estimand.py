"""The macro estimand: WHICH classes mIoU/AA average over, and why it is pinned once.

`miou` / `average_accuracy` average over the classes present in `y_true`. With a shifting
checkerboard split that set changes between seeds, so a mean over seeds averages different
quantities. `bandsim.metrics.common_class_set` fixes the set; these tests pin its behaviour, prove
the consolidated version reproduces the two hand-copies it replaces, and lock the phase5 contract
that `eval_mlp` keeps its legacy meaning when no class set is given.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bandsim.io import disjoint_block_split
from bandsim.metrics import (common_class_set, macro_over, per_class_recall, per_class_iou,
                             average_accuracy, miou)

_GT_PATH = os.path.join(_ROOT, "data", "indian_pines", "Indian_pines_gt.mat")


def _indian_pines_gt():
    if not os.path.exists(_GT_PATH):
        pytest.skip("Indian Pines ground truth not present")
    from bandsim.io import load_mat_cube
    return load_mat_cube(_GT_PATH, key="indian_pines_gt").astype(int)


def _two_class_gt():
    """30x30, class 1 everywhere, class 2 in a patch that lands on the TEST side for offsets 0 AND 1.

    The placement is not arbitrary and is asserted below: `disjoint_block_split` sends cells of even
    `(bi + bj)` parity to TRAIN, so a patch in cell (0,0) — the obvious choice — is never scored at
    all. Cell (0,1) is test for both offsets, and rows 2-7 / cols 12-17 stay clear of the guard band
    eating the cell's edges."""
    gt = np.ones((30, 30), int)
    gt[2:8, 12:18] = 2
    for off in (0, 1):
        _, te = disjoint_block_split(gt, block=10, guard=1, offset=off)
        assert np.any(gt[te] == 2), f"fixture broken: class 2 not in the test split at offset {off}"
    return gt


# ------------------------------------------------------------------------- common_class_set
def test_a_class_that_misses_one_test_split_is_excluded():
    """The whole point: a class present in the scene but not in EVERY test split must not enter the
    macro average, because the seeds that do contain it would then be averaging over more classes
    than the seeds that do not."""
    gt = _two_class_gt()
    tr, _ = disjoint_block_split(gt, block=10, guard=1, offset=0)
    ys, xs = np.where(tr)
    gt[ys[0], xs[0]] = 3                             # class 3: a single TRAIN pixel for offset 0
    keep, present = common_class_set(gt, block=10, offsets=[0], guard=1, num_classes=3)
    assert keep == [0, 1], f"class 3 (0-based 2) never reaches the test split: {keep}"
    assert present[2] == 0 and present[0] == present[1] == 1
    assert all(isinstance(c, int) and 0 <= c < 3 for c in keep), "class_set must be 0-based ints"


def test_common_class_set_refuses_a_set_it_cannot_define():
    gt = np.ones((30, 30), int)                      # one class only -> no macro set possible
    with pytest.raises(ValueError, match="no macro metric"):
        common_class_set(gt, block=10, offsets=[0], guard=1, num_classes=2)
    with pytest.raises(ValueError, match="at least one offset"):
        common_class_set(gt, block=10, offsets=[], guard=1, num_classes=2)


def test_a_num_classes_larger_than_the_dataset_is_harmless():
    """Documented safety property: surplus classes appear in no split, so they are excluded rather
    than silently counted. (A num_classes SMALLER than the data truncates, which is why it is a
    parameter and not inferred.)"""
    gt = _two_class_gt()
    keep_exact, _ = common_class_set(gt, block=10, offsets=[0, 1], guard=1, num_classes=2)
    keep_over, _ = common_class_set(gt, block=10, offsets=[0, 1], guard=1, num_classes=9)
    assert keep_exact == keep_over == [0, 1]


def test_the_indian_pines_macro_set_is_14_and_stable_across_seed_lists():
    """The measured fact the phase 2 docstring quotes. It matters that the set does NOT depend on how
    many seeds were run: both stragglers appear within the first two offsets, so 0..1, 0..4 and 0..9
    all give the same 14. (It is not universally stable — offsets 3,4,5,6 alone give all 16 — which
    is why the set is recorded in every CSV and provenance sidecar.)"""
    gt = _indian_pines_gt()
    sets = {}
    for k in (2, 5, 10):
        keep, present = common_class_set(gt, block=10, offsets=list(range(k)), guard=1)
        sets[k] = keep
        assert len(keep) == 14, f"seeds 0..{k-1} gave {len(keep)} classes, expected 14"
        excluded = sorted(set(range(16)) - set(keep))
        assert excluded == [6, 8], f"expected GT labels 7 and 9 to fall out, got {excluded}"
    assert sets[2] == sets[5] == sets[10]
    # ...and the all-16 alternative really is only those four adjacent offsets
    all16 = [o for o in range(10) if len(common_class_set(gt, 10, [o], 1)[0]) == 16]
    assert all16 == [3, 4, 5, 6], f"the all-16 offsets moved: {all16}"


def test_the_consolidated_definition_reproduces_the_two_hand_copies_it_replaces():
    """A merge is only faithful if it agrees with BOTH originals on real data. phase1 and
    phase2_degradation each carried their own copy (differing in signature) before bandsim.metrics
    had one; this is the evidence the consolidation changed no number.

    `num_classes` is passed EXPLICITLY where the signature allows it. phase2_degradation's copy
    defaults to its module-level `NUM_CLASSES`, which other modules have historically reassigned from
    the outside (phase 6 set it to 9; tests/test_experiment_guards.py now guards against that), so
    reading the ambient value would make this test depend on what ran before it — it did, and failed
    only in a full-suite run. That the copies inherit mutable global state and the consolidated
    version takes a parameter is part of the argument for consolidating them.
    """
    import inspect
    gt = _indian_pines_gt()
    offsets = [0, 1, 2, 3, 4]
    mine, mine_present = common_class_set(gt, 10, offsets, guard=1, num_classes=16)
    checked = 0
    for mod_name in ("phase1_indian_pines", "phase2_degradation"):
        try:
            mod = __import__(mod_name)
        except Exception as e:                      # optional heavy deps (sklearn/torch)
            pytest.skip(f"{mod_name} not importable here: {e}")
        fn = getattr(mod, "common_class_set", None)
        if fn is None:
            continue                                # already migrated to bandsim.metrics
        kw = {"num_classes": 16} if "num_classes" in inspect.signature(fn).parameters else {}
        theirs, theirs_present = fn(gt, 10, offsets, **kw)
        assert list(theirs) == list(mine), f"{mod_name} disagrees on the class set"
        assert np.array_equal(np.asarray(theirs_present), mine_present), \
            f"{mod_name} disagrees on the per-class split counts"
        checked += 1
    if checked == 0:
        pytest.skip("both copies already migrated -- nothing left to cross-check")


# ------------------------------------------------------------------------------- macro_over
def test_macro_over_is_the_mean_over_exactly_the_given_classes():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 1, 2, 2, 2])
    aa, mi = macro_over(y, p, [0, 1], num_classes=3)
    iou = per_class_iou(y, p, 3)
    rec = per_class_recall(y, p, 3)
    assert mi == pytest.approx(float(np.mean(iou[[0, 1]])))
    assert aa == pytest.approx(float(np.mean(rec[[0, 1]])))
    # dropping a class changes the answer -- i.e. the class set is really load-bearing
    assert macro_over(y, p, [0, 1, 2], num_classes=3)[1] != pytest.approx(mi)


def test_macro_over_lets_a_mismatched_class_set_show_up_as_nan():
    """If the class set was derived from splits other than the one being scored, the result is NaN
    rather than a plausible number computed over fewer classes. Loud beats silently-different."""
    y = np.array([0, 0, 1, 1])                      # class 2 absent from y_true
    p = np.array([0, 0, 1, 1])
    assert np.isnan(macro_over(y, p, [0, 1, 2], num_classes=3)[1])
    assert np.isnan(macro_over(y, p, [0, 1, 2], num_classes=3)[0])


def test_macro_over_rejects_an_out_of_range_or_empty_class_set():
    y = p = np.array([0, 1])
    for bad in ([], [0, 5], [-1]):
        with pytest.raises(ValueError):
            macro_over(y, p, bad, num_classes=3)


# -------------------------------------------------------------------------- per_class_recall
def test_per_class_recall_is_nan_for_absent_classes_and_nanmeans_to_AA():
    y = np.array([0, 0, 1, 1])                      # class 2 absent
    p = np.array([0, 1, 1, 1])
    rec = per_class_recall(y, p, 3)
    assert np.isnan(rec[2]), "an absent class must be NaN, not a spurious 0%"
    assert float(np.nanmean(rec)) == pytest.approx(average_accuracy(y, p, 3))


# ------------------------------------------------------------- the phase5 contract on eval_mlp
def test_main_writes_the_macro_set_both_conventions_and_a_sample_sd(tmp_path, monkeypatch):
    """Run the entrypoint with the trainer STUBBED and assert on what it WRITES.

    A metric that is correct in `eval_mlp` but mis-keyed on its way into the CSV is still a wrong
    deliverable, and unit tests on the helpers cannot see that. Training is replaced by an untrained
    model, so this exercises the whole write path — class set, both mIoU conventions, per-class
    columns, provenance — in seconds and independently of machine load. (Technique borrowed from the
    phase 3 session's note: test the entrypoint on its artefacts, not on its arithmetic.)"""
    _indian_pines_gt()                                   # skip early if the dataset is absent
    import torch                                         # noqa: F401  (import cost only)
    import phase2_cross_sensor as CS
    from bandsim.model import MLPBaseline

    def _no_train(Xtr, ytr, seed, epochs=60, hidden=256, lr=1e-3, bs=256):
        m = MLPBaseline(np.asarray(Xtr).shape[1], CS.NUM_CLASSES, hidden=hidden)
        m.eval()
        return m

    monkeypatch.setattr(CS, "train_mlp", _no_train)
    monkeypatch.setattr(CS, "PAPER_DIR", str(tmp_path))
    (tmp_path / "figs").mkdir()
    monkeypatch.setattr(sys, "argv", ["phase2_cross_sensor.py", "--seeds", "0", "1",
                                      "--epochs", "1", "--device", "cpu", "--jobs", "1"])
    CS.main()

    import csv as _csv
    import json as _json
    summary = list(_csv.DictReader(open(tmp_path / "results_phase2_cross_sensor.csv")))
    raw = list(_csv.DictReader(open(tmp_path / "results_phase2_cross_sensor_raw.csv")))
    assert len(summary) == 3 and len(raw) == 6, "3 band-sets x 2 seeds"
    for row in summary:
        assert row["macro_classes"] == "14", f"the macro set must reach the CSV: {row}"
        assert row["mIoU_mean"] and row["mIoU_present_mean"], "both conventions must be written"
    for row in raw:
        assert row["macro_classes"] == "14"
        assert row["mIoU"] and row["mIoU_present"] and row["AA"] and row["AA_present"]
        # the two classes outside the macro set are exactly the ones that may be blank per seed
        blank = {k for k in row if k.startswith("iou_c") and row[k] == ""}
        assert blank <= {"iou_c7", "iou_c9"}, f"unexpected absent class: {blank}"

    prov = _json.load(open(tmp_path / "results_phase2_cross_sensor.csv.provenance.json"))
    macro = prov["extra"]["macro_estimand"]
    assert macro["excluded_gt_labels"] == [7, 9]
    assert len(macro["class_set_gt_labels"]) == 14
    assert sorted(macro["class_set_gt_labels"] + macro["excluded_gt_labels"]) == list(range(1, 17))

    # The reported spread must be the SAMPLE SD, recomputed from the per-seed rows -- which is the
    # only reason it can be checked at all. And the check must be shown to have POWER: if this run's
    # values make ddof=0 and ddof=1 agree to the CSV's two decimals, the assertion above would pass
    # against either formula and would silently stop guarding anything. (Both points are the phase
    # 1/2/4 session's mutation finding: ddof was revertible at three call sites with no test moving.)
    for row in summary:
        vals = [float(r["mIoU"]) for r in raw if r["condition"] == row["sensor"]]
        assert len(vals) == 2
        sample, population = np.std(vals, ddof=1), np.std(vals)
        # The power criterion is what SURVIVES THE COLUMN'S ROUNDING, not the raw gap: this run's
        # spreads are small (0.0211 vs 0.0149 for one condition) yet still land on different strings
        # at two decimals. Comparing raw magnitudes instead rejected a run whose guard was fine.
        assert f"{sample:.2f}" != f"{population:.2f}", (
            f"this run cannot distinguish ddof=1 from ddof=0 for {row['sensor']} once written at two "
            f"decimals ({sample:.4f} -> {sample:.2f}) -- the assertion below has no power here")
        assert float(row["mIoU_sd"]) == pytest.approx(sample, abs=0.005), \
            f"{row['sensor']}: mIoU_sd is not the sample SD (population would be {population:.2f})"


def test_eval_mlp_without_a_class_set_keeps_its_legacy_meaning():
    """phase5_ab_flagship imports eval_mlp and reads ["mIoU"]. Adding the class-set parameter must
    not move that number when the parameter is not passed."""
    import torch
    import phase2_cross_sensor as CS
    from bandsim.model import MLPBaseline
    torch.manual_seed(0)
    model = MLPBaseline(4, CS.NUM_CLASSES, hidden=8)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = rng.integers(0, CS.NUM_CLASSES, size=64)
    legacy = CS.eval_mlp(model, X, y)
    assert legacy["mIoU"] == pytest.approx(legacy["mIoU_present"])
    assert legacy["AA"] == pytest.approx(legacy["AA_present"])
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).argmax(1).numpy()
    assert legacy["mIoU"] == pytest.approx(miou(y, pred, CS.NUM_CLASSES))
    # ...and passing a set DOES change the macro pair, while leaving OA/kappa alone
    present = sorted(int(c) for c in np.unique(y))[:5]
    fixed = CS.eval_mlp(model, X, y, class_set=present)
    assert fixed["OA"] == pytest.approx(legacy["OA"])
    assert fixed["kappa"] == pytest.approx(legacy["kappa"])
    assert fixed["mIoU_present"] == pytest.approx(legacy["mIoU_present"])
    assert fixed["mIoU"] == pytest.approx(float(np.mean(
        per_class_iou(y, pred, CS.NUM_CLASSES)[np.asarray(present)])))
