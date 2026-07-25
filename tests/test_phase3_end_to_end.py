"""End-to-end guards for phase3_atmosphere: run main(), then assert on what it WROTE.

WHY THIS FILE EXISTS. tests/test_phase3_atmosphere_guards.py covers the module-level helpers
thoroughly, and a mutation campaign showed that was not enough: 35 of 39 mutations survived, and
the survivors formed an exact pattern -- *every* mutation inside `run_seed`'s closures and inside
`main()`'s writers lived. Extracting product_multiplier / build_eval_input / summary_rows /
validate_args / provenance_extra / control_rng / check_* made those functions reachable, but the
code that CALLS them and the code that EMITS their results stayed as unreachable as before. Testing
a helper does not test the pipeline that uses it; deleting a call site is invisible to a test of the
callee.

So this file runs the real `main()` and the real `run_seed`, and asserts on the artefacts. It kills
the defect class the helper tests structurally cannot see:
  * a call site deleted (validate_args, check_results_complete, check_mask_invariant,
    provenance_extra all extracted precisely so they could be tested -- and nothing checked main()
    still calls them);
  * `--tau-sweep` passing the module default instead of the swept tau, so the sweep does not sweep
    while its CSV reports five distinct taus;
  * the supplementary arm evaluated with correct_radiometry=True, making it a copy of primary;
  * the ceiling row computed WITH the core mask, zeroing the 19.4/6.5 fixed-core headline;
  * proposed_nomask handed `absent` instead of `[]`, so the only measurement of the mask mechanism
    becomes a copy of the column it is meant to be compared against;
  * retention divided by the other method's clean score; policy labels swapped; a column dropped.

NOTHING IS TRAINED. The three trainers are stubbed with tiny deterministic modules, so this runs in
seconds on CPU and asserts structure and arithmetic, not model quality. The 6S LUT and the real
wavelength axis ARE used -- the band-loss geometry is the thing under test.
"""
import csv
import json
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

import phase3_atmosphere as P3                                            # noqa: E402
from bandsim.io import AVIRIS_WL_NM                                       # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(P3.TABLE), reason="6S table not present")

N_CLASSES = 16
SIDE = 40


class _StubMLP(torch.nn.Module):
    """B2 stand-in: logits are a deterministic linear read of the spectrum.

    Deterministic and input-DEPENDENT on purpose -- a constant-output stub would make every
    condition score identically and hide any mutation that changes which bands reach the model.
    """

    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(n_bands, dtype=torch.float32) / n_bands,
                                    requires_grad=False)
        self.n_classes = n_classes

    def forward(self, x):
        s = (x * self.w).cumsum(1)
        return torch.stack([s[:, i::self.n_classes].sum(1) for i in range(self.n_classes)], 1)


class _StubProposed(_StubMLP):
    """Proposed stand-in: same read, but the present-mask must change the ARGMAX.

    Projecting the mask onto classes rather than adding `present_mask.sum()` as a scalar: a
    constant added to every logit leaves argmax untouched, so a stub built that way reports
    identical mIoU whether or not a group is masked, and every mask-related assertion below would
    pass vacuously. (That is exactly how the first version of this file failed.)
    """

    def forward(self, x, present_mask):
        base = super().forward(x)
        pm = present_mask.float()
        g = pm.shape[1]
        proj = torch.stack([pm[:, i % g] for i in range(self.n_classes)], 1)
        return base + 50.0 * proj


@pytest.fixture
def ran(tmp_path, monkeypatch):
    """Run main() once under --smoke into a temporary paper dir; return the parsed artefacts."""
    n_bands = len(AVIRIS_WL_NM)
    rng = np.random.default_rng(0)
    cube = rng.uniform(1000.0, 5000.0, size=(SIDE, SIDE, n_bands))
    # every 10x10 block carries all 16 classes, so guard=1 cannot empty a class from either split
    gt = np.tile(np.arange(1, N_CLASSES + 1).reshape(4, 4), (SIDE // 4, SIDE // 4))

    monkeypatch.setattr(P3, "load_data", lambda: (cube, gt))
    monkeypatch.setattr(P3, "train_mlp",
                        lambda *a, **k: _StubMLP(n_bands, N_CLASSES).eval())
    monkeypatch.setattr(P3, "pretrain_sgmae", lambda *a, **k: None)
    monkeypatch.setattr(P3, "finetune_proposed", lambda *a, **k: None)
    monkeypatch.setattr(P3, "GroupedCrossBandAttention",
                        lambda *a, **k: _StubProposed(n_bands, N_CLASSES).eval())
    monkeypatch.setattr(P3.hw, "setup", lambda **k: None)
    monkeypatch.setattr(P3.hw, "info", lambda: "stub")
    monkeypatch.setattr(P3.hw, "device", lambda: torch.device("cpu"))

    paper = tmp_path / "paper"
    (paper / "figs").mkdir(parents=True)
    monkeypatch.setattr(P3, "PAPER_DIR", str(paper))
    monkeypatch.setattr(P3, "P", lambda rel: os.path.join(str(paper), rel))
    monkeypatch.setattr(sys, "argv",
                        ["phase3_atmosphere.py", "--smoke", "--tau-sweep", "0.3", "0.6",
                         "--random-control", "--jobs", "1", "--device", "cpu"])
    P3.main()

    def read(name):
        p = paper / name
        assert p.exists(), f"{name} was not written"
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    prov_path = paper / "results_phase3_atmosphere_smoke.csv.provenance.json"
    return {
        "dir": paper,
        "summary": read("results_phase3_atmosphere_smoke.csv"),
        "perseed": read("results_phase3_perseed_smoke.csv"),
        "sweep": read("results_phase3_tau_sweep_smoke.csv"),
        "control": read("results_phase3_random_control_smoke.csv"),
        "prov": json.loads(prov_path.read_text()) if prov_path.exists() else None,
    }


# --------------------------------------------------------------------------------------------
# smoke isolation — the bug class that once INVERTED this phase's published conclusion
# --------------------------------------------------------------------------------------------
def test_smoke_writes_only_suffixed_artefacts(ran):
    """A --smoke run must never touch a real deliverable. A 1-seed sanity check silently replaced
    the multi-seed table the paper quotes, once, and nothing in the CSV recorded the swap."""
    written = {p.name for p in ran["dir"].rglob("*") if p.is_file()}
    unsuffixed = [n for n in written
                  if n.startswith(("results_phase3", "fig_robust")) and "_smoke" not in n]
    assert not unsuffixed, f"a --smoke run wrote unsuffixed artefacts: {sorted(unsuffixed)}"
    assert "fig_robust_vs_cwv_smoke.pdf" in written


# --------------------------------------------------------------------------------------------
# the summary table
# --------------------------------------------------------------------------------------------
def test_summary_has_every_condition_and_the_ceiling_row(ran):
    conds = [r["condition"] for r in ran["summary"]]
    assert conds == ["benchmark200_full", "clean", "cwv0.5", "cwv2.0", "cwv4.0"], \
        "a condition vanished from the published table"


def test_ceiling_row_really_drops_the_core_mask(ran):
    """benchmark200_full must be evaluated with apply_core_mask=False. If it silently carries the
    core mask, the fixed-core cost collapses to zero -- and the paper's "most of the margin predates
    any CWV shift" claim (19.4 vs 6.5 mIoU) is computed from exactly this difference."""
    by = {r["condition"]: r for r in ran["summary"]}
    ceiling, clean = by["benchmark200_full"], by["clean"]
    assert int(ceiling["n_missing_bands"]) == 0, "the ceiling row is missing bands; core mask leaked in"
    # 10, not 8, since the 2026-07-20 axis correction: every centre moved +0.2..37.7 nm, so two
    # more bands fall inside the fixed water-vapour core windows.
    assert int(clean["n_missing_bands"]) == 10, "clean must carry the 10 hard-masked core bands"
    assert float(ceiling["proposed_miou_mean"]) != float(clean["proposed_miou_mean"]), \
        "ceiling == clean, so the fixed-core cost is identically zero and the headline is void"


def test_summary_reports_the_real_band_loss_geometry(ran):
    """n_missing_bands / n_absent_groups are the information-parity claim AS DATA. Hardcoding them
    would make the parity argument unfalsifiable from the artefact."""
    by = {r["condition"]: r for r in ran["summary"]}
    assert [int(by[f"cwv{c}"]["n_missing_bands"]) for c in P3.CWVS] == [19, 32, 40]  # corrected axis
    # [0, 0, 0] on the corrected axis: at CWV 4.0 no whole group crosses the absence threshold any
    # more (the 40 missing bands spread differently). The shared-layer phase-3 note "mask mechanism
    # inert in 3 of 4 conditions" is therefore 4 of 4 on this axis -- the margin phase 3 reports is
    # a transformer-vs-MLP result under band corruption, never mask-mechanism evidence.
    assert [int(by[f"cwv{c}"]["n_absent_groups"]) for c in P3.CWVS] == [0, 0, 0]
    assert int(by["clean"]["n_absent_groups"]) == 0


def test_retention_is_each_method_against_its_own_clean_score(ran):
    """Retention divided by the OTHER method's clean score survives every helper test, and it is
    quoted in both the CSV and the printed summary."""
    by = {r["condition"]: r for r in ran["summary"]}
    clean = by["clean"]
    assert clean["proposed_retention"] == "1.00" and clean["b2_retention"] == "1.00", \
        "the clean row must anchor retention at 1.00 by construction"
    for c in P3.CWVS:
        row = by[f"cwv{c}"]
        want_p = float(row["proposed_miou_mean"]) / float(clean["proposed_miou_mean"])
        want_b = float(row["b2_miou_mean"]) / float(clean["b2_miou_mean"])
        assert float(row["proposed_retention"]) == pytest.approx(want_p, abs=0.011)
        assert float(row["b2_retention"]) == pytest.approx(want_b, abs=0.011)


def test_summary_is_recomputable_from_the_per_seed_rows(ran):
    """The per-seed CSV exists so a reader can rebuild the table and run a paired test. If it
    cannot reproduce the summary, it is decoration."""
    by_seed = {}
    for r in ran["perseed"]:
        by_seed.setdefault(r["condition"], []).append(r)
    for row in ran["summary"]:
        cond = row["condition"]
        if cond not in by_seed:
            continue
        vals = [float(r["proposed_miou"]) for r in by_seed[cond]]
        assert float(row["proposed_miou_mean"]) == pytest.approx(np.mean(vals), abs=0.011), \
            f"{cond}: summary mean does not match the per-seed rows it claims to summarise"
        diffs = [float(r["paired_diff"]) for r in by_seed[cond]]
        assert float(row["paired_diff_mean"]) == pytest.approx(np.mean(diffs), abs=0.011)


def test_per_seed_paired_diff_has_the_documented_sign(ran):
    """paired_diff is proposed - b2. A sign flip would invert every comparison the paper makes."""
    for r in ran["perseed"]:
        want = float(r["proposed_miou"]) - float(r["b2_miou"])
        assert float(r["paired_diff"]) == pytest.approx(want, abs=1e-6), \
            "paired_diff is not (proposed - b2); the sign or the operands changed"


def test_per_seed_ceiling_row_does_not_report_corrected_values_as_uncorrected(ran):
    rows = [r for r in ran["perseed"] if r["condition"] == "benchmark200_full"]
    assert rows, "the ceiling row is missing from the per-seed file"
    for r in rows:
        assert r["proposed_uncorrected"] == "-" and r["b2_uncorrected"] == "-"


# --------------------------------------------------------------------------------------------
# the two arms, and the mask ablation
# --------------------------------------------------------------------------------------------
def test_the_supplementary_arm_is_not_a_copy_of_the_primary_arm(ran):
    """If eval_condition is ever called with correct_radiometry=True for the supplementary arm, the
    uncorrected columns become a duplicate of the primary ones and the whole product distinction
    silently disappears while the CSV still shows two sets of numbers."""
    by = {r["condition"]: r for r in ran["summary"]}
    same = [c for c in P3.CWVS
            if by[f"cwv{c}"]["proposed_gasonly_uncorrected_mean"] == by[f"cwv{c}"]["proposed_miou_mean"]]
    assert not same, f"the uncorrected column equals the corrected one at CWV={same}"


def test_the_mask_ablation_is_inert_exactly_where_no_group_is_absent(ran):
    """proposed_nomask is the ONLY measurement of the mask mechanism. It must equal proposed where
    nothing is absent (by construction) and differ where something is -- if it is handed `absent`
    instead of `[]`, it becomes a copy of the column it exists to be compared against."""
    by = {r["condition"]: r for r in ran["summary"]}
    for c in P3.CWVS:
        row = by[f"cwv{c}"]
        nomask, prop = row["proposed_nomask_mean"], row["proposed_miou_mean"]
        if int(row["n_absent_groups"]) == 0:
            assert nomask == prop, f"CWV={c}: no group absent, yet the ablation changed the score"
        else:
            assert nomask != prop, (
                f"CWV={c}: {row['n_absent_groups']} group(s) absent but the ablation is identical; "
                f"the mask mechanism is not actually being switched off")


# --------------------------------------------------------------------------------------------
# the optional arms
# --------------------------------------------------------------------------------------------
def test_the_tau_sweep_actually_sweeps(ran):
    """THE highest-risk defect in the file. `physics_missing_bands` is keyword-only precisely so a
    sweep cannot silently threshold at the module default -- but that guards the CALLEE. If the call
    site inside run_seed passes `tau_missing` instead of `tau`, the CSV reports distinct taus whose
    rows are identical, and the withdrawn sweep claim would be re-made from an artefact that never
    swept anything."""
    rows = ran["sweep"]
    taus = sorted({float(r["tau"]) for r in rows})
    assert taus == [0.3, 0.6], f"the sweep grid is {taus}, not what was requested"
    for c in P3.CWVS:
        per_tau = {float(r["tau"]): int(r["n_missing_bands"]) for r in rows
                   if float(r["cwv"]) == c}
        assert per_tau[0.3] < per_tau[0.6], (
            f"CWV={c}: a higher TAU did not remove more bands ({per_tau}); the sweep is not "
            f"varying the threshold it reports")


def test_tau_sweep_columns_are_not_transposed(ran):
    """tau and cwv under each other's headers survives every arithmetic check."""
    for r in ran["sweep"]:
        assert float(r["tau"]) in (0.3, 0.6), f"tau column holds {r['tau']}"
        assert float(r["cwv"]) in tuple(P3.CWVS), f"cwv column holds {r['cwv']}"


def test_random_control_reports_both_policies_matched_on_band_count(ran):
    """The control answers 'does the IDENTITY of the lost bands matter, at the same COUNT'. Swapped
    policy labels would invert its conclusion; an unmatched count would answer a different
    question."""
    rows = ran["control"]
    assert {r["policy"] for r in rows} == {"physics_6s", "uniform_random"}
    by = {}
    for r in rows:
        by.setdefault(float(r["cwv"]), {})[r["policy"]] = r
    physics_counts = {c: int(v["physics_6s"]["n_missing_bands"]) for c, v in by.items()}
    # [19, 32, 40] on the corrected axis (was [15, 27, 37] on the stretched one): the geometry is
    # a function of the axis, exactly as the phase3/8F shared-layer entry warned.
    assert [physics_counts[c] for c in P3.CWVS] == [19, 32, 40], \
        "the physics rows are not carrying the real 6S band counts; labels may be swapped"
    for c, v in by.items():
        assert v["physics_6s"]["n_missing_bands"] == v["uniform_random"]["n_missing_bands"], \
            f"CWV={c}: the arms are not count-matched, so they answer different questions"


# --------------------------------------------------------------------------------------------
# wiring: helpers extracted to be testable must still be CALLED
# --------------------------------------------------------------------------------------------
def test_main_actually_stamps_the_full_provenance_payload(ran):
    """provenance_extra is asserted on in the helper tests. Nothing checked main() still passes it,
    and `stamp(..., extra={})` leaves every one of those assertions true while the artefact records
    nothing."""
    prov = ran["prov"]
    assert prov is not None, "main() did not stamp the summary CSV"
    extra = prov.get("extra", {})
    for k in ("cube_sha256", "lut_sha256", "lut_keys_used", "cube_physical_quantity_verified",
              "n_absent_groups_per_condition", "wavelength_axis_provenance"):
        assert k in extra, f"main() stamped a payload without {k}; provenance_extra is unwired"
    assert extra["cube_physical_quantity_verified"] is False
    assert extra["lut_keys_used"], "lut_keys_used is empty; the resolved column is not recorded"


def test_provenance_records_which_classes_the_miou_averages_over(ran):
    """"mIoU" names two different quantities across this repo unless the class set is recorded.
    Phases 1/2 score on the classes common to every split; phase 3 now does too, and the artefact
    has to say so or a reader cannot tell which table rows are comparable."""
    extra = ran["prov"]["extra"]
    assert extra["macro_class_set_0based"], "the macro class set is not recorded"
    assert extra["macro_n_classes"] == len(extra["macro_class_set_0based"])
    assert "common_class_set" in extra["macro_class_set_note"]
    assert extra["classes_per_split_count"], "per-class split counts must be recorded too"


def test_provenance_records_the_thresholds_actually_used_not_the_module_defaults(ran):
    """Stamping TAU_MISSING instead of args.tau_missing records a value the run may not have used.
    The sweep grid is the tell: it exists only as a CLI argument."""
    extra = ran["prov"]["extra"]
    assert extra["tau_sweep"] == [0.3, 0.6], \
        "the stamped sweep grid is not the one the run was given"
    assert extra["random_control"] is True
    assert extra["tau_missing"] == P3.TAU_MISSING          # default here, but it must be present
    assert "group_absent_frac" in extra


def test_main_rejects_an_invalid_configuration(monkeypatch, tmp_path):
    """validate_args was extracted so it could be tested -- and nothing checked main() calls it.
    A deleted call site leaves every validate_args test passing."""
    monkeypatch.setattr(sys, "argv", ["phase3_atmosphere.py", "--smoke", "--groups", "1"])
    monkeypatch.setattr(P3.hw, "setup", lambda **k: None)
    with pytest.raises(SystemExit):
        P3.main()


# --------------------------------------------------------------------------------------------
# run_seed itself (called directly, still nothing trained)
# --------------------------------------------------------------------------------------------
def _run_seed_direct(monkeypatch, **over):
    n_bands = len(AVIRIS_WL_NM)
    rng = np.random.default_rng(1)
    cube = rng.uniform(1000.0, 5000.0, size=(SIDE, SIDE, n_bands))
    gt = np.tile(np.arange(1, N_CLASSES + 1).reshape(4, 4), (SIDE // 4, SIDE // 4))
    monkeypatch.setattr(P3, "train_mlp", lambda *a, **k: _StubMLP(n_bands, N_CLASSES).eval())
    monkeypatch.setattr(P3, "pretrain_sgmae", lambda *a, **k: None)
    monkeypatch.setattr(P3, "finetune_proposed", lambda *a, **k: None)
    monkeypatch.setattr(P3, "GroupedCrossBandAttention",
                        lambda *a, **k: _StubProposed(n_bands, N_CLASSES).eval())
    monkeypatch.setattr(P3.hw, "device", lambda: torch.device("cpu"))
    kw = dict(n_groups=10, epochs=1)
    kw.update(over)
    return P3.run_seed(0, cube, gt, **kw)


def test_run_seed_applies_the_hard_core_mask_in_every_condition(monkeypatch):
    """If `core` is ever all-True, the absorption cores are never masked and the clean reference
    stops carrying the fixed removal -- which is the confound the clean/benchmark200_full split
    exists to separate."""
    res = _run_seed_direct(monkeypatch)
    assert res["clean"]["n_missing_bands"] == 10, "clean lost the hard core mask"  # 10 on the corrected axis
    assert res[4.0]["n_missing_bands"] == 40  # corrected axis


def test_run_seed_standardises_on_TRAIN_statistics_only(monkeypatch):
    """Standardising with test-set mu/sd is leakage, and it would additionally erase the
    low-variance mechanism the supplementary docstring spends 25 lines explaining.

    Checked by CAPTURING the (mu, sd) run_seed actually hands to build_eval_input and comparing
    them against the train-region statistics, exactly. A first version of this test tried to detect
    leakage behaviourally, by giving the test region a different scale -- that cannot work here,
    because the checkerboard split interleaves blocks across the whole scene, so the train region
    always spans the same distribution as the test region. Capturing the arguments tests the claim
    directly instead of hoping a proxy is sensitive to it.
    """
    n_bands = len(AVIRIS_WL_NM)
    rng = np.random.default_rng(2)
    cube = rng.uniform(1000.0, 5000.0, size=(SIDE, SIDE, n_bands))
    gt = np.tile(np.arange(1, N_CLASSES + 1).reshape(4, 4), (SIDE // 4, SIDE // 4))

    seen = []
    real = P3.build_eval_input
    monkeypatch.setattr(P3, "build_eval_input",
                        lambda X, T, m, mu, sd: seen.append((mu, sd)) or real(X, T, m, mu, sd))
    monkeypatch.setattr(P3, "train_mlp", lambda *a, **k: _StubMLP(n_bands, N_CLASSES).eval())
    monkeypatch.setattr(P3, "pretrain_sgmae", lambda *a, **k: None)
    monkeypatch.setattr(P3, "finetune_proposed", lambda *a, **k: None)
    monkeypatch.setattr(P3, "GroupedCrossBandAttention",
                        lambda *a, **k: _StubProposed(n_bands, N_CLASSES).eval())
    monkeypatch.setattr(P3.hw, "device", lambda: torch.device("cpu"))
    P3.run_seed(0, cube, gt, n_groups=10, epochs=1)

    assert seen, "build_eval_input was never called; the capture is not on the real path"
    tr, te = P3.disjoint_block_split(gt, block=10, guard=1, offset=0)
    want_mu = cube[tr].mean(0)
    want_sd = cube[tr].std(0) + 1e-8
    test_mu = cube[te].mean(0)
    for mu, sd in seen:
        assert np.allclose(mu, want_mu), "standardisation did not use the TRAIN-region mean"
        assert np.allclose(sd, want_sd), "standardisation did not use the TRAIN-region sd"
    # and the two regions must actually differ, or the assertion above proves nothing
    assert not np.allclose(want_mu, test_mu), \
        "train and test means coincide here, so this test could not detect leakage"


def test_run_seed_masks_bands_the_atmosphere_removes(monkeypatch):
    """The missing bands must actually reach the model as standardized 0, in every condition."""
    res = _run_seed_direct(monkeypatch)
    assert res[4.0]["n_missing_bands"] > res[0.5]["n_missing_bands"] > res["clean"]["n_missing_bands"]
    assert res["clean"]["proposed_nomask"] == res["clean"]["proposed"]
