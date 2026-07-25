#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment INTEGRITY harness — guards against SILENT failures (the class of bug where a phase
crashes but a stale output file is read as if fresh, e.g. the phase4R missing-import crash).

For every phase script it asserts, from a clean timestamp:
  1. exit code 0                      (the script did not crash)
  2. each expected output was WRITTEN AFTER the run started (mtime >= t0)  -> not a stale file
  3. output CSV parses, is non-empty, and contains NO NaN/inf              -> no silent corruption
  4. metric values are in-range and NOT all-zero                          -> training progressed

Run: python experiments/integrity_check.py            (fast: smoke configs)
Exit code is non-zero if ANY phase fails integrity.
"""
import os, sys, csv, time, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
PAPER = os.path.join(ROOT, "paper")
PY = sys.executable

# (label, argv after the script, [expected output files under paper/])
# NOTE: "Phase0 synthetic PoC" was removed once, when experiment_synthetic_multiseed.py was archived
# during the reorg. It is RESTORED (2026-07-20) and now stamps provenance: the controlled synthetic
# mechanism check is a claim the paper makes, so it must be reproducible rather than cited from
# 07-02 numbers no script in the tree could regenerate.
# The --smoke entries expect *_smoke.csv. A smoke run no longer writes the canonical deliverable
# (tests/test_smoke_isolation.py enforces that), so asking for the unsuffixed name here would find
# the last FULL run's file, judge it older than t0, and report "STALE ... silent crash?" for a phase
# that in fact ran perfectly — the harness inventing the exact failure mode it exists to detect.
PHASES = [
    # Was `--seeds 0 --trials 3` pointed at the CANONICAL results_multiseed.csv, i.e. this harness
    # would have replaced the 5-seed Phase-0 deliverable (and its .tex macros and .pdf) with a
    # 1-seed run the first time anyone ran it. The script now has --smoke like its neighbours, so
    # tests/test_smoke_isolation.py covers it from here on.
    #
    # 2026-07-20 integration: every row below now runs suffix-protected (--smoke or, for phase1,
    # its own _fewsplits guard); the clobber vector this comment used to describe is closed. It was:
    # invoked with `--seeds 0` while their expected output is the UNSUFFIXED deliverable, so running
    # this harness replaces four more paper artefacts with 1-seed runs. Each needs a --smoke flag of
    # its own; changing this list alone does not fix it.
    ("Phase0 synthetic PoC", "experiment_synthetic_multiseed.py --smoke",
     ["results_multiseed_smoke.csv", "results_multiseed_smoke_raw.csv"]),
    # phase1 has no --seeds flag (933e00a renamed the axes to --split-offsets/--model-seeds), so
    # the old entry crashed argparse rather than running. One offset trips phase1's own
    # fewer-than-5-splits guard, which suffixes every artefact _fewsplits -- that guard IS its
    # smoke isolation, so the canonical deliverable is never written from here.
    ("Phase1 IndianPines", "phase1_indian_pines.py --split-offsets 0 --model-seeds 0",
     ["results_phase1_table1_fewsplits.csv"]),
    ("Phase2 degradation", "phase2_degradation.py --smoke", ["results_phase2_curve_smoke.csv"]),
    # Was `--seeds 0 --epochs 12` pointed at the CANONICAL results_phase2_cross_sensor.csv, so this
    # harness overwrote the deliverable it exists to protect with a 1-seed run. The script now has
    # its own --smoke flag, which is what makes tests/test_smoke_isolation.py cover it; this entry
    # only has to stop asking for the unsuffixed name. Both artefacts are listed because a summary
    # of means cannot support the paired comparisons the panel's claims rest on.
    ("Phase2 cross-sensor", "phase2_cross_sensor.py --smoke",
     ["results_phase2_cross_sensor_smoke.csv", "results_phase2_cross_sensor_smoke_raw.csv"]),
    ("Phase3 atmosphere", "phase3_atmosphere.py --smoke", ["results_phase3_atmosphere_smoke.csv"]),
    # Was `--seeds 0 --groups 5 10 --epochs 12` pointed at the CANONICAL results_phase2_group_
    # ablation.csv, i.e. the harness OVERWROTE the deliverable it exists to protect: what sat in
    # paper/ afterwards was a 1-seed 12-epoch two-group run (n_seeds=1, every SD 0.00, no G=20 row,
    # no provenance sidecar) under a STATUS_REPORT line quoting G in {5,10,20} = 53.4/57.2/57.1 —
    # three numbers that file does not contain. The script now has --smoke like its neighbours, so
    # tests/test_smoke_isolation.py covers it automatically from here on.
    #
    # 2026-07-20 integration: the five rows this comment used to list as clobber-open (Phase0,
    # Phase1, Phase2 cross-sensor, Phase5, Phase6) all run suffix-protected now -- --smoke flags
    # from #18/#7/#13+W2/#16, and phase1's own _fewsplits guard. Kept as a marker: a NEW entry must
    # arrive with script-side isolation, not a bare `--seeds 0` against a canonical path.
    ("Phase2 group ablation", "phase2_group_ablation.py --smoke",
     ["results_phase2_group_ablation_G5-10_smoke.csv",
      "results_phase2_group_ablation_G5-10_smoke_raw.csv"]),
    ("Phase4 C/D ablation", "phase4_ablation.py --smoke", ["results_phase4_ablation_smoke.csv"]),
    ("Phase4R reliability", "phase4R_reliability.py --smoke", ["results_phase4R_reliability_smoke.csv"]),
    ("Phase5 A+B flagship", "phase5_ab_flagship.py --smoke", ["results_phase5_ab_flagship_smoke.csv"]),
    # --smoke, not `--seeds 0 --epochs 12`: the latter named the UNSUFFIXED canonical outputs, so
    # running an integrity check REPLACED paper/results_phase6_synthetic.csv with a 1-seed result.
    ("Phase6 synthetic", "phase6_second_dataset.py --dataset synthetic --smoke", ["results_phase6_synthetic_smoke.csv"]),
    # The real-data phases below carry the paper's headline claims and were missing from this list,
    # so the harness printed "every phase" while covering 8 of 17. Their output names are taken from
    # each script's own P(f"...{sfx}.csv") expression, not guessed from the phase name.
    ("Phase7 efficiency", "phase7_efficiency.py --smoke", ["results_phase7_efficiency_smoke.csv"]),
    ("Phase8 CloudSEN12", "phase8_cloudsen12.py --smoke",
     ["results_phase8_cloudsen12_curve_smoke.csv", "results_phase8_cloudsen12_scenarios_smoke.csv",
      "results_phase8_cloudsen12_perclass_smoke.csv"]),
    ("Phase8R reliability", "phase8R_reliability.py --smoke", ["results_phase8R_reliability_smoke.csv"]),
    ("Phase8D difficulty", "phase8D_difficulty.py --smoke", ["results_phase8D_difficulty_smoke.csv"]),
    ("Phase8E DOFA", "phase8E_dofa.py --smoke", ["results_phase8E_dofa_smoke.csv"]),
    ("Phase8F EMIT single", "phase8F_emit.py --smoke", ["results_phase8F_emit_smoke.csv"]),
    ("Phase8F EMIT multi", "phase8F_multi.py --smoke",
     ["results_phase8F_multi_smoke.csv", "results_phase8F_multi_perband_smoke.csv"]),
    ("Phase8G EMIT reliability", "phase8G_emit_reliability.py --smoke",
     ["results_phase8G_emit_reliability_smoke.csv"]),
]


def csv_finite_and_sane(path):
    """Return (ok, msg): CSV parses, has data rows, all numeric cells finite, not all zero."""
    with open(path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return False, "no data rows"
    nums = []
    for r in rows[1:]:
        for cell in r:
            try:
                v = float(cell)
            except ValueError:
                continue
            if v != v or v in (float("inf"), float("-inf")):   # NaN / inf
                return False, f"non-finite cell '{cell}'"
            nums.append(v)
    if not nums:
        return False, "no numeric cells"
    if all(v == 0 for v in nums):
        return False, "all-zero metrics (training did not progress?)"
    # metric sanity: mIoU/OA style columns should be within [-1, 200]
    if max(abs(v) for v in nums) > 1e4:
        return False, f"out-of-range value {max(nums)}"
    return True, f"{len(nums)} finite numeric cells"


def run_phase(label, argv, outputs):
    script = argv.split()[0]
    cmd = [PY, os.path.join("experiments", script)] + argv.split()[1:]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "OMP_NUM_THREADS": "8"})
    dt = time.time() - t0
    # 1. exit code
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "(no stderr)"
        return False, f"CRASH exit={proc.returncode}: {tail}"
    problems = []
    for rel in outputs:
        p = os.path.join(PAPER, rel)
        # 2. fresh output (written after run started) -> not a stale file
        if not os.path.exists(p):
            problems.append(f"missing {rel}"); continue
        if os.path.getmtime(p) < t0 - 1:
            problems.append(f"STALE {rel} (mtime before run -> silent crash?)"); continue
        # 3+4. finite, non-empty, sane
        if rel.endswith(".csv"):
            ok, msg = csv_finite_and_sane(p)
            if not ok:
                problems.append(f"{rel}: {msg}")
    if problems:
        return False, "; ".join(problems)
    return True, f"exit0, fresh+finite outputs ({dt:.0f}s)"


def main():
    print("=" * 74)
    print("EXPERIMENT INTEGRITY HARNESS — every phase: runs, writes FRESH, finite, sane")
    print("=" * 74)
    # Back up EVERY result artefact (csv + tex + figures) and byte-restore afterwards,
    # git-independently. The --smoke phases no longer need this — they write *_smoke.* now — but the
    # Every entry is suffix-protected now (--smoke or phase1's _fewsplits guard). The pre-run
    # backup stays anyway: it is the second line of defence for the day a script's isolation
    # regresses, and it also sweeps up the *_smoke.* files, which is harmless -- they are restored
    # to the bytes this same harness just wrote.
    import glob
    backup = {}
    for p in (glob.glob(os.path.join(PAPER, "*.csv")) + glob.glob(os.path.join(PAPER, "*.tex"))
              + glob.glob(os.path.join(PAPER, "figs", "*.pdf"))):
        backup[p] = open(p, "rb").read()
    # phases that need the precomputed 6S table are SKIPPED (not failed) when it is absent,
    # so the harness stays green on machines without the sixs env (e.g. this GPU container).
    have_6s = os.path.exists(os.path.join(ROOT, "data", "srf_cache", "T_6s_grid.npz"))
    needs_6s = {"phase3_atmosphere.py", "phase5_ab_flagship.py"}
    # Same treatment for the real-data phases: SKIP loudly when the dataset is absent rather than
    # reporting a CRASH, so the harness stays meaningful on a machine that has only part of the data.
    # A skip is printed, never silently omitted -- an unrun phase must not look like a passing one.
    have_s2 = os.path.isdir(os.path.join(ROOT, "data", "cloudsen12"))
    needs_s2 = {"phase8_cloudsen12.py", "phase8R_reliability.py", "phase8D_difficulty.py",
                "phase8E_dofa.py"}
    import glob as _g
    have_emit = bool(_g.glob(os.path.join(ROOT, "data", "emit*", "*RFL*.nc")))
    needs_emit = {"phase8F_emit.py", "phase8F_multi.py", "phase8G_emit_reliability.py"}
    try:
        results = []
        for label, argv, outputs in PHASES:
            script = argv.split()[0]
            if script in needs_6s and not have_6s:
                print(f"[SKIP] {label:22s} needs 6S table (data/srf_cache/T_6s_grid.npz) — absent")
                continue
            if script in needs_s2 and not have_s2:
                print(f"[SKIP] {label:22s} needs CloudSEN12 (data/cloudsen12/) — absent")
                continue
            if script in needs_emit and not have_emit:
                print(f"[SKIP] {label:22s} needs EMIT granules (data/emit*/*RFL*.nc) — absent")
                continue
            ok, msg = run_phase(label, argv, outputs)
            results.append(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {label:22s} {msg}")
    finally:
        for p, data in backup.items():                 # restore canonical results
            open(p, "wb").write(data)
        print("(restored canonical result files clobbered by smoke runs)")
    n_fail = results.count(False)
    print("=" * 74)
    print(f"INTEGRITY: {results.count(True)}/{len(results)} phases genuinely ran with fresh valid output")
    if n_fail:
        print(f"*** {n_fail} PHASE(S) FAILED INTEGRITY — investigate before trusting their results ***")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
