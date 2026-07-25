#!/usr/bin/env python3
"""2x2: wavelength-vs-learned positional encoding X band-group dropout, SGMAE held fixed.

WHY THIS EXISTS.  phase2's headline contrast is a crossover, not a win: Proposed - B4 runs
+5.32 mIoU at m=0 down to -10.65 at m=6 (5 seeds; every level's 95% paired interval excludes
zero, so the sign change is real and not noise).  Routing that contrast through B6 splits the
modest net into two much larger opposing effects:

    B6 - B4        (HCS -> SGMAE, learned PE both sides)     +3.64  ->  -28.75
    Proposed - B6  (learned+nodrop -> wavelength+drop)       +1.68  ->  +18.10
    ---------------------------------------------------------------------------
    Proposed - B4  (the headline)                            +5.32  ->  -10.65

So spectral-group masked pretraining ON ITS OWN is catastrophic for missing-band robustness,
and something in the Proposed arm more than repays it.  That is a far more interesting claim
than the headline -- but phase2 cannot support it, because "Proposed - B6" moves TWO factors
at once: the positional encoding (wavelength vs learned) AND band-group dropout (on vs off).
Which one repays the SGMAE debt is exactly what the paper wants to assert, and it is not
answerable from any shipped artefact.  Hence this script, which fills the two empty cells:

                        learned PE          wavelength PE
    no dropout          B6      (upstream)  NEW
    dropout             NEW                 Proposed  (upstream)

Reading the result: the PE main effect is (wavePE_nodrop - learnedPE_nodrop), the dropout main
effect is (learnedPE_drop - learnedPE_nodrop), and the interaction is whatever the four cells
do not explain additively.  A large interaction is the scientifically interesting outcome and
is the reason both empty cells are run rather than just one -- with a single new cell the
design is saturated and the interaction is unidentifiable, so a strong "wavelength PE only
pays off once dropout is present" effect would be silently absorbed into a main effect.

WHY ALL FOUR CELLS ARE RECOMPUTED rather than two of them joined from results_phase2_raw.csv:
joining would compare cells scored on different band-drop realisations, since phase2 hands
every method its own fresh generator.  Recomputing costs two extra models per seed and buys a
genuinely paired design plus a free correctness check -- the two cells that DO exist upstream
must reproduce phase2's numbers, and --verify fails loudly if they do not, which is what would
happen if the seeding contract below were mirrored even slightly wrong.

This writes results_phase2X_* and touches no canonical phase2 deliverable.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.phase2_degradation as P2  # noqa: E402
import experiments.phase6_second_dataset as P6  # noqa: E402
from bandsim import hw  # noqa: E402
from bandsim.grouping import contiguous_groups, group_center_wavelengths  # noqa: E402
from bandsim.io import disjoint_block_split  # noqa: E402
from bandsim.model import GroupedCrossBandAttention  # noqa: E402
from bandsim.provenance import stamp  # noqa: E402

# name -> (pe_type, group_dropout, upstream phase2 `kind` this cell must reproduce or None)
#
# pe_type "sinusoidal" IS the wavelength-conditioned PE -- bandsim.model's own name for the
# proposed ingredient.  Spelling it "wavelength" here and translating at construction keeps the
# table readable without inventing a second vocabulary for the model to misinterpret.
ARMS = {
    "learnedPE_nodrop": ("learned", False, "b6"),
    "wavePE_nodrop":    ("sinusoidal", False, None),
    "learnedPE_drop":   ("learned", True, None),
    "wavePE_drop":      ("sinusoidal", True, "proposed"),
}

# --sampling-baselines arm set: architecture HELD FIXED (wavelength PE + SGMAE = the Proposed
# recipe), only the training-time channel-sampling augmentation varies. This answers the reviewer
# objection "your band-group dropout = ChannelViT HCS" head-to-head on redundant HSI bands, and
# tests whether spectral redundancy changes the natural-image conclusion.
#   drop = our band-group dropout (== wavePE_drop == Proposed; sampling='drop' is bit-identical);
#   hcs  = ChannelViT Hierarchical Channel Sampling (ICLR'24) on OUR architecture;
#   dcs  = DiChaViT Diverse Channel Sampling COMPONENT (NeurIPS'24, sampling only, no CDL/TDL).
SAMPLING_ARMS = {
    "drop": ("sinusoidal", "drop", None),
    "hcs":  ("sinusoidal", "hcs", None),
    "dcs":  ("sinusoidal", "dcs", None),
}


def _prep_checkerboard(seed, cube, gt, n_groups):
    """phase2's split: the one the Indian Pines 2x2 was measured on, so verification can run.

    Its five "seeds" are one-pixel offsets of a single checkerboard whose test sets overlap at
    mean pairwise IoU 0.440 -- intervals computed over them are too NARROW. That is a known and
    documented defect, kept here only because reproducing results_phase2_raw.csv requires the
    identical split; --split blocks is the better design and is the default everywhere else.
    """
    Xtr, ytr, Xte, yte, Xte_raw, mu, sd = P2.prep(
        cube, gt, block=P2.SPLIT_BLOCK, offset=seed, return_raw=True)
    wl = P2.AVIRIS_WL_NM
    groups = contiguous_groups(Xtr.shape[1], n_groups)
    return (Xtr, ytr, Xte, yte, Xte_raw, mu, sd, wl, groups,
            group_center_wavelengths(wl, groups))


def _prep_blocks(seed, cube, gt, wl, n_groups):
    """phase6's split: disjoint blocks with a one-pixel guard band, mirrored line for line.

    guard=1 is what the checkerboard lacks -- it keeps train and test pixels from touching, so the
    test sets of different offsets do not share pixels the way the checkerboard's do.
    """
    tr, te = disjoint_block_split(gt, block=10, guard=1, offset=seed)
    Xtr = cube[tr]; ytr = gt[tr].astype(int) - 1
    Xte = cube[te]; yte = gt[te].astype(int) - 1
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xte_raw = Xte.astype(np.float32)
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)
    groups = contiguous_groups(cube.shape[-1], n_groups)
    return (Xtr, ytr, Xte, yte, Xte_raw, mu, sd, wl, groups,
            group_center_wavelengths(wl, groups))


def run_seed_2x2(seed, prepared, max_missing, trials, epochs, class_set, num_classes,
                 wl_scale=None, arms=None):
    """Mirror of P2.run_seed's contract, restricted to the four SGMAE cells.

    Every deviation from phase2 here would silently decouple this 2x2 from the artefacts it is
    meant to explain, so the three load-bearing lines are called out:
      * the split comes from `prepared` and is offset by seed -- see _prep_* above;
      * torch.manual_seed(seed + 101) before EACH construction -- phase2 reseeds per model so
        that B4/B6/Proposed are paired rather than sharing one advancing stream.  phase6 uses the
        SAME +101 deliberately, so an arm built here initialises identically down either path;
      * a FRESH default_rng(seed + 999) per cell -- so all four cells face the IDENTICAL set of
        random band-drop realisations.  One shared generator would advance between cells and
        turn part of every reported gap into sampling noise.

    num_classes is threaded EXPLICITLY rather than read off P2.NUM_CLASSES. phase6 documents what
    the global costs: a 9-class cube built a 9-output head that was then scored with miou(..., 16),
    making classes 9..15 impossible to predict and contributing IoU 0 each -- a plausible, silently
    deflated mIoU rather than an error.
    """
    Xtr, ytr, Xte, yte, Xte_raw, mu, sd, wl, groups, cwl = prepared

    # sinusoidal_wavelength_pe normalises by wl_scale=2500 -- AVIRIS's upper bound -- and takes no
    # override, so a narrow-range sensor gets a compressed, less separable encoding. Rescaling the
    # centres here is EXACTLY equivalent and touches no other phase:
    #     sinusoidal_wavelength_pe(cwl, d, wl_scale=S) == sinusoidal_wavelength_pe(cwl*2500/S, d)
    # verified to 0.0 max abs difference. cwl reaches nothing but the PE -- degradation_curve gets
    # the untouched `wl` for its own use -- so this changes the arm under test and nothing else.
    if wl_scale is not None:
        cwl = np.asarray(cwl, float) * (2500.0 / float(wl_scale))

    rows, curves = [], {}
    for arm, (pe_type, aug, _up) in (arms or ARMS).items():
        torch.manual_seed(seed + 101)
        m = GroupedCrossBandAttention(groups, cwl, num_classes, pe_type=pe_type)
        P2.pretrain_sgmae(m, Xtr, groups, seed, epochs=max(1, epochs // 2))
        # No num_classes here: the head was sized at construction, SGMAE is label-free and the
        # finetune reads the head off the model. phase6 says the same in as many words.
        # aug is a BOOL (-> group_dropout, the PE×dropout 2×2) or a sampling-mode STRING
        # ('drop'/'hcs'/'dcs' -> the channel-sampling baseline comparison). Both go through
        # finetune_proposed; only the training-time group-masking augmentation differs, so the
        # architecture is held fixed and the arms are compute-matched. sampling='drop' is
        # bit-identical to group_dropout=True (same _vec_group_subset call, same RNG), so the
        # 'drop' baseline anchors the sampling comparison to the Proposed arm.
        if isinstance(aug, bool):
            P2.finetune_proposed(m, Xtr, ytr, groups, seed, epochs=epochs, group_dropout=aug)
        else:
            P2.finetune_proposed(m, Xtr, ytr, groups, seed, epochs=epochs, sampling=aug)

        # kind="proposed", NOT the arm name.  degradation_curve dispatches the forward pass on
        # `kind in ("proposed","b4","b6")`; an unrecognised string would route these attention
        # models down the plain-MLP path and score four subtly wrong curves without erroring.
        # The label is corrected on the recorded rows immediately below.
        mark = len(rows)
        curves[arm] = P2.degradation_curve(
            "proposed", m, Xte, yte, groups, wl, max_missing, trials,
            np.random.default_rng(seed + 999), class_set=class_set, record=rows,
            Xte_raw=Xte_raw, mu=mu, sd=sd, num_classes=num_classes)
        for r in rows[mark:]:
            r["kind"] = arm
            r["seed"] = seed
    return curves, rows


def paired(rows, a, b, max_missing):
    """Per-level paired mean difference a-b with a 95% t interval, seeds as the unit."""
    from scipy import stats
    by = {}
    for r in rows:
        by.setdefault((r["kind"], int(r["seed"])), {}).setdefault(
            int(r["missing_groups"]), []).append(float(r["miou"]))
    seeds = sorted({s for k, s in by if k == a})
    out = []
    for m in range(max_missing + 1):
        A = np.array([np.mean(by[(a, s)][m]) for s in seeds])
        B = np.array([np.mean(by[(b, s)][m]) for s in seeds])
        d = A - B
        # ddof=1: these 5 seeds are a sample from the seed distribution, not the population.
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        out.append((m, float(d.mean()), float(d.mean() - h), float(d.mean() + h)))
    return out


def verify_against_phase2(rows, raw_path, tol):
    """The two upstream cells must reproduce results_phase2_raw.csv.

    Returns a list of human-readable discrepancies; empty means the seeding contract mirrored
    correctly.  A missing upstream file is reported, not swallowed -- silently skipping the one
    check that can catch a mis-mirrored contract would defeat the point of running it.
    """
    p = Path(raw_path)
    if not p.exists():
        return [f"{p} absent -- cannot verify the two upstream cells"]
    up = {}
    for r in csv.DictReader(p.open()):
        up.setdefault((r["kind"], int(r["seed"]), int(r["missing_groups"])), []).append(
            float(r["miou"]))
    mine = {}
    for r in rows:
        mine.setdefault((r["kind"], int(r["seed"]), int(r["missing_groups"])), []).append(
            float(r["miou"]))
    bad = []
    for arm, (_pe, _gd, upstream) in ARMS.items():
        if upstream is None:
            continue
        checked = 0
        for (k, s, m), v in sorted(mine.items()):
            if k != arm:
                continue
            ref = up.get((upstream, s, m))
            if ref is None:
                bad.append(f"{arm}: phase2 has no {upstream} seed={s} m={m}")
                continue
            got, exp = float(np.mean(v)), float(np.mean(ref))
            checked += 1
            if abs(got - exp) > tol:
                bad.append(f"{arm} vs {upstream} seed={s} m={m}: {got:.4f} != {exp:.4f} "
                           f"(delta {got - exp:+.4f} > tol {tol})")
        if checked == 0:
            bad.append(f"{arm}: nothing compared against {upstream} -- check is vacuous")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dataset", default="indian_pines",
                    choices=["indian_pines", "pavia", "salinas", "synthetic"],
                    help="scene to run the 2x2 on; non-Indian scenes route through phase6's "
                         "loader, which pairs each cube with ITS wavelength axis")
    # Two splits, chosen explicitly rather than defaulted per dataset. checkerboard exists to
    # reproduce results_phase2_raw.csv and carries its 0.440 test-set overlap; blocks is phase6's
    # guard-banded design and is what a fresh scene should use. Running indian_pines under BOTH
    # asks whether the conclusion survives the split, which is a different question from whether
    # it survives the scene.
    ap.add_argument("--split", default=None, choices=["checkerboard", "blocks"],
                    help="default: checkerboard for indian_pines (verifiable), blocks otherwise")
    ap.add_argument("--wl-scale", default=None,
                    help="normalisation for the wavelength PE: 'auto' uses this sensor's own max "
                         "group centre, a number sets it explicitly, omitted keeps the model's "
                         "hardcoded 2500 (AVIRIS's top). Only 'auto'/explicit changes the PE arm")
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--max-missing", type=int, default=6)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--sampling-baselines", action="store_true",
                    help="instead of the PE×dropout 2×2, run the channel-sampling comparison "
                         "{our band-group drop, ChannelViT-HCS, DiChaViT-DCS} on a FIXED "
                         "wavelength-PE+SGMAE architecture; writes results_phase2X_sampling_*.csv")
    # 1e-4, not 1e-6: phase2 writes miou to four decimals, so a value that round-trips through
    # results_phase2_raw.csv can legitimately differ from the in-memory float by up to 5e-5. A
    # tighter tolerance does not detect a mirrored-contract bug, it just fails on every row while
    # printing pairs that are visibly identical ("67.4087 != 67.4087") -- which is what 1e-6 did.
    ap.add_argument("--verify-tol", type=float, default=1e-4,
                    help="max |mIoU| drift allowed against phase2's upstream cells; the floor is "
                         "phase2's own CSV write precision, not machine epsilon")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the upstream reproduction check (it is the only guard against a "
                         "mis-mirrored seeding contract -- use only when phase2 raw is stale)")
    args = ap.parse_args()

    split = args.split or ("checkerboard" if args.dataset == "indian_pines" else "blocks")
    if split == "checkerboard" and args.dataset != "indian_pines":
        # Fail closed rather than quietly running P2.prep against a foreign cube: its wavelength
        # axis is AVIRIS_WL_NM by construction, so a Pavia cube would be scored against Indian
        # Pines centres and the wavelength-PE arm -- the arm under test -- would read an axis that
        # has nothing to do with its data.
        raise SystemExit(f"--split checkerboard is Indian-Pines-only (it hardcodes phase2's "
                         f"AVIRIS axis); use --split blocks for {args.dataset}")

    hw.setup(deterministic=True, prefer=args.device)
    print("HW:", hw.info())

    if split == "checkerboard":
        cube, gt = P2.load_data()
        wl, num_classes, axis_status = P2.AVIRIS_WL_NM, P2.NUM_CLASSES, "phase2 AVIRIS_WL_NM"
        uniq_off = sorted({int(s) % P2.SPLIT_BLOCK for s in args.seeds})
        class_set, _present = P2.common_class_set(gt, P2.SPLIT_BLOCK, uniq_off)
    else:
        cube, gt, wl, num_classes, axis_status = P6.load_dataset(args.dataset)
        # class_set stays None on this path, matching phase6: it scores every class the cube
        # declares. common_class_set is a checkerboard-specific repair for offsets that do not all
        # see the same classes, and applying it to a different split would restrict the macro
        # average on a rule the split does not need.
        class_set = None
    print(f"dataset={args.dataset} split={split} cube {cube.shape} | classes={num_classes} | "
          f"axis={axis_status} | groups={args.groups} | seeds={args.seeds} | "
          f"epochs={args.epochs} | arms={list(ARMS)}")

    t0 = time.time()
    rows = []
    for i, s in enumerate(args.seeds):
        ts = time.time()
        prepared = (_prep_checkerboard(s, cube, gt, args.groups) if split == "checkerboard"
                    else _prep_blocks(s, cube, gt, wl, args.groups))
        wls = (float(np.max(prepared[9])) if args.wl_scale == "auto"
               else float(args.wl_scale) if args.wl_scale else None)
        _c, r = run_seed_2x2(s, prepared, args.max_missing, args.trials, args.epochs,
                             class_set, num_classes, wl_scale=wls,
                             arms=(SAMPLING_ARMS if args.sampling_baselines else ARMS))
        rows += r
        print(f"  seed {s} done ({time.time() - ts:.0f}s, {i + 1}/{len(args.seeds)})", flush=True)

    P = lambda n: Path(__file__).resolve().parents[1] / "paper" / n  # noqa: E731
    # The default run keeps its bare filename so the committed artefact and everything citing it
    # stay put; every other combination names itself. Without this a Pavia run would overwrite the
    # Indian Pines deliverable in place and nothing in the file would say which scene it holds.
    # stem: the sampling-baseline comparison is a DIFFERENT experiment (fixed architecture, varied
    # channel sampling) and must never overwrite the PE×dropout 2×2 deliverable.
    stem = "sampling" if args.sampling_baselines else "pe_dropout"
    variant = "" if (args.dataset == "indian_pines" and split == "checkerboard"
                     and not args.wl_scale and not args.sampling_baselines) \
        else f"_{args.dataset}_{split}"
    if args.wl_scale:
        variant += f"_wls{args.wl_scale}"
    raw_out = P(f"results_phase2X_{stem}{variant}_raw{args.out_tag}.csv")
    with raw_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "seed", "missing_groups", "miou"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # Verification is only meaningful against the run phase2 actually shipped. On any other
    # dataset or split there are no upstream b6/proposed cells to reproduce, so the check is
    # skipped by CONSTRUCTION -- recorded as "n/a", never as "ok", so the sidecar cannot be read
    # as a passed check that never ran.
    # The sampling-baseline comparison has NO phase2 upstream cells, so verification is n/a there
    # (as it is for any non-Indian/non-checkerboard run).
    verifiable = (args.dataset == "indian_pines" and split == "checkerboard"
                  and not args.sampling_baselines)
    problems = ([] if (args.no_verify or not verifiable)
                else verify_against_phase2(rows, P("results_phase2_raw.csv"), args.verify_tol))
    verify_state = ("n/a (no upstream cells for this dataset/split)" if not verifiable
                    else "skipped" if args.no_verify else (problems or "ok"))

    if args.sampling_baselines:
        # Channel-sampling comparison on a fixed architecture. Positive = the first-named arm is
        # MORE robust. hcs_vs_drop / dcs_vs_drop directly answer "is our band-group dropout ≈ the
        # channel-sampling baselines on redundant HSI"; a near-zero, non-significant contrast means
        # our dropout is a legitimate instance of the strong family (ChannelViT's own point).
        contrasts = {
            "hcs_minus_drop": ("hcs", "drop"),
            "dcs_minus_drop": ("dcs", "drop"),
            "dcs_minus_hcs":  ("dcs", "hcs"),
        }
    else:
        contrasts = {
            "PE_main_effect_nodrop":  ("wavePE_nodrop", "learnedPE_nodrop"),
            "PE_effect_with_drop":    ("wavePE_drop", "learnedPE_drop"),
            "dropout_main_effect":    ("learnedPE_drop", "learnedPE_nodrop"),
            "dropout_effect_wavePE":  ("wavePE_drop", "wavePE_nodrop"),
            "both_vs_neither":        ("wavePE_drop", "learnedPE_nodrop"),
        }
    summary = {n: paired(rows, a, b, args.max_missing) for n, (a, b) in contrasts.items()}

    sum_out = P(f"results_phase2X_{stem}{variant}{args.out_tag}.csv")
    with sum_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "missing_groups", "delta_miou", "ci_lo", "ci_hi"])
        for n, vals in summary.items():
            for m, d, lo, hi in vals:
                w.writerow([n, m, f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    print(f"\n  2x2 paired contrasts, n={len(args.seeds)} seeds "
          f"(* = 95% interval excludes zero)\n")
    print(f"  {'contrast':24s} " + " ".join(f"m={m}".rjust(8)
                                            for m in range(args.max_missing + 1)))
    for n, vals in summary.items():
        cells = [f"{d:+7.2f}" + ("*" if (lo > 0 or hi < 0) else " ") for _m, d, lo, hi in vals]
        print(f"  {n:24s} " + " ".join(c.rjust(8) for c in cells))

    arms_used = SAMPLING_ARMS if args.sampling_baselines else ARMS
    inter = {}
    if args.sampling_baselines:
        # Absolute per-arm mIoU by level, so "do the sampling strategies tie?" is readable directly
        # (the contrasts above are their paired differences). Averaged per level over seeds.
        import collections as _c
        by = _c.defaultdict(lambda: _c.defaultdict(list))
        for r in rows:
            by[r["kind"]][int(r["missing_groups"])].append(float(r["miou"]))
        print("\n  absolute mIoU by arm (mean over seeds):")
        print(f"  {'arm':6s} " + " ".join(f"m={m}".rjust(7) for m in range(args.max_missing + 1)))
        for arm in arms_used:
            cells = [f"{np.mean(by[arm][m]):6.2f}" for m in range(args.max_missing + 1)]
            print(f"  {arm:6s} " + " ".join(c.rjust(7) for c in cells))
        print("\n  read: hcs_minus_drop / dcs_minus_drop near zero & non-significant ⇒ our "
              "band-group dropout is a legitimate instance of the strong channel-sampling family "
              "(ChannelViT's own claim); a positive, significant contrast ⇒ redundant HSI bands "
              "change the natural-image conclusion (publishable either way).")
    else:
        # Interaction at EVERY level (it peaks mid-range, near zero at both ends).
        for m in range(args.max_missing + 1):
            pe_no = summary["PE_main_effect_nodrop"][m][1]
            pe_dr = summary["PE_effect_with_drop"][m][1]
            inter[f"m={m}"] = {"PE_without_dropout": round(pe_no, 3),
                               "PE_with_dropout": round(pe_dr, 3),
                               "interaction": round(pe_dr - pe_no, 3)}
        print("\n  interaction (PE effect WITH dropout - PE effect WITHOUT), per level:")
        for k, v in inter.items():
            print(f"    {k}: {v['PE_without_dropout']:+6.2f} -> {v['PE_with_dropout']:+6.2f} "
                  f"(interaction {v['interaction']:+6.2f})")
        peak = max(inter.items(), key=lambda kv: abs(kv[1]["interaction"]))
        print(f"    peak interaction at {peak[0]}: {peak[1]['interaction']:+.2f} mIoU")

        # Additivity: the two main effects must sum to the corner-to-corner contrast if additive.
        print("\n  additivity check (dropout_main + PE_with_drop vs both_vs_neither):")
        for m in range(args.max_missing + 1):
            add = summary["dropout_main_effect"][m][1] + summary["PE_effect_with_drop"][m][1]
            got = summary["both_vs_neither"][m][1]
            print(f"    m={m}: {add:+7.2f} vs {got:+7.2f}  (residual {add - got:+.3f})")

    stamp(sum_out, args, extra={"arms": {k: list(v) for k, v in arms_used.items()},
                                "interaction": inter,
                                "dataset": args.dataset, "split": split,
                                "wavelength_axis": axis_status, "num_classes": int(num_classes),
                                "wl_scale": args.wl_scale or "model default (2500)",
                                "upstream_verification": verify_state,
                                "n_raw_rows": len(rows),
                                "runtime_s": round(time.time() - t0, 1)})
    stamp(raw_out, args, extra={"n_raw_rows": len(rows)})

    if problems:
        print("\n  UPSTREAM REPRODUCTION FAILED -- the 2x2 does not describe phase2's models:")
        for p in problems[:12]:
            print("   ", p)
        # Non-zero exit: the artefacts are written (they are still evidence of what ran) but no
        # caller should treat a 2x2 that fails to reproduce its own upstream cells as a result.
        return 1
    # "no problems" and "verified" are different states: --no-verify also yields an empty list.
    # Printing the stronger of the two would put an unearned reproduction claim in the log that
    # a later reader has no way to distinguish from a real one.
    status = (f"upstream verification {verify_state}" if isinstance(verify_state, str)
              else "upstream cells reproduced")
    print(f"\n  wrote {sum_out.name} + {raw_out.name} ({time.time() - t0:.0f}s, {status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
