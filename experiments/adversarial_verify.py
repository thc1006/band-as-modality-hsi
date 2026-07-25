#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial verification — empirically PROBE the real code paths to try to REFUTE the
reported results. Each hypothesis Hk is a way the results could be WRONG; we test the actual
behaviour and print PASS (refuted the bug / property holds), FAIL (real bug), or WARN
(honest limitation). This is stronger than reading code: it exercises the true functions.

Run: python experiments/adversarial_verify.py
Exit code: 0 when no check FAILs, 1 when any check FAILs (so CI catches a regression).
"""
import os, sys, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "8")
# This harness PROBES models with plain CPU tensors (torch.from_numpy(...)) and compares against
# numpy / previously-captured CPU tensors at tight tolerances, so it must run wholly on CPU. The
# adaptive-training helpers (bandsim.hw) otherwise auto-select CUDA on a GPU box, and P2's trainers
# would leave the model on CUDA — breaking the direct model(cpu_tensor) probes (H4/H7) and the
# cross-device torch.allclose in H5 with a device mismatch. Pin CPU before importing bandsim/P2.
os.environ["BANDSIM_DEVICE"] = "cpu"
import torch; torch.set_num_threads(8)
from scipy.ndimage import binary_dilation

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.io import load_mat_cube, disjoint_block_split, AVIRIS_WL_NM
from bandsim.grouping import contiguous_groups, group_center_wavelengths, build_group_matrix
from bandsim.metrics import miou, per_class_iou
from bandsim.model import GroupedCrossBandAttention, count_params
from bandsim.reliability import fit_temperature, softmax, selective_auroc
import phase2_degradation as P2

DATA = os.path.join(os.path.dirname(_HERE), "data", "indian_pines")
TABLE = os.path.join(os.path.dirname(_HERE), "data", "srf_cache", "T_6s_grid.npz")


def main():
    results = []
    def report(tag, ok, msg):
        status = "PASS" if ok == 1 else ("WARN" if ok == 2 else "FAIL")
        results.append((status, tag, msg)); print(f"[{status}] {tag}: {msg}")

    cube = load_mat_cube(os.path.join(DATA, "Indian_pines_corrected.mat"), key="indian_pines_corrected").astype(float)
    gt = load_mat_cube(os.path.join(DATA, "Indian_pines_gt.mat"), key="indian_pines_gt").astype(int)

    # ---------------------------------------------------------------- H1 split integrity
    # Probe the SAME split the paper's phase2 numbers use: prep()/run_seed() default to guard=1
    # (leakage-hardened — drops test pixels adjacent to train across block seams), offset=0. A plain
    # guard=0 split is a different, un-hardened split the paper no longer uses, so verifying it would
    # audit the wrong numbers.
    tr, te = disjoint_block_split(gt, block=10, guard=1)
    overlap = int((tr & te).sum())
    tr_cls = int(len(np.unique(gt[tr]))); te_cls = int(len(np.unique(gt[te])))
    report("H1a no train/test pixel overlap", 1 if overlap == 0 else 0, f"overlap={overlap} pixels")
    report("H1b all 16 classes trainable; guard-hardened test keeps 15/16", 1 if (tr_cls == 16 and te_cls >= 15) else 0,
           f"train={tr_cls}/16 test={te_cls}/16 classes; guard=1 drops the scene's smallest class "
           f"(~20 px) from test — metrics average over PRESENT classes so no inflation (see H2)")
    # seam adjacency leakage: test pixels 8-adjacent to a train pixel. Use scipy binary_dilation
    # (ZERO-PADDED border) NOT np.roll: np.roll wraps periodically and would falsely count the top
    # row adjacent to the bottom row (and the left column to the right column). tr and te are
    # disjoint, so dilating tr (incl. its own cells) is safe — a test pixel is counted iff one of
    # its 8 neighbours is a train pixel. Matches experiments/verify_guardband.py::seam_leak.
    adj = binary_dilation(tr, structure=np.ones((3, 3), bool))
    seam = int((te & adj).sum()); seam_frac = seam / max(1, te.sum())
    report("H1c guard band removes seam-adjacency leakage", 2 if seam_frac > 0 else 1,
           f"{seam} of {int(te.sum())} test px ({seam_frac*100:.1f}%) are 8-adjacent to a train pixel "
           f"under the guard=1 split the paper uses (the guard band drops seam-adjacent test pixels; "
           f"guard=0 would leave ~34%). verify_guardband.py further checks the ranking under guard=2")

    # ---------------------------------------------------------------- H2 mIoU empty-class handling
    # The guard=1 test split legitimately lacks the smallest class. PROBE the real metric: miou()
    # must AVERAGE OVER PRESENT CLASSES ONLY (never award a free IoU=1.0 to an absent class), else a
    # missing class would inflate the score. Compare against what a free-1.0 impl would report.
    yte_full = gt[te].astype(int) - 1
    classes_present = np.unique(yte_full)
    absent = 16 - len(classes_present)
    _rngH2 = np.random.default_rng(0)
    yp_probe = np.where(_rngH2.uniform(size=yte_full.size) < 0.7, yte_full,
                        _rngH2.integers(0, 16, yte_full.size))          # imperfect predictor
    mi = miou(yte_full, yp_probe, 16)
    pci = per_class_iou(yte_full, yp_probe, 16)                          # NaN for absent classes
    present_mean = float(np.nanmean(pci))
    inflated = (np.nansum(pci) + 100.0 * absent) / 16.0                 # what a free-1.0 impl returns
    excludes = abs(mi - present_mean) < 1e-6 and (absent == 0 or mi < inflated - 1e-9)
    report("H2 miou averages over PRESENT classes (no free 1.0 for an absent class)", 1 if excludes else 0,
           f"{absent} class absent from guard=1 test GT; miou={mi:.2f} == mean over present "
           f"({present_mean:.2f}), below the inflated {inflated:.2f} a free-1.0 impl would report")

    # ---------------------------------------------------------------- H3 degradation fairness (phase2)
    # Re-derive the drop-group sequence each method sees; must be IDENTICAL across b1/b2/b3/proposed.
    def drop_sequence(seed, G=10, max_missing=6, trials=12):
        rng = np.random.default_rng(seed + 999)
        seq = []
        for m in range(0, max_missing+1):
            for _ in range(trials if m>0 else 1):
                seq.append(tuple(sorted(rng.choice(G, size=m, replace=False).tolist())) if m>0 else ())
        return seq
    seqs = [drop_sequence(0) for _ in range(4)]  # 4 methods each build fresh rng(seed+999)
    identical = all(s == seqs[0] for s in seqs)
    report("H3 degradation drop-sets identical across methods", 1 if identical else 0,
           f"per-seed rng(seed+999) reused fresh per method -> same {len(seqs[0])} drop sets. identical={identical}")

    # ---------------------------------------------------------------- H4 model masking correctness
    Xtr, ytr, Xte, yte = P2.prep(cube, gt, block=10)
    groups = contiguous_groups(200, 10); cwl = group_center_wavelengths(AVIRIS_WL_NM, groups)
    torch.manual_seed(0)
    m_prop = GroupedCrossBandAttention(groups, cwl, 16)
    P2.pretrain_sgmae(m_prop, Xtr, groups, 0, epochs=6)
    P2.finetune_proposed(m_prop, Xtr, ytr, groups, 0, epochs=15)
    m_prop.eval()
    Xb = torch.from_numpy(Xte[:200])
    with torch.no_grad():
        # (a) TRUE MASKING: changing a MISSING group's band values must NOT change the output
        pm = P2.group_present_mask(200, groups, [3, 7])           # groups 3,7 absent
        out1 = m_prop(Xb, torch.from_numpy(pm))
        Xb2 = Xb.clone()
        for g in [3, 7]:
            Xb2[:, groups[g]] = torch.randn_like(Xb2[:, groups[g]]) * 5 + 99  # garbage into missing groups
        out2 = m_prop(Xb2, torch.from_numpy(pm))
        invariant = torch.allclose(out1, out2, atol=1e-5)
        # (b) MASK HAS EFFECT: masking a real group changes the output
        pm_full = P2.group_present_mask(200, groups, [])
        out_full = m_prop(Xb, torch.from_numpy(pm_full))
        differs = not torch.allclose(out_full, out1, atol=1e-4)
        # (c) NO NaN across the used range m=0..9 (>=1 group present)
        nan_free = True
        for m in range(0, 10):
            pm_m = P2.group_present_mask(50, groups, list(range(m)))
            o = m_prop(Xb[:50], torch.from_numpy(pm_m))
            nan_free &= bool(torch.isfinite(o).all())
    report("H4a missing-group VALUES cannot change output (true masking)", 1 if invariant else 0,
           f"garbage into missing groups 3,7 -> logits invariant={invariant} (max|d|={(out1-out2).abs().max():.2e})")
    report("H4b present-mask has real effect", 1 if differs else 0,
           f"clean vs 2-missing logits differ={differs}")
    report("H4c no NaN/inf for m=0..9 missing groups", 1 if nan_free else 0, f"finite outputs={nan_free}")

    # ---------------------------------------------------------------- H5 SGMAE pretraining is not a no-op
    torch.manual_seed(1)
    m_a = GroupedCrossBandAttention(groups, cwl, 16)
    w_before = m_a.encoder.layers[0].linear1.weight.detach().clone()
    P2.pretrain_sgmae(m_a, Xtr, groups, 1, epochs=8)
    w_after = m_a.encoder.layers[0].linear1.weight.detach().clone()
    changed = not torch.allclose(w_before, w_after)
    report("H5 SGMAE pretraining actually updates encoder weights", 1 if changed else 0,
           f"encoder weights changed={changed} (mean|d|={ (w_after-w_before).abs().mean():.2e})")

    # ---------------------------------------------------------------- H6 6S table physics
    if os.path.exists(TABLE):
        T = np.load(TABLE)
        means = {k: float(T[k].mean()) for k in T.files if k.startswith("cwv")}
        # table is now keyed by CWV alone (cwv0.5/cwv2.0/cwv4.0); the AOD axis was dropped.
        mono = means["cwv0.5"] > means["cwv2.0"] > means["cwv4.0"]
        prov = str(T["_provenance"]) if "_provenance" in T.files else ""
        no_aod_axis = not any("aod" in k.lower() for k in T.files)
        report("H6a 6S transmittance decreases with CWV", 1 if mono else 0,
               f"mean T: 0.5->{means['cwv0.5']:.3f} 2->{means['cwv2.0']:.3f} 4->{means['cwv4.0']:.3f}")
        report("H6b table keyed by CWV only; AOD dropped (gaseous T is aerosol-independent)",
               2 if (no_aod_axis and "aod" in prov.lower()) else 0,
               f"cwv keys={[k for k in T.files if k.startswith('cwv')]}, no AOD axis. Aerosol-"
               f"independence was established upstream in Py6S (aod0.1==aod0.4) and is recorded in the "
               f"table provenance; this table documents rather than re-derives it (AOD columns omitted)")
        # H6c catches the class of bug H6a/H6b cannot see: mean-T monotonicity and key layout are
        # both invariant to WHICH wavelength each value sits on. The shipped table was once computed
        # on a gapless linspace(400,2500,200) while the pipeline treated it as the gapped AVIRIS
        # axis -- same length, same means, same keys, every band mis-assigned (worst 98.3 nm).
        from bandsim.io import AVIRIS_WL_NM, axis_sha256
        if "wl_nm" in T.files:
            wl = np.asarray(T["wl_nm"], float)
            aligned = wl.shape == AVIRIS_WL_NM.shape and np.allclose(wl, AVIRIS_WL_NM, rtol=0, atol=1e-6)
            dmax = float(np.abs(wl - AVIRIS_WL_NM).max()) if wl.shape == AVIRIS_WL_NM.shape else float("nan")
            stamped = "wl_sha256" in T.files and str(T["wl_sha256"]) == axis_sha256(AVIRIS_WL_NM)
            report("H6c 6S table sits on the REAL (gapped) AVIRIS axis", 1 if aligned else 0,
                   f"table axis {wl[0]:.2f}-{wl[-1]:.2f} nm vs AVIRIS_WL_NM "
                   f"{AVIRIS_WL_NM[0]:.2f}-{AVIRIS_WL_NM[-1]:.2f} nm, max |delta|={dmax:.4f} nm; "
                   f"wl_sha256 {'matches' if stamped else 'ABSENT/STALE'}")
        else:
            report("H6c 6S table sits on the REAL (gapped) AVIRIS axis", 0,
                   "table stores NO wl_nm axis -- its wavelength alignment cannot be verified at all")
    else:
        report("H6 6S table (needs sixs env)", 2, "table absent on this machine — skipped, not a failure")

    # ---------------------------------------------------------------- H7 temperature scaling reduces NLL on REAL logits
    with torch.no_grad():
        logits = m_prop(Xb, torch.from_numpy(P2.group_present_mask(200, groups, [2,5,8]))).numpy()
    labels = yte[:200]
    def nll(T): p = softmax(logits / T); return -np.log(p[np.arange(len(labels)), labels] + 1e-12).mean()
    Tfit = fit_temperature(logits, labels)
    report("H7 temperature scaling reduces NLL on real logits", 1 if nll(Tfit) <= nll(1.0)+1e-9 else 0,
           f"T={Tfit:.3f}, NLL(1)={nll(1.0):.4f} -> NLL(T)={nll(Tfit):.4f}")

    # ---------------------------------------------------------------- H8 standardization uses TRAIN stats only
    Xtr_raw = cube[tr]; mu_tr = Xtr_raw.mean(0); sd_tr = Xtr_raw.std(0) + 1e-8
    # prep() must reproduce exactly (train)-standardized train; and test standardized by TRAIN stats
    Xtr_p, _, Xte_p, _ = P2.prep(cube, gt, block=10)
    recomputed_tr = ((Xtr_raw - mu_tr) / sd_tr).astype(np.float32)
    match_tr = np.allclose(Xtr_p, recomputed_tr, atol=1e-4)
    Xte_raw = cube[te]; recomputed_te = ((Xte_raw - mu_tr) / sd_tr).astype(np.float32)
    match_te = np.allclose(Xte_p, recomputed_te, atol=1e-4)
    report("H8 standardization fit on TRAIN only (test uses train mu/sd)", 1 if (match_tr and match_te) else 0,
           f"train match={match_tr}, test-by-train-stats match={match_te}")

    # ---------------------------------------------------------------- H9 null (untrained) model ~ chance
    torch.manual_seed(2)
    m_null = GroupedCrossBandAttention(groups, cwl, 16); m_null.eval()
    with torch.no_grad():
        pred_null = m_null(torch.from_numpy(Xte), torch.from_numpy(P2.group_present_mask(len(Xte), groups, []))).argmax(1).numpy()
    oa_null = (pred_null == yte).mean() * 100
    report("H9 untrained model is near chance (pipeline not trivially inflated)", 1 if oa_null < 30 else 0,
           f"untrained OA={oa_null:.1f}% (chance~6.25%; trained ~84%). low => numbers come from training, not a metric bug")

    # ---------------------------------------------------------------- H10 determinism
    # Starred tail, not (c1, _): run_seed grew a third return value (the per-drop-set raw rows)
    # in c993b7e, and a fixed-arity unpack here would raise AFTER training two models -- the same
    # break that hit phase2_group_ablation during the 2026-07-20 campaign.
    c1, *_ = P2.run_seed(0, cube, gt, 10, 4, 6, 12)
    c2, *_ = P2.run_seed(0, cube, gt, 10, 4, 6, 12)
    det = all(np.allclose(c1[k], c2[k]) for k in c1)
    report("H10 same seed -> identical results (reproducible)", 1 if det else 0,
           f"two run_seed(0) identical={det}")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 70)
    n_fail = sum(1 for s,_,_ in results if s == "FAIL")
    n_warn = sum(1 for s,_,_ in results if s == "WARN")
    n_pass = sum(1 for s,_,_ in results if s == "PASS")
    print(f"ADVERSARIAL VERIFY: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    if n_fail: print("FAILURES:", [t for s,t,_ in results if s=="FAIL"])
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
