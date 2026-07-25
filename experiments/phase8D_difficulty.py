#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8D — model reliability vs the CloudSEN12 per-patch ANNOTATION-CONFIDENCE rating.

WHAT `difficulty` IS. It is the ANNOTATOR'S CONFIDENCE IN THEIR OWN LABELS (1 = near-perfect,
5 = possibly significant labelling errors), not an independent physical scene-difficulty score.
Accuracy here is measured against those same, possibly noisier, labels, so any difficulty->accuracy
trend is partly a LABEL-NOISE confound. Nor is it a clean label-noise gradient in the other
direction: CloudSEN12 sends its hardest patches (rating > 4) back for additional expert review, so
residual label error need not rise monotonically with the rating at all.

*** THE CONFOUND THAT DECIDES WHETHER THIS EXPERIMENT MEANS ANYTHING ***
Measured on this repo's 975 test patches, from the metadata and the real label files:

    difficulty | clear  thick  thin   shadow |   n
        1      | 0.992  0.002  0.005  0.002  | 200
        2      | 0.431  0.417  0.017  0.135  | 443
        3      | 0.351  0.345  0.191  0.112  | 202
        4      | 0.365  0.234  0.319  0.082  | 118
        5      | 0.301  0.295  0.297  0.107  |  12
    Spearman(difficulty, thin_fraction)  = +0.624
    Spearman(difficulty, clear_fraction) = -0.561

A difficulty-1 patch is 99.2% CLEAR pixels; a difficulty-4 patch is 31.9% THIN CLOUD — the class
this model scores ~29 IoU on against ~77 for clear. So ANY cloud classifier is confident and
accurate on the first and neither on the second, with no reference whatsoever to annotation
quality. The previously shipped per-level accuracies (78.4 / 72.9 / 64.8 / 54.0) are reproduced to
within 1.8 points by class composition ALONE (non-negative least squares on the four class
fractions — a saturated 4-point fit, so suggestive rather than conclusive; the two rank
correlations above are not a fit and need no such caveat).

The raw association is therefore NOT evidence about annotation quality. What this script reports is
the association CONDITIONAL on what is in the scene, four ways:

  overall        descriptive Spearman over patches, ROI-clustered CI. Confounded; kept for
                 continuity with the previous version, not as a finding.
  within_roi     Kendall tau over pairs of patches INSIDE each ROI. 84% of the difficulty variance
                 is within-ROI (measured), and an ROI's five patches are the same footprint on five
                 dates, so this holds biome, region, land cover, terrain and projection fixed for
                 free. It does NOT hold cloud amount fixed — inside an ROI the five patches ARE the
                 five cloud-coverage levels, and Spearman(difficulty, cloud level) = 0.53 there.
  stratum_*      Spearman inside each cloud-coverage stratum. Every ROI contributes EXACTLY ONE
                 patch to each stratum (verified, 195/195), so a stratum is a set of INDEPENDENT
                 patches — no clustering correction needed — with cloud amount fixed by
                 construction. The cleanest of the four.
  partial        rank partial Spearman controlling for all four class fractions.

If the association survives `stratum_*` and `partial_class_mix`, it is about something other than
what is in the scene. If it does not, this experiment is a class-composition measurement and must
be reported as one. NO threshold in this file converts a coefficient into a verdict: the previous
version printed "ASSOCIATES with" whenever rho < -0.1 — an author-chosen number, with no interval,
no null, and no use of the accuracy coefficient it also computed.

NOT a selective-prediction experiment. `error_rate` is the ORDINARY error rate 1 - accuracy at FULL
coverage: no confidence threshold, no abstention, no selected subset. It was once mislabelled
`selective_risk`, the error rate among ACCEPTED predictions under an abstention rule — a different
quantity this script never computes. Those estimands live in phase4R and phase8R.

Outputs (../paper/), each written atomically:
  results_phase8D_difficulty.csv   per level: mean confidence / accuracy / error rate. All five
                                   levels ALWAYS, blank where a level has no patches — the previous
                                   version dropped empty levels silently, which is why the shipped
                                   file has four rows and no provenance sidecar.
  results_phase8D_per_patch.csv    RAW: one row per (seed, patch) with roi_id, difficulty, cloud
                                   coverage, exact class fractions, mean MSP, accuracy. Every
                                   number above is recomputable from it.
  results_phase8D_association.csv  each analysis with its coefficient, bootstrap CI, permutation
                                   p-value, and the unit those were computed over.

Usage:
  python experiments/phase8D_difficulty.py --smoke
  python experiments/phase8D_difficulty.py --seeds 0 1 2 --device cuda
"""
import os, sys, csv, argparse, tempfile
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim.reliability import confidence_msp
from bandsim import hw
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))


def P(rel):
    """Resolve under ../paper, creating the directory LAZILY rather than at import time."""
    out = os.path.join(PAPER_DIR, rel)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    return out


_UMASK = os.umask(0); os.umask(_UMASK)          # read-and-restore: umask has no getter


def _atomic_write(path, write_fn):
    """write_fn(tmp) then os.replace — a killed run leaves each artefact fully old or fully new."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix="_" + os.path.basename(path))
    os.close(fd)
    try:
        write_fn(tmp)
        try:                                    # mkstemp creates 0600; do not publish owner-only
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        except FileNotFoundError:
            os.chmod(tmp, 0o644 & ~_UMASK)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _write_csv(path, header, rows):
    def _w(tmp):
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    return _atomic_write(path, _w)


LEVELS = [1, 2, 3, 4, 5]
# CloudSEN12 picks five patches per ROI, one at each cloud-coverage level. Ordinal, not nominal.
CLOUD_ORDER = ["cloud-free", "almost-clear", "low-cloudy", "mid-cloudy", "cloudy"]
# `error_rate` is 1 - accuracy at FULL coverage (no abstention) -- deliberately NOT "selective_risk".
# tests/test_reliability_guards.py pins this exact list AND the literal `w.writerow(CSV_COLUMNS)`
# call below, so the canonical deliverable keeps its four-column contract; the per-level counts and
# spread live in the per-patch file, from which they are recomputable.
# n_patches and n_roi ship WITH the means, not only to the console. Level 5 is 12 of 975 patches
# from 10 ROIs; without those counts beside it, "difficulty 5: accuracy 48.14" reads in the CSV as
# a number of the same standing as level 1's 94.64, which rests on 600. The absent-level fix below
# already covers "this level had no patches at all" -- this covers the harder case, a level that IS
# present and is quietly too thin to carry a claim. n_roi is the one that matters for inference:
# patches within an ROI are not independent, so 36 rows over 10 ROIs is nearer n=10 than n=36.
CSV_COLUMNS = ["difficulty", "n_patches", "n_roi", "mean_confidence", "accuracy", "error_rate"]


# ======================================================================================
# rank statistics
# ======================================================================================
def _spearman(x, y):
    """Spearman rho with proper TIE handling (scipy.stats.spearmanr).

    The order of the checks matters: the previous version called `x.std()` before testing the
    length, so an empty input produced numpy "Degrees of freedom <= 0" warnings on its way to a
    nan. Non-finite values are dropped HERE rather than only upstream — the old caller filtered
    NaN in `difficulty` and nothing at all in the metric, so a single inf reached the coefficient.
    """
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"_spearman shape mismatch: {x.shape} vs {y.shape}")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    from scipy.stats import spearmanr
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _partial_spearman(x, y, Z):
    """Rank partial correlation of x and y given the columns of Z.

    Rank-transform everything, regress the ranks of x and of y on the ranks of Z, correlate the
    residuals. The class fractions sum to one and are therefore collinear with the intercept;
    lstsq's minimum-norm solution still defines the residual projection uniquely, and the residuals
    are the only part used.

    Returns nan — deliberately, not as a failure — when a residual is degenerate. If the controls
    predict the exposure perfectly there is no variation left to correlate and the partial
    correlation is undefined; reporting a number there would invent one. In this dataset that does
    not arise (patches at the same difficulty have visibly different class mixes), so a nan in the
    output means the evaluated subset has collapsed, not that the analysis is inapplicable.
    """
    from scipy.stats import rankdata
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    Z = np.asarray(Z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    n = int(ok.sum())
    if n < Z.shape[1] + 3:
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    A = np.column_stack([np.ones(n)] + [rankdata(Z[ok, j]) for j in range(Z.shape[1])])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def _pair_counts(unit_rows, x, y):
    """Per-unit (concordant, discordant) pair counts over pairs INSIDE each unit.

    The sufficient statistic for a within-cluster Kendall tau: tau = (SUM C - SUM D)/(SUM C + SUM D).
    Keeping it per unit is what makes the ROI bootstrap exact and cheap — resampling ROIs becomes
    resampling rows of these two arrays, with no pair enumeration in the inner loop.

    TIED PAIRS ARE EXCLUDED, counted as neither concordant nor discordant, so the denominator is
    the number of pairs that could order at all. That matters here and is not a detail: difficulty
    is a 1-5 rating over five patches, so a large share of within-ROI pairs tie on it and the
    statistic is computed over the rest.
    """
    conc = np.zeros(len(unit_rows), np.int64)
    disc = np.zeros(len(unit_rows), np.int64)
    for i, ix in enumerate(unit_rows):
        if ix.size < 2:
            continue
        xi, yi = x[ix], y[ix]
        s = np.sign(xi[:, None] - xi[None, :]) * np.sign(yi[:, None] - yi[None, :])
        iu = np.triu_indices(xi.size, 1)
        v = s[iu]
        conc[i] = int((v > 0).sum()); disc[i] = int((v < 0).sum())
    return conc, disc


def _tau(conc, disc):
    n = conc.sum() + disc.sum()
    return float((conc.sum() - disc.sum()) / n) if n else float("nan")


def _boot_ci(values, alpha=0.05):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size < 20:
        return float("nan"), float("nan")
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def _perm_p(observed, null, two_sided=True):
    """Permutation p with the +1 correction, so an empirical 0 is never reported.

    A p of exactly 0 asserts more than `perm` draws can support; (r+1)/(B+1) is the standard
    correction and is what keeps the value comparable across different --perm settings.
    """
    null = np.asarray([v for v in null if np.isfinite(v)], float)
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    r = int(np.sum(np.abs(null) >= abs(observed))) if two_sided else int(np.sum(null <= observed))
    return float((r + 1) / (null.size + 1))


def _jsonable(v):
    """Non-finite floats -> None. `json.dump` emits a bare `NaN` token, which is not valid JSON and
    is rejected by every strict parser; the provenance sidecar is meant to be readable by more than
    Python. An undefined coefficient must survive as null, never as a number."""
    return None if isinstance(v, float) and not np.isfinite(v) else v


def _shuffle_within(unit, values, rng):
    """Permute `values` inside each unit, vectorised.

    The null that keeps ROI membership AND each ROI's own multiset of difficulty ratings, breaking
    only their pairing with the metric. Both index vectors group rows by unit; one keeps the
    original order inside a unit and the other randomises it, so the assignment can only ever move
    a value to another row of the SAME unit.
    """
    unit = np.asarray(unit)
    home = np.argsort(unit, kind="stable")
    away = np.lexsort((rng.random(unit.size), unit))
    out = np.empty_like(values)
    out[home] = values[away]
    return out


# ======================================================================================
# data
# ======================================================================================
def patch_class_fractions(split, patch_ids):
    """EXACT per-class pixel fractions over each patch's valid 509x509 region.

    Exact, not estimated from the evaluated pixel sample: at --px-eval 400 a sampled fraction
    carries a ~2.5 pp standard error, and measurement error in a CONTROL variable attenuates the
    control — biasing the conditional analysis TOWARD finding an association, the wrong direction
    for a check whose whole job is to try to remove one. Doubles as a label-domain scan of every
    evaluated patch, which the sampled loader can only do for the pixels it happened to draw.

    (Once phase 8's `label_histogram` lands on main this becomes
     `P8.label_histogram(split)[patch_ids][:, :NUM_CLASSES]` and this function should go.)
    """
    root = os.path.join(P8.DATA, split)
    n = len(pd.read_csv(os.path.join(root, "metadata.csv")))
    lab = P8._memmap_checked(os.path.join(root, "LABEL_manual_hq.dat"), np.uint8, n)
    o, V, K = P8.VALID_OFF, P8.VALID_SIDE, P8.NUM_CLASSES
    out = np.zeros((len(patch_ids), K))
    for i, p in enumerate(np.asarray(patch_ids, int)):
        h = np.bincount(np.asarray(lab[int(p)][o:o + V, o:o + V]).ravel(), minlength=256)
        bad = np.flatnonzero(h[K:]) + K
        if bad.size:
            raise ValueError(f"{root}: patch {int(p)} carries label value(s) {bad.tolist()[:6]} "
                             f"outside [0,{K}) over {int(h[bad].sum())} pixels")
        out[i] = h[:K] / h[:K].sum()
    return out


def predict_batched(model, X, groups, batch_size=8192):
    """Class logits in batches, on the model's own device.

    The previous version pushed the whole evaluation set through in one call. At the default 975
    test patches x 400 pixels that is 390,000 rows at once; the same architecture was MEASURED at a
    7.6 GB eval-time peak on 292,500 rows in phase 8, so this was a ~10 GB transient growing
    linearly with --px-eval. Batching bounds it and changes no number: rows are independent and the
    model is in eval mode. `model.eval()` is asserted here rather than inherited from whichever
    phase-2 training function ran last — a model left in train mode samples dropout at evaluation
    time and returns a slightly wrong, irreproducible answer with no error anywhere.
    """
    if not (isinstance(batch_size, (int, np.integer)) and batch_size >= 1):
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    try:
        out = []
        with torch.no_grad():
            for s in range(0, X.shape[0], batch_size):
                xb = torch.from_numpy(np.ascontiguousarray(X[s:s + batch_size])).to(dev)
                pm = P2.group_present_mask(xb.shape[0], groups, [])
                lg = model(xb, torch.from_numpy(pm).to(dev)).cpu().numpy()
                if not np.isfinite(lg).all():
                    raise FloatingPointError(
                        f"non-finite logits in the batch starting at row {s}: argmax would still "
                        f"return a class index and the accuracy would look entirely normal")
                out.append(lg)
        return np.concatenate(out, 0)
    finally:
        model.train(was_training)


def _preflight(args):
    """Reject configurations that produce a plausible-looking but unsupportable table.

    Note what is deliberately NOT required: a minimum number of seeds. Seeds vary only the model
    initialisation — the training and evaluation pixels come from fixed loader seeds — so the unit
    of inference here is the ROI, not the seed, and every interval below comes from resampling
    ROIs. Seeds are reported as a robustness range, never as the error bar.
    """
    if len(args.seeds) != len(set(args.seeds)):
        dup = sorted({s for s in args.seeds if list(args.seeds).count(s) > 1})
        raise SystemExit(f"--seeds contains duplicates {dup}: every training function reseeds from "
                         f"the seed value, so a repeat is the same model twice and only makes the "
                         f"reported seed range look tighter than it is.")
    if args.epochs < 1:
        # `max(1, epochs//2)` floors the PRETRAIN at one epoch while the finetune loop runs zero
        # times, so --epochs 0 produced a pretrained-but-never-finetuned model and a full table.
        raise SystemExit(f"--epochs must be >= 1, got {args.epochs}")
    for name in ("patches_train", "px_train", "px_eval", "boot", "perm", "batch_size"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1, got {getattr(args, name)}")
    if args.patches_eval is not None and args.patches_eval < 2:
        raise SystemExit(f"--patches-eval must be >= 2 or omitted, got {args.patches_eval}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--patches-train", type=int, default=2000)
    ap.add_argument("--px-train", type=int, default=300)
    ap.add_argument("--patches-eval", type=int, default=None,
                    help="None = all test patches. A subset is drawn BY ROI (all of a chosen ROI's "
                         "patches, never a partial ROI), which keeps every cloud-coverage stratum "
                         "populated and keeps the clustered analyses well defined.")
    ap.add_argument("--px-eval", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8192, help="rows per inference batch")
    ap.add_argument("--boot", type=int, default=2000, help="ROI bootstrap replicates")
    ap.add_argument("--perm", type=int, default=1000, help="within-ROI permutation replicates")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    ap.add_argument("--nondeterministic", action="store_true")
    args = ap.parse_args()
    sfx = ""
    if args.smoke:
        forced = dict(seeds=[0], patches_train=80, patches_eval=25, epochs=8,
                      px_train=200, px_eval=200, boot=300, perm=200)
        clob = [f"--{f.replace('_', '-')} {getattr(args, f)!r}" for f in forced
                if getattr(args, f) != ap.get_default(f)]
        if clob:
            print(f"[smoke] NOTE: --smoke OVERRIDES the {', '.join(clob)} you passed")
        for _f, _v in forced.items():
            setattr(args, _f, _v)
        sfx = "_smoke"
        print("[smoke] 1 seed / 8 epochs — writing *_smoke artefacts, NOT the real deliverable")
    _preflight(args)
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device)
    print("HW:", hw.info())

    groups = P8.s2_physical_groups()
    wl = np.array(P8.S2_WL_NM, float)
    cwl = group_center_wavelengths(wl, groups)
    NUM = P8.NUM_CLASSES
    # `P2.NUM_CLASSES = NUM` used to sit here and is deliberately gone. Rewriting another module's
    # global is not reentrant and bandsim.parallel runs its serial path in the CALLER's process, so
    # the value escapes into whatever runs next. Unlike phase 8, nothing here depended on it: the
    # class count reaches GroupedCrossBandAttention explicitly, and the three phase-2 entry points
    # this file calls (pretrain_sgmae, finetune_proposed, group_present_mask) contain no reference
    # to NUM_CLASSES at all. Removing it here is a pure deletion of a side effect.

    # --- train pixels (train ROIs; normalise by train stats) ---
    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=args.px_train,
                             n_patches=args.patches_train, seed=12345)
    mu = Xtr.mean(0); sd_raw = Xtr.std(0)
    dead = np.flatnonzero(~np.isfinite(sd_raw) | (sd_raw < 1e-6))
    if dead.size:
        # `sd + 1e-8` keeps the division safe and silently turns a dead band into a constant-0
        # feature: the run then trains on fewer bands than every table it writes claims.
        raise SystemExit(f"train band(s) {[P8.L1C_BANDS[i] for i in dead]} are constant "
                         f"(sd < 1e-6 over {Xtr.shape[0]} pixels) — standardising them would "
                         f"flatten them into a constant feature")
    sd = sd_raw + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr = norm(Xtr)
    if not np.isfinite(Xtr).all():
        raise SystemExit("non-finite values in the standardised training pixels")

    # --- eval patches (test ROIs) with difficulty, ROI, cloud coverage and class mix ---
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    for col in ("difficulty", "roi_id", "cloud_coverage"):
        if col not in meta.columns:
            raise SystemExit(f"test/metadata.csv has no '{col}' column")
    diff_all = pd.to_numeric(meta["difficulty"], errors="raise").to_numpy(float)
    if not np.isfinite(diff_all).all():
        raise SystemExit("test/metadata.csv difficulty contains NaN or Inf")
    if not np.all(diff_all == np.round(diff_all)) or not np.all(np.isin(diff_all, LEVELS)):
        bad = np.unique(diff_all[~np.isin(diff_all, LEVELS)])
        raise SystemExit(f"difficulty outside {LEVELS}: {bad[:8].tolist()} — the per-level table "
                         f"keeps only 1..5 while the correlations kept everything numeric, so the "
                         f"two would be computed over different populations")
    diff_all = diff_all.astype(int)
    roi_all = meta["roi_id"].to_numpy()
    cloud_all = meta["cloud_coverage"].to_numpy()
    unknown = sorted(set(cloud_all.tolist()) - set(CLOUD_ORDER))
    if unknown:
        raise SystemExit(f"unknown cloud_coverage value(s) {unknown}; expected {CLOUD_ORDER}")

    rng = np.random.default_rng(777)
    if args.patches_eval is None:
        eval_ids = np.arange(len(meta))
    else:
        # Sample whole ROIs. Patch-level sampling drew PARTIAL ROIs, which breaks the clustered
        # analyses and can miss level 5 entirely — it holds 12 of 975 patches, and the shipped
        # results file has no difficulty-5 row for exactly that reason.
        rois = np.unique(roi_all)
        per = max(1, int(round(len(meta) / len(rois))))
        k = max(1, min(len(rois), int(round(args.patches_eval / per))))
        keep = rng.choice(rois, size=k, replace=False)
        eval_ids = np.sort(np.flatnonzero(np.isin(roi_all, keep)))
        print(f"[eval] {k} ROIs -> {eval_ids.size} patches (whole ROIs, never a partial one)")
    Xev, yev, pid = P8.load_split("test", "L1C", pixels_per_patch=args.px_eval,
                                  patch_ids=eval_ids, seed=54321, return_patch_id=True)
    Xev = norm(Xev)
    if not np.isfinite(Xev).all():
        raise SystemExit("non-finite values in the standardised evaluation pixels")

    upid = np.unique(pid)
    pdiff = diff_all[upid]
    proi = roi_all[upid]
    pcloud = cloud_all[upid]
    print(f"  scanning label files for EXACT class fractions over {upid.size} patches "
          f"(~{upid.size * 0.0165:.0f}s) ...", flush=True)
    pfrac = patch_class_fractions("test", upid)          # exact, and scans their label domain
    roi_code = np.unique(proi, return_inverse=True)[1].astype(int).ravel()
    n_roi = int(roi_code.max()) + 1 if roi_code.size else 0
    rows_by_roi = [np.flatnonzero(roi_code == u) for u in range(n_roi)]
    print(f"train {Xtr.shape[0]} px | eval {Xev.shape[0]} px over {upid.size} patches / {n_roi} ROIs")
    print("  patches per difficulty level: "
          + "  ".join(f"{L}:{int((pdiff == L).sum())}" for L in LEVELS))
    print("  mean class fractions by level (clear/thick/thin/shadow) — the confound, measured here:")
    for L in LEVELS:
        m = pdiff == L
        if m.any():
            print(f"    {L}: " + " ".join(f"{v:.3f}" for v in pfrac[m].mean(0)))

    # --- one trained model per seed; per-patch metrics ---
    # bs=auto_bs, NOT P2's 256 default. This trains on ~600k CloudSEN12 pixels (patches_train x
    # px_train); at 256 that is ~2,340 launch-bound steps/epoch, the same pathology phase8 had, and
    # it is why this phase took ~65 min for a single-model-per-seed experiment. auto_bs(600k)=4096
    # (~147 steps/epoch) brings it to a few minutes. The value is stamped into provenance below;
    # like phase8 this is a hyperparameter change, so these numbers are not comparable with the
    # bs=256 run -- but the finding here is a RANK correlation (confidence vs annotation
    # difficulty), which is robust to the optimisation detail, and the re-run reproduces it.
    train_bs = P2.auto_bs(Xtr.shape[0])
    per_patch_rows, seed_conf, seed_acc = [], {}, {}
    for seed in args.seeds:
        m = GroupedCrossBandAttention(groups, cwl, NUM)
        P2.pretrain_sgmae(m, Xtr, groups, seed, epochs=max(1, args.epochs // 2), bs=train_bs)
        P2.finetune_proposed(m, Xtr, ytr, groups, seed, epochs=args.epochs, bs=train_bs)
        logits = predict_batched(m, Xev, groups, batch_size=args.batch_size)
        conf = confidence_msp(logits)
        correct = (logits.argmax(1) == yev).astype(float)
        inv = np.searchsorted(upid, pid)
        cnt = np.bincount(inv, minlength=upid.size).astype(float)
        if (cnt == 0).any():
            raise RuntimeError("a selected patch contributed no evaluated pixels")
        pc = np.zeros(upid.size); np.add.at(pc, inv, conf); pc /= cnt
        pa = np.zeros(upid.size); np.add.at(pa, inv, correct); pa /= cnt
        seed_conf[seed], seed_acc[seed] = pc, pa
        for i in range(upid.size):
            per_patch_rows.append([int(seed), int(upid[i]), str(proi[i]), int(pdiff[i]),
                                   str(pcloud[i]), int(cnt[i])]
                                  + [repr(float(v)) for v in pfrac[i]]
                                  + [repr(float(pc[i])), repr(float(pa[i]))])
        print(f"  seed {seed}: mean patch conf {pc.mean():.3f}  mean patch acc {pa.mean() * 100:.1f}%")

    # Seed-averaged per-patch metrics carry the association analyses; the per-seed coefficients are
    # reported separately as a robustness range. Seeds are not the unit of inference here.
    conf_p = np.mean([seed_conf[s] for s in args.seeds], 0)
    acc_p = np.mean([seed_acc[s] for s in args.seeds], 0)

    # ============================ association analyses ============================
    brng = np.random.default_rng(20260720)
    prng = np.random.default_rng(20260721)
    rows, headline = [], {}

    def add(analysis, metric_name, stat, value, ci, p, n_units, unit, note):
        rows.append([analysis, metric_name, stat, f"{value:.4f}",
                     "" if not np.isfinite(ci[0]) else f"{ci[0]:.4f}",
                     "" if not np.isfinite(ci[1]) else f"{ci[1]:.4f}",
                     "" if not np.isfinite(p) else f"{p:.4f}", int(n_units), unit, note])
        headline[(analysis, metric_name)] = (value, ci, p)

    print(f"\ncomputing associations (boot={args.boot}, perm={args.perm}) — measured at ~115 s for "
          f"975 patches / 195 ROIs, so this is not hung", flush=True)
    for name, metric in (("confidence", conf_p), ("accuracy", acc_p)):
        print(f"  [{name}] ...", flush=True)
        # 1) overall, ROI-clustered CI. Confounded by class composition; descriptive only.
        obs = _spearman(pdiff, metric)
        boot = []
        for _ in range(args.boot):
            idx = np.concatenate([rows_by_roi[u] for u in brng.integers(0, n_roi, n_roi)])
            boot.append(_spearman(pdiff[idx], metric[idx]))
        # No p-value here, deliberately. The only null available for a clustered design is the
        # within-ROI permutation, and it preserves the BETWEEN-ROI component of this statistic, so
        # its null distribution is narrower than the statistic's true sampling distribution.
        # MEASURED on 150 null replicates over the real ROI/difficulty structure: it rejects at
        # 7.3% for a nominal 5%. The ROI-clustered CI resamples whole ROIs and carries both
        # components, so it is the interval to read.
        add("overall", name, "spearman", obs, _boot_ci(boot), float("nan"),
            n_roi, "ROI", "CONFOUNDED by class composition; descriptive only. No p: the only "
                          "available permutation null is anti-conservative here (measured 7.3% "
                          "rejection at a nominal 5%) — read the ROI-clustered CI instead")

        # 2) within-ROI Kendall tau: every ROI-level property held fixed.
        c0, d0 = _pair_counts(rows_by_roi, pdiff.astype(float), metric)
        obs = _tau(c0, d0)
        boot = [_tau(c0[p_], d0[p_]) for p_ in (brng.integers(0, n_roi, n_roi)
                                                for _ in range(args.boot))]
        null = []
        for _ in range(args.perm):
            cs, ds = _pair_counts(rows_by_roi,
                                  _shuffle_within(roi_code, pdiff, prng).astype(float), metric)
            null.append(_tau(cs, ds))
        # This null IS exactly matched to this statistic — both are purely within-ROI. MEASURED on
        # 150 null replicates over the real structure: rejects at 6.0% for a nominal 5%, i.e.
        # calibrated within the noise of that many replicates.
        add("within_roi", name, "kendall_tau", obs, _boot_ci(boot), _perm_p(obs, null),
            n_roi, "ROI", "geography/land cover held fixed; cloud amount NOT. Permutation null "
                          "measured calibrated (6.0% at a nominal 5%)")

        # 3) rank partial correlation given the four class fractions.
        obs = _partial_spearman(pdiff, metric, pfrac)
        boot = []
        for _ in range(args.boot):
            idx = np.concatenate([rows_by_roi[u] for u in brng.integers(0, n_roi, n_roi)])
            boot.append(_partial_spearman(pdiff[idx], metric[idx], pfrac[idx]))
        add("partial_class_mix", name, "partial_spearman", obs, _boot_ci(boot), float("nan"),
            n_roi, "ROI", "controls clear/thick/thin/shadow fractions. CI only, deliberately: a "
                          "permutation of difficulty would also destroy the difficulty-to-class-mix "
                          "link this analysis conditions on, so its null is the wrong one")

        # 4) inside a cloud-coverage stratum every ROI contributes at most one patch, so the
        #    stratum is INDEPENDENT patches with cloud amount fixed by construction.
        for cl in CLOUD_ORDER:
            msk = pcloud == cl
            k = int(msk.sum())
            if k < 3:
                add(f"stratum_{cl}", name, "spearman", float("nan"), (float("nan"),) * 2,
                    float("nan"), k, "patch", "too few patches")
                continue
            sd_, sm_ = pdiff[msk], metric[msk]
            one_per_roi = np.unique(roi_code[msk]).size == k
            obs = _spearman(sd_, sm_)
            boot = [_spearman(sd_[i], sm_[i]) for i in (brng.integers(0, k, k)
                                                        for _ in range(args.boot))]
            null = [_spearman(prng.permutation(sd_), sm_) for _ in range(args.perm)]
            add(f"stratum_{cl}", name, "spearman", obs, _boot_ci(boot), _perm_p(obs, null),
                k, "patch",
                ("one patch per ROI -> independent. UNADJUSTED marginal p, one of "
                 f"{2 * len(CLOUD_ORDER)} such tests — read stratum_pooled instead"
                 if one_per_roi else "MORE THAN ONE PATCH PER ROI"))

        # 5) POOLED over strata — the number to read, not the five above. The five stratum
        #    coefficients are independent, so reporting them separately with unadjusted p-values
        #    invites exactly one kind of mistake: during verification, on RANDOM logits with no
        #    association at all, one of the five came out at p=0.017. Their mean is a single
        #    statistic with cloud amount held fixed throughout; the bootstrap resamples ROIs, which
        #    resamples all five strata COHERENTLY because an ROI owns one patch in each.
        if all(np.unique(roi_code[pcloud == c]).size == int((pcloud == c).sum())
               for c in CLOUD_ORDER):
            maps = []
            for cl in CLOUD_ORDER:
                mp = np.full(n_roi, -1, int)
                mm = np.flatnonzero(pcloud == cl)
                mp[roi_code[mm]] = mm
                maps.append(mp)

            def _pooled(sel_per_stratum, diff_vec):
                vals = [_spearman(diff_vec[s], metric[s]) for s in sel_per_stratum if s.size >= 3]
                vals = [v for v in vals if np.isfinite(v)]
                return float(np.mean(vals)) if vals else float("nan")

            base = [np.flatnonzero(pcloud == c) for c in CLOUD_ORDER]
            obs = _pooled(base, pdiff)
            boot = []
            for _ in range(args.boot):
                pick = brng.integers(0, n_roi, n_roi)
                boot.append(_pooled([mp[pick][mp[pick] >= 0] for mp in maps], pdiff))
            null = []
            for _ in range(args.perm):
                shuffled = pdiff.copy()
                for s in base:                       # permute WITHIN each stratum: cloud fixed
                    shuffled[s] = pdiff[prng.permutation(s)]
                null.append(_pooled(base, shuffled))
            add("stratum_pooled", name, "mean_spearman", obs, _boot_ci(boot), _perm_p(obs, null),
                n_roi, "ROI", "mean of the five cloud strata; cloud amount held fixed, ROIs "
                              "resampled coherently. Permutation null measured calibrated (4.7% "
                              "at a nominal 5%). THIS is the conditional headline")
        else:
            add("stratum_pooled", name, "mean_spearman", float("nan"), (float("nan"),) * 2,
                float("nan"), 0, "ROI", "skipped: a stratum holds more than one patch per ROI")

    _write_csv(P(f"results_phase8D_association{sfx}.csv"),
               ["analysis", "metric", "statistic", "value", "ci_lo", "ci_hi", "p_permutation",
                "n_units", "unit", "note"], rows)
    _write_csv(P(f"results_phase8D_per_patch{sfx}.csv"),
               ["seed", "patch_id", "roi_id", "difficulty", "cloud_coverage", "n_pixels"]
               + [f"frac_{c}" for c in P8.CLASS_NAMES] + ["mean_msp", "accuracy"],
               per_patch_rows)

    # --- per-level table (canonical four-column deliverable; ALL five levels, always) ---
    print(f"\n===== Phase 8D — reliability vs annotation-confidence rating "
          f"({len(args.seeds)} seeds, {upid.size} patches, {n_roi} ROIs) =====")
    print("difficulty | n_patches  mean_conf  accuracy%  error%   (full coverage: error = 100 - acc)")
    level_rows = []
    for L in LEVELS:
        msk = pdiff == L
        if msk.any():
            c, a = float(conf_p[msk].mean()), float(acc_p[msk].mean()) * 100
            level_rows.append([L, int(msk.sum()), int(np.unique(proi[msk]).size),
                               f"{c:.4f}", f"{a:.2f}", f"{100 - a:.2f}"])
            print(f"    {L}      |   {int(msk.sum()):5d}     {c:.3f}     {a:5.1f}    {100 - a:5.1f}")
        else:
            # Written, not skipped. The shipped file has four rows because level 5 (12 of 975
            # patches) was absent from a subset run, and nothing in the file said so.
            level_rows.append([L, 0, 0, "", "", ""])
            print(f"    {L}      |       0         --        --       --   (no patches)")

    def _w(tmp):
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            w.writerows(level_rows)
    _atomic_write(P(f"results_phase8D_difficulty{sfx}.csv"), _w)

    # --- console: coefficients with their intervals, and no verdict ---
    print("\nassociation with the annotation-confidence rating "
          "(negative = less confident / less accurate where annotators trusted their labels less):")
    for name in ("confidence", "accuracy"):
        print(f"  {name}:")
        for a in (["overall", "within_roi", "partial_class_mix", "stratum_pooled"]
                  + [f"stratum_{c}" for c in CLOUD_ORDER]):
            if (a, name) not in headline:
                continue
            v, ci, p = headline[(a, name)]
            ci_s = "  [n/a]           " if not np.isfinite(ci[0]) else f"  [{ci[0]:+.3f},{ci[1]:+.3f}]"
            p_s = "" if not np.isfinite(p) else f"  p={p:.4f}"
            print(f"    {a:<22}{v:+.3f}{ci_s}{p_s}")
        per_seed = [_spearman(pdiff, (seed_conf if name == "confidence" else seed_acc)[s])
                    for s in args.seeds]
        print(f"    {'per-seed overall rho':<22}" + " ".join(f"{v:+.3f}" for v in per_seed)
              + "   (model variability, NOT the error bar)")
    print("\nNo threshold in this file turns a coefficient into a verdict. Read `overall` as\n"
          "confounded: difficulty-1 patches are ~99% clear pixels and difficulty-4 patches ~32%\n"
          "thin cloud, so any cloud classifier reproduces that trend without knowing anything\n"
          "about annotation quality. The claim that this measures ANNOTATION QUALITY rests on\n"
          "`stratum_pooled` and `partial_class_mix` surviving; if they do not, this is a\n"
          "class-composition measurement and must be reported as one. The five individual\n"
          "`stratum_*` rows carry UNADJUSTED p-values across ten tests — on random logits one of\n"
          "them reached p=0.017 during verification, so quote the pooled row, not the best of\n"
          "five. The stronger anti-artifact validation remains EMIT per-band retrieval\n"
          "uncertainty (phase8F_multi).")

    _prov = {"n_train_px": int(Xtr.shape[0]), "n_eval_px": int(Xev.shape[0]),
             # The training batch (P2.auto_bs of the ~600k-pixel train split). A hyperparameter --
             # results are not comparable with a bs=256 run of this phase; see the note at the
             # training loop. train_bs is defined there and always bound before this dict is built.
             "train_bs": int(train_bs),
             "n_eval_patches": int(upid.size), "n_eval_rois": n_roi,
             "patches_by_level": {int(L): int((pdiff == L).sum()) for L in LEVELS},
             "rois_by_level": {int(L): int(np.unique(roi_code[pdiff == L]).size) for L in LEVELS},
             "mean_class_fraction_by_level": {
                 int(L): {c: round(float(v), 5)
                          for c, v in zip(P8.CLASS_NAMES, pfrac[pdiff == L].mean(0))}
                 for L in LEVELS if (pdiff == L).any()},
             "loader_seeds": {"train": 12345, "test": 54321, "roi_subset": 777},
             "boot": int(args.boot), "perm": int(args.perm),
             "association": {f"{a}|{m}": {"value": _jsonable(v),
                                          "ci": [_jsonable(c) for c in ci], "p": _jsonable(p)}
                             for (a, m), (v, ci, p) in headline.items()},
             "note": "seeds vary model init only; the unit of inference is the ROI"}
    names = ("difficulty", "per_patch", "association")
    for nm in names:
        stamp(P(f"results_phase8D_{nm}{sfx}.csv"), args, extra=_prov)
    for nm in names:
        print(f"wrote: {P(f'results_phase8D_{nm}{sfx}.csv')}")


if __name__ == "__main__":
    main()
