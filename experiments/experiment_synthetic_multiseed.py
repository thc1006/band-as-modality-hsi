#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 — CONTROLLED SYNTHETIC mechanism check, multi-seed.

WHAT THIS IS. A 12-dimensional correlated-feature world with 4 classes, used to ask ONE question:
when whole features go missing, does (a) training with group masking and (b) reconstructing the
missing features from the observed ones recover accuracy that zero-filling loses? It is a mechanism
illustration. The paper's actual missing-band claims live in phases 2, 3 and 8, on real data.

WHAT THE THIRD CURVE IS — READ THIS BEFORE QUOTING IT. It is the group-dropout MLP evaluated with a
CONDITIONAL-GAUSSIAN imputer fitted to the training covariance. It was labelled "Proposed (SGMAE
imputation + attn)" in the figure, the CSV column, the LaTeX macros and docs/status/STATUS_REPORT.md
("proposed=91"). There is no autoencoder, no mask token, no reconstruction loss and no attention
anywhere in this file; only two MLPs are trained, and the second and third curves come from the SAME
model — visible in the old deliverable, where `band_dropout` and `proposed` are identical at 0
missing bands (94.88 +/- 3.16 in both columns). The curves are now named for what they are.

That matters more than a label, because of how this world is built. Pixels are
`x = class_mean + z @ B_load.T + Gaussian noise`: a linear-Gaussian factor model. The imputer's
assumed model IS the generator's model, so the third curve is close to the CEILING a
covariance-based reconstruction can reach here, not a prediction of what a learned SGMAE does on
real spectra. Read it as "reconstruction from inter-feature redundancy is worth a lot when the
redundancy is real and linear", and let phases 2/3/8 carry the claim about the actual architecture.

MASKS. Every config is evaluated on the SAME missing-feature sets. Each config used to consume from
one shared generator in turn, so the three curves were scored on three DIFFERENT random mask sets,
and the comparison was not paired. That is not a small effect here: enumerating all 12 single-feature
drops, baseline mIoU runs from 33.3 to 97.9 (SD 17.4), so averaging 12 unpaired draws leaves about
+/-7 mIoU of pure mask noise on any pairwise difference at m=1 — larger than the differences being
measured. Masks are now drawn once per missing-count and every config sees all of them.

They are also ENUMERATED rather than sampled wherever the space is affordable. `C(12,6)=924` is the
largest set here and the whole m=0..6 space is 2510 masks, all under the repo's ENUMERATION_CAP, so
the curve is the EXACT mean over every mask of that size and carries no mask sampling error at all.
Sampling 12 masks with replacement covered about 7.8 of the 12 single-feature cases and 17% of the 66
two-feature cases, so which features happened to be drawn moved the curve.

PAIRED ABLATION. The baseline and group-dropout models are built from the same spawned init stream
and the same shuffle stream, and differ only in whether the augmentation is applied — the
augmentation draws from its own third stream. Previously one generator served init, minibatch order
AND the mask draws, so the two arms diverged in shuffle order from the second epoch on, and their
initial weights came from `seed+101` vs `seed+202`; the full-band gap therefore carried optimisation
variance on top of the augmentation being studied (the shipped run shows the dropout arm at
94.88 +/- 3.16 against the baseline's 97.67 +/- 0.39 — an 8x spread with no way to attribute it).
`SeedSequence([world_seed, run_seed]).spawn(...)` also removes the offset collision: `--seeds 0 101`
gave seed 0's dropout model and seed 101's baseline the same stream, 202.

ERROR BANDS. The world is FIXED by --world-seed, on purpose: every run studies the same problem. The
band is the SAMPLE standard deviation (ddof=1) across run seeds of pixel sampling, weight init and
minibatch order. With masks enumerated it contains no mask Monte-Carlo error. It does NOT contain
variation over worlds (class separation, factor loadings, noise level, missingness mechanism), so it
does not support a claim about synthetic problems in general.

FURTHER LIMITS, so the figure is not over-read:
  * Per-pixel classification, not segmentation. There is no image, no neighbourhood, no boundary;
    mIoU here is the mean per-class IoU of an i.i.d. vector classifier.
  * No radiometry. The 12 features have no wavelengths (the unused `np.linspace(0.49, 2.19, 12)`
    axis and the "Sentinel-2-like" label are gone: real Sentinel-2 has 13 non-uniformly spaced bands
    at three ground resolutions). Values are not bounded to [0,1] — 44% of pixels have at least one
    feature outside it — because nothing here models reflectance.
  * Train-time and test-time missingness DIFFER by construction: training masks whole 4-feature
    groups, evaluation removes individual features uniformly at random. Generalising across that gap
    is part of what is being measured, not an oversight.
  * Zero-fill after standardisation is indistinguishable from "exactly the training mean". No model
    here receives a missingness indicator.

Outputs (under ../paper/), atomic; --smoke writes *_smoke.* so it cannot replace them:
  figs/fig_degradation_multiseed.pdf  - mIoU vs #missing features, mean +/- sample SD
  results_multiseed.csv               - per-missing mean/sd for the 3 configs
  results_multiseed_raw.csv           - one row per (seed, config, missing count, mask)
  results_multiseed.tex               - LaTeX macros, all prefixed \\ms* (see below)

Usage:
  python experiments/experiment_synthetic_multiseed.py                 # seeds 0..4
  python experiments/experiment_synthetic_multiseed.py --seeds 0 1 2 3 4 5 6 7
  python experiments/experiment_synthetic_multiseed.py --smoke         # fast sanity, *_smoke.*
"""
import os
import csv
import argparse
from itertools import combinations
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the project's audited metric + robustness summary instead of a local copy. This file used to
# say that and then define its own `miou`, which handed a free IoU of 1.0 to every class absent from
# the ground truth — on y_true=[0,0,1,1] / y_pred=[0,0,0,0] with 4 classes the local copy returned
# 62.50 where the audited metric returns 25.00. Identical on this experiment's own data (all four
# classes are always present in 4000 uniform draws), so no published number moves; the point is that
# there is now one metric in the repo rather than two that agree by luck.
_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.metrics import audc, miou
from bandsim import parallel
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)

# ---- fixed synthetic-world constants -------------------------------------------------
C = 12          # correlated features (NOT wavelengths; see the module docstring)
K = 4           # classes
R = 4           # latent rank -> inter-feature correlation (the redundancy being exploited)
H = 64          # MLP hidden units
N_TR = 6000
N_TE = 4000
GROUPS = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]  # train-time masking groups
CLASS_NAMES = ["Clear", "Thin cloud", "Thick cloud", "Shadow"]       # recorded in provenance

# Enumerate every mask of a given size when the space is at most this large; sample beyond it. Same
# constant and the same rule as phase2_degradation, so "was this point an exact mean over all masks
# or a Monte-Carlo estimate?" has one answer across the repo. At C=12 every m is under the cap.
ENUMERATION_CAP = 1024

# ONE source of truth for the configs: CSV columns, LaTeX macros, the figure legend and the tests all
# derive from this. A name is DATA here, not prose, so a test can assert on it — a guard must not be
# a source-substring search, which cannot tell a claim from its own retraction.
CONFIGS = (
    ("base",   "Baseline MLP + zero-fill",                "#c0392b", "-o", "Base"),
    ("drop",   "Group-dropout MLP + zero-fill",           "#e67e22", "-s", "Drop"),
    ("impute", "Group-dropout MLP + Gaussian imputation", "#1f6f3a", "-^", "Impute"),
)
KEYS = tuple(k for k, _, _, _, _ in CONFIGS)


def _sd1(a):
    """SAMPLE standard deviation (ddof=1) over run seeds; NaN for a single seed.

    numpy's default ddof=0 treats the seeds that happened to run as the whole population; at n=5 it
    is 89% of the sample SD, an error bar 11% too small. NaN rather than 0.00 at n=1, because a
    one-seed run has no measurable spread and `0.0` reads as perfect reproducibility."""
    a = np.asarray(a, float)
    return float(a.std(ddof=1)) if a.size > 1 else float("nan")


def _fmt(v, nd=2):
    """NaN -> empty cell: unambiguously "not computed", and keeps integrity_check's finite-cell scan
    meaning what it says."""
    v = float(v)
    return "" if v != v else f"{v:.{nd}f}"


def _atomic_write(path, write_fn):
    """Write through a temp file + os.replace, so a killed run cannot leave a truncated deliverable
    that still parses."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as f:
        write_fn(f)
    os.replace(tmp, path)


def tex_macros(stats, audcs, n_seeds, world_seed, max_missing, n_masks):
    """The LaTeX macros this phase emits, as a {name: body} dict.

    A dict rather than a pile of f.write calls so a test can assert on the NAMES, which is where two
    defects lived. \\audcProp was defined by BOTH this file and paper/results_phase2_summary.tex with
    different meanings (91.4 for the Gaussian imputer here, 56.5 for the real proposed model on
    Indian Pines): \\newcommand twice is a LaTeX error, and a \\providecommand workaround would
    silently print one phase's number under the other's name. And \\baseSixMS said "Six" while
    indexing stats[...][max_missing], so `--max-missing 3` wrote the 3-missing value into a macro the
    paper reads as six. Everything is now namespaced \\ms*, the count is a macro of its own, and no
    name claims a value it does not hold."""
    out = {"msNSeeds": str(int(n_seeds)), "msWorldSeed": str(int(world_seed)),
           "msMaxMissing": str(int(max_missing)), "msMasksPerSeed": str(int(n_masks))}
    for key, _, _, _, tag in CONFIGS:
        mean, sd = stats[key]
        out[f"ms{tag}Full"] = f"{mean[0]:.1f}\\,$\\pm$\\,{_fmt(sd[0], 1) or 'n/a'}"
        out[f"ms{tag}AtMax"] = (f"{mean[max_missing]:.1f}\\,$\\pm$\\,"
                                f"{_fmt(sd[max_missing], 1) or 'n/a'}")
        out[f"msAudc{tag}"] = f"{audcs[key]:.1f}"
    return out


def build_world(world_seed):
    """Class signatures + shared loadings — the fixed 'ground-truth' world.

    No wavelength axis is returned. One was computed (`np.linspace(0.49, 2.19, 12)`), labelled
    Sentinel-2-like, unpacked by the caller and never used by the generator, the model or the
    metric; keeping it invited the experiment being read as a multispectral simulation when it is a
    12-dimensional correlated-feature problem."""
    rng = np.random.default_rng(world_seed)
    base = np.stack([
        0.10 + 0.05 * np.sin(np.linspace(0, 2.0, C)),
        0.45 + 0.10 * np.cos(np.linspace(0, 3.0, C)),
        0.75 + 0.05 * np.cos(np.linspace(0.5, 2.5, C)),
        0.08 + 0.03 * np.linspace(0, 1, C),
    ])
    B_load = rng.normal(0, 1, size=(C, R)) * 0.06
    return base, B_load


def onehot(y, k=K):
    o = np.zeros((y.size, k)); o[np.arange(y.size), y] = 1; return o


def train_mlp(Xs, y, init_rng, shuffle_rng, aug_rng=None, epochs=60, lr=0.2, bs=256):
    """One-hidden-layer MLP. THREE separate generators, which is what makes the two arms comparable.

    `init_rng` draws the weights, `shuffle_rng` the epoch permutations, and `aug_rng` — present only
    for the group-dropout arm — the augmentation. Pass the SAME init and shuffle streams to both arms
    and they differ by the augmentation alone. One generator used to serve all three, so the arm that
    drew masks fell out of step with the other's minibatch order after the first epoch, and their
    initial weights came from different offsets entirely."""
    W1 = init_rng.normal(0, 0.3, (C, H)); b1 = np.zeros(H)
    W2 = init_rng.normal(0, 0.3, (H, K)); b2 = np.zeros(K)
    Y = onehot(y); n = Xs.shape[0]
    for _ in range(epochs):
        idx = shuffle_rng.permutation(n)
        for s in range(0, n, bs):
            bi = idx[s:s + bs]; xb = Xs[bi].copy(); yb = Y[bi]
            if aug_rng is not None:
                for r in range(xb.shape[0]):
                    ndrop = aug_rng.integers(0, 3)
                    if ndrop:
                        gs = aug_rng.choice(len(GROUPS), size=ndrop, replace=False)
                        for g in gs:
                            xb[r, GROUPS[g]] = 0.0
            z1 = xb @ W1 + b1; a1 = np.maximum(0, z1)
            z2 = a1 @ W2 + b2
            z2 -= z2.max(1, keepdims=True); e = np.exp(z2); p = e / e.sum(1, keepdims=True)
            g2 = (p - yb) / xb.shape[0]
            gW2 = a1.T @ g2; gb2 = g2.sum(0)
            g1 = (g2 @ W2.T) * (z1 > 0); gW1 = xb.T @ g1; gb1 = g1.sum(0)
            W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1
    model = (W1, b1, W2, b2)
    # A diverged model does not raise: argmax over NaN logits returns 0, every pixel is predicted
    # class 0, and the run reports a low-but-finite mIoU that looks like a hard problem.
    for name, arr in zip(("W1", "b1", "W2", "b2"), model):
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError(
                f"training produced non-finite {name} ({int((~np.isfinite(arr)).sum())} entries): "
                f"the run would have reported a plausible mIoU from an all-class-0 prediction.")
    return model


def predict(model, Xs):
    W1, b1, W2, b2 = model
    a1 = np.maximum(0, Xs @ W1 + b1); z2 = a1 @ W2 + b2
    return z2.argmax(1)


def _assert_learned(model, Xs, y, who, floor_margin=10.0):
    """Reject a model that trained without learning — the failure a finiteness check cannot see.

    A finite-parameter check catches a NaN blow-up, and that is NOT the failure mode this
    architecture actually has. Measured here: at lr=1e12 the ReLU units die, every parameter stays
    FINITE, the network collapses to predicting 3 of 4 classes, and the run reports a
    plausible-looking low mIoU with nothing raised. So check the thing that matters — that the model
    beats chance on its own training set — rather than the thing that is easy to check.

    The floor is chance + 10 points (35% at K=4). A legitimate run clears it by a wide margin even
    at one epoch: 73.6% after 1 epoch and 98.7% after the default 60 on this world."""
    acc = float(np.mean(predict(model, Xs) == y)) * 100.0
    floor = 100.0 / K + floor_margin
    if acc < floor:
        raise RuntimeError(
            f"{who}: training collapsed — {acc:.1f}% accuracy on its OWN training set against "
            f"{100.0 / K:.0f}% chance (floor {floor:.0f}%). Every parameter is finite, so a "
            f"non-finite check would have passed this and the degradation curve would have been "
            f"reported as a hard problem rather than a broken run.")
    return acc


def masks_for(n_feat, m, trials, rng):
    """Every mask of size m when the space is affordable, otherwise `trials` UNIQUE sampled ones.

    `rng.choice(..., replace=False)` only guarantees no repeat WITHIN one mask; drawing `trials` of
    them independently repeats masks across trials, so 12 draws covered ~7.8 of the 12 single-feature
    cases (65%) and ~11 of the 66 two-feature cases (17%). Enumeration removes the question; where
    the space is too large to enumerate, the samples are at least distinct."""
    if m == 0:
        return [()]
    n_sets = comb(n_feat, m)
    if n_sets <= max(int(trials), ENUMERATION_CAP):
        return [tuple(c) for c in combinations(range(n_feat), m)]
    target = min(int(trials), n_sets)
    seen, out, guard = set(), [], 0
    while len(out) < target:
        cand = tuple(sorted(rng.choice(n_feat, size=m, replace=False).tolist()))
        if cand not in seen:
            seen.add(cand); out.append(cand)
        guard += 1
        if guard > 200 * target:
            raise RuntimeError(f"could not draw {target} distinct masks of size {m} from "
                               f"{n_sets} possibilities")
    return out


def impute(xs, mask, Sigma, gmean):
    """Conditional-Gaussian reconstruction of the masked features from the observed ones.

    `solve` rather than `Smo @ inv(Soo)`. Measured on this world the two agree to 8e-14 at the worst
    observed condition number (2306), so this moves no number — it is the form that stays correct if
    the covariance ever becomes less friendly."""
    miss = np.asarray(mask, int)
    if miss.size == 0:
        return xs.copy()
    obs = np.array([i for i in range(Sigma.shape[0]) if i not in set(mask)], int)
    if obs.size == 0:
        raise ValueError("every feature is masked: nothing left to condition on")
    Soo = Sigma[np.ix_(obs, obs)]; Smo = Sigma[np.ix_(miss, obs)]
    Wc = np.linalg.solve(Soo, Smo.T).T
    if not np.all(np.isfinite(Wc)):
        raise FloatingPointError(f"non-finite imputation weights for mask {mask} "
                                 f"(cond(Soo)={np.linalg.cond(Soo):.3g})")
    out = xs.copy()
    out[:, miss] = gmean[miss] + (xs[:, obs] - gmean[obs]) @ Wc.T
    return out


def run_once(seed, world, world_seed, max_missing=6, trials=12, epochs=60):
    """One independent run: sample data, train both models, evaluate every config on shared masks."""
    base, B_load = world
    # Independent, reproducible streams. `seed + 101` / `seed + 202` offsets collide across runs:
    # --seeds 0 101 gave seed 0's dropout model and seed 101's baseline the same stream (202).
    data_ss, init_ss, shuffle_ss, aug_ss, mask_ss = np.random.SeedSequence(
        [int(world_seed), int(seed)]).spawn(5)
    data_rng = np.random.default_rng(data_ss)

    def sample(n):
        y = data_rng.integers(0, K, size=n)
        z = data_rng.normal(0, 1, size=(n, R))
        x = base[y] + z @ B_load.T + data_rng.normal(0, 0.02, size=(n, C))
        return x.astype(np.float64), y

    Xtr, ytr = sample(N_TR)
    Xte, yte = sample(N_TE)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr_s = (Xtr - mu) / sd; Xte_s = (Xte - mu) / sd
    Sigma = np.cov(Xtr_s, rowvar=False) + 1e-4 * np.eye(C)
    gmean = Xtr_s.mean(0)

    # PAIRED: same init, same minibatch order; the dropout arm additionally draws from aug_ss.
    # default_rng(SeedSequence) is deterministic, so two generators built from the same spawned
    # SeedSequence produce identical streams -- that is what makes this an ablation of one variable.
    models = {
        "base": train_mlp(Xtr_s, ytr, np.random.default_rng(init_ss),
                          np.random.default_rng(shuffle_ss), None, epochs=epochs),
        "drop": train_mlp(Xtr_s, ytr, np.random.default_rng(init_ss),
                          np.random.default_rng(shuffle_ss), np.random.default_rng(aug_ss),
                          epochs=epochs),
    }

    for name, model in models.items():
        _assert_learned(model, Xtr_s, ytr, f"seed {seed} / {name}")

    mask_rng = np.random.default_rng(mask_ss)
    curves = {k: [] for k in KEYS}
    raw = []
    for m in range(0, max_missing + 1):
        # ONE mask list per missing count, shared by every config: the comparison is paired by
        # construction rather than by three generators happening to agree.
        per = {k: [] for k in KEYS}
        for mask in masks_for(C, m, trials, mask_rng):
            x0 = Xte_s.copy()
            if mask:
                x0[:, list(mask)] = 0.0
            xi = impute(x0, mask, Sigma, gmean) if mask else x0
            vals = {"base": miou(yte, predict(models["base"], x0), K),
                    "drop": miou(yte, predict(models["drop"], x0), K),
                    "impute": miou(yte, predict(models["drop"], xi), K)}
            for k in KEYS:
                per[k].append(vals[k])
                raw.append((int(seed), k, m, "|".join(map(str, mask)), float(vals[k])))
        for k in KEYS:
            curves[k].append(float(np.mean(per[k])))
    return {k: np.array(curves[k], float) for k in KEYS}, raw


def _validate(args):
    """Every pure-argument error, rejected before any training. `--trials 0` used to produce
    `np.mean([])` -> NaN and a RuntimeWarning, and `--max-missing 13` raised inside `rng.choice`
    only after both models had trained."""
    if len(args.seeds) < 1:
        raise ValueError("--seeds needs at least one seed.")
    if len(set(args.seeds)) != len(args.seeds):
        dup = sorted({s for s in args.seeds if list(args.seeds).count(s) > 1})
        raise ValueError(f"--seeds must be unique, got duplicates {dup}: a repeated seed is counted "
                         f"twice in every mean while n_seeds claims independent runs.")
    if any(s < 0 for s in args.seeds):
        raise ValueError("--seeds must be non-negative (SeedSequence entropy must be >= 0).")
    if not 0 <= args.max_missing < C:
        raise ValueError(f"--max-missing must be in [0, {C - 1}], got {args.max_missing}: with all "
                         f"{C} features masked there is nothing left to condition the imputer on.")
    if args.trials < 1:
        raise ValueError(f"--trials must be >= 1, got {args.trials}: it is the number of masks drawn "
                         f"wherever the space is too large to enumerate.")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}.")
    if args.jobs is not None and args.jobs < 1:
        raise ValueError(f"--jobs must be >= 1 if given, got {args.jobs}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--world-seed", type=int, default=20260630)
    ap.add_argument("--max-missing", type=int, default=6)
    ap.add_argument("--trials", type=int, default=12,
                    help="masks per missing count WHERE THE SPACE IS TOO LARGE TO ENUMERATE "
                         f"(unused at C={C}, where every count is under ENUMERATION_CAP)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true", help="1 seed, few epochs, quick sanity")
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent seed workers (default: adaptive; also BANDSIM_WORKERS)")
    args = ap.parse_args()

    # A smoke run writes *_smoke.* so it cannot replace the deliverable. experiments/
    # integrity_check.py ran this as `--seeds 0 --trials 3` pointed at the CANONICAL
    # results_multiseed.csv/.tex/.pdf, so the integrity harness would have replaced a 5-seed
    # deliverable with a 1-seed run and left \msNSeeds saying so in small print.
    out_tag = ""
    if args.smoke:
        args.seeds = [0]; args.epochs = 8; args.max_missing = 2
        out_tag = "_smoke"
        print("SMOKE RUN: 1 seed / 8 epochs / <=2 missing — writing *_smoke.*. Sanity check, not results.")
    _validate(args)

    world = build_world(args.world_seed)
    xs = np.arange(0, args.max_missing + 1)
    runs = {k: [] for k in KEYS}
    raw_rows = []
    # numpy-only Phase 0: fan seeds across CPU cores (the tiny numpy MLP has no GPU path, and
    # bandsim.parallel already caps OMP/MKL threads per worker, so hw.setup -- which configures
    # torch -- has nothing to do here).
    results = parallel.run_jobs(
        run_once, args.seeds,
        shared=dict(world=world, world_seed=args.world_seed, max_missing=args.max_missing,
                    trials=args.trials, epochs=args.epochs),
        prefer="cpu", jobs=args.jobs, label="phase0/seed")
    for sd, (curves, raw) in zip(args.seeds, results):
        for k in KEYS:
            runs[k].append(curves[k])
        raw_rows.extend(raw)
        print(f"seed {sd:>5}: " + "  ".join(f"{k}={curves[k].mean():5.1f}" for k in KEYS)
              + "   (mean over missing counts)")

    stats = {k: (np.mean(np.stack(runs[k]), 0),
                 np.array([_sd1(np.stack(runs[k])[:, i]) for i in range(len(xs))]))
             for k in KEYS}
    n = len(args.seeds)
    masks_at = {int(i): len(masks_for(C, int(i), args.trials, np.random.default_rng(0)))
                for i in xs}
    n_masks = sum(masks_at.values())

    # ---- summary csv --------------------------------------------------------------
    # `world_seed` is deliberately NOT a column: integrity_check.csv_finite_and_sane rejects any
    # cell above 1e4 as an out-of-range metric, and 20260630 is one. It travels in the provenance
    # sidecar (both CSVs are stamped) and in \msWorldSeed instead, which is where a reader looks for
    # run identity anyway. Everything else here is small enough to be a metric or a count.
    def _write_summary(f):
        w = csv.writer(f)
        w.writerow(["missing_features"] + sum([[f"{k}_mean", f"{k}_sd_ddof1"] for k in KEYS], [])
                   + ["n_seeds", "epochs", "n_masks_at_this_count", "seed_sd_ddof"])
        for i in xs:
            w.writerow([int(i)] + sum([[_fmt(stats[k][0][i]), _fmt(stats[k][1][i])] for k in KEYS], [])
                       + [n, args.epochs, masks_at[int(i)], 1])
    _atomic_write(P(f"results_multiseed{out_tag}.csv"), _write_summary)

    # ---- raw evidence: one row per (seed, config, missing count, mask) --------------
    # Aggregates alone cannot be re-analysed. This also answers "does it matter WHICH features go
    # missing?" directly -- on this world, enormously: baseline mIoU spans 33.3 to 97.9 across the
    # 12 single-feature drops, which is why unpaired masks were not a cosmetic problem.
    def _write_raw(f):
        w = csv.writer(f)
        w.writerow(["run_seed", "config", "missing_features", "mask", "miou"])
        for sd, key, m, mask, val in raw_rows:
            w.writerow([sd, key, m, mask, f"{val:.6f}"])
    _atomic_write(P(f"results_multiseed{out_tag}_raw.csv"), _write_raw)

    audcs = {k: audc(xs, stats[k][0]) for k in KEYS}

    # ---- LaTeX macros (names and namespacing rationale: see tex_macros) --------------
    macros = tex_macros(stats, audcs, n, args.world_seed, args.max_missing, n_masks)

    def _write_tex(f):
        f.write(f"% auto-generated by experiment_synthetic_multiseed.py "
                f"(world_seed={args.world_seed}, seeds={args.seeds}, epochs={args.epochs})\n")
        f.write("% the third config is a Gaussian conditional imputer, NOT SGMAE and NOT attention "
                "-- see the module docstring before quoting it as the proposed method\n")
        for name, body in macros.items():
            f.write(f"\\newcommand{{\\{name}}}{{{body}}}\n")
    _atomic_write(P(f"results_multiseed{out_tag}.tex"), _write_tex)

    # ---- figure --------------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for key, label, color, style, _ in CONFIGS:
        mean, sd = stats[key]
        ax.plot(xs, mean, style, color=color, lw=1.8, ms=4, label=label)
        ax.fill_between(xs, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Number of missing features")
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"Missing-feature robustness (synthetic, {n} seeds)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=6.2, frameon=False, loc="lower left")
    # Say what the band is ON the figure: it was a population SD over seeds with nothing saying so.
    ax.text(0.98, 0.97, r"band: $\pm$1 sample SD over seeds", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.6, color="0.35")
    fig.tight_layout(); fig.savefig(P(f"figs/fig_degradation_multiseed{out_tag}.pdf")); plt.close(fig)

    # BOTH csvs are stamped. The raw file is the evidence a reader re-analyses, and without its own
    # sidecar it would carry no record of which world produced it once `world_seed` stopped being a
    # column (see the summary writer).
    provenance = {"fabricated_data": True,
                      "world": "synthetic mechanism check, not a dataset",
                      "not_wavelengths": "the 12 features carry no wavelength or radiometric meaning",
                      "n_features": C, "n_classes": K, "n_shared_factors": R,
                      "class_names": CLASS_NAMES,
                      "train_time_masking": f"whole groups {GROUPS}",
                      "test_time_masking": "individual features, enumerated where affordable",
                      "configs": {k: lab for k, lab, _, _, _ in CONFIGS},
                      "third_config_is_not_sgmae": (
                          "conditional-Gaussian imputation fitted to the training covariance; the "
                          "generator is linear-Gaussian, so this is close to a ceiling for "
                          "covariance-based reconstruction, not a prediction for a learned SGMAE"),
                  "masks_per_seed": n_masks, "masks_by_missing_count": masks_at,
                  "enumeration_cap": ENUMERATION_CAP, "seed_sd_ddof": 1,
                  "raw_evidence_csv": f"results_multiseed{out_tag}_raw.csv"}
    ok_summary = stamp(P(f"results_multiseed{out_tag}.csv"), args, extra=provenance)
    ok_raw = stamp(P(f"results_multiseed{out_tag}_raw.csv"), args, extra=provenance)

    print(f"\nDONE — Phase 0 ({n} seeds, {n_masks} masks per config per seed, world "
          f"{args.world_seed})")
    print("missing   " + "   ".join(f"{k}(mean+/-sd)" for k in KEYS))
    for i in xs:
        print(f"{i:5d}   " + "   ".join(
            f"{stats[k][0][i]:5.1f}+/-{stats[k][1][i]:4.1f}" for k in KEYS))
    print("AUDC: " + "  ".join(f"{k}={audcs[k]:.1f}" for k in KEYS))
    print("NOTE: the third config is a Gaussian conditional imputer, not SGMAE and not attention; "
          "this world is linear-Gaussian, so it sits near a ceiling rather than forecasting real "
          "data. See the module docstring.")
    print(f"\nwrote: {P(f'figs/fig_degradation_multiseed{out_tag}.pdf')}")
    print(f"       {P(f'results_multiseed{out_tag}.csv')}")
    print(f"       {P(f'results_multiseed{out_tag}_raw.csv')}  ({len(raw_rows)} rows)")
    print(f"       {P(f'results_multiseed{out_tag}.tex')}")
    if ok_summary is None or ok_raw is None:
        print("PROVENANCE FAILED: the artefacts above are on disk but UNATTRIBUTED — do not cite.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
