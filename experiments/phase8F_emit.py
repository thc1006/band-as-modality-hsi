#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8F (single-granule) — SUPERSEDED BY phase8F_multi.py. Kept as a module, not as evidence.

DO NOT CITE THIS FILE'S NUMBERS. The experiment it runs is an in-sample, single-granule
correlation; `experiments/phase8F_multi.py` is the version whose result the paper uses, and it
exists precisely because an adversarial review found this design could not support the claim. It
holds out SPATIAL blocks (train on 70% of pixels, measure error on the disjoint 30%, standardised
with TRAIN statistics only), runs several granules across biomes, converts the error back to
reflectance units, adds a training-free PCA baseline and a brightness-partial correlation, and
reports per-band as explicitly post-hoc.

What that rerun MEASURED is the reason this file's headline is withdrawn rather than merely
softened: under the spatial split the per-pixel Spearman moves a lot and in BOTH directions --
India +0.42 -> +0.55, sahara +0.089 -> -0.007, us_midwest +0.22 -> +0.15. Two of three granules
lose their positive sign entirely. Part of the old "positive everywhere" reading came from holding
out pixels that sat among the training pixels; here there is no holdout at all, so the
reconstruction error is a TRAINING RESIDUAL, not a generalisation signal.

WHY THIS FILE STILL EXISTS: phase8F_multi does `import phase8F_emit as F` to reuse
`recon_error_matrix`, which is therefore a LIVE dependency of the current result. Changes to that
function change the paper's numbers. Everything else here is history.

WHAT EMIT's UNCERTAINTY IS, STATED CORRECTLY. Earlier revisions of this docstring called it
"INDEPENDENT physical retrieval uncertainty" and an "external, physics-based ground-truth". It is
neither. EMIT L2A reflectance and its uncertainty are produced by the SAME ISOFIT
optimal-estimation retrieval: the reflectance is the retrieval's estimate and the uncertainty is
the square root of the diagonal of that retrieval's posterior covariance. They share the
at-sensor radiance, the atmospheric correction, the instrument noise model, the surface prior and
the RTM's wavelength-dependent sensitivity. So the uncertainty is EXTERNAL TO OUR MODEL but NOT
independent of the reflectance our model is trained on, and it is not measured error against a
reference. The defensible term is an external, model-derived RETRIEVAL-UNCERTAINTY PROXY. A
correlation with it is evidence that our error tracks something real in the retrieval, and it is
not validation against ground truth.

DATA LICENCE: an earlier revision said "MIT-licensed reflectance". That is wrong -- the MIT licence
in this repository covers this repository's own code. Cite the EMIT L2A product
(doi:10.5067/EMIT/EMITL2ARFL.001) and NASA's data-use terms instead.

Data: data/emit/emit_sample.npz (50k pixels x 244 retained bands + per-pixel/per-band uncertainty),
extracted from ONE EMIT L2A granule via earthaccess. Note the npz also carries `lat`, `lon` and
`worldcover`, which this script ignores -- they are what a spatial split and a surface-class
stratification would need, and phase8F_multi uses that kind of information.

Output (../paper/):
  results_phase8F_emit.csv    per-band: wavelength, EMIT uncertainty, our reconstruction error
Any run whose configuration is not the canonical one (including --smoke, which drops --groups to
12) writes a `_smoke`/`_nonCanonical` suffix instead: re-partitioning the bands changes every
per-band error, so it is a different experiment rather than a shorter one.

Usage:
  python experiments/phase8F_emit.py --smoke
  python experiments/phase8F_emit.py --seeds 0 1 2 --device cuda
"""
import os, sys, csv, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase2_degradation as P2
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim import hw
from bandsim.provenance import stamp, file_sha256

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(PAPER_DIR, exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)
EMIT_NPZ = os.path.join(os.path.dirname(_HERE), "data", "emit", "emit_sample.npz")
# The configuration the canonical CSV name is allowed to represent. Any other setting writes a
# suffixed file: --groups in particular re-partitions the spectrum and changes every per-band error.
CANONICAL = {"groups": 20, "epochs": 40, "max_px": None}


def _spearman(x, y):
    """Spearman rho with proper TIE handling (scipy.stats.spearmanr).

    The old argsort-of-argsort mis-ranked ties (arbitrary order instead of average rank). That was
    described as "inflating correlation"; it can move it in EITHER direction, since the arbitrary
    order may agree or disagree with the other variable. SciPy averages tied ranks.

    Non-finite handling is explicit rather than inherited. scipy's default nan_policy is
    'propagate', so a single NaN returned NaN -- and the caller then took np.nanmean over seeds,
    which SKIPS that seed and reports a mean over fewer runs than it claims. Infinity was worse: it
    is orderable, so it ranked as the largest value and produced a plausible finite rho from
    corrupt data. Pairs are dropped only where either side is non-finite, and how many were dropped
    is returned so a caller can refuse to report a correlation computed on a remnant.

    Returns (rho, n_used). rho is nan if fewer than 3 valid pairs remain or either side is constant.
    """
    from scipy.stats import spearmanr
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), n
    r = spearmanr(x, y).correlation
    return (float(r) if r == r else float("nan")), n


def load_emit(path, max_px=None, rng_seed=0):
    """Load the EMIT extract and ASSERT the data contract before anything downstream trusts it.

    The npz is a regenerable extract, so "the extraction script already cleaned it" is a property
    of a script that is not in this file's control. EMIT L2A uses -9999 for nodata and can carry
    -0.01 where reflectance was not estimated inside deep water-vapour absorption; a single such
    value poisons the per-band mean and sd, and the resulting correlation is a number that looks
    fine. Checked here, once, loudly. (Measured on the shipped extract: all finite, no fill values,
    uncertainty non-negative, wavelengths strictly increasing -- so these cost nothing today and
    fail loudly if a regenerated extract is dirty.)
    """
    with np.load(path, allow_pickle=False) as d:
        missing = {"reflectance", "uncertainty", "wavelengths"} - set(d.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)} (have {sorted(d.files)})")
        R = d["reflectance"].astype(np.float32)
        U = d["uncertainty"].astype(np.float32)
        wl = d["wavelengths"].astype(float)
    if R.ndim != 2 or U.shape != R.shape:
        raise ValueError(f"reflectance {R.shape} and uncertainty {U.shape} must be the same 2-D shape")
    if wl.ndim != 1 or wl.size != R.shape[1]:
        raise ValueError(f"wavelengths {wl.shape} must have one entry per band ({R.shape[1]})")
    if not np.isfinite(R).all() or not np.isfinite(U).all():
        raise ValueError("reflectance/uncertainty contain non-finite values")
    if (R <= -0.005).any():
        raise ValueError(
            f"reflectance contains EMIT fill values (min {R.min():.4f}): -9999 is nodata and -0.01 "
            f"marks bands where reflectance was not estimated. Drop them upstream.")
    if (U < 0).any():
        raise ValueError(f"uncertainty must be non-negative (posterior sd), min {U.min():.4g}")
    if not np.all(np.diff(wl) > 0):
        raise ValueError("wavelengths must be strictly increasing (band order is assumed downstream)")
    if max_px is not None and max_px < R.shape[0]:
        sub = np.random.default_rng(rng_seed).choice(R.shape[0], max_px, replace=False)
        R, U = R[sub], U[sub]
    return R, U, wl


def _check_partition(X, groups):
    """`groups` must tile 0..X.shape[1]-1 exactly once. Returns the band count.

    Not a formality. The old code sized the output as `max(max(g)) + 1`, so a grouping that did not
    reach the last band produced a NARROWER matrix than X, and the caller then correlated it
    column-by-column against a per-band uncertainty vector of the FULL width -- every band silently
    compared against a different band's uncertainty. A grouping with a HOLE left that band's column
    all-NaN, which `np.nanmean` turns into a NaN that `np.nanmean` upstream then skips, quietly
    dropping the band from the correlation instead of failing. Overlapping groups silently
    overwrote each other, last-writer-wins. None of the three raised.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be (n_pixels, n_bands), got shape {X.shape}")
    if not len(groups):
        raise ValueError("groups is empty")
    flat = np.concatenate([np.asarray(g).ravel() for g in groups]) if groups else np.array([], int)
    expected = np.arange(X.shape[1])
    if flat.size != expected.size or not np.array_equal(np.sort(flat), expected):
        raise ValueError(
            f"groups must cover every band of X exactly once: X has {X.shape[1]} bands, groups "
            f"cover {flat.size} indices ({np.unique(flat).size} distinct, range "
            f"{flat.min() if flat.size else '-'}..{flat.max() if flat.size else '-'}). A partial "
            f"or overlapping grouping misaligns the returned matrix against any per-band vector "
            f"it is compared with.")
    return X.shape[1]


@torch.no_grad()
def recon_error_matrix(model, X, groups, bs=4096):
    """(N, n_band) per-pixel per-band SGMAE reconstruction error: for each group, mask ONLY that
    group, reconstruct its bands from the OTHER (present) groups, record |pred - true| per band.

    WHAT IS MEASURED, precisely: a band's error is recorded while its WHOLE GROUP is hidden, not
    while that band alone is hidden. It is therefore a group-conditioned reconstruction difficulty,
    and it depends on the partition -- how many bands the group holds, whether it straddles an
    absorption feature, whether it sits at the spectrum's edge. Changing --groups changes every
    number this returns. Do not describe it as an intrinsic per-band property; a leave-one-band-out
    sweep would be needed for that.

    LIVE DEPENDENCY: phase8F_multi.py imports this function for the result the paper uses. Its
    output is in the units of X -- standardised, when the caller standardised -- and callers that
    compare against reflectance-unit uncertainty must convert back (phase8F_multi multiplies by the
    train sd). Changes here change published numbers.
    """
    n_band = _check_partition(X, groups)
    if bs <= 0:
        raise ValueError(f"bs must be positive, got {bs}")
    if not np.isfinite(X).all():
        raise ValueError(
            "X contains non-finite values. EMIT L2A uses -9999 for nodata and can carry -0.01 "
            "where reflectance was not estimated (deep water-vapour absorption); standardising "
            "such a cube propagates NaN/inf into every reconstruction error and then into a "
            "correlation that still looks like a number. Mask or drop those pixels/bands upstream.")
    dev = next(model.parameters()).device
    G = len(groups)
    E = np.full((X.shape[0], n_band), np.nan, np.float32)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    # EVALUATION MODE, restored afterwards. `@torch.no_grad()` only stops autograd; it does not
    # switch Dropout or BatchNorm. Today P2.pretrain_sgmae ends with model.eval() and the model is
    # built with dropout=0.0, so this is a no-op on the numbers -- which is exactly why it is worth
    # pinning here rather than relying on the last line of a function in another module.
    was_training = model.training
    model.eval()
    try:
        for g in range(G):
            for s in range(0, X.shape[0], bs):
                xb = Xt[s:s + bs].to(dev)
                b = xb.shape[0]
                masked = torch.zeros(b, G, dtype=torch.bool, device=dev)
                masked[:, g] = True
                pred = model.reconstruct(xb, masked)                       # (b, G, S)
                if pred.ndim != 3 or pred.shape[0] != b or pred.shape[1] != G \
                        or pred.shape[2] < len(groups[g]):
                    raise ValueError(
                        f"reconstruct() returned {tuple(pred.shape)}; expected (b={b}, G={G}, "
                        f"S>={len(groups[g])}). A change to the group padding or the reconstruct "
                        f"API would otherwise map predictions onto the wrong bands silently.")
                for li, band in enumerate(groups[g]):
                    E[s:s + b, int(band)] = torch.abs(
                        pred[:, g, li] - xb[:, int(band)]).cpu().numpy()
    finally:
        model.train(was_training)
    if np.isnan(E).any():
        raise ValueError("reconstruction left NaN entries; some band was never reconstructed")
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-px", type=int, default=None, help="cap pixels (default: all in the npz)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    ap.add_argument("--nondeterministic", action="store_true")
    args = ap.parse_args()
    if args.groups < 2 or args.epochs <= 0 or len(set(args.seeds)) != len(args.seeds):
        ap.error("--groups must be >= 2, --epochs > 0, and --seeds must be distinct")
    # --smoke changes --groups (20 -> 12), which changes the band grouping the SGMAE reconstructs
    # through and therefore every per-band error in the CSV. That is a different experiment, not a
    # shorter one, and it used to be written to the deliverable's filename.
    #
    # THE SAME ARGUMENT APPLIES TO --groups ITSELF, and only --smoke was guarded: `--groups 5`
    # re-partitions the bands exactly as smoke does and still landed on the canonical path. Any
    # non-canonical configuration now gets its own suffix, so the deliverable can only be produced
    # by the configuration it claims to represent.
    sfx = ""
    if args.smoke:
        args.seeds = [0]; args.epochs = 8; args.max_px = 4000; args.groups = 12
        sfx = "_smoke"
        print("[smoke] 1 seed / 8 epochs / 12 groups — writing *_smoke artefacts, NOT the deliverable")
    elif (args.groups, args.epochs, args.max_px) != (CANONICAL["groups"], CANONICAL["epochs"],
                                                     CANONICAL["max_px"]):
        sfx = "_nonCanonical"
        print(f"[non-canonical] groups={args.groups} epochs={args.epochs} max_px={args.max_px} "
              f"differs from the canonical {CANONICAL} — writing *_nonCanonical artefacts. "
              f"Re-partitioning the bands changes every per-band error, so this is a DIFFERENT "
              f"experiment and must not overwrite the deliverable.")
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device)
    dev = hw.device(args.device)
    print("HW:", hw.info())

    R, U, wl = load_emit(EMIT_NPZ, max_px=args.max_px)
    n_band = R.shape[1]
    mu = R.mean(0); sd = R.std(0) + 1e-8
    Rn = ((R - mu) / sd).astype(np.float32)                            # per-band standardize
    groups = contiguous_groups(n_band, args.groups)
    cwl = group_center_wavelengths(wl, groups)
    P2.NUM_CLASSES = 2                                                 # dummy (SGMAE uses no labels)
    band_unc = U.mean(0)                                               # (n_band,) real per-band uncertainty
    print(f"EMIT: {R.shape[0]} px x {n_band} bands ({wl.min():.0f}-{wl.max():.0f} nm) | {args.groups} groups")

    band_err_seeds, sp_band, sp_pix = [], [], []
    for seed in args.seeds:
        # SEED BEFORE CONSTRUCTION. P2.pretrain_sgmae seeds torch INSIDE itself, i.e. AFTER the
        # model's parameters have already been drawn -- so `seed` controlled the data order and the
        # masking but never the initialisation. Measured consequence: two models built from the same
        # RNG state differ, so an init depended on HOW MANY models had been constructed earlier in
        # the loop, and "seed=1" alone did not reproduce "seed=1" inside a 3-seed run.
        # NOTE: experiments/phase8F_multi.py:590 has the same ordering and is the LIVE experiment.
        torch.manual_seed(seed)
        np.random.seed(seed)
        # `.to(dev)` explicitly. `dev` was resolved at startup and then never used: whether
        # --device cuda took effect depended entirely on P2.pretrain_sgmae happening to move the
        # model, and recon_error_matrix reads the device back off the parameters. That is an
        # invisible cross-module contract; placing the model is this script's job.
        m = GroupedCrossBandAttention(groups, cwl, 2).to(dev)
        P2.pretrain_sgmae(m, Rn, groups, seed, epochs=args.epochs)
        E = recon_error_matrix(m, Rn, groups)                         # (N, n_band) TRAINING residual
        band_err = E.mean(axis=0)                                     # per-band recon error
        pix_err = E.mean(axis=1)                                      # per-pixel recon error
        pix_unc = U.mean(1)                                           # per-pixel aggregate uncertainty
        band_err_seeds.append(band_err)
        rb, nb = _spearman(band_unc, band_err)
        rp, npix = _spearman(pix_unc, pix_err)
        if nb != n_band or npix != R.shape[0]:
            raise ValueError(f"non-finite pairs were dropped before correlating "
                             f"(bands {nb}/{n_band}, pixels {npix}/{R.shape[0]})")
        sp_band.append(rb); sp_pix.append(rp)
        print(f"  seed {seed}: Spearman(EMIT_unc, recon_err) per-band={rb:+.3f} per-pixel={rp:+.3f}")

    band_err_seeds = np.asarray(band_err_seeds)                       # (n_seeds, n_band)
    band_err_mean = band_err_seeds.mean(0)
    # ---- csv ----
    # `recon_error_std_pop` and the raw-reflectance column are new. The error is measured in
    # STANDARDISED units (the cube was z-scored per band) while emit_uncertainty is a posterior sd
    # in REFLECTANCE units, so the two columns were not on a common scale and a per-band rank
    # comparison between them was reordered by the per-band sd. Spearman is rank-based, so a single
    # monotone rescaling would not matter -- but dividing each band by its OWN sd is not one
    # transform, it is a different divisor per band, and it does change the ranking. The raw column
    # is the comparable one; phase8F_multi multiplies by the train sd for exactly this reason.
    order = np.argsort(wl)
    with open(P(f"results_phase8F_emit{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wavelength_nm", "emit_uncertainty_reflectance", "recon_error_standardised",
                    "recon_error_reflectance", "recon_error_std_pop_over_seeds", "n_seeds"])
        for i in order:
            w.writerow([f"{wl[i]:.1f}", f"{band_unc[i]:.5f}", f"{band_err_mean[i]:.5f}",
                        f"{band_err_mean[i] * sd[i]:.5f}",
                        f"{band_err_seeds[:, i].std():.5f}", len(args.seeds)])
    # Per-seed correlations, which only ever existed in a print. The per-pixel column decided the
    # old verdict and was never written anywhere a reader could recompute or stratify it.
    with open(P(f"results_phase8F_emit_perseed{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "spearman_perband", "spearman_perpixel"])
        for s, rb, rp in zip(args.seeds, sp_band, sp_pix):
            w.writerow([s, f"{rb:+.5f}", f"{rp:+.5f}"])

    mb, mp = float(np.mean(sp_band)), float(np.mean(sp_pix))
    sb, spx = float(np.std(sp_band)), float(np.std(sp_pix))
    print(f"\n===== Phase 8F (SUPERSEDED; in-sample, one granule) — {len(args.seeds)} seeds =====")
    print(f"Spearman(EMIT per-band uncertainty, per-band reconstruction error)  = {mb:+.3f} "
          f"(sd {sb:.3f} over seeds)")
    print(f"Spearman(EMIT per-pixel uncertainty, per-pixel reconstruction error) = {mp:+.3f} "
          f"(sd {spx:.3f} over seeds)")
    print(f"  per-seed per-band : {['%+.3f' % v for v in sp_band]}")
    print(f"  per-seed per-pixel: {['%+.3f' % v for v in sp_pix]}")
    # NO VERDICT IS PRINTED, and the threshold that produced one is gone. It read
    # `'IS GROUNDED in' if mp > 0.2 else ...` -- an unjustified hardcoded cutoff turning 0.199 and
    # 0.201 into opposite scientific conclusions, asserting a causal grounding claim, from an
    # IN-SAMPLE correlation on ONE granule, in a file whose successor measured the sign flipping
    # under a proper spatial split (sahara +0.089 -> -0.007). It also silently switched the endpoint:
    # the module, the CSV and the filename all say PER-BAND, while the verdict was decided on
    # PER-PIXEL because the per-band result was heterogeneous. Reporting the estimate and its
    # spread, and pointing at the file that did the work properly, is what this run can support.
    print("\nNo verdict is issued here. These are IN-SAMPLE correlations (the SGMAE was trained on")
    print("the very pixels it is scored on, so the error is a training residual) from ONE granule,")
    print("whose pixels are spatially autocorrelated -- 50k pixels are not 50k independent samples,")
    print("and the seed spread above is optimisation variance, not sampling uncertainty.")
    print("Per-band and per-pixel are DIFFERENT endpoints; per-band is heterogeneous (VNIR/NIR")
    print("positive, SWIR absorption bands reverse). For the held-out, multi-granule result with a")
    print("PCA baseline and a brightness-partial correlation, see phase8F_multi.py -- and note it")
    print("found two of three granules lose the positive sign once the split is spatial.")
    # The whole claim is "our error tracks EMIT's own uncertainty", so WHICH granule sample was read
    # is the input identity: emit_sample.npz is a regenerable extract, and a different draw of pixels
    # moves both columns. Hash it, and record the two Spearman scalars the CSV does not carry.
    stamp(P(f"results_phase8F_emit{sfx}.csv"), args,
          extra={"emit_npz": EMIT_NPZ, "emit_npz_sha256": file_sha256(EMIT_NPZ),
                 "n_px": int(R.shape[0]), "n_bands": int(n_band),
                 "wavelength_nm_range": [float(wl.min()), float(wl.max())],
                 "spearman_perband": mb, "spearman_perband_sd_over_seeds": sb,
                 "spearman_perpixel": mp, "spearman_perpixel_sd_over_seeds": spx,
                 "spearman_perband_per_seed": [float(v) for v in sp_band],
                 "spearman_perpixel_per_seed": [float(v) for v in sp_pix],
                 "canonical_config": CANONICAL,
                 "is_canonical_config": sfx == "",
                 "superseded_by": "experiments/phase8F_multi.py",
                 "status": (
                     "SUPERSEDED. In-sample (trained and scored on the same pixels, so the error is "
                     "a training residual) and single-granule (spatially autocorrelated pixels, so "
                     "n_px is not the effective sample size). Retained because phase8F_multi imports "
                     "recon_error_matrix from it. Do not cite these numbers."),
                 "uncertainty_semantics": (
                     "EMIT L2A reflectance and its uncertainty come from the SAME ISOFIT "
                     "optimal-estimation retrieval (uncertainty = sqrt of the posterior covariance "
                     "diagonal). External to our model, NOT independent of the reflectance it is "
                     "trained on, and not measured error against a reference. It is a "
                     "retrieval-uncertainty PROXY, not physical ground truth."),
                 "error_units": (
                     "recon_error_standardised is in per-band z units; recon_error_reflectance "
                     "multiplies by the per-band sd and is the column comparable to "
                     "emit_uncertainty_reflectance."),
                 "endpoints": (
                     "per-band and per-pixel are DIFFERENT endpoints, neither pre-registered here. "
                     "The removed verdict used per-band naming and a per-pixel decision rule."),
                 "seed_semantics": (
                     "seeds are re-seeded BEFORE model construction as of this revision; earlier "
                     "runs seeded only inside pretrain_sgmae, so initialisation was not controlled "
                     "by the seed label. The sd over seeds is optimisation variance, not sampling "
                     "uncertainty.")})
    print(f"wrote: {P(f'results_phase8F_emit{sfx}.csv')}")


if __name__ == "__main__":
    main()
