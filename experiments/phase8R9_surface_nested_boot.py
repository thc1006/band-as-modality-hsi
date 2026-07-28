#!/usr/bin/env python
"""E5 (round-5 review 3.5), CORRECTED nested scene-component bootstrap of the SURFACE breach.

The estimator the paper reports for the surface axis is the mean over the crossed design (model seeds x
calibration splits) of the component-equal target joint risk. This bootstrap therefore RE-COMPUTES THAT
AGGREGATE per replicate (an earlier version mistakenly pooled single-run risks, which is a prediction-like
mixture, not the CI of the mean). Each replicate:

  * draws ONE bright (target) + one dark (source) scene-component bootstrap cohort that is SHARED across all
    model seeds, so cross-model scene-difficulty covariance is preserved (independent per-seed resampling
    would destroy it);
  * for M random temperature/calibration splits of the UNIQUE dark components (temp and calib never share an
    original component, so no leak), and for EVERY model seed, re-fits the temperature, re-selects the CRC
    threshold, and reads the CERTIFIED component-equal triple (joint risk, coverage, feasibility) from the
    shared conformal_risk_control API -- never a hand-written accepted-and-wrong mean, so an infeasible /
    abstain-all threshold is recorded as such rather than silently counted as zero risk;
  * averages the feasible joint risks over the (seed x split) cells -> one aggregate replicate estimate.

The percentile interval of the B aggregate estimates is the CI. We also report the point estimate on the
un-bootstrapped cohort, the bootstrap bias, coverage, the CRC feasibility rate, and a THREE-WAY relation to
the target (entirely above / entirely below / includes). Scene-component bootstrap unit; duplicated
components get fresh group ids. Parallel across the CPU-affinity cores via a portable Pool initializer
(each worker loads the dumps itself -- no reliance on fork COW), one BLAS thread per worker.
Offline from scenedump_surface.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

os.environ["OMP_NUM_THREADS"] = "1"                        # force (not setdefault): the pool is the parallelism
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.reliability import fit_temperature, conformal_risk_control

DUMP_DIR = os.path.join(_HERE, "..", "paper", "scenedump_surface")
ALPHA = 0.10
BASE_SEED = 20260724
_DATA = []                                                 # per-seed arrays, set by the pool initializer
_DARK = _BRIGHT = None                                     # shared unique component ids (identical across seeds)


def require_finite(name, x):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def ci_relation(lo, hi, tgt):
    """Three-way relation of a CI to the target (P0 fix: a CI entirely BELOW the target must NOT be called
    'includes'). Returns (short relation, verdict sentence)."""
    if lo > tgt:
        return "entirely above", "evidence of excess risk (CI entirely above target)"
    if hi < tgt:
        return "entirely below", "evidence of risk BELOW target (CI entirely below)"
    return "includes", "inconclusive relative to target (CI includes it)"


def parse_seed(path):
    m = re.search(r"seed[_-]?(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse a model seed from {path!r}")
    return int(m.group(1))


def _load_raw(path):
    d = np.load(path)
    lg, y, comp, it = d["logits"], d["y"].astype(np.int64), d["comp"], d["is_target"]
    require_finite(f"{os.path.basename(path)} logits", lg)
    if not (lg.ndim == 2 and len(lg) == len(y) == len(comp) == len(it)):
        raise ValueError(f"{path}: mis-aligned arrays {lg.shape} {y.shape} {comp.shape} {it.shape}")
    if not np.isin(np.unique(it), [0, 1]).all():
        raise ValueError(f"{path}: is_target must be strictly 0/1, got {np.unique(it)[:8]}")
    if not np.isin(np.unique(y), np.arange(lg.shape[1])).all():
        raise ValueError(f"{path}: labels outside [0,{lg.shape[1]})")
    return lg.astype(np.float32), y, comp.astype(np.int64), it.astype(bool)


def _prep_arrays(lg, y, comp):
    order = np.argsort(comp, kind="stable")                # per-component pixel indices in ONE O(n log n) pass
    cs = comp[order]
    uniq, starts = np.unique(cs, return_index=True)
    ends = np.append(starts[1:], len(cs))
    cidx = {int(uniq[i]): order[starts[i]:ends[i]] for i in range(len(uniq))}
    return dict(lg=lg, y=y, cidx=cidx)


def _init_worker(paths):
    """Portable (fork/forkserver/spawn): each worker loads the dumps itself and caches the shared unique
    dark/bright component ids. Assumes main() already validated alignment across dumps."""
    global _DATA, _DARK, _BRIGHT
    _DATA = []
    dark = bright = None
    for p in paths:
        lg, y, comp, it = _load_raw(p)
        _DATA.append(_prep_arrays(lg, y, comp))
        if dark is None:
            uc = np.unique(comp)
            frac = {int(c): float(it[comp == c].mean()) for c in uc}
            dark = np.array([c for c in uc if frac[c] == 0.0], dtype=np.int64)
            bright = np.array([c for c in uc if frac[c] == 1.0], dtype=np.int64)
    _DARK, _BRIGHT = dark, bright


def gather(ids, cidx, lg, y):
    """Pixels of the (possibly duplicated) component ids; each drawn instance gets a fresh unique group id so
    bootstrap-duplicated components count as separate exchangeable units."""
    if len(ids) == 0:
        raise ValueError("empty component bootstrap sample")
    idxs, gs = [], []
    for k, c in enumerate(ids):
        ii = cidx[int(c)]
        idxs.append(ii); gs.append(np.full(len(ii), k, dtype=np.int32))
    ii = np.concatenate(idxs)
    return lg[ii], y[ii], np.concatenate(gs)


def _cell(dd, tb, cb, bb):
    """One (model seed, split) cell: fit temperature on the temp cohort, select the CRC threshold on the
    calib cohort, read the CERTIFIED component-equal triple on the bright eval cohort. Returns (joint%,
    coverage%, feasible)."""
    lg, y, cidx = dd["lg"], dd["y"], dd["cidx"]
    xt, yt, _ = gather(tb, cidx, lg, y)
    T = fit_temperature(xt, yt)
    require_finite("temperature", [T])
    xc, yc, gc = gather(cb, cidx, lg, y)
    pc = softmax(xc / T, axis=1); require_finite("calib prob", pc)
    corr_c = (pc.argmax(1) == yc).astype(int)
    xe, ye, ge = gather(bb, cidx, lg, y)
    pe = softmax(xe / T, axis=1); require_finite("eval prob", pe)
    corr_e = (pe.argmax(1) == ye).astype(int)
    crc = conformal_risk_control(corr_c, pc.max(1), corr_e, pe.max(1), alpha=ALPHA,
                                 calib_group=gc, eval_group=ge)
    if not crc["feasible"]:
        return np.nan, crc["eval_group_coverage"] * 100.0, False
    return crc["eval_group_joint_risk"] * 100.0, crc["eval_group_coverage"] * 100.0, True


def _split(rng, dark):
    nt = max(2, int(round(len(dark) * 0.4)))               # temp / calib (temp fit disjoint from CRC calib)
    perm = rng.permutation(dark)
    temp_c, calib_c = perm[:nt], perm[nt:]
    tb = rng.choice(temp_c, len(temp_c), replace=True)     # bootstrap each set's composition (no shared original)
    cb = rng.choice(calib_c, len(calib_c), replace=True)
    return tb, cb


def one_replicate(task):
    """One nested bootstrap replicate: a SHARED bright + dark scene cohort across all model seeds, M random
    splits, and the AGGREGATE (seed x split) mean joint risk -- the estimator the paper reports."""
    seed_seq, M = task
    rng = np.random.default_rng(seed_seq)
    bb = rng.choice(_BRIGHT, len(_BRIGHT), replace=True)    # shared target bootstrap for all model seeds
    joints, covs, feas = [], [], []
    for _ in range(M):
        tb, cb = _split(rng, _DARK)                        # shared dark split for all model seeds this sub-draw
        for dd in _DATA:
            j, c, f = _cell(dd, tb, cb, bb)
            joints.append(j); covs.append(c); feas.append(1.0 if f else 0.0)
    joints = np.asarray(joints, float)
    theta = float(np.nanmean(joints)) if np.isfinite(joints).any() else np.nan
    return theta, float(np.mean(covs)), float(np.mean(feas))


def point_estimate(M0=10):
    """The un-bootstrapped aggregate estimate: mean over seeds x M0 fixed splits of the joint risk on the
    FULL (non-resampled) components -- the reference the bootstrap CI is a sampling distribution FOR."""
    js = []
    for sp in range(M0):
        rng = np.random.default_rng((BASE_SEED, 7777, sp))
        nt = max(2, int(round(len(_DARK) * 0.4)))
        perm = rng.permutation(_DARK)
        temp_c, calib_c = perm[:nt], perm[nt:]
        for dd in _DATA:
            j, _, f = _cell(dd, temp_c, calib_c, _BRIGHT)  # full cohorts, no composition bootstrap
            if f:
                js.append(j)
    return float(np.mean(js)) if js else np.nan


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--splits-per-rep", type=int, default=1)      # M: random calib splits averaged per replicate (M=1 is conservative on the split axis)
    ap.add_argument("--m0", type=int, default=5)                  # fixed splits behind the un-bootstrapped point estimate
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8R9_surface_nested_boot"))
    args = ap.parse_args()

    # ---- fail-closed dump discovery + cross-seed ALIGNMENT ----
    paths = sorted(glob.glob(os.path.join(DUMP_DIR, "*.npz")), key=parse_seed)
    if len(paths) < 3:
        raise RuntimeError(f"need >= 3 surface dumps, found {len(paths)} in {DUMP_DIR}")
    seed_ids = [parse_seed(p) for p in paths]
    if len(set(seed_ids)) != len(seed_ids):
        raise RuntimeError(f"duplicate seed dumps: {seed_ids}")
    ref = None
    for p in paths:
        lg, y, comp, it = _load_raw(p)
        if ref is None:
            ref = (y, comp, it)
            uc = np.unique(comp)
            frac = {int(c): float(it[comp == c].mean()) for c in uc}
            n_dark = int(sum(frac[int(c)] == 0.0 for c in uc))
            n_bright = int(sum(frac[int(c)] == 1.0 for c in uc))
            n_mixed = int(len(uc) - n_dark - n_bright)
        else:
            np.testing.assert_array_equal(y, ref[0], err_msg=f"{p}: y mis-aligned across dumps")
            np.testing.assert_array_equal(comp, ref[1], err_msg=f"{p}: comp mis-aligned across dumps")
            np.testing.assert_array_equal(it, ref[2], err_msg=f"{p}: is_target mis-aligned across dumps")
    try:
        ncpu = len(os.sched_getaffinity(0))                # process CPU-affinity mask (not a full CFS-quota guarantee)
    except (AttributeError, OSError):
        ncpu = min(8, os.cpu_count() or 8)
    workers = max(1, min(ncpu - 1, args.B))
    print(f"surface nested bootstrap (CORRECTED aggregate estimator): {len(paths)} dumps seeds {seed_ids}; "
          f"components dark(pure) {n_dark}, bright(pure) {n_bright}, mixed-excluded {n_mixed}; "
          f"B={args.B} x {args.splits_per_rep} splits/rep, {len(paths)} seeds/cell, {workers} cores", flush=True)

    _init_worker(paths)                                    # load + prep ONCE in main (validated above)
    theta_hat = point_estimate(args.m0)

    child = np.random.SeedSequence(BASE_SEED).spawn(args.B)
    tasks = [(c, args.splits_per_rep) for c in child]
    if sys.platform.startswith("linux"):
        ctx, pool_kw = mp.get_context("fork"), {}          # workers inherit _DATA/_DARK/_BRIGHT via COW -- no re-load
    else:
        ctx = mp.get_context("spawn")                      # portable fallback: each worker loads the dumps itself
        pool_kw = dict(initializer=_init_worker, initargs=(paths,))
    with ctx.Pool(workers, **pool_kw) as pool:
        res = pool.map(one_replicate, tasks, chunksize=max(1, len(tasks) // (workers * 4)))
    theta = np.array([r[0] for r in res], float)
    cov = np.array([r[1] for r in res], float)
    feas = np.array([r[2] for r in res], float)
    ok = np.isfinite(theta)
    if ok.sum() < 0.9 * len(theta):
        print(f"  WARNING: {(~ok).sum()}/{len(theta)} replicates had no feasible cell (dropped)", flush=True)
    theta = theta[ok]

    lo, hi = np.percentile(theta, [2.5, 97.5])
    blo, bhi = 2 * theta_hat - hi, 2 * theta_hat - lo      # basic (reverse-percentile) CI
    tgt = ALPHA * 100
    rel, verdict = ci_relation(lo, hi, tgt)
    print(f"\n  point estimate (un-bootstrapped seed x split mean) theta_hat = {theta_hat:.2f}%", flush=True)
    print(f"  bootstrap mean {theta.mean():.2f}%  bias {theta.mean() - theta_hat:+.2f}  "
          f"mean coverage {cov.mean():.0f}%  CRC-feasible {feas.mean() * 100:.0f}% of cells", flush=True)
    print(f"  percentile 95% CI [{lo:.1f}, {hi:.1f}]   basic 95% CI [{blo:.1f}, {bhi:.1f}]", flush=True)
    print(f"  -> relation to {tgt:.0f}% target: {rel}; surface verdict: {verdict}", flush=True)

    try:
        commit = subprocess.check_output(["git", "-C", _HERE, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    summary = dict(git_commit=commit, base_seed=BASE_SEED, B=int(args.B), splits_per_rep=int(args.splits_per_rep),
                   model_seed_ids=seed_ids, n_dark=n_dark, n_bright=n_bright, n_mixed_excluded=n_mixed,
                   theta_hat=theta_hat, bootstrap_mean=float(theta.mean()), bias=float(theta.mean() - theta_hat),
                   mean_coverage=float(cov.mean()), feasible_rate=float(feas.mean()),
                   percentile_ci=[float(lo), float(hi)], basic_ci=[float(blo), float(bhi)],
                   target=tgt, relation=rel, n_feasible_replicates=int(ok.sum()))
    with open(args.out + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(args.out + "_draws.npy", theta)
    print(f"  wrote {args.out}_summary.json + _draws.npy", flush=True)


if __name__ == "__main__":
    main()
