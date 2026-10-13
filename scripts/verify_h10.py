#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-runnable checks behind the H-10 (train once per model seed) claims.

WHY THIS EXISTS. experiments/phase8R_reliability.py (and phase8R2) train each model seed ONCE and evaluate
it on every split (run_two_stage), where the pre-H-10 code retrained it in every cell. That is a refactor
only if the rows come out the same, and the docstrings that say so cite measurements. This script holds
the measurements that can be re-run from the repository, so they can be repeated on another machine
rather than trusted; experiments/H10_TRAIN_ONCE_VERIFICATION.md records their results on the 4x P100 box
and names the session-only probes that are not in here.

    python scripts/verify_h10.py prep INPUTS.npz                 # the flagship's inputs, as main() builds them
    python scripts/verify_h10.py train-hash --inputs INPUTS.npz  # SHA-256 of a model seed's trained weights
    python scripts/verify_h10.py rows --inputs INPUTS.npz        # one cell: per-cell vs two-stage rows
    python scripts/verify_h10.py ab-run old|new REPO OUT.pkl     # a campaign's rows at FULL precision
    python scripts/verify_h10.py ab-compare A.pkl B.pkl          # exact comparison of two ab-run pickles
    python scripts/verify_h10.py cpu-repeat --threads 2          # CPU repeatability of the 6-cell campaign
    python scripts/verify_h10.py cpu-thread-logits               # CPU logits vs the intra-op thread count
    python scripts/verify_h10.py chunking --inputs INPUTS.npz    # GPU: one forward vs the same rows chunked

train-hash, rows and chunking take --inputs INPUTS.npz or --synthetic N (a generated 13-band problem with
N training pixels); choose the GPU with CUDA_VISIBLE_DEVICES. ab-run runs REPO's own main() (so `old` can
be a worktree of the pre-H-10 code; --script phase8R2 for the land-cover flagship), checks that every
experiments/ and bandsim/ module came from REPO, and pickles the in-memory rows, because the CSVs it also
writes (tag _verifyAB_<mode>) keep 6 significant digits. rows and ab-compare exit non-zero when they find
a difference or compared nothing, train-hash when its in-process repeats (--repeat 2 or more) differ, and
ab-run when its capture did not fire exactly once; at the default --repeat 1 train-hash compares nothing
(repeats_identical: null) and only prints the hashes, for comparing across processes. cpu-repeat,
cpu-thread-logits and chunking are measurements and report what they find.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import pickle
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _paths(repo=_ROOT):
    for p in (repo, os.path.join(repo, "experiments")):     # experiments ends up first, as in the scripts
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _groups():
    import phase8_cloudsen12 as P8
    from bandsim.grouping import group_center_wavelengths
    groups = P8.s2_physical_groups()
    return groups, group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)


def prep(args):
    """phase8R_reliability.main()'s inputs at its default arguments -- the same loader calls and seeds,
    the scene-level leak guard, and the TRAIN-statistics normalisation -- cached, because reading the
    CloudSEN12 memmaps takes minutes."""
    _paths()
    import pandas as pd
    import phase8_cloudsen12 as P8
    import phase8R_reliability as P8R
    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    train_products = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    ids = np.arange(len(meta))
    ids = ids[~meta["s2_id"].isin(train_products).to_numpy()[ids]]
    X_l1c, y_te, pid = P8.load_split("test", "L1C", pixels_per_patch=400, patch_ids=ids, seed=54321,
                                     return_patch_id=True)
    X_l2a, y_l2a = P8.load_split("test", "L2A", pixels_per_patch=400, patch_ids=ids, seed=54321)
    if not np.array_equal(y_te, y_l2a):
        raise ValueError("L1C/L2A pixel misalignment")
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)          # noqa: E731
    # The scene-component ids are strings (an object array, which np.load refuses without pickle). Their
    # integer codes from np.unique keep the sorted order split_test_rois permutes, so the splits are the
    # same ones main() draws, and the file needs no pickle.
    cid = np.unique(P8R.scene_component_ids("test")[pid], return_inverse=True)[1]
    np.savez(args.out, Xtr=norm(Xtr), ytr=ytr, l1c=norm(X_l1c), l2a=norm(X_l2a), yte=y_te, cid=cid)
    print(json.dumps({"wrote": args.out, "train_px": int(Xtr.shape[0]), "test_px": int(X_l1c.shape[0])}))
    return 0


def _inputs(inputs=None, synthetic=None):
    if inputs:
        z = np.load(inputs)
        return (z["Xtr"], z["ytr"], {"l1c": z["l1c"], "l2a": z["l2a"]}, z["yte"], z["cid"],
                f"flagship inputs ({os.path.basename(inputs)})")
    rng = np.random.default_rng(0)                    # the generator tests/test_h10_train_once.py uses
    W = rng.standard_normal((13, 4))

    def draw(n):
        X = rng.standard_normal((n, 13)).astype(np.float32)
        return X, (X @ W + 0.8 * rng.standard_normal((n, 4))).argmax(1)
    Xtr, ytr = draw(synthetic)
    X1, yte = draw(2400)
    return (Xtr, ytr, {"l1c": X1, "l2a": (X1 * 1.15 + 0.3).astype(np.float32)}, yte,
            np.repeat(np.arange(24), 100), f"synthetic ({synthetic} training px)")


def _setup(threads, need_cuda=False):
    _paths()
    import torch
    from bandsim import hw
    hw.setup(deterministic=True, threads=threads)
    dev = hw.device()
    if need_cuda and dev.type != "cuda":
        raise SystemExit("this check needs a CUDA device")
    groups, cwl = _groups()
    where = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
             "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "threads": torch.get_num_threads(), "torch": torch.__version__}
    return groups, cwl, where


def _sha(model):
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_hash(args):
    """Train one model seed with train_models on this device (--repeat times in THIS process) and print
    the full SHA-256 of each model's weights. Equal hashes across GPUs / processes / thread counts /
    repetitions = bit-identical training. Only --repeat 2 or more compares anything here; at 1,
    repeats_identical is null."""
    groups, cwl, where = _setup(args.threads)
    import phase8R_reliability as P8R
    Xtr, ytr, _, _, _, what = _inputs(args.inputs, args.synthetic)
    reps = []
    for _ in range(args.repeat):
        models = P8R.train_models(args.seed, Xtr, ytr, groups, cwl, 0.8, args.epochs)
        reps.append({k: _sha(m) for k, m in models.items()})
    same = all(r == reps[0] for r in reps)
    print(json.dumps(dict(where, inputs=what, seed=args.seed, epochs=args.epochs, sha256=reps[0],
                          repeats_in_process=args.repeat,
                          repeats_identical=same if args.repeat > 1 else None)))
    return 0 if same else 1


def _diff_rows(a, b):
    """Every field of two row lists compared exactly (NaN equals NaN, types must match)."""
    out = []
    if len(a) != len(b):
        out.append(("row count", len(a), len(b)))
    for ra, rb in zip(a, b):
        if ra.keys() != rb.keys():
            out.append(("keys", sorted(ra), sorted(rb)))
            continue
        for k in ra:
            va, vb = ra[k], rb[k]
            if type(va) is not type(vb):
                out.append((ra.get("method"), ra.get("state"), ra.get("arm"), k, "type", repr(va), repr(vb)))
            elif isinstance(va, float) and math.isnan(va) and math.isnan(vb):
                continue
            elif va != vb:
                out.append((ra.get("method"), ra.get("state"), ra.get("arm"), k, va, vb))
    return out


def rows(args):
    """One cell, both protocols, in this process and at full precision: run_seed (train + evaluate) vs
    eval_job against train_job's pickled bytes -- i.e. the real transport (on a GPU: GPU -> CPU -> bytes
    -> GPU) between two trainings of the same seed."""
    groups, cwl, where = _setup(args.threads)
    import phase8R_reliability as P8R
    Xtr, ytr, Xte, yte, cid, what = _inputs(args.inputs, args.synthetic)
    job = (args.split, args.seed)
    per_cell = P8R.run_seed(job, Xtr, ytr, Xte, yte, cid, groups, cwl, 0.8, args.epochs, target=0.1)
    blob = P8R.train_job(args.seed, Xtr, ytr, groups, cwl, 0.8, args.epochs)
    two_stage = P8R.eval_job(job, {args.seed: blob}, Xte, yte, cid, groups, target=0.1)
    diffs = _diff_rows(per_cell, two_stage)
    print(json.dumps(dict(where, inputs=what, cell=list(job), epochs=args.epochs, rows=len(per_cell),
                          fields=sum(len(r) for r in per_cell), differing=len(diffs),
                          first_differences=[list(map(str, d)) for d in diffs[:5]],
                          blob_mb=round(len(blob) / 1e6, 2))))
    return 1 if diffs or not per_cell else 0


def _check_origin(repo):
    """Every loaded bandsim module, and every module loaded straight from an experiments/ directory,
    must come from REPO -- so `ab-run old` cannot quietly run any of the current code."""
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = os.path.abspath(f)
        ours = (name == "bandsim" or name.startswith("bandsim.")
                or os.path.basename(os.path.dirname(f)) == "experiments")
        if ours and not f.startswith(repo + os.sep):
            bad.append(f"{name}: {f}")
    if bad:
        raise SystemExit("modules imported from outside " + repo + ":\n  " + "\n  ".join(bad))


def ab_run(args):
    """Run REPO's main() unchanged and pickle the rows it computes, before any CSV formatting."""
    repo, out = os.path.abspath(args.repo), os.path.abspath(args.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to reuse it (a stale pickle would pass a comparison)")
    _paths(repo)
    os.chdir(repo)
    import phase8R_reliability as P8R
    from bandsim import parallel
    target = P8R
    if args.script == "phase8R2":
        import phase8R2_landcover_reliability as R2
        target = R2
    _check_origin(repo)
    meta = {"mode": args.mode, "script": args.script, "repo": repo, "argv": list(sys.argv),
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    fired = []
    if args.mode == "old":
        if hasattr(P8R, "run_two_stage"):
            raise SystemExit("`old` expects the pre-H-10 (per-cell) code")
        real = parallel.run_jobs

        def capture(fn, items, *a, **k):
            res = real(fn, items, *a, **k)
            if getattr(fn, "__name__", "") == "run_seed":
                fired.append(1)
                with open(out, "wb") as f:
                    pickle.dump(dict(meta, items=list(items), results=res), f)
            return res
        parallel.run_jobs = capture
    else:
        if not hasattr(P8R, "run_two_stage"):
            raise SystemExit("`new` expects the H-10 (two-stage) code")
        real = P8R.run_two_stage

        def capture(eval_fn, jobs, **k):
            res, compute = real(eval_fn, jobs, **k)
            fired.append(1)
            with open(out, "wb") as f:
                pickle.dump(dict(meta, items=list(jobs), results=res, compute=compute), f)
            return res, compute
        P8R.run_two_stage = capture
    sys.argv = [f"{args.script}.py", "--split-seeds", *map(str, args.split_seeds),
                "--model-seeds", *map(str, args.model_seeds), "--out-tag", f"_verifyAB_{args.mode}"]
    if args.script == "phase8R2":
        sys.argv += ["--axis", args.axis]
    target.main()
    _check_origin(repo)
    if len(fired) != 1:
        raise SystemExit(f"the capture fired {len(fired)} times (expected exactly once): nothing comparable written")
    return 0


def ab_compare(args):
    with open(args.a, "rb") as f:
        raw_a = f.read()
    with open(args.b, "rb") as f:
        raw_b = f.read()
    if os.path.realpath(args.a) == os.path.realpath(args.b) or raw_a == raw_b:
        raise SystemExit("A and B are the same run (same file or same bytes): there is nothing to compare")
    a, b = pickle.loads(raw_a), pickle.loads(raw_b)
    if {a.get("mode"), b.get("mode")} != {"old", "new"}:
        raise SystemExit(f"compare one `old` run with one `new` run, got {a.get('mode')!r} and {b.get('mode')!r}")
    if a.get("script") and b.get("script") and a["script"] != b["script"]:
        raise SystemExit(f"runs of different scripts: {a['script']} vs {b['script']}")
    if a["items"] != b["items"]:
        raise SystemExit(f"different cells: {a['items']} vs {b['items']}")
    if len(a["results"]) != len(a["items"]) or len(b["results"]) != len(b["items"]):
        raise SystemExit("a pickle holds a different number of results than cells")
    diffs, n_rows, n_fields = [], 0, 0
    for job, ra, rb in zip(a["items"], a["results"], b["results"]):
        n_rows += len(ra)
        n_fields += sum(len(r) for r in ra)
        diffs += [(tuple(job),) + d for d in _diff_rows(ra, rb)]
    print(json.dumps({"a": {k: a.get(k) for k in ("mode", "script", "repo", "time_utc")},
                      "b": {k: b.get(k) for k in ("mode", "script", "repo", "time_utc")},
                      "cells": len(a["items"]), "rows": n_rows, "fields": n_fields,
                      "differing_at_full_precision": len(diffs),
                      "first_differences": [list(map(str, d)) for d in diffs[:10]],
                      "compute": b.get("compute") or a.get("compute")}))
    return 1 if diffs or n_fields == 0 else 0


def cpu_repeat(args):
    """CPU repeatability: the 6-cell campaign (split seeds 0-2 x model seeds 0-1, synthetic 3000-px problem,
    2 epochs) run per-cell (run_seed through the pool) and two-stage (run_two_stage), both at --threads per
    worker and 3 workers, --repeat times; reports how many repetitions disagree at full precision."""
    os.environ["BANDSIM_DEVICE"] = "cpu"
    _paths()
    import phase8R_reliability as P8R
    from bandsim import parallel
    groups, cwl = _groups()
    Xtr, ytr, Xte, yte, cid, what = _inputs(synthetic=3000)
    cells = [(ss, ms) for ss in (0, 1, 2) for ms in (0, 1)]
    tr = dict(Xtr=Xtr, ytr=ytr, groups=groups, cwl=cwl, subsample_frac=0.8, epochs=2)
    per_rep = []
    for _ in range(args.repeat):
        a = parallel.run_jobs(P8R.run_seed, cells, shared=dict(tr, Xte=Xte, yte=yte, rid_te=cid, target=0.1),
                              jobs=3, threads=args.threads, log=None)
        b, _ = P8R.run_two_stage(P8R.eval_job, cells, train_shared=tr,
                                 eval_shared=dict(Xte=Xte, yte=yte, rid_te=cid, groups=groups, target=0.1),
                                 n_jobs=3, threads=args.threads, log=None)
        per_rep.append(sum(len(_diff_rows(x, y)) for x, y in zip(a, b)))
    print(json.dumps({"inputs": what, "threads_per_worker": args.threads, "MKL_CBWR": os.environ.get("MKL_CBWR"),
                      "repetitions": args.repeat, "repetitions_differing": sum(1 for n in per_rep if n),
                      "differing_fields_per_repetition": per_rep}))
    return 0


def cpu_thread_logits(args):
    """How far CPU logits move with the intra-op thread count: one model per kind, trained at one thread on
    the synthetic problem, then logits_at on --rows standard-normal pixels at each thread count; the
    largest absolute difference from the logits at the first count. The answer depends on --rows: on
    the P100 box's 8 vCPUs nothing moved at 2,400 or 60,000 rows."""
    os.environ["BANDSIM_DEVICE"] = "cpu"
    _paths()
    import torch
    from bandsim import hw
    hw.setup(deterministic=True, threads=1, prefer="cpu")
    import phase8R_reliability as P8R
    groups, cwl = _groups()
    Xtr, ytr, _, _, _, what = _inputs(synthetic=3000)
    models = P8R.train_models(0, Xtr, ytr, groups, cwl, 0.8, 2)
    X = np.random.default_rng(1).standard_normal((args.rows, 13)).astype(np.float32)
    out = {}
    for kind, m in models.items():
        ref = None
        for t in args.thread_counts:
            torch.set_num_threads(t)
            lg = P8R.logits_at(kind, m, X, groups, [])
            ref = lg if ref is None else ref
            out[f"{kind}@{t}"] = float(np.abs(lg.astype(np.float64) - ref).max())
    print(json.dumps({"models_trained_on": what, "rows": args.rows, "thread_counts": args.thread_counts,
                      "max_abs_logit_diff_vs_first_thread_count": out}))
    return 0


def chunking(args):
    """GPU: logits of one forward over all evaluation rows vs the same rows in chunks (largest absolute
    logit difference per chunk size). Chunking would bound phase8R2's evaluation peak, but only if it left
    the numbers alone."""
    groups, cwl, where = _setup(args.threads, need_cuda=True)
    import phase8R_reliability as P8R
    Xtr, ytr, Xte, _, _, what = _inputs(args.inputs, args.synthetic)
    models = P8R.train_models(args.seed, Xtr, ytr, groups, cwl, 0.8, args.epochs)
    X = Xte["l1c"]
    out = {}
    for kind, m in models.items():
        whole = P8R.logits_at(kind, m, X, groups, [])
        for c in args.chunks:
            parts = np.concatenate([P8R.logits_at(kind, m, X[i:i + c], groups, []) for i in range(0, len(X), c)])
            out[f"{kind}@{c}"] = {"bitwise_equal": bool(np.array_equal(whole, parts)),
                                  "max_abs_logit_diff": float(np.abs(whole.astype(np.float64) - parts).max())}
    print(json.dumps(dict(where, inputs=what, rows=int(len(X)), chunks=out)))
    return 0


def _positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("out")
    for name in ("train-hash", "rows", "chunking"):
        p = sub.add_parser(name)
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--inputs")
        src.add_argument("--synthetic", type=_positive)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--epochs", type=_positive, default=40)
        p.add_argument("--threads", type=_positive, default=None)
        if name == "train-hash":
            p.add_argument("--repeat", type=_positive, default=1)
        if name == "rows":
            p.add_argument("--split", type=int, default=0)
        if name == "chunking":
            p.add_argument("--chunks", type=_positive, nargs="+", default=[65536, 32768, 16384])
    p = sub.add_parser("ab-run")
    p.add_argument("mode", choices=["old", "new"])
    p.add_argument("repo")
    p.add_argument("out")
    p.add_argument("--script", choices=["phase8R", "phase8R2"], default="phase8R")
    p.add_argument("--axis", choices=["surface", "geography"], default="surface")
    p.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--model-seeds", type=int, nargs="+", default=list(range(10)))
    p = sub.add_parser("ab-compare")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("cpu-repeat")
    p.add_argument("--threads", type=_positive, default=2)
    p.add_argument("--repeat", type=_positive, default=20)
    p = sub.add_parser("cpu-thread-logits")
    p.add_argument("--thread-counts", type=_positive, nargs="+", default=[1, 2, 4])
    p.add_argument("--rows", type=_positive, default=60000)
    args = ap.parse_args()
    return {"prep": prep, "train-hash": train_hash, "rows": rows, "ab-run": ab_run, "ab-compare": ab_compare,
            "cpu-repeat": cpu_repeat, "cpu-thread-logits": cpu_thread_logits, "chunking": chunking}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
