"""bandsim.parallel — fan independent per-seed (per-job) work across all GPUs + CPU cores.

Why
---
Every experiment here is many INDEPENDENT small jobs: `for seed in seeds: run_seed(seed)`,
and within a phase several such phases. Each job is tiny (a <100k-param model on a few
thousand pixels) and cannot by itself saturate even one GPU. The throughput win is to run
the jobs CONCURRENTLY — spread across the 2 V100s (round-robin, oversubscribed) and the
available CPU cores — instead of one after another.

`run_jobs(target, items, shared=...)` replaces a serial `[target(it, **shared) for it in
items]` loop with a process pool. It is a drop-in: results are returned in item order, so
the caller's downstream aggregation is unchanged.

Determinism
-----------
Each job is fully self-seeded and runs in its own process, so the result of item i does NOT
depend on how many workers run or which GPU it lands on (the 2 V100s are identical). Running
parallel therefore produces the same aggregate output as running serial — with one
pre-existing caveat: torch's GPU multi-head-attention backward has no deterministic kernel,
so the attention models already vary run-to-run on GPU whether serial or parallel (the CPU
path is deterministic). Force serial with BANDSIM_WORKERS=1 for the reference path.

This module keeps only the stdlib at import time (no torch/numpy) so the worker initializer
can pin CUDA_VISIBLE_DEVICES / thread env BEFORE torch is imported in the child.

Environment overrides
---------------------
  BANDSIM_WORKERS       int    force the number of concurrent workers (1 = serial)
  BANDSIM_DEVICE        str    cpu | cuda | auto  (device preference; default auto)
  BANDSIM_GPU_OVERSUB   int    workers per GPU when on CUDA (default 2)
  BANDSIM_THREADS       int    intra-op/BLAS threads per worker (default cores//workers)
"""
from __future__ import annotations
import os
import sys
import inspect
import importlib.util
from concurrent.futures import ProcessPoolExecutor


# =======================================================================================
# planning
# =======================================================================================
class Plan:
    """Resolved fan-out plan for a set of jobs."""
    def __init__(self, workers, device_mode, n_gpus, oversub, threads_per_worker):
        self.workers = workers                 # number of concurrent worker processes
        self.device_mode = device_mode         # "cuda" or "cpu"
        self.n_gpus = n_gpus
        self.oversub = oversub
        self.threads_per_worker = threads_per_worker

    @property
    def serial(self):
        return self.workers <= 1

    def describe(self):
        if self.serial:
            return f"serial (1 worker, {self.device_mode}, {self.threads_per_worker} threads)"
        if self.device_mode == "cuda":
            return (f"{self.workers} workers over {self.n_gpus} GPU(s) "
                    f"(~{self.oversub}/GPU), {self.threads_per_worker} threads each")
        return f"{self.workers} CPU workers, {self.threads_per_worker} threads each"


def _int_env(name, default):
    v = os.environ.get(name)
    if v:
        try:
            return int(v)
        except ValueError:
            pass
    return default


def plan_jobs(n_items, prefer=None, jobs=None, threads=None):
    """Decide how many workers, on what devices, with how many threads each.

    Detection (cores, GPUs) is delegated to bandsim.hw, imported lazily so this module stays
    torch-free at top level.
    """
    from . import hw                                    # lazy: pulls in torch
    cores = hw.available_cores()
    # reserve 1-2 cores for basic OS/daemon operation (user policy); the rest feed workers.
    reserve = max(0, _int_env("BANDSIM_RESERVE_CORES", 1))
    cores = max(1, cores - reserve)
    ng = hw.n_gpus()

    prefer = (prefer or os.environ.get("BANDSIM_DEVICE") or "auto").lower()
    device_mode = "cuda" if (prefer in ("auto", "cuda") or prefer.startswith("cuda")) and ng > 0 else "cpu"

    if threads is None:                                   # BANDSIM_THREADS forces per-worker
        threads = _int_env("BANDSIM_THREADS", 0) or None  # threads for BOTH serial & parallel,
    #                                                       so the two paths match bit-for-bit.
    forced = _int_env("BANDSIM_WORKERS", jobs if jobs is not None else 0)

    if device_mode == "cuda":
        oversub = max(1, _int_env("BANDSIM_GPU_OVERSUB", 2))
        default_workers = min(n_items, max(1, ng * oversub))
    else:
        oversub = 0
        # a couple of threads per CPU worker (numpy BLAS + torch intra-op), the rest as workers
        tpw_guess = threads if threads else (2 if cores >= 4 else 1)
        default_workers = min(n_items, max(1, cores // max(1, tpw_guess)))

    workers = forced if forced > 0 else default_workers
    workers = max(1, min(workers, n_items))

    if threads:
        tpw = max(1, int(threads))
    else:
        tpw = max(1, cores // workers)
    return Plan(workers, device_mode, ng, oversub, tpw)


# =======================================================================================
# worker side
# =======================================================================================
_MODULE_CACHE = {}


def _import_by_path(path):
    """Import a .py file as a module (cached per worker); NOT run as __main__, so the file's
    `if __name__ == '__main__': main()` guard does not fire."""
    path = os.path.abspath(path)
    mod = _MODULE_CACHE.get(path)
    if mod is not None:
        return mod
    modname = "_bandsim_job_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[path] = mod
    return mod


def _init_worker(counter, device_mode, n_gpus, threads, deterministic):
    """Pool initializer: claim a worker ordinal, pin a GPU (or hide GPUs) + thread budget,
    all via env vars BEFORE torch is imported in this child; then configure the process
    (threads + determinism) exactly like the serial main would via hw.setup()."""
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    if device_mode == "cuda" and n_gpus > 0:
        # Map this worker's ordinal onto the PHYSICAL GPU indices the PARENT was restricted to (its
        # inherited CUDA_VISIBLE_DEVICES), not a bare 0-based logical index. A plain str(idx % n_gpus)
        # DISCARDS an externally-set restriction: launched with CUDA_VISIBLE_DEVICES=1 the parent uses
        # physical GPU1, but "0" would send every worker to physical GPU0 (OOM if another job owns it).
        # No external restriction -> `visible` is None and the old 0-based physical assignment stands.
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible = [d for d in inherited.split(",") if d != ""] if inherited != "" else None
        os.environ["CUDA_VISIBLE_DEVICES"] = (visible[idx % len(visible)] if visible
                                              else str(idx % n_gpus))   # worker sees its GPU as cuda:0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                  # CPU worker: hide GPUs
    os.environ["BANDSIM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ.setdefault("BANDSIM_GPU_ENV_QUIET", "1")
    # env is pinned; now import torch (no CUDA context yet) and apply the SAME process config
    # the serial path uses: intra-op threads + determinism flags. Per-job seeding happens
    # inside the target (train_* reseed), so seed here is a harmless placeholder.
    from . import hw
    hw.setup(seed=0, deterministic=deterministic, threads=threads)


def _run_one(path, name, item, shared):
    """Worker entry: import the target module by path, resolve `name`, call it."""
    mod = _import_by_path(path)
    fn = getattr(mod, name)
    return fn(item, **shared)


# =======================================================================================
# driver
# =======================================================================================
def run_jobs(target, items, shared=None, prefer=None, jobs=None, threads=None,
             deterministic=True, log=print, label="job"):
    """Run `target(item, **shared)` for each item, concurrently, results in item order.

    Parameters
    ----------
    target : a MODULE-LEVEL function (so it can be re-imported in a worker). Its first
             positional arg is the item (e.g. a seed); remaining args come from `shared`.
    items  : list of item values (e.g. seeds).
    shared : dict of keyword args passed to every call (e.g. dict(cube=..., gt=..., epochs=...)).
             Must be picklable (numpy arrays are fine).
    prefer : device preference "cpu"|"cuda"|"auto" (default: BANDSIM_DEVICE or auto).
    jobs   : force worker count (overrides the plan; BANDSIM_WORKERS also does).
    threads: force intra-op threads per worker.

    Returns
    -------
    list of results, one per item, in the same order as `items`.
    """
    shared = dict(shared or {})
    items = list(items)
    plan = plan_jobs(len(items), prefer=prefer, jobs=jobs, threads=threads)

    if plan.serial:
        # exact serial path (single process) — reference/deterministic/debug behaviour.
        from . import hw
        if plan.device_mode == "cpu":
            # ensure hw.device() calls INSIDE the target also resolve to CPU (the parallel path
            # hides GPUs via CUDA_VISIBLE_DEVICES in _init_worker; the serial path must too, else
            # a target that calls hw.device() would still pick cuda). Fixes --jobs 1 --device cpu.
            os.environ["BANDSIM_DEVICE"] = "cpu"
        hw.setup(deterministic=deterministic, threads=plan.threads_per_worker,
                 prefer=(prefer or ("cpu" if plan.device_mode == "cpu" else None)))
        if log:
            log(f"[parallel] {label}: {plan.describe()}")
        return [target(it, **shared) for it in items]

    # resolve the target to (file, name) so workers can re-import it without pickling __main__.
    path = inspect.getfile(target)
    name = target.__name__
    if log:
        log(f"[parallel] {label}: {plan.describe()}  ({len(items)} items)")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")                        # spawn: safe with CUDA
    counter = ctx.Value("i", 0)
    results = [None] * len(items)
    with ProcessPoolExecutor(max_workers=plan.workers, mp_context=ctx,
                             initializer=_init_worker,
                             initargs=(counter, plan.device_mode, plan.n_gpus,
                                       plan.threads_per_worker, deterministic)) as ex:
        futs = {ex.submit(_run_one, path, name, it, shared): i for i, it in enumerate(items)}
        done = 0
        for fut in _as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()                    # re-raises worker exceptions with traceback
            done += 1
            if log:
                log(f"[parallel] {label} {done}/{len(items)} done (item={items[i]})")
    return results


def _as_completed(futs):
    from concurrent.futures import as_completed
    return as_completed(futs)
