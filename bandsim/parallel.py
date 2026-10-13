"""bandsim.parallel — fan independent per-seed (per-job) work across all GPUs + CPU cores.

Why
---
Every experiment here is many INDEPENDENT small jobs: `for seed in seeds: run_seed(seed)`,
and within a phase several such phases. Each job is tiny (a <100k-param model on a few
thousand pixels) and cannot by itself saturate even one GPU. The throughput win is to run
the jobs CONCURRENTLY — spread across every visible GPU (round-robin, oversubscribed) and the
available CPU cores — instead of one after another.

`run_jobs(target, items, shared=...)` replaces a serial `[target(it, **shared) for it in
items]` loop with a process pool. It is a drop-in: results are returned in item order, so
the caller's downstream aggregation is unchanged.

Determinism
-----------
Each job is fully self-seeded, so its result does not depend on which worker process or which
(identical) GPU runs it. On the CPU it CAN depend on the per-worker intra-op THREAD budget, which
the plan derives from the worker count: the reduction order of CPU kernels -- hence the last bits
of CPU training and inference -- changes with the thread count, so a different --jobs /
BANDSIM_THREADS / campaign size can give different, equally valid numbers (pass `threads=`, or set
BANDSIM_THREADS, to hold it fixed). And at two CPU threads per worker (the only count above one
measured) a fixed budget was still not enough for bitwise repeatability: measured 2026-10-13 (torch
2.6, oneMKL 2024.2), a synthetic 6-cell CPU campaign run twice (once per-cell, once two-stage)
disagreed in 14 of 160 repetitions over five batches (0 to 7 per batch of 20 or 40), and six runs
of a small campaign on real CloudSEN12 pixels disagreed once. Wherever the differing fields were
recorded they were b2 rows: the fitted temperatures and the CRC thresholds derived from them (in
the one synthetic mismatch kept at full precision, by 1e-9..4e-9 and up to 2.6e-8, relative) and,
in the real-data one, also AURC/AUROC -- at temperature 1 too (up to 1.9e-7 relative), which
depends on the logits alone, so there the b2 logits themselves had moved (by an amount not
recorded). The mechanism was not isolated -- oneMKL's CNR mode (MKL_CBWR=AUTO) did not remove it (2
of 40) -- while at one thread per worker the same synthetic comparison was exact in 120 of 120
repetitions, so exact CPU comparisons are made with ONE thread per worker (scripts/verify_h10.py
cpu-repeat). On CUDA with deterministic=True (hw.setup forces the math SDPA kernel and
deterministic algorithms, warn-only), on the 4x P100 box (2026-10-13), the flagship's model seed 0
trained bit-identically on each of the four GPUs at 1 and at 4 intra-op threads, in separate
processes and twice within one (experiments/H10_TRAIN_ONCE_VERIFICATION.md); an op without a
deterministic kernel would only warn -- none did -- so these are observations on the current models
and stack, not a guarantee. Force serial with BANDSIM_WORKERS=1 for the reference path.

This module keeps only the stdlib at import time (no torch/numpy), so importing it loads neither. A
spawned worker, however, first re-imports the parent's main script, which in the experiment scripts loads
numpy and torch before the worker initializer runs. CUDA_VISIBLE_DEVICES still takes effect: the CUDA
driver reads it at the process's first CUDA call (even torch.cuda.is_available() makes one), and nothing
in those imports makes one -- the initializer has to stay ahead of any. hw.setup then sets torch's
intra-op threads. numpy's OpenBLAS pool, however, was sized when numpy loaded, from the environment the
parent passed down: OPENBLAS_NUM_THREADS (or OpenBLAS's other overrides, OPENBLAS_DEFAULT_NUM_THREADS and
the legacy GOTO_NUM_THREADS), else OMP_NUM_THREADS -- which an experiment script's own hw.setup sets, if
unset, to the parent's thread count (an exported BANDSIM_THREADS, else all cores) -- else all cores. So
it follows the parent, not the worker's budget: 8 threads in a "1-thread" worker on the P100 box
(2026-10-13). That is an open issue, not something this module pins.

Environment overrides
---------------------
  BANDSIM_WORKERS       int    force the number of concurrent workers (1 = serial)
  BANDSIM_DEVICE        str    cpu | cuda | auto  (device preference; default auto)
  BANDSIM_GPU_OVERSUB   int    workers per GPU when on CUDA. Default is SIZE-AWARE (hw.gpu_oversub_default):
                               <2*budget->1 (OOM-safe), <8*budget->2 (V100-32/A100-40), else 3
                               (A100-80/H100), in decimal GB. Set N to force any size.
  BANDSIM_PER_WORKER_GB float  per-worker VRAM budget (decimal GB) the size-aware default buckets against
                               (default 8, or the per_worker_gb a caller passes for heavier jobs); when
                               set it replaces BOTH -- including a script's own larger budget (phase8R2's
                               evaluation asks for 11), so a smaller value can run that out of memory.
                               Raise it for heavier jobs (fewer workers/GPU; inf = one per GPU), lower it
                               for tiny ones. NaN, 0, negatives and non-numbers are ignored.
  BANDSIM_RESERVE_CORES int    cores held back from workers for the OS (default 1)
  BANDSIM_THREADS       int    torch intra-op threads per worker (default cores//workers). numpy's
                               OpenBLAS pool in a spawned worker follows the parent's environment
                               instead (an exported value reaches it through the parent's hw.setup;
                               the per-worker budget does not -- see above)
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
    def __init__(self, workers, device_mode, n_gpus, oversub, threads_per_worker, gpu_gb=None,
                 oversub_forced=False, budget_gb=None, workers_forced=False):
        self.workers = workers                 # number of concurrent worker processes
        self.device_mode = device_mode         # "cuda" or "cpu"
        self.n_gpus = n_gpus
        self.oversub = oversub                 # workers per GPU the plan allows (CUDA)
        self.threads_per_worker = threads_per_worker
        self.gpu_gb = list(gpu_gb or [])       # total memory (GB) per visible GPU (for the size-aware plan)
        self.oversub_forced = oversub_forced   # True when BANDSIM_GPU_OVERSUB set workers-per-GPU
        self.budget_gb = budget_gb             # per-worker VRAM budget the size-aware default used
        self.workers_forced = workers_forced   # True when jobs= / BANDSIM_WORKERS set the worker count

    @property
    def serial(self):
        return self.workers <= 1

    @property
    def per_gpu(self):
        """Most workers sharing one GPU under the round-robin GPU assignment (0 on the CPU)."""
        if self.device_mode != "cuda" or not self.n_gpus:
            return 0
        return -(-self.workers // self.n_gpus)

    @property
    def over_budget(self):
        """A forced worker count that puts more workers on a GPU than the plan's workers-per-GPU -- the
        size-aware one, or the one BANDSIM_GPU_OVERSUB forces, which a forced worker count overrides."""
        return self.device_mode == "cuda" and self.workers_forced and self.per_gpu > self.oversub

    def _limit(self):
        """What set the workers-per-GPU a forced worker count is held against (describe(), run_jobs)."""
        if self.oversub_forced:
            return "BANDSIM_GPU_OVERSUB sets"
        if not self.gpu_gb:
            return "used when GPU memory is unknown"
        if self.budget_gb is not None:
            return f"its {self.budget_gb:g}GB/worker budget allows"
        return "the size-aware plan allows"

    def describe(self):
        if self.serial:
            return f"serial (1 worker, {self.device_mode}, {self.threads_per_worker} threads)"
        if self.device_mode == "cuda":
            gb = self.gpu_gb
            memnote = (f", {gb[0]:.0f}GB" if gb and len({round(g) for g in gb}) == 1
                       else (f", {min(gb):.0f}-{max(gb):.0f}GB" if gb else ""))
            if self.oversub_forced:
                how = f"forced {self.oversub}/GPU by BANDSIM_GPU_OVERSUB"
            elif not gb:
                how = f"{self.oversub}/GPU, GPU memory unknown"
            else:
                how = f"size-aware ~{self.oversub}/GPU" + (f" at {self.budget_gb:g}GB/worker"
                                                            if self.budget_gb is not None else "")
            if self.workers_forced:
                how += f"; worker count forced by jobs=/BANDSIM_WORKERS: up to {self.per_gpu}/GPU"
                if self.over_budget:
                    how += f", MORE than the {self.oversub}/GPU {self._limit()}"
            return (f"{self.workers} workers over {self.n_gpus} GPU(s){memnote} "
                    f"({how}), {self.threads_per_worker} threads each")
        return f"{self.workers} CPU workers, {self.threads_per_worker} threads each"


def _int_env(name, default):
    v = os.environ.get(name)
    if v:
        try:
            return int(v)
        except ValueError:
            pass
    return default


def plan_jobs(n_items, prefer=None, jobs=None, threads=None, per_worker_gb=None):
    """Decide how many workers, on what devices, with how many threads each.

    Detection (cores, GPUs) is delegated to bandsim.hw, imported lazily so this module stays
    torch-free at top level. `per_worker_gb` is the caller's per-worker VRAM budget for the size-aware
    workers-per-GPU default (hw.gpu_oversub_default); None keeps the default budget.
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
    #                                                       so the two paths use the same count, which
    #                                                       their equality needs (inferred, not compared
    #                                                       directly: see "Determinism" for what was
    #                                                       observed on CUDA and on the CPU).
    forced = _int_env("BANDSIM_WORKERS", jobs if jobs is not None else 0)

    if per_worker_gb is not None:
        hw.per_worker_budget_gb(per_worker_gb)            # a NaN / non-positive budget is a caller bug
    oversub_forced, budget, mem = False, None, None
    if device_mode == "cuda":
        # oversub PER GPU: size-aware by default (small GPU -> 1 to avoid OOM, V100/32G -> 2 as
        # measured optimal here, big GPU -> a little more), overridable with BANDSIM_GPU_OVERSUB.
        mem = hw.gpu_mem_gb()                             # read ONCE: the plan and its description agree
        env_over = _int_env("BANDSIM_GPU_OVERSUB", 0)     # 0/unset => auto (hw.gpu_oversub_default)
        if env_over > 0:
            oversub, oversub_forced = env_over, True
        elif mem:
            budget = hw.per_worker_budget_gb(per_worker_gb)
            oversub = max(1, hw.gpu_oversub_default(per_worker_gb=per_worker_gb, mem=mem))
        else:                                             # memory unreadable: 1 per GPU, and no budget
            oversub = 1                                   # took part in that (gpu_oversub_default's rule)
        default_workers = min(n_items, max(1, ng * oversub))
    else:
        oversub = 0
        # a couple of torch intra-op threads per CPU worker, the rest as workers (numpy's BLAS pool in a
        # spawned worker follows the parent instead -- see the module docstring)
        tpw_guess = threads if threads else (2 if cores >= 4 else 1)
        default_workers = min(n_items, max(1, cores // max(1, tpw_guess)))

    workers = forced if forced > 0 else default_workers
    workers = max(1, min(workers, n_items))

    if threads:
        tpw = max(1, int(threads))
    else:
        tpw = max(1, cores // workers)
    return Plan(workers, device_mode, ng, oversub, tpw, gpu_gb=mem,
                oversub_forced=oversub_forced, budget_gb=budget, workers_forced=forced > 0)


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
    """Pool initializer: claim a worker ordinal, pin a GPU (or hide GPUs) + thread budget via env
    vars before this child's first CUDA call -- which is what CUDA_VISIBLE_DEVICES needs; torch itself
    may already be imported (see the module docstring) -- then configure the process (threads +
    determinism) exactly like the serial main would via hw.setup()."""
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
    # env is pinned (torch may already be imported, but no CUDA call has been made); now apply the
    # SAME process config the serial path uses: intra-op threads + determinism flags. Per-job
    # seeding happens inside the target (train_* reseed), so seed here is a harmless placeholder.
    from . import hw
    hw.setup(seed=0, deterministic=deterministic, threads=threads)


def _release_cuda_cache():
    """Hand this process's cached-but-unused CUDA blocks back to the driver at a JOB BOUNDARY.

    WHY. A pool worker runs many jobs, and PyTorch's caching allocator keeps the blocks it frees reserved
    for the process, so a worker's footprint grew over its first jobs: on 16 GiB P100s (nvidia-smi, one
    flagship worker per GPU, 2026-10-13) two workers went 5,478 -> 6,986 -> 8,526 MiB over their first
    three jobs (the other two stayed at 4,526), and over a 100-cell campaign every worker held 7,038 MiB
    -- against 5,478 MiB for a single job run alone. Two workers at 8,526 MiB would hold more than the
    card's 16,384. Releasing at the job boundary bounds a worker's footprint by ONE job's peak plus its
    CUDA context: the full 10x10 flagship at two workers per GPU then peaked at no more than 11,574 MiB
    per card in four runs. Pure memory bookkeeping: no tensor value or RNG state is touched, and it
    only ever makes MORE memory available, so it could change a kernel choice only where a workspace
    allocation would otherwise have failed (the flagship's 10x10 CSVs were byte-identical with and
    without it).

    torch.cuda.empty_cache() is itself a no-op until the process has initialised CUDA; the explicit
    is_initialized() check only keeps that true whatever the torch version, and reading sys.modules
    (rather than importing) keeps this module torch-free in a process that never imported torch.
    Errors are NOT swallowed -- see _call_releasing."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _call_releasing(fn, item, shared):
    """Run one job, then release the CUDA cache (_release_cuda_cache) -- on success AND on failure, so
    the next job of this worker starts from a clean pool (and, for a serial run in the caller's process,
    a following stage's workers do not share the GPU with this one's cache).

    After a SUCCESSFUL job a release error propagates: the device is then in a bad state. After a
    FAILED job the job's own error is the one re-raised, carrying any release error as a note, so the
    root cause is never replaced by a secondary CUDA error. (On that path the failing job's frames are
    still referenced by its traceback, so only what is already unreferenced can be released.)"""
    try:
        result = fn(item, **shared)
    except BaseException as job_error:
        try:
            _release_cuda_cache()
        except Exception as release_error:
            job_error.add_note(f"[parallel] releasing the CUDA cache after this failure also raised "
                               f"{release_error!r}")
        raise
    _release_cuda_cache()
    return result


def _run_one(path, name, item, shared):
    """Worker entry: import the target module by path, resolve `name`, and run the job through
    _call_releasing, so the NEXT job in this worker starts from a clean CUDA pool."""
    return _call_releasing(getattr(_import_by_path(path), name), item, shared)


# =======================================================================================
# driver
# =======================================================================================
def run_jobs(target, items, shared=None, prefer=None, jobs=None, threads=None,
             deterministic=True, log=print, label="job", per_worker_gb=None, on_plan=None):
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
    per_worker_gb: VRAM budget of ONE job, in decimal GB, for the size-aware workers-per-GPU default
             (see hw.gpu_oversub_default); None keeps the default. It changes how many workers share a
             GPU and -- unless `threads` is given -- therefore each worker's CPU thread budget, so it
             changes no number only where the jobs' results do not depend on that budget.
    on_plan: optional callable, called once with the Plan this call actually runs, before any job --
             so a caller can record it (e.g. in provenance) rather than re-plan and hope it matches.

    Returns
    -------
    list of results, one per item, in the same order as `items`.
    """
    shared = dict(shared or {})
    items = list(items)
    plan = plan_jobs(len(items), prefer=prefer, jobs=jobs, threads=threads, per_worker_gb=per_worker_gb)
    if on_plan is not None:
        on_plan(plan)
    if plan.over_budget and log:
        log(f"[parallel] WARNING {label}: {plan.workers} forced workers on {plan.n_gpus} GPU(s) put up to "
            f"{plan.per_gpu} on one GPU, above the {plan.oversub}/GPU {plan._limit()} -- this can run out "
            f"of GPU memory")

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
        # Released per job here too: a serial stage runs in the CALLER's process, which would otherwise
        # keep that stage's cached GPU memory while a following pooled stage's workers share its GPU.
        return [_call_releasing(target, it, shared) for it in items]

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
