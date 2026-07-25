#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 7 (efficiency) — MEASURED model cost, replacing the paper's unverified "deployment
goals" (Table II) with real numbers. It answers the HALF of the reviewers' critique this script
can actually measure: FLOPs / memory / latency / INT8, on this host, for these four models.

It does NOT measure DISTILLATION. There is no teacher, no student and no KD loss anywhere in
this file — B4/B6 are architecture ablations, not distilled students. The frozen submission's
Table II listed a "distilled student" row as a deployment GOAL (its caption says so); nothing
here upgrades that to a measurement, and the paper must either drop the claim or keep it
explicitly in future work. A script that silently inherits a critique it does not answer is how
"goals" become "results".

MEASUREMENT DISCIPLINE (why this script is shaped the way it is)
----------------------------------------------------------------
A latency table is a physical measurement of a machine, and the first version of this script
was not one: it wrapped `time.time()` around a fixed 100-iteration loop, once, for all four
models in a SINGLE process, on whatever thread count the box happened to give it, and reported
one bare number per cell. Four independent things were wrong with that, and all four are now
closed:

1. TIMING METHOD. `torch.utils.benchmark.Timer` replaces the manual loop. It amortizes timer
   overhead into blocks, synchronizes CUDA around the measurement, pins the thread count for
   the duration, and returns REPLICATES rather than one total. We call `adaptive_autorange`
   (which keeps sampling until iqr/median falls under a threshold) `--repeats` times and pool
   every replicate, so each reported latency is a MEDIAN over many samples and ships with its
   IQR. A single number with no dispersion cannot be checked by anyone; a median+IQR can.

2. PROCESS ISOLATION. Each model is measured in its OWN subprocess (`--worker`). In one
   process, model k's timing inherits model k-1's caching allocator state, cuDNN autotune
   cache, warm cuBLAS handles and warm instruction/data caches — the LAST model benchmarked
   always looks better than the first for reasons that have nothing to do with the model. A
   fresh process gets a fresh CUDA context and a fresh allocator, so the four rows are
   comparable to each other. It also makes peak-memory ATTRIBUTION unambiguous (no other model
   is resident at all, rather than merely evicted to CPU). Note the word: attribution, not
   totality — see `peak_mem_mb` for what the allocator can and cannot see. And note the limit:
   a fresh process resets SOFTWARE state only. GPU temperature, boost-clock residency and page
   cache carry across workers, and the four models are always measured in the same order, so
   the rows are comparable in every respect this script CAN control. `gpu_clock_sm_mhz` /
   `gpu_temp_c` are recorded per worker to make the residual drift auditable rather than assumed
   away.

3. PINNED ENVIRONMENT. CPU thread count and default dtype are pinned explicitly and RECORDED
   (`bench_threads`, `torch_threads`, `interop_threads`, `default_dtype` columns). An unpinned
   CPU latency is a measurement of the box's current load, not of the model, and two runs of it
   are not comparable. In `--strict` we refuse to run when the requested thread count cannot be
   honoured, because that silently changes the meaning of every CPU column.

4. CONTENDED MACHINE. Before doing anything, the script probes GPU utilisation, the number of
   other CUDA compute processes and the CPU load average. If the box is busy, EVERY latency
   number on it is noise, so the run REFUSES to produce paper artefacts unless
   `--allow-contended` is passed explicitly — and even then every row is stamped
   `timing_validity=CONTENDED` so a number measured under load can never be quietly quoted as
   clean. `--smoke` is exempt (it validates the machinery, and its output is already suffixed
   away from the deliverables).

   Three things this gate could NOT previously say, all closed:
     * The CPU signal divided the HOST's load average (there is no cgroup-virtualised
       /proc/loadavg) by OUR cgroup quota. That is a unit error, and it is not academic: on the
       box this was written for it read 7.57/8 = 0.95 "busy" — over the 0.60 threshold, refusing
       every run — while the host was 7.57/36 = 21% busy and the cgroup's own accounting showed
       0.04% of scheduling periods throttled over the container's lifetime. A gate that misfires
       this way does not protect the numbers; it teaches you to pass `--allow-contended`, which
       is the real fail-open. The threshold is now on the host-wide ratio, and BOTH ratios are
       recorded.
     * A signal that could not be MEASURED was treated as a signal that was quiet. It still does
       not refuse on one (it cannot call a box busy on evidence it does not have), but the
       verdict becomes `ok-unverified` rather than `ok`, and which signals were blind travels in
       the provenance. "I checked and it was fine" and "I could not check" are different claims.
     * Nothing checked whether WE got the CPU we pinned. `cgroup_throttle` is sampled before and
       after; if the CFS scheduler throttled us during the run, every CPU latency in it measures
       the quota rather than the model, and that is now a strict failure.

MEASURED PER MODEL (MLP baseline, Proposed grouped cross-band attention, B4/B6 variants)
----------------------------------------------------------------------------------------
  - parameters, fp32 state-dict size (MB)
  - MACs (fvcore) reported ONLY when the count is COMPLETE. fvcore's `.total()` still returns
    a (smaller) number when it silently skips ops it cannot count (e.g. a Transformer's
    scaled_dot_product_attention) or submodules it never traced, so we check
    `unsupported_ops()` / `uncalled_modules()` and report "n/a" (never that undercount, never
    0) whenever either is non-empty; the `macs_note` column records why. Because the fvcore
    total is over the WHOLE profiling batch, we also record per-sample MACs
    (`macs_per_sample` = total / batch) and the `batch_size`. Convention: fvcore counts one
    fused multiply-add (MAC) as 1 (a MAC count, not 2xMAC). MACs are the ONE metric that is
    allowed to be n/a in strict mode, because the reason is understood and documented.
  - peak GPU memory for a batch, in a process where NO other model exists. Reported as
    baseline (this model's params + its own inputs), activation (incremental forward cost),
    total peak, and the allocator's RESERVED arenas; "n/a" on a non-CUDA device. Each model's
    worker builds ONLY the inputs that model actually consumes — the MLP worker never allocates
    a present-mask, so no unused tensor sits in the allocator inflating its baseline. All of
    those are ALLOCATOR numbers: `device_mem_used_mb` is the separate, much larger figure for
    "what this process occupies on the card", taken from the driver's own per-PID accounting,
    because the CUDA context is hundreds of MB that no allocator counts. The allocator peak and
    the device footprint are both true and answer different questions; `peak_mem_note` carries
    the distinction into the CSV, and the device figure is 'n/a' rather than approximated when
    the driver cannot attribute it to this PID.
  - latency on CUDA and on CPU, as TWO different quantities, because they are two different
    quantities and a column called "latency" silently picks one:
      * `lat_*_ms`        — STEADY-STATE amortized service time (median ms/batch + IQR), from
                            torch.utils.benchmark. It times a block of `number_per_run` forwards
                            between two clock reads and divides, so on CUDA the launch queue
                            pipelines call k+1 behind call k. `*_number_per_run` is recorded:
                            when it is 1 (typical on CPU) this IS a per-call number; when it is
                            100 (typical on GPU) it is not.
      * `lat_*_single_ms` — SINGLE-REQUEST latency: one forward per timed interval with a full
                            synchronization boundary on each side. This is what a deployment or
                            edge claim is about. A scoping measurement on a V100 put it ~11% above
                            the steady-state number for the Proposed model — taken on a BUSY box,
                            so treat that as an order of magnitude for why both columns exist,
                            not as a result. The point is that the gap is small enough to stay
                            invisible and large enough to matter in a deployment claim.
    Throughput (samples/s) is derived from the steady-state number, which is the one it belongs
    to. The GPU-labelled columns are "n/a" when no GPU is used — we never report a CPU number
    under a GPU label. `accel_device` records the device actually used.
  - PARTIAL post-training DYNAMIC INT8 quantization (CPU inference): int8 size, CPU latency,
    and segmentation accuracy (mIoU) vs the fp32 model on the real Indian Pines test set. The
    `quant_scope` column is derived by INSPECTING the quantized module tree after the fact —
    it lists the modules that actually became `torch.ao.nn.quantized.dynamic.*` and the ones
    that stayed fp32 — so "partial dynamic INT8" is a statement of what was measured, not a
    claim about what was intended. `quant_api` / `quant_validated_torch` pin the API and the
    torch version this path was validated against (see quantize_int8).
  - missing-band accuracy over EVERY missing-group subset when they can all be enumerated. A
    single fixed prefix (groups 0..k-1) is the cheapest subset to accidentally tell a story
    about, and five subsets out of C(10,6)=210 — three of them hand-picked — is a descriptive
    statistic over an arbitrary corner, not an estimate of anything. At the default settings all
    210 are evaluated, which turns mean/median/std from a sample statistic into the POPULATION:
    there is no sampling error left to argue about and no seed that could have drawn a flattering
    set. The cost is one forward over the test set per subset per precision — cheap here because
    the test set is small, and it scales linearly, so check it before raising
    --subset-exhaustive-max on a bigger dataset. `missing_subset_coverage` says which regime a row is in
    ("210/210 exhaustive" vs "64/1000 sampled (6.4%)") so it is never inferred, and the prefix
    stays FIRST so the historical number remains comparable.

FAILURE POLICY
--------------
No measurement is ever produced by an exception handler. When a metric raises, the worker
records the exception under `errors` and leaves the metric ABSENT; the driver renders absent as
"n/a" and copies the exception text into the `status` column. In `--strict` (the default) a
missing REQUIRED metric, a crashed worker, an unhonoured thread pin, a latency whose relative
IQR exceeds `--max-rel-iqr`, or an "INT8" row where nothing was actually quantized all abort
the run with a nonzero exit and write NO paper artefact (the diagnostic CSV goes to the
workdir instead), so a benchmark that measured nothing can never look like one that ran.

That sentence was once broader than the gate behind it. Four ways a table could pass while
having measured nothing, all reproduced against the real function and all now closed:
  * `strict_failures([])` returned []. An empty table, or a table missing `Proposed`, or one
    holding a single model, passed every check — the gate iterated rows and asked nothing about
    the SET of them. It now requires exactly EXPECTED_MODELS, with no duplicates.
  * the "nothing was quantized" guard read `quant_n_quantized`, and skipped itself entirely when
    that value was ABSENT — so the one failure mode where the scope report itself broke was the
    one it could not see. `quant_scope`/`quant_n_quantized` are REQUIRED now.
  * the worker already computed `required_failed` and the driver discarded it, re-deriving the
    verdict from absent columns. Two failure semantics, only one consulted. It is consulted now.
  * `interop_threads` was recorded and never checked, and the INT8 latency had no replicate
    count in the CSV at all, so `--min-replicates` could not apply to it.
An artefact from a torch other than the one the INT8 path was validated against is also a
failure now (`--allow-unvalidated-quant` to accept it): recording a version is not pinning one.

Outputs (../paper/):
  results_phase7_efficiency.csv
  results_phase7_efficiency.tex   (LaTeX macros)
  results_phase7_efficiency.csv.provenance.json
--smoke writes all three under a `_smoke` suffix. It lowers --bs and the measurement budget, and
every latency/memory number is a function of both, so smoke output must never land on the paths
above.

Usage:
  python experiments/phase7_efficiency.py                     # full (refuses on a busy box)
  python experiments/phase7_efficiency.py --epochs 30 --bs 256
  python experiments/phase7_efficiency.py --smoke             # tiny/quick machinery check
  python experiments/phase7_efficiency.py --device cpu        # force CPU (GPU columns -> n/a)
  python experiments/phase7_efficiency.py --allow-contended   # override the busy-box refusal
"""
import os, sys, csv, io, json, math, time, shutil, argparse, itertools, statistics, subprocess, tempfile
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention, MLPBaseline, count_params
from bandsim.io import AVIRIS_WL_NM
from bandsim.metrics import miou
from bandsim import hw
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
def P(rel):
    """Path under paper/, creating the directory on demand.

    The mkdir used to run at IMPORT. Every measurement worker imports this module, and so does
    every test collection — none of which should touch the filesystem to find out what a path
    is. Deferring it costs nothing and keeps `import phase7_efficiency` side-effect-free."""
    os.makedirs(PAPER_DIR, exist_ok=True)
    return os.path.join(PAPER_DIR, rel)


# --------------------------------------------------------------------------------------
# quantisation API pin (item 6)
# --------------------------------------------------------------------------------------
# `torch.ao.quantization.quantize_dynamic` is the LEGACY eager-mode dynamic-PTQ path; PyTorch's
# quantisation work has moved to TorchAO. We deliberately do NOT migrate here: TorchAO is not a
# dependency of this project (`import torchao` fails in the pinned environment), and adding a
# quantisation library to produce one table column would change the lockfile the rest of the
# results were produced under. So we PIN instead: the path below is validated against the torch
# version named here, the running version is recorded in every row, and — crucially — the scope
# is VERIFIED after the fact by walking the quantized module tree (quantized_scope_report), so
# the reported INT8 number always states exactly which operators it covers.
QUANT_API = "torch.ao.quantization.quantize_dynamic (legacy eager-mode dynamic PTQ)"
QUANT_VALIDATED_TORCH = "2.6.0+cu124"


# The table IS these four rows. Declared here rather than derived from whatever the run happened
# to produce, because a gate that asks "is every row I got valid?" cannot notice the row it never
# got: `strict_failures([])` used to return [] — an empty benchmark passing as a complete one.
# _run_models asserts its roster against this, so the two cannot drift apart silently.
EXPECTED_MODELS = ("MLP (B1)", "B4 attn", "B6 attn", "Proposed")


# --------------------------------------------------------------------------------------
# formatting helpers — a MISSING measurement (None / NaN) renders as "n/a", NEVER 0.
# None  = "could not be computed" (e.g. fvcore missing / tracing failed / metric raised).
# NaN   = "not applicable / skipped here" (e.g. GPU metric on a CPU-only host).
# Both are surfaced honestly; neither is ever coerced to a fabricated 0.
# --------------------------------------------------------------------------------------
def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt(v, spec):
    """Format v with `spec`, but render a missing value (None/NaN) as 'n/a' instead of
    crashing on None or fabricating a number."""
    return "n/a" if _missing(v) else format(v, spec)


def _cell(v):
    """CSV cell: 'n/a' for a missing value (None/NaN), 3-dp for floats, else the value."""
    if _missing(v):
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else v


# --------------------------------------------------------------------------------------
# machine-contention gate — a latency measured on a busy box is noise, not a result
# --------------------------------------------------------------------------------------
def _nvidia_smi(query):
    """Run one `nvidia-smi --query-...` and return the CSV rows, or None if unavailable.

    Returns None (not []) when nvidia-smi is missing/failing, so callers can distinguish
    "no GPU load" from "could not tell" — the second must never be read as the first."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def gpu_state():
    """Per-GPU clocks / temperature / power / utilisation, or None when nvidia-smi cannot say.

    Recorded per WORKER, not once per run. Subprocess isolation resets software state and
    nothing else: the four models are measured in a fixed order on one physical card, and a card
    that has warmed up or dropped a boost bin between the first worker and the fourth makes the
    last row slower for reasons that have nothing to do with the model. Randomising the order
    would trade that bias for run-to-run incomparability; recording the clocks makes the bias
    VISIBLE, which is the honest option at this cost."""
    rows = _nvidia_smi("--query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
                       "utilization.gpu")
    if not rows:
        return None
    keys = ("index", "clock_sm_mhz", "clock_sm_max_mhz", "temp_c", "power_w", "util_pct")
    out = []
    for r in rows:
        parts = [p.strip() for p in r.split(",")]
        if len(parts) == len(keys):
            out.append(dict(zip(keys, parts)))
    return out or None


def process_gpu_mem_mb():
    """Device memory attributed to THIS PROCESS by the driver (MB), or None when it cannot be.

    This is the deployment-footprint number — allocator bytes plus the CUDA context, which is
    several hundred MB that no allocator counts and that every process pays.

    It is deliberately NOT `torch.cuda.mem_get_info`. That returns free/total for the DEVICE, so
    `total - free` is the sum over every process on the card. Measured while writing this: it
    reported 324 MB on a GPU where this process had allocated nothing at all and another tenant
    held 306 MB. A column named for this model's footprint, reporting a co-tenant's, is precisely
    the failure mode the rest of this file exists to prevent — and shared GPUs are the normal
    case here, not the exception.

    Returns None rather than a wrong number when the PID cannot be matched (nvidia-smi reports
    HOST pids, and inside a PID-namespaced container ours will not be among them). Absent is a
    reportable state; wrong is not."""
    rows = _nvidia_smi("--query-compute-apps=pid,used_memory")
    if not rows:
        return None
    me = os.getpid()
    for r in rows:
        parts = [p.strip() for p in r.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) == me:
                return float(parts[1])          # nounits => MiB
        except ValueError:
            continue
    return None


def cgroup_cpu_throttle():
    """`(nr_periods, nr_throttled)` from this process's CPU cgroup, or None if unavailable.

    The only signal here scoped to our CGROUP rather than to the whole machine. The load average
    cannot say whether we got the CPU we pinned — it is a host-wide count of runnable tasks, and
    inside a container it is not even virtualised. The CFS throttle counter is exact about the
    cgroup: if it advances while a timing loop runs, the scheduler took the CPU away
    mid-measurement and every CPU latency in that run is a measurement of the quota, not of the
    model.

    NOT per-process. If another job shares this container, its CPU use throttles the cgroup and
    shows up here — which is the right answer for our purposes (a throttled cgroup deschedules
    THIS process too) but the wrong claim to make in the other direction: a clean counter says
    the cgroup was not throttled, not that nothing else was running.

    This matters most at the setting people reach for first. `--bench-threads` equal to the full
    cgroup quota leaves nothing for the main thread, the allocator or the CUDA driver's own
    threads, so the process spends the run at its ceiling — which is why the default now leaves
    two cores. Handles cgroup v2 (`/sys/fs/cgroup/cpu.stat`) and v1 (`.../cpu/cpu.stat`)."""
    for path in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat"):
        try:
            with open(path, encoding="utf-8") as f:
                kv = dict(ln.split()[:2] for ln in f if len(ln.split()) >= 2)
            if "nr_periods" in kv and "nr_throttled" in kv:
                return int(kv["nr_periods"]), int(kv["nr_throttled"])
        except Exception:
            continue
    return None


def _proc_stat_busy():
    """Per-CPU (busy_jiffies, total_jiffies) from /proc/stat, keyed by cpu index. {} if unreadable.

    busy = everything that is not idle+iowait. /proc/stat is host-wide and NOT namespaced, so on a
    cpuset-pinned container the per-cpu rows still report every tenant's use of each physical core
    -- which is exactly what is wanted: the contention on a core I run on is real whether it comes
    from me or from a neighbour that shares it."""
    out = {}
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            for ln in f:
                if not ln.startswith("cpu") or len(ln) < 4 or not ln[3].isdigit():
                    continue
                p = ln.split()
                c = int(p[0][3:])
                v = [int(x) for x in p[1:]]
                idle = v[3] + (v[4] if len(v) > 4 else 0)   # idle + iowait
                out[c] = (sum(v) - idle, sum(v))
    except Exception:
        return {}
    return out


def own_cpuset_busy_per_core(sample_s=0.4):
    """Mean busy fraction over the cores THIS process may actually run on, or None if unmeasurable.

    This is the signal the CPU gate should use, and it is neither of the two the probe already had.
    load_per_host_core (loadavg / 36) counts runnable tasks across the whole machine, so a host kept
    busy by other tenants on cores this cgroup cannot touch reads as contention that cannot reach
    us. load_per_core (loadavg / 8) is the documented unit error. The real question -- "are the
    cores I am pinned to busy right now?" -- is answered by differencing /proc/stat over exactly
    os.sched_getaffinity(0), which on this box is 8 of the host's 36. A quiet cpuset returns ~0.08
    even while the host sits at 0.62/core, and a genuinely oversubscribed cpuset still returns ~1.

    Best-effort: returns None (never 0.0) when affinity or /proc/stat is unavailable, so
    contention_reasons falls back to the host-wide proxy rather than treating "could not measure"
    as "quiet"."""
    try:
        cpus = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None
    if not cpus:
        return None
    a = _proc_stat_busy()
    if not a:
        return None
    time.sleep(sample_s)
    b = _proc_stat_busy()
    fracs = []
    for c in cpus:
        if c in a and c in b:
            dt = b[c][1] - a[c][1]
            db = b[c][0] - a[c][0]
            if dt > 0:
                fracs.append(max(0.0, min(1.5, db / dt)))
    if not fracs:
        return None
    return sum(fracs) / len(fracs)


def probe_contention():
    """Snapshot of how busy this machine is RIGHT NOW.

    Signals, because each misses a different kind of load:
      * `gpu_util_max`        — peak SM utilisation across visible GPUs. Catches another job's
                                kernels, which is what actually steals GPU latency from us.
      * `n_compute_apps`      — number of CUDA compute processes. Catches a job that is between
                                kernel launches at the instant we look (util momentarily 0).
      * `load_per_host_core`  — 1-minute load average over the cores it is MEASURED over. See
                                below; this is the one the threshold uses.
      * `load_per_core`       — the same load over the cores this process may use. Kept because
                                it is the interesting number when the two differ, but it is NOT
                                a contention threshold: dividing a host-wide numerator by a
                                cgroup-quota denominator mixes units and reads a quiet 36-core
                                host as a saturated 8-core one (7.57/36 = 21% became 0.95/core).
      * `cgroup_throttle`     — see cgroup_cpu_throttle; sampled here, DIFFERENCED across the run.
    Every field is best-effort; `None` means "could not measure", never "zero" — and callers must
    keep those apart, which is what contention_unknowns exists for."""
    utils, mem = None, None
    rows = _nvidia_smi("--query-gpu=utilization.gpu,memory.used")
    if rows is not None:
        try:
            parsed = [[float(x) for x in r.split(",")] for r in rows]
            utils = [p[0] for p in parsed]
            mem = [p[1] for p in parsed]
        except Exception:
            utils, mem = None, None
    apps = _nvidia_smi("--query-compute-apps=pid")
    try:
        load1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        load1 = None
    cores = hw.available_cores()
    host_cores = os.cpu_count() or cores
    return {
        "nvidia_smi": rows is not None,
        "gpu_util_pct": utils,
        "gpu_util_max": (max(utils) if utils else None),
        "gpu_mem_used_mb": mem,
        "n_compute_apps": (len(apps) if apps is not None else None),
        "loadavg_1m": load1,
        "cores": cores,
        "host_cores": host_cores,
        "load_per_core": (load1 / cores if load1 is not None and cores else None),
        "load_per_host_core": (load1 / host_cores if load1 is not None and host_cores else None),
        # Busy fraction of the cores this process is actually pinned to -- the CPU signal the gate
        # prefers, because it is the only one that distinguishes "the host is busy on cores I cannot
        # use" from "my cores are busy". See own_cpuset_busy_per_core.
        "own_cpuset_busy": own_cpuset_busy_per_core(),
        "cgroup_throttle": cgroup_cpu_throttle(),
    }


def contention_reasons(probe, want_cuda, gpu_util_max_pct=15.0, load_per_core_max=0.6,
                       expected_own_compute_apps=0):
    """Human-readable reasons this box is too busy to benchmark on. Empty list == quiet enough.

    `expected_own_compute_apps` lets the caller discount CUDA contexts it knows are its own: the
    PRE-flight probe runs before we touch CUDA (so 0), while a post-run probe must not flag the
    driver's own idle context as someone else's job.
    GPU signals are only applied when the bench will actually use CUDA; the CPU load signal
    always applies, because the CPU columns are measured on every run.

    The CPU threshold prefers `own_cpuset_busy` — the measured busy fraction of the cores this
    process is pinned to — and only falls back to `load_per_host_core` (loadavg over ALL host
    cores) when the cpuset signal could not be measured. It NEVER uses `load_per_core` (loadavg
    over the cgroup quota): the load average is host-wide and un-namespaced, so dividing a
    numerator counted over 36 cores by a denominator of 8 turned a host at 21% into "0.95/core,
    too busy" and refused every run while the cgroup was throttled 0.04% of the time. host_per_core
    fixed the units but is still a whole-machine average: it refuses when other tenants load cores
    this cpuset cannot touch, which on a shared box (steady host loadavg ~22/36 = 0.62) refuses
    forever even though the pinned cores sit near 8% busy. own_cpuset_busy answers the only
    question that bears on our latency -- are MY cores busy -- and the post-run cgroup-throttle
    check is the backstop for a neighbour that bursts after this snapshot. A gate wrong in the
    refusing direction is not conservative; it just trains you to pass --allow-contended."""
    out = []
    if want_cuda:
        u = probe.get("gpu_util_max")
        if u is not None and u > gpu_util_max_pct:
            out.append(f"GPU utilisation {u:.0f}% > {gpu_util_max_pct:.0f}% "
                       f"(per-GPU: {probe.get('gpu_util_pct')})")
        n = probe.get("n_compute_apps")
        if n is not None and n > expected_own_compute_apps:
            out.append(f"{n} CUDA compute process(es) on this box, expected at most "
                       f"{expected_own_compute_apps} of our own")
    # CPU gate, in order of precision. own_cpuset_busy measures the cores this process is pinned
    # to, so it is authoritative: a host loaded to 0.62/core by tenants on cores we cannot use does
    # not steal our latency, and refusing on it just teaches --allow-contended. Use it when it
    # could be measured; fall back to load_per_host_core (loadavg / all host cores), then to the
    # hand-built-probe ratio, only when the better signal is absent. NEVER load_per_core (the
    # loadavg / cgroup-quota unit error).
    own = probe.get("own_cpuset_busy")
    if own is not None:
        if own > load_per_core_max:
            hint = ""
            lph = probe.get("load_per_host_core")
            if lph is not None:
                hint = f" (host is {lph:.2f}/core; the pinned cores are the busy ones)"
            out.append(f"pinned CPUs are {own:.2f} busy/core > {load_per_core_max:.2f}{hint}")
        return out
    lpc, over = probe.get("load_per_host_core"), probe.get("host_cores")
    if lpc is None and over is None:
        # A probe carrying no host-core information at all — i.e. built by hand rather than by
        # probe_contention. Use the one ratio it does have rather than silently checking nothing;
        # the unit error this guards against is only possible when BOTH are available and the
        # wrong one is chosen.
        lpc, over = probe.get("load_per_core"), probe.get("cores")
    if lpc is not None and lpc > load_per_core_max:
        out.append(f"CPU load {probe['loadavg_1m']:.2f} over {over} core(s) "
                   f"= {lpc:.2f}/core > {load_per_core_max:.2f} "
                   f"(this process may use {probe.get('cores')} of them)")
    return out


def contention_unknowns(probe, want_cuda):
    """Signals that could NOT be measured. An unmeasurable signal is not a quiet one.

    contention_reasons deliberately ignores a None — it must not call a box busy on evidence it
    does not have, and refusing to run whenever nvidia-smi is absent would make this script
    unusable on hosts where it is fine. But the other half was missing: the run then stamped
    every row `timing_validity=ok`, which asserts a check that never happened. These reasons do
    not stop the run; they downgrade the verdict to `ok-unverified` and travel into the
    provenance, so "I checked and it was quiet" stays distinguishable from "I could not look"."""
    out = []
    if want_cuda:
        if not probe.get("nvidia_smi"):
            out.append("nvidia-smi unavailable: neither GPU utilisation nor foreign CUDA "
                       "processes were checked")
        else:
            if probe.get("gpu_util_max") is None:
                out.append("GPU utilisation could not be parsed from nvidia-smi")
            if probe.get("n_compute_apps") is None:
                out.append("the CUDA compute-process list could not be read")
    if probe.get("load_per_host_core") is None and probe.get("load_per_core") is None:
        out.append("no load average on this platform: CPU contention was not checked")
    return out


# --------------------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------------------
def model_size_mb(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def flops_macs(model, inputs):
    """MACs per forward via fvcore, returned ONLY when the count is COMPLETE.

    Returns ``(macs, unsupported, uncalled, error)``:
      * ``macs``        — int total MACs when fvcore traced the model AND every op was
                          supported AND every submodule was reached; otherwise ``None``.
      * ``unsupported`` — dict {op_name: count} of ops fvcore could not count, or ``None`` when
                          fvcore is absent / tracing raised (i.e. no analysis happened at all).
      * ``uncalled``    — sorted list of submodule names fvcore never reached, or ``None`` (as
                          above).
      * ``error``       — the exception TEXT when nothing could be analysed, else ``None``.
                          Previously both causes collapsed into the same 'fvcore unavailable or
                          tracing failed' note, so "the package is not installed" and "tracing
                          raised on this model" — one trivial, one a real finding — were
                          indistinguishable in the artefact. Discarding the exception is the
                          cheapest possible way to make an n/a unauditable.

    Why ``None`` on ANY unsupported op / uncalled module: ``fca.total()`` happily returns a
    number even when it silently skips ops it does not understand (e.g. a Transformer's
    ``scaled_dot_product_attention``) or submodules it never traced (e.g. a self-attention
    ``out_proj``). Disabling the warnings does NOT make ``.total()`` fail — it just returns a
    smaller number. That number is an UNDERCOUNT, not a real MAC count, so we refuse to report
    it: callers MUST render ``None`` as 'n/a'/NaN and NEVER as 0 (a 0 would fabricate "zero
    compute"). The unsupported/uncalled lists are returned so the reason is auditable.

    Convention: fvcore's FlopCountAnalysis counts one fused multiply-add (MAC) as 1, so a
    completed count is a MAC count — a FLOP count under the "FLOP = 2xMAC" convention would be
    roughly twice as large.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        return None, None, None, f"fvcore not importable: {type(e).__name__}: {e}"
    try:
        fca = FlopCountAnalysis(model, inputs)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = int(fca.total())
        unsupported = dict(fca.unsupported_ops())     # {op: count}, empty when fully supported
        uncalled = sorted(fca.uncalled_modules())     # [] when every submodule was reached
    except Exception as e:                            # tracing raised -> no trustworthy count
        return None, None, None, f"fvcore tracing raised: {type(e).__name__}: {e}"
    if unsupported or uncalled:
        return None, unsupported, uncalled, None      # incomplete -> undercount, report 'n/a'
    return total, {}, [], None                        # complete -> a real MAC total


def _macs_note(macs, unsupported, uncalled, error=None):
    """One-line CSV note explaining a MACs cell — 'complete' for a real count, or WHY it is
    'n/a' — so an n/a is auditable instead of mysterious (see flops_macs: silent undercount)."""
    if macs is not None:
        return "complete"
    if unsupported is None and uncalled is None:
        return error or "fvcore unavailable or tracing failed"
    parts = []
    if unsupported:
        parts.append("unsupported ops: " + ",".join(sorted(unsupported)))
    if uncalled:
        parts.append("uncalled modules: " + ",".join(uncalled))
    detail = (" (" + "; ".join(parts) + ")") if parts else ""
    return "incomplete count -> n/a" + detail


def summarize_times(times_s, batch):
    """Replicate wall-times (seconds PER ITERATION) -> the dispersion-carrying summary we report.

    A benchmark cell is a distribution, not a number. We keep the median (robust to the one slow
    replicate every machine produces), the IQR and the relative IQR (the dimensionless noise
    figure the strict gate thresholds on), the range, and the replicate count — so a reader can
    tell a tight measurement from a coin flip."""
    ts = sorted(float(t) for t in times_s)
    if not ts:
        raise ValueError("no timing replicates were collected")
    med = statistics.median(ts)
    p25, p75 = (float(x) for x in np.percentile(ts, [25, 75]))
    iqr = p75 - p25
    return {
        "median_ms": med * 1e3,
        "iqr_ms": iqr * 1e3,
        "p25_ms": p25 * 1e3,
        "p75_ms": p75 * 1e3,
        "min_ms": ts[0] * 1e3,
        "max_ms": ts[-1] * 1e3,
        "mean_ms": statistics.fmean(ts) * 1e3,
        "rel_iqr": (iqr / med) if med > 0 else float("nan"),
        "n": len(ts),
        "throughput_sps": (batch / med) if med > 0 else float("nan"),
    }


def bench_latency(model, inputs, dev, num_threads, repeats=5, warmup=50,
                  min_run_time=0.5, max_run_time=10.0, target_rel_iqr=0.05, label=""):
    """Latency via `torch.utils.benchmark.Timer` — REPEATED, ADAPTIVE, thread-pinned.

    Replaces a manual `time.time()` loop, which got four things wrong at once: it timed a fixed
    iteration count regardless of how long the op took, it never amortized timer overhead, it
    never synchronized CUDA except once at each end (so any launch-queue drift landed in the
    number), and it produced exactly one sample.

    What happens here instead:
      * an explicit `warmup` loop first, because the FIRST call pays one-time costs that belong
        to nobody's steady-state latency — lazy CUDA init, cuBLAS handle creation, cuDNN
        algorithm selection, allocator growth;
      * `Timer(num_threads=...)` pins the intra-op thread count FOR THE DURATION of the
        measurement, so a differently-loaded box cannot silently change what is being measured;
      * `adaptive_autorange` picks a block size that makes timer overhead <0.1% and then keeps
        sampling until iqr/median drops under `target_rel_iqr` (or `max_run_time` is hit) —
        this is the "blocking + adaptive" part a hand-rolled loop cannot do;
      * that is repeated `repeats` times and ALL replicates are pooled, so the reported median
        is drawn from many samples spread over the run rather than one contiguous window that
        could sit entirely inside another job's quiet period.

    WHAT THIS IS NOT. On a CUDA-capable host `torch.utils.benchmark`'s default timer synchronizes
    CUDA — but around each BLOCK, not around each call. `adaptive_autorange` first picks a
    `number_per_run`, then times `number_per_run` forwards between two clock reads and divides.
    Measured here: 100 on a V100 for these models, 1 on CPU. At 100, calls pipeline — the launch
    queue is filling call k+1 while call k executes — so the result is steady-state amortized
    SERVICE TIME, not what one synchronized request costs. A scoping measurement on a busy V100
    put the gap around 11% for the Proposed model — not a result (the box was loaded), but enough
    to establish that it is small enough to stay invisible and large enough to matter in a
    deployment claim. So `bench_single_request` measures the other quantity and both go in the
    table under names that say which is which. `number_per_run` is returned so a reader can see
    when the two must coincide (it is 1) rather than taking anyone's word for it.

    A second consequence worth stating: at number_per_run=100 the reported rel_iqr is dispersion
    ACROSS BLOCKS of 100 averaged calls, so it is a much tighter number than call-to-call
    variation. `--max-rel-iqr` is therefore a weaker filter on GPU than it looks; the
    single-request cell carries the honest per-call dispersion and has its own threshold.

    Returns the dict from `summarize_times` plus the measurement settings actually used.
    Raises on failure — callers decide whether the metric was required."""
    from torch.utils.benchmark import Timer, Measurement

    model.eval()
    batch = int(inputs[0].shape[0])
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            model(*inputs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        measurements = []
        for _ in range(max(1, int(repeats))):
            timer = Timer(stmt="model(*inputs)",
                          globals={"model": model, "inputs": inputs},
                          num_threads=int(num_threads),
                          label=label or "forward", description=dev.type)
            measurements.append(timer.adaptive_autorange(
                threshold=float(target_rel_iqr),
                min_run_time=float(min_run_time),
                max_run_time=float(max_run_time)))
    # Measurement.times normalizes each raw block time by its iteration count, so pooling across
    # repeats (which may have chosen different block sizes) compares like with like.
    pooled = [t for m in Measurement.merge(measurements) for t in m.times]
    out = summarize_times(pooled, batch)
    out.update(repeats=int(repeats), warmup=int(warmup), num_threads=int(num_threads),
               min_run_time_s=float(min_run_time), target_rel_iqr=float(target_rel_iqr),
               n_measurements=len(measurements),
               number_per_run=";".join(str(n) for n in
                                       sorted({int(m.number_per_run) for m in measurements})))
    return out


def bench_single_request(model, inputs, dev, num_threads, n=200, warmup=50):
    """SINGLE-REQUEST latency: ONE forward per timed interval, synchronized on both sides.

    The quantity a deployment or edge-latency claim is about — what one request costs, launch
    overhead and synchronization included — as opposed to what a request costs when 99 others
    are already in flight (that is `bench_latency`). Both belong in the table; neither is a
    substitute for the other, and a column called simply "latency" quietly picks one. MLPerf's
    single-stream scenario is the first quantity; a throughput benchmark is the second.

    Deliberately NOT torch.utils.benchmark: its whole design is to amortize timer overhead across
    a block, which is exactly the property that has to be removed here. The cost is that timer
    overhead (~sub-microsecond for perf_counter) is no longer amortized — negligible against the
    millisecond-scale forwards here, and the honest trade for a per-call boundary.

    Thread count is pinned for the duration and restored, so this cell is measured under the same
    configuration as the others rather than inheriting whatever the caller left set."""
    model.eval()
    batch = int(inputs[0].shape[0])
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(int(num_threads))
    try:
        with torch.no_grad():
            for _ in range(max(0, int(warmup))):
                model(*inputs)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            times = []
            for _ in range(max(2, int(n))):
                t0 = time.perf_counter()
                model(*inputs)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                times.append(time.perf_counter() - t0)
    finally:
        torch.set_num_threads(prev_threads)
    out = summarize_times(times, batch)
    out.update(n_requests=len(times), warmup=int(warmup), num_threads=int(num_threads),
               number_per_run="1")
    return out


def peak_mem_mb(model, inputs, dev, warmup=2):
    """Peak ALLOCATOR memory (MB) for ONE forward of THIS model. Dict, or None on a non-CUDA host.

    WHAT THIS IS, exactly: bytes the PyTorch caching allocator has handed out to live tensors.
    It is NOT the memory this model needs on a device, and the difference is not a rounding
    error — the CUDA context alone is several hundred MB that no allocator counts and every
    process pays, against an allocator peak of ~41 MB for the Proposed model. Quoting the
    allocator figure as "the GPU this model needs" would be wrong by more than the entire model.
    So both are reported, from sources that can actually attribute what they measure:

      * ``baseline``    — allocator bytes live right BEFORE the forward: this model's
                          parameters/buffers plus its own profiling inputs.
      * ``total``       — allocator PEAK during the forward (params + inputs + activations).
      * ``activation``  — ``total - baseline``, the forward's incremental activation cost.
      * ``reserved``    — peak bytes the allocator RESERVED from the driver (its arenas; >= total,
                          and the number that actually has to fit alongside another process).
      * ``device_used`` — driver-reported memory for THIS PID, CUDA context included: the
                          deployment-footprint question's answer. ``None`` (never a substitute
                          number) when the PID cannot be matched -- see process_gpu_mem_mb.
      * ``note``        — the caveat itself, carried into the CSV so the column cannot be read
                          out of context.

    ATTRIBUTION is unambiguous here because this runs in a worker process holding exactly one
    model: no other model's weights can be folded into `baseline`, and no unused tensor from a
    sibling model's input set is resident. (In-process, `torch.cuda.empty_cache()` cannot free
    memory held by live tensors, so the previous single-process version could only approximate
    this by evicting the other models to CPU.) That is a claim about attribution, not totality.

    `warmup` forwards run BEFORE the reset so a one-off first-call workspace (cuDNN/cuBLAS
    scratch, lazy module state) cannot be attributed to `activation`. For these models the cold
    and warm numbers are identical to 3 dp — hw.setup pins the math SDP kernel and disables cuDNN
    autotune, so there is no workspace to allocate lazily. The warmup is insurance against the
    model where that stops being true, not a correction to this one."""
    if dev.type != "cuda":
        return None
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            model(*inputs)
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        baseline = torch.cuda.memory_allocated(dev)  # this model's params + its own inputs
        model(*inputs)
        torch.cuda.synchronize(dev)
        peak = torch.cuda.max_memory_allocated(dev)  # params + inputs + activations at the peak
        reserved = torch.cuda.max_memory_reserved(dev)
    device_used = process_gpu_mem_mb()                # THIS process, or None -- never the device's
    return {
        "total": peak / 1e6,
        "baseline": baseline / 1e6,
        "activation": (peak - baseline) / 1e6,
        "reserved": reserved / 1e6,
        "device_used": device_used,
        "note": ("allocator-tracked live tensors only; excludes the CUDA context and any "
                 "non-PyTorch allocation -- device_mem_used_mb is the whole-process figure"),
    }


# --------------------------------------------------------------------------------------
# accuracy (fp32 vs partial-int8) on real test pixels
# --------------------------------------------------------------------------------------
def group_present_mask(n, groups, drop_group_ids):
    """(n, G) bool present-mask with the given group ids dropped for all rows."""
    m = np.ones((n, len(groups)), bool)
    for g in drop_group_ids:
        m[:, g] = False
    return m


def zero_missing(X, groups, drop_group_ids):
    """Zero the bands of the dropped groups — how a channel-stack MLP sees a missing group."""
    Xc = X.copy()
    for g in drop_group_ids:
        Xc[:, groups[g]] = 0.0
    return Xc


@torch.no_grad()
def acc_mlp(model, X, y, dev, num_classes, groups=None, drop=()):
    Xc = zero_missing(X, groups, list(drop)) if drop else X
    pred = model(torch.from_numpy(Xc).to(dev)).argmax(1).cpu().numpy()
    return miou(y, pred, num_classes)


@torch.no_grad()
def acc_prop(model, X, y, groups, dev, num_classes, drop=()):
    pm = group_present_mask(X.shape[0], groups, list(drop))
    pred = model(torch.from_numpy(X).to(dev), torch.from_numpy(pm).to(dev)).argmax(1).cpu().numpy()
    return miou(y, pred, num_classes)


def missing_subsets(n_groups, k, n_subsets=5, seed=0, exhaustive_max=512):
    """DISTINCT missing-group subsets of size k — ALL of them when they can be enumerated.

    Measuring missing-band robustness on `range(k)` alone reports the behaviour of one corner of
    the group space — and it is the corner most likely to be the one someone looked at while
    developing the model, i.e. exactly the subset a robustness claim should NOT rest on. So we
    always evaluate, in this order:
      * the PREFIX  (groups 0..k-1)      — kept first so the historical number stays visible and
                                            comparable to earlier runs of this script;
      * the SUFFIX  (the last k groups)  — the opposite end of the spectrum;
      * a STRIDE    (evenly spaced)      — interleaved rather than contiguous, which is a
                                            genuinely different failure mode for a model that
                                            pools over neighbouring groups;
      * then every remaining subset, or seeded random ones when there are too many.

    `n_subsets=None` means EXHAUSTIVE when C(G,k) <= exhaustive_max. That is the default at the
    settings this script ships with, and it is the point: C(10,6) is 210, five of which is 2.4%
    coverage with three of the five hand-chosen. mean/std over that is a description of an
    arbitrary corner; mean/std over all 210 is the population, with no sampling error to argue
    about and no seed that could have drawn a flattering set. Enumeration also removes a silent
    failure the sampler had — it rejection-sampled with a fixed 1000-draw guard and then returned
    however many it happened to get (asking for all 252 of C(10,5) yielded 244, quietly). When
    sampling IS unavoidable the shortfall is still real, so nothing infers coverage from the
    count: `subset_coverage` states it.

    Deterministic in `seed`. Degenerate k is handled rather than silently producing nonsense:
    k<=0 -> [()] (nothing missing), k>=n_groups -> [everything] (only one such subset exists)."""
    G, k = int(n_groups), int(k)
    if k <= 0:
        return [()]
    if k >= G:
        return [tuple(range(G))]
    total = math.comb(G, k)
    if n_subsets is None:
        want = total if total <= int(exhaustive_max) else int(exhaustive_max)
    else:
        want = int(n_subsets)
    want = max(1, min(want, total))
    ordered = [tuple(range(k)),                                        # prefix
               tuple(range(G - k, G)),                                 # suffix
               tuple(sorted(np.linspace(0, G - 1, k).round().astype(int).tolist()))]  # stride
    out = []
    for s in ordered:
        if len(set(s)) == k and s not in out:
            out.append(s)
    if want >= total:
        seen = set(out)
        out.extend(c for c in itertools.combinations(range(G), k) if c not in seen)
        return out[:total]
    rng = np.random.default_rng(seed)
    seen = set(out)
    guard, limit = 0, 200 * want + 1000
    while len(out) < want and guard < limit:
        guard += 1
        s = tuple(sorted(int(x) for x in rng.choice(G, size=k, replace=False)))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:want]


def subset_coverage(n_groups, k, subsets):
    """How much of the missing-group space a row actually covers, as a string for the CSV.

    "mean over 5 subsets" and "mean over all 210" are different claims and a reader must never
    have to infer which one a number is. Stating it also makes a sampler shortfall visible
    instead of silent."""
    G, k = int(n_groups), int(k)
    n = len(subsets)
    if k <= 0 or k >= G:
        return f"{n}/1 (the only subset of this size)"
    total = math.comb(G, k)
    if n >= total:
        return f"{n}/{total} exhaustive"
    return f"{n}/{total} sampled ({100.0 * n / total:.1f}%)"


def stat_block(values):
    """mean/median/std/min/max/p05 over the per-subset accuracies, plus the raw values.

    The SPREAD is the point: a robustness number quoted without it cannot be distinguished from
    a number picked because it was the best of several. `median` and `p05` are here because
    exhaustive enumeration makes them meaningful — over all 210 subsets the 5th percentile is
    the honest "how bad does this get" figure, and a mean alone hides a tail."""
    v = [float(x) for x in values]
    return {
        "mean": statistics.fmean(v),
        "median": statistics.median(v),
        "std": (statistics.stdev(v) if len(v) > 1 else 0.0),
        "min": min(v),
        "max": max(v),
        "p05": float(np.percentile(v, 5)) if len(v) > 1 else float(v[0]),
        "values": v,
    }


# --------------------------------------------------------------------------------------
# PARTIAL dynamic INT8 quantisation + VERIFIED scope
# --------------------------------------------------------------------------------------
def quantize_int8(model):
    """PARTIAL post-training DYNAMIC INT8 quantization for CPU inference.

    Dynamically quantizes nn.Linear weights to int8 — it is NOT a fully int8 network. Precisely:
    weights are quantized ONCE, ahead of time; activations are quantized ON THE FLY, per batch,
    from the observed range at runtime, and the integer matmul happens inside the quantized
    kernel. What stays floating point is the module's Python-level INPUT and OUTPUT, and every op
    outside a quantized Linear. (An earlier version of this comment said activations "stay
    floating point", which reads as "no activation is ever quantized" — the opposite of how
    dynamic quantisation works, and the reason its accuracy can move at all.) Targeted:
      * MLPBaseline: every nn.Linear (the whole MLP stack).
      * GroupedCrossBandAttention: ONLY the outer linears (embed / classifier / decoder); the
        nn.TransformerEncoder is deliberately LEFT IN FP32. Quantizing the attention's Linear
        weights makes torch's transformer fast-path read a quantized weight as a method
        ('function' has no attribute 'device'). Outer-linear dynamic quant is the safe, standard
        subset and still shrinks + speeds the model.
    Callers must report `quantized_scope_report(fp32, q)` alongside any int8 number: what was
    INTENDED here is not evidence of what was achieved, and the report is derived from the
    resulting module tree. See QUANT_API / QUANT_VALIDATED_TORCH for the pinned API."""
    from torch.ao.quantization import quantize_dynamic, default_dynamic_qconfig
    m = model.to("cpu").eval()
    if isinstance(m, GroupedCrossBandAttention):
        spec = {n: default_dynamic_qconfig for n, mod in m.named_modules()
                if isinstance(mod, nn.Linear) and not n.startswith("encoder")}
        return quantize_dynamic(m, spec, dtype=torch.qint8)
    return quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)


def quantized_scope_report(fp32_model, qmodel):
    """What was ACTUALLY quantized — inspected from the resulting module tree, not declared.

    An "INT8" cell that does not say what it covers is uninterpretable: a 4x size drop means one
    thing if the whole network is int8 and quite another if three outer linears are. This walks
    the quantized model, collects every module whose class lives under `torch.ao.nn.quantized`,
    and separately lists the nn.Linear modules that survived in fp32. It also notes the
    MultiheadAttention in-projection, which is a raw Parameter rather than an nn.Linear and so
    is not even a candidate for `quantize_dynamic` — a real limit of this path that a
    module-count alone would hide.

    Returns a dict with the counts, the names, and a one-line `scope` string for the CSV."""
    quantized, left_fp32 = [], []
    for name, mod in qmodel.named_modules():
        if name.endswith("._packed_params"):
            continue
        cls = type(mod)
        if cls.__module__.startswith("torch.ao.nn.quantized"):
            quantized.append((name or "<root>", cls.__name__))
        elif isinstance(mod, nn.Linear):
            left_fp32.append(name or "<root>")
    mha = [n for n, m in fp32_model.named_modules() if isinstance(m, nn.MultiheadAttention)]
    parts = []
    if quantized:
        parts.append("int8-dynamic: " + ",".join(n for n, _ in quantized)
                     + f" ({len(quantized)} module(s) -> {sorted({c for _, c in quantized})})")
    else:
        parts.append("int8-dynamic: NOTHING WAS QUANTIZED")
    if left_fp32:
        parts.append(f"fp32 nn.Linear left: {','.join(left_fp32)} ({len(left_fp32)})")
    if mha:
        parts.append(f"fp32 MultiheadAttention in_proj (raw Parameter, not nn.Linear): "
                     f"{','.join(mha)}")
    return {
        "n_quantized": len(quantized),
        "n_left_fp32": len(left_fp32),
        "quantized_modules": [n for n, _ in quantized],
        "left_fp32_modules": left_fp32,
        "scope": "; ".join(parts),
    }


# --------------------------------------------------------------------------------------
# worker: measure ONE model, in ITS OWN process
# --------------------------------------------------------------------------------------
# REQUIRED metrics — a run that cannot produce these measured nothing useful, and in --strict it
# must fail loudly rather than emit a table of 'n/a' that looks like a completed benchmark.
# `macs` is deliberately NOT required: fvcore genuinely cannot count scaled_dot_product_attention,
# the reason is recorded in `macs_note`, and demanding it would force us to report an undercount.
#
# `quant_scope` / `quant_n_quantized` are required, and that is not bookkeeping. The gate that
# catches an "INT8" row describing the fp32 model reads `quant_n_quantized == 0` — and skipped
# itself entirely when the value was ABSENT, so the single case where the scope report itself
# broke was the one case it could not see. A guard whose precondition is the thing that failed
# is not a guard.
# `dynint8_lat_cpu_n` is required for the same shape of reason: the INT8 latency shipped a
# relative IQR but no replicate count, so `--min-replicates` had nothing to apply to it.
REQUIRED_ALWAYS = ("params", "size_mb", "lat_cpu_ms", "lat_cpu_single_ms", "thr_cpu",
                   "miou_fp32_clean", "miou_fp32_miss_mean",
                   "dynint8_size_mb", "dynint8_lat_cpu_ms", "dynint8_lat_cpu_n",
                   "quant_scope", "quant_n_quantized",
                   "miou_dynint8_clean", "miou_dynint8_miss_mean")
REQUIRED_CUDA = ("lat_gpu_ms", "lat_gpu_single_ms", "thr_gpu",
                 "peak_mem_mb", "peak_mem_baseline_mb", "peak_mem_activation_mb")
# Metrics allowed to be absent, each for a REASON that is recorded next to them. This is the
# whole list — anything not here and not required is an oversight, not a policy.
OPTIONAL_DOCUMENTED = ("macs", "macs_per_sample",           # fvcore cannot count SDPA (macs_note)
                       "peak_mem_reserved_mb", "device_mem_used_mb",  # driver may not report
                       "gpu_clock_sm_mhz", "gpu_temp_c")              # nvidia-smi may be absent


def _build_model(spec, groups, cwl):
    if spec["kind"] == "mlp":
        return MLPBaseline(int(spec["n_bands"]), int(spec["num_classes"]))
    return GroupedCrossBandAttention(groups, cwl, int(spec["num_classes"]),
                                     pe_type=spec.get("pe_type", "sinusoidal"))


def _pin_environment(spec):
    """Pin the things that make two runs comparable, and report what was actually pinned.

    Thread count and dtype are not cosmetic here: an unpinned CPU latency measures the box's
    current load, and a default dtype changed elsewhere in the process would silently alter every
    tensor created afterwards. `set_num_interop_threads` can only be set before the inter-op pool
    starts, which a fresh worker process guarantees — one more thing single-process benchmarking
    could not do. Returns the EFFECTIVE values so the caller can refuse when a pin did not take."""
    want = int(spec["bench_threads"])
    torch.set_default_dtype(torch.float32)
    dev = hw.setup(seed=int(spec["seed"]), deterministic=True,
                   prefer=spec["device"], threads=want)
    try:
        torch.set_num_interop_threads(want)
    except Exception:
        pass                                   # already initialized; report the effective value
    return dev, {
        "bench_threads_requested": want,
        "torch_threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
        "default_dtype": str(torch.get_default_dtype()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "usable_cores": hw.available_cores(),
        "host_cores": os.cpu_count(),
    }


def worker_main(spec_path):
    """Measure ONE model in this (fresh) process and write its result JSON. Returns an exit code.

    Everything this process touches belongs to one model: its own CUDA context, its own caching
    allocator, its own cuDNN autotune cache, and ONLY the input tensors that model consumes. That
    last point matters for more than tidiness — the MLP takes no present-mask, so in this design
    no present-mask is ever allocated in the MLP's process and its peak-memory baseline cannot
    include a tensor it never reads.

    Failure policy: a metric that raises is recorded under `errors` and left ABSENT from the
    result. It is never replaced by a value, because a number whose provenance is an exception
    handler is indistinguishable from a measurement."""
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    res = {"name": spec["name"], "kind": spec["kind"], "index": spec["index"],
           "worker_pid": os.getpid(), "driver_pid": os.getppid(),
           "errors": {}, "required_failed": []}

    def attempt(key, fn, required):
        """Run one measurement. On success return its value; on failure record WHY and return
        None so the metric stays absent. Never invents a fallback number."""
        try:
            return fn()
        except Exception as e:                       # noqa: BLE001 - recorded, never converted
            res["errors"][key] = f"{type(e).__name__}: {e}"
            if required:
                res["required_failed"].append(key)
            return None

    dev, pinned = _pin_environment(spec)
    res["pinned"] = pinned
    res["accel_device"] = dev.type

    with np.load(spec["data_npz"]) as data:
        Xte, yte = data["Xte"], data["yte"]
    with open(spec["groups_json"], encoding="utf-8") as f:      # read ONCE, closed explicitly
        group_info = json.load(f)
    groups = [np.asarray(g, dtype=int) for g in group_info["groups"]]
    cwl = group_info["cwl"]
    ncls = int(spec["num_classes"])
    is_attn = spec["kind"] == "attn"
    res["gpu_state"] = gpu_state()          # clocks/temp at THIS worker's turn in the sequence

    # Shape/dtype contract, checked HERE rather than surfacing as a confusing failure three
    # measurements later. A worker whose inputs are wrong should say so about its inputs.
    if Xte.ndim != 2 or Xte.shape[1] != int(spec["n_bands"]):
        raise ValueError(f"Xte has shape {Xte.shape}, expected (N, {spec['n_bands']})")
    if not np.isfinite(Xte).all():
        raise ValueError("Xte contains non-finite values -- every latency below would be a "
                         "measurement of denormal/NaN handling, not of the model")
    flat = np.concatenate([np.asarray(g).ravel() for g in groups]) if groups else np.array([], int)
    if len(flat) != len(set(flat.tolist())) or len(flat) != Xte.shape[1]:
        raise ValueError(f"the {len(groups)} groups do not partition the {Xte.shape[1]} bands "
                         f"(covered {len(flat)}, distinct {len(set(flat.tolist()))})")

    model = _build_model(spec, groups, cwl)
    # weights_only=True: refuse to unpickle arbitrary objects from a checkpoint. This path loads a
    # local state_dict we just wrote, so the risk is low, but phase8E already loads DOFA this way
    # and there is no reason for the two checkpoint paths to differ in safety posture.
    model.load_state_dict(torch.load(spec["state_dict"], map_location="cpu", weights_only=True))
    model.eval()
    res["params"] = count_params(model)
    res["size_mb"] = model_size_mb(model)

    bs = min(int(spec["bs"]), Xte.shape[0])
    res["batch_size"] = bs

    def make_inputs(device):
        """Inputs for THIS model only. The MLP branch returns a 1-tuple: no present-mask is
        created, so nothing unused can sit in the allocator during its memory measurement."""
        xb = torch.from_numpy(Xte[:bs]).to(device)
        if not is_attn:
            return (xb,)
        pm = torch.from_numpy(group_present_mask(bs, groups, [])).to(device)
        return (xb, pm)

    # ---- MACs on CPU, BEFORE any GPU allocation: fvcore's tracing allocates intermediates, and
    # doing it on the GPU would fold them into this model's peak-memory row.
    cpu_inp = make_inputs("cpu")
    macs_out = attempt("macs", lambda: flops_macs(model, cpu_inp), required=False)
    macs, m_unsup, m_uncalled, m_err = macs_out if macs_out is not None else (None, None, None, None)
    res["macs"] = macs
    res["macs_per_sample"] = (macs / bs) if macs is not None else None
    res["macs_note"] = _macs_note(macs, m_unsup, m_uncalled, m_err)

    bench_kw = dict(num_threads=int(spec["bench_threads"]), repeats=int(spec["repeats"]),
                    warmup=int(spec["warmup"]), min_run_time=float(spec["min_run_time"]),
                    max_run_time=float(spec["max_run_time"]),
                    target_rel_iqr=float(spec["target_rel_iqr"]))

    # ---- GPU phase (only on CUDA; never a CPU number under a GPU label) ----
    # `.get` with defaults, not `[...]`: the worker protocol is a JSON contract that outlives any
    # one driver version, and a spec written before a knob existed must still measure rather than
    # die with a KeyError naming a field its author never heard of.
    single_kw = dict(num_threads=int(spec["bench_threads"]), n=int(spec.get("single_n", 200)),
                     warmup=int(spec["warmup"]))
    if dev.type == "cuda":
        model = model.to(dev)
        gpu_inp = make_inputs(dev)
        mem = attempt("peak_mem", lambda: peak_mem_mb(model, gpu_inp, dev), required=True)
        if mem is not None:
            res["peak_mem_mb"] = mem["total"]
            res["peak_mem_baseline_mb"] = mem["baseline"]
            res["peak_mem_activation_mb"] = mem["activation"]
            res["peak_mem_reserved_mb"] = mem["reserved"]
            res["device_mem_used_mb"] = mem["device_used"]
            res["peak_mem_note"] = mem["note"]
        g = attempt("lat_gpu", lambda: bench_latency(model, gpu_inp, dev,
                                                     label=f"{spec['name']}/gpu", **bench_kw),
                    required=True)
        if g is not None:
            res["lat_gpu"] = g
        gs = attempt("lat_gpu_single",
                     lambda: bench_single_request(model, gpu_inp, dev, **single_kw), required=True)
        if gs is not None:
            res["lat_gpu_single"] = gs
        del gpu_inp
        model = model.to("cpu")
        torch.cuda.empty_cache()               # release this model's GPU blocks before CPU work

    # ---- CPU phase (always measured, always CPU-labelled) ----
    c = attempt("lat_cpu", lambda: bench_latency(model, cpu_inp, torch.device("cpu"),
                                                 label=f"{spec['name']}/cpu", **bench_kw),
                required=True)
    if c is not None:
        res["lat_cpu"] = c
    cs = attempt("lat_cpu_single",
                 lambda: bench_single_request(model, cpu_inp, torch.device("cpu"), **single_kw),
                 required=True)
    if cs is not None:
        res["lat_cpu_single"] = cs

    # ---- fp32 accuracy: clean + SEVERAL missing-group subsets ----
    n_sub = spec.get("n_missing_subsets", 5)
    subsets = missing_subsets(len(groups), int(spec["max_missing"]),
                              None if n_sub is None else int(n_sub), int(spec["subset_seed"]),
                              exhaustive_max=int(spec.get("subset_exhaustive_max", 512)))
    res["missing_subsets"] = [list(s) for s in subsets]
    res["missing_subset_coverage"] = subset_coverage(len(groups), int(spec["max_missing"]), subsets)
    acc_dev = dev
    model = model.to(acc_dev)

    def _acc(m, drop, device):
        if is_attn:
            return acc_prop(m, Xte, yte, groups, device, ncls, drop=drop)
        return acc_mlp(m, Xte, yte, device, ncls, groups=groups, drop=drop)

    a = attempt("miou_fp32_clean", lambda: _acc(model, (), acc_dev), required=True)
    if a is not None:
        res["miou_fp32_clean"] = a
    ms = attempt("miou_fp32_miss",
                 lambda: stat_block([_acc(model, s, acc_dev) for s in subsets]), required=True)
    if ms is not None:
        res["miou_fp32_miss"] = ms

    # ---- PARTIAL dynamic INT8 (CPU) with a VERIFIED scope ----
    model = model.to("cpu")
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    res["quant_api"] = QUANT_API
    res["quant_torch_version"] = torch.__version__
    res["quant_validated_torch"] = QUANT_VALIDATED_TORCH
    # WHICH kernels the int8 Linears dispatch to. fbgemm (x86), qnnpack (ARM) and onednn are
    # different implementations with different performance, and this was not recorded at all --
    # an INT8 speedup that cannot name the kernel that produced it is not reproducible on another
    # host, and the engine is chosen by the platform rather than by anything in this file.
    res["quant_engine"] = str(getattr(torch.backends.quantized, "engine", None))
    qmodel = attempt("quantize", lambda: quantize_int8(model), required=True)
    if qmodel is not None:
        rep = attempt("quant_scope", lambda: quantized_scope_report(model, qmodel), required=True)
        if rep is not None:
            res["quant_scope"] = rep["scope"]
            res["quant_n_quantized"] = rep["n_quantized"]
            res["quant_n_left_fp32"] = rep["n_left_fp32"]
        v = attempt("dynint8_size_mb", lambda: model_size_mb(qmodel), required=True)
        if v is not None:
            res["dynint8_size_mb"] = v
        q = attempt("dynint8_lat_cpu",
                    lambda: bench_latency(qmodel, cpu_inp, torch.device("cpu"),
                                          label=f"{spec['name']}/int8-cpu", **bench_kw),
                    required=True)
        if q is not None:
            res["dynint8_lat_cpu"] = q
        a = attempt("miou_dynint8_clean",
                    lambda: _acc(qmodel, (), torch.device("cpu")), required=True)
        if a is not None:
            res["miou_dynint8_clean"] = a
        ms = attempt("miou_dynint8_miss",
                     lambda: stat_block([_acc(qmodel, s, torch.device("cpu")) for s in subsets]),
                     required=True)
        if ms is not None:
            res["miou_dynint8_miss"] = ms

    with open(spec["out"], "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    # Exit 0 as long as the PROTOCOL completed: the driver owns the strictness decision, and it
    # needs this JSON (with its `errors`) to say WHICH measurement failed and why.
    return 0


# --------------------------------------------------------------------------------------
# driver: flatten a worker result into a CSV row
# --------------------------------------------------------------------------------------
def _lat(res, key, field):
    """Pull one field out of a latency summary, or None if that measurement never happened.

    `n` (the pooled replicate count) is returned as an int: it is a count, and a '4.000' in the
    CSV invites reading it as a duration."""
    d = res.get(key)
    if not isinstance(d, dict) or field not in d:
        return None
    v = d[field]
    return int(v) if field in ("n", "n_measurements", "n_requests", "repeats") else v


def result_to_row(res, spec, driver_pid):
    """One worker's JSON -> one CSV row. Absent metrics stay absent (-> 'n/a'), never zero."""
    miss_fp = res.get("miou_fp32_miss") or {}
    miss_q8 = res.get("miou_dynint8_miss") or {}
    subs = res.get("missing_subsets") or []
    pin = res.get("pinned") or {}
    errs = res.get("errors") or {}
    status = "ok" if not errs else "; ".join(f"{k}={v}" for k, v in sorted(errs.items()))
    # nvidia-smi index 0, which is torch's cuda:0 only while CUDA_VISIBLE_DEVICES is unset --
    # true for this driver (it passes `dev.type`, never an index) but not a property of the
    # world. The full per-GPU list goes into the provenance; this column is a drift indicator,
    # not an attribution.
    gpu = (res.get("gpu_state") or [{}])[0]
    return {
        "name": res.get("name"), "kind": res.get("kind"),
        "params": res.get("params"), "size_mb": res.get("size_mb"),
        "macs": res.get("macs"), "macs_per_sample": res.get("macs_per_sample"),
        "batch_size": res.get("batch_size"), "macs_note": res.get("macs_note"),
        "accel_device": res.get("accel_device"),
        "bench_threads": pin.get("bench_threads_requested"),
        "torch_threads": pin.get("torch_threads"),
        "interop_threads": pin.get("interop_threads"),
        "default_dtype": pin.get("default_dtype"),
        "usable_cores": pin.get("usable_cores"),
        "host_cores": pin.get("host_cores"),
        "gpu_clock_sm_mhz": gpu.get("clock_sm_mhz"), "gpu_temp_c": gpu.get("temp_c"),
        "peak_mem_mb": res.get("peak_mem_mb"),
        "peak_mem_baseline_mb": res.get("peak_mem_baseline_mb"),
        "peak_mem_activation_mb": res.get("peak_mem_activation_mb"),
        "peak_mem_reserved_mb": res.get("peak_mem_reserved_mb"),
        "device_mem_used_mb": res.get("device_mem_used_mb"),
        "peak_mem_note": res.get("peak_mem_note"),
        "lat_gpu_ms": _lat(res, "lat_gpu", "median_ms"),
        "lat_gpu_iqr_ms": _lat(res, "lat_gpu", "iqr_ms"),
        "lat_gpu_rel_iqr": _lat(res, "lat_gpu", "rel_iqr"),
        "lat_gpu_n": _lat(res, "lat_gpu", "n"),
        "lat_gpu_n_measurements": _lat(res, "lat_gpu", "n_measurements"),
        "lat_gpu_number_per_run": _lat(res, "lat_gpu", "number_per_run"),
        "lat_gpu_single_ms": _lat(res, "lat_gpu_single", "median_ms"),
        "lat_gpu_single_iqr_ms": _lat(res, "lat_gpu_single", "iqr_ms"),
        "lat_gpu_single_rel_iqr": _lat(res, "lat_gpu_single", "rel_iqr"),
        "lat_gpu_single_n": _lat(res, "lat_gpu_single", "n_requests"),
        "thr_gpu": _lat(res, "lat_gpu", "throughput_sps"),
        "lat_cpu_ms": _lat(res, "lat_cpu", "median_ms"),
        "lat_cpu_iqr_ms": _lat(res, "lat_cpu", "iqr_ms"),
        "lat_cpu_rel_iqr": _lat(res, "lat_cpu", "rel_iqr"),
        "lat_cpu_n": _lat(res, "lat_cpu", "n"),
        "lat_cpu_n_measurements": _lat(res, "lat_cpu", "n_measurements"),
        "lat_cpu_number_per_run": _lat(res, "lat_cpu", "number_per_run"),
        "lat_cpu_single_ms": _lat(res, "lat_cpu_single", "median_ms"),
        "lat_cpu_single_iqr_ms": _lat(res, "lat_cpu_single", "iqr_ms"),
        "lat_cpu_single_rel_iqr": _lat(res, "lat_cpu_single", "rel_iqr"),
        "lat_cpu_single_n": _lat(res, "lat_cpu_single", "n_requests"),
        "thr_cpu": _lat(res, "lat_cpu", "throughput_sps"),
        "repeats": spec.get("repeats"), "warmup_iters": spec.get("warmup"),
        "min_run_time_s": spec.get("min_run_time"),
        "quant_api": res.get("quant_api"), "quant_engine": res.get("quant_engine"),
        "quant_torch_version": res.get("quant_torch_version"),
        "quant_validated_torch": res.get("quant_validated_torch"),
        "quant_scope": res.get("quant_scope"),
        "quant_n_quantized": res.get("quant_n_quantized"),
        "quant_n_left_fp32": res.get("quant_n_left_fp32"),
        "dynint8_size_mb": res.get("dynint8_size_mb"),
        "dynint8_lat_cpu_ms": _lat(res, "dynint8_lat_cpu", "median_ms"),
        "dynint8_lat_cpu_iqr_ms": _lat(res, "dynint8_lat_cpu", "iqr_ms"),
        "dynint8_lat_cpu_rel_iqr": _lat(res, "dynint8_lat_cpu", "rel_iqr"),
        "dynint8_lat_cpu_n": _lat(res, "dynint8_lat_cpu", "n"),
        "dynint8_lat_cpu_number_per_run": _lat(res, "dynint8_lat_cpu", "number_per_run"),
        "miou_fp32_clean": res.get("miou_fp32_clean"),
        "miou_dynint8_clean": res.get("miou_dynint8_clean"),
        "missing_k": spec.get("max_missing"), "n_missing_subsets": len(subs),
        "missing_subset_coverage": res.get("missing_subset_coverage"),
        "missing_subsets": " | ".join("+".join(str(g) for g in s) for s in subs),
        "miou_fp32_miss_mean": miss_fp.get("mean"), "miou_fp32_miss_std": miss_fp.get("std"),
        "miou_fp32_miss_median": miss_fp.get("median"), "miou_fp32_miss_p05": miss_fp.get("p05"),
        "miou_fp32_miss_min": miss_fp.get("min"), "miou_fp32_miss_max": miss_fp.get("max"),
        "miou_fp32_miss_prefix": (miss_fp.get("values") or [None])[0],
        "miou_fp32_miss_values": ";".join(f"{v:.3f}" for v in (miss_fp.get("values") or [])),
        "miou_dynint8_miss_mean": miss_q8.get("mean"), "miou_dynint8_miss_std": miss_q8.get("std"),
        "miou_dynint8_miss_median": miss_q8.get("median"), "miou_dynint8_miss_p05": miss_q8.get("p05"),
        "miou_dynint8_miss_min": miss_q8.get("min"), "miou_dynint8_miss_max": miss_q8.get("max"),
        "miou_dynint8_miss_prefix": (miss_q8.get("values") or [None])[0],
        "miou_dynint8_miss_values": ";".join(f"{v:.3f}" for v in (miss_q8.get("values") or [])),
        # The worker's own verdict on its REQUIRED metrics. It always computed this and the driver
        # always ignored it, re-deriving the answer from absent columns -- two failure semantics
        # with only one consulted, which is how a metric that raised inside a nested `attempt`
        # could go missing without any column being obviously empty.
        "required_failed": ";".join(res.get("required_failed") or []),
        "worker_pid": res.get("worker_pid"), "driver_pid": driver_pid,
        "isolation": ("subprocess" if res.get("worker_pid") not in (None, driver_pid)
                      else "IN-PROCESS"),
        "timing_validity": res.get("timing_validity", "ok"),
        "status": status,
    }


CSV_COLS = ["name", "kind", "params", "size_mb", "macs", "macs_per_sample", "batch_size",
            "macs_note", "accel_device", "bench_threads", "torch_threads", "interop_threads",
            "default_dtype", "usable_cores", "host_cores", "gpu_clock_sm_mhz", "gpu_temp_c",
            "peak_mem_mb", "peak_mem_baseline_mb",
            "peak_mem_activation_mb", "peak_mem_reserved_mb", "device_mem_used_mb",
            "peak_mem_note",
            "lat_gpu_ms", "lat_gpu_iqr_ms", "lat_gpu_rel_iqr",
            "lat_gpu_n", "lat_gpu_n_measurements", "lat_gpu_number_per_run",
            "lat_gpu_single_ms", "lat_gpu_single_iqr_ms", "lat_gpu_single_rel_iqr",
            "lat_gpu_single_n", "thr_gpu",
            "lat_cpu_ms", "lat_cpu_iqr_ms", "lat_cpu_rel_iqr",
            "lat_cpu_n", "lat_cpu_n_measurements", "lat_cpu_number_per_run",
            "lat_cpu_single_ms", "lat_cpu_single_iqr_ms", "lat_cpu_single_rel_iqr",
            "lat_cpu_single_n", "thr_cpu",
            "repeats", "warmup_iters", "min_run_time_s", "quant_api", "quant_engine",
            "quant_torch_version", "quant_validated_torch", "quant_scope", "quant_n_quantized",
            "quant_n_left_fp32", "dynint8_size_mb", "dynint8_lat_cpu_ms",
            "dynint8_lat_cpu_iqr_ms", "dynint8_lat_cpu_rel_iqr", "dynint8_lat_cpu_n",
            "dynint8_lat_cpu_number_per_run", "miou_fp32_clean",
            "miou_dynint8_clean", "missing_k", "n_missing_subsets", "missing_subset_coverage",
            "missing_subsets",
            "miou_fp32_miss_mean", "miou_fp32_miss_std", "miou_fp32_miss_median",
            "miou_fp32_miss_p05", "miou_fp32_miss_min",
            "miou_fp32_miss_max", "miou_fp32_miss_prefix", "miou_fp32_miss_values",
            "miou_dynint8_miss_mean", "miou_dynint8_miss_std", "miou_dynint8_miss_median",
            "miou_dynint8_miss_p05", "miou_dynint8_miss_min",
            "miou_dynint8_miss_max", "miou_dynint8_miss_prefix", "miou_dynint8_miss_values",
            "required_failed",
            "worker_pid", "driver_pid", "isolation", "timing_validity", "status"]


def write_rows(fh, rows):
    """Write the table to an ALREADY-OPEN handle.

    Taking a handle rather than a path is deliberate: `tests/test_smoke_isolation.py` reads this
    module's AST and requires every paper/ artefact to be written through a call it recognises,
    with a path expression that varies with --smoke. Hiding the deliverable's `open()` inside a
    helper would make the guard blind to exactly the path it exists to protect, so the `open(P(
    ...{sfx}...))` stays inline at the call site and only the row formatting lives here."""
    w = csv.DictWriter(fh, fieldnames=CSV_COLS)
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(r.get(k)) for k in CSV_COLS})


# --------------------------------------------------------------------------------------
# strict gate — a benchmark that measured nothing must not look like one that ran
# --------------------------------------------------------------------------------------
def strict_failures(rows, dev_type, max_rel_iqr, check_dispersion=True, min_replicates=8,
                    expected_models=None, allow_unvalidated_quant=False,
                    max_rel_iqr_single=None):
    """Everything that disqualifies these rows from being a publication-grade measurement.

    Independent classes, because each can hold while the others fail:
      * the TABLE is not the table — a model row missing, duplicated or unexpected. This one is
        about the set of rows rather than any row, which is why iterating rows could not see it:
        `strict_failures([])` returned [], so an empty benchmark passed as a complete one, as did
        a table with no `Proposed` in it. Off unless `expected_models` is supplied, because the
        function is also used on hand-built single rows in tests; `main` always supplies it;
      * a REQUIRED metric is missing (its error text is in `status`);
      * the WORKER's own required-metric verdict (`required_failed`) is non-empty. The worker
        computed this all along and the driver threw it away;
      * a row was not measured in its own subprocess, so it is not comparable to its neighbours;
      * a thread pin did not take, so the CPU columns are not the pinned configuration and are
        not comparable to another run of this script;
      * a latency is not enough samples to have a dispersion at all, or is too dispersed to be a
        measurement (relative IQR over threshold) — the post-hoc counterpart to the pre-flight
        contention probe, catching load that arrived after the gate;
      * an "INT8" row where the scope report proves nothing was quantized, i.e. an int8 size and
        latency that describe the fp32 model wearing a different column name;
      * an INT8 number produced on a torch other than the one that path was VALIDATED against.
        Recording a version is not pinning one, and the row already carried both halves of the
        comparison without anyone making it.
    Returns a list of strings; empty means the table may be published."""
    required = list(REQUIRED_ALWAYS) + (list(REQUIRED_CUDA) if dev_type == "cuda" else [])
    out = []
    if expected_models is not None:
        want = list(expected_models)
        names = [r.get("name") for r in rows]
        absent = [m for m in want if m not in names]
        unexpected = sorted({n for n in names if n not in want})
        dupes = sorted({n for n in names if names.count(n) > 1})
        if absent:
            out.append(f"the table is missing {len(absent)} of the {len(want)} expected model "
                       f"row(s): {absent} -- a benchmark of a SUBSET of the models is not this "
                       f"benchmark (rows present: {names or 'NONE AT ALL'})")
        if unexpected:
            out.append(f"unexpected model row(s) {unexpected}; this table is defined as exactly "
                       f"{want}")
        if dupes:
            out.append(f"duplicate model row(s) {dupes} -- two measurements of one model would be "
                       f"silently averaged or overwritten by anything reading this by name")
    for r in rows:
        who = r.get("name", "?")
        for k in required:
            if _missing(r.get(k)):
                out.append(f"{who}: required measurement '{k}' is missing "
                           f"(status: {r.get('status')})")
        if r.get("isolation") != "subprocess":
            out.append(f"{who}: not measured in an isolated subprocess "
                       f"(worker_pid={r.get('worker_pid')} driver_pid={r.get('driver_pid')})")
        rf = r.get("required_failed")
        if rf:
            out.append(f"{who}: the worker recorded its own REQUIRED measurement(s) as failed: "
                       f"{rf} (status: {r.get('status')})")
        want, got = r.get("bench_threads"), r.get("torch_threads")
        if want is not None and got is not None and int(want) != int(got):
            out.append(f"{who}: requested {want} bench threads but torch reports {got} — the CPU "
                       f"columns are not the pinned configuration (pass --bench-threads {got})")
        # The INTER-op pool was recorded and never checked. It defaults to the HOST's core count
        # regardless of any cgroup quota, so on a capped box it is the one pin most likely not to
        # have taken -- and `set_num_interop_threads` can only be applied once, before any
        # parallel work, so a failure to apply it is silent by construction.
        got_i = r.get("interop_threads")
        if want is not None and got_i is not None and int(want) != int(got_i):
            out.append(f"{who}: requested {want} threads but torch reports {got_i} INTER-op "
                       f"threads — the inter-op pool is not the pinned configuration (it "
                       f"defaults to the host's core count, ignoring the cgroup quota)")
        if not allow_unvalidated_quant:
            ran, val = r.get("quant_torch_version"), r.get("quant_validated_torch")
            if ran and val and str(ran) != str(val):
                out.append(f"{who}: the INT8 path is validated against torch {val} but this run "
                           f"used {ran} — recording a version is not pinning one "
                           f"(--allow-unvalidated-quant to accept it deliberately)")
        if check_dispersion:
            for key, lab in (("lat_cpu_rel_iqr", "CPU"), ("lat_gpu_rel_iqr", "GPU"),
                             ("dynint8_lat_cpu_rel_iqr", "int8 CPU")):
                if key.startswith("lat_gpu") and dev_type != "cuda":
                    continue
                v = r.get(key)
                if _missing(v):
                    continue
                if float(v) > max_rel_iqr:
                    out.append(f"{who}: {lab} latency relative IQR {float(v):.3f} > "
                               f"{max_rel_iqr:.3f} — too noisy to report")
            # Single-request dispersion gets its OWN, looser threshold. These samples are single
            # forwards with no block averaging, so their natural spread is genuinely wider than
            # the steady-state cells' -- holding them to the same number would either reject good
            # measurements or force the steady-state threshold up to where it filters nothing.
            if max_rel_iqr_single is not None:
                for key, lab in (("lat_cpu_single_rel_iqr", "CPU single-request"),
                                 ("lat_gpu_single_rel_iqr", "GPU single-request")):
                    if key.startswith("lat_gpu") and dev_type != "cuda":
                        continue
                    v = r.get(key)
                    if _missing(v):
                        continue
                    if float(v) > max_rel_iqr_single:
                        out.append(f"{who}: {lab} latency relative IQR {float(v):.3f} > "
                                   f"{max_rel_iqr_single:.3f} — too noisy to report")
            # A median over two samples is not a median. Enforce that the repeats actually
            # produced replicates, so "median + IQR" is a real distribution rather than a label
            # on a measurement budget that collapsed to one block.
            for key, lab in (("lat_cpu_n", "CPU"), ("lat_gpu_n", "GPU"),
                             ("dynint8_lat_cpu_n", "int8 CPU")):
                if key.startswith("lat_gpu") and dev_type != "cuda":
                    continue
                v = r.get(key)
                if _missing(v):
                    continue
                if int(v) < min_replicates:
                    out.append(f"{who}: {lab} latency pooled only {int(v)} replicate(s) < "
                               f"{min_replicates} — raise --repeats/--min-run-time")
        if r.get("quant_n_quantized") is not None and int(r["quant_n_quantized"]) == 0:
            out.append(f"{who}: quantisation produced 0 quantized modules — the reported int8 "
                       f"size/latency would describe the fp32 model")
    return out


# --------------------------------------------------------------------------------------
# train here, measure THERE — the only place that spawns workers
# --------------------------------------------------------------------------------------
def _run_models(args, dev, workdir, validity, meta):
    """Train the four models in THIS process, then measure each one in its OWN subprocess.

    Returns ``(rows, hard_failures)``. A hard failure is a worker that died or produced no result
    file at all — distinct from a worker that ran and recorded a per-metric error, which comes
    back inside the row's `status`. `meta` is filled in with dataset facts the provenance stamp
    needs (it is an out-parameter so this whole stage stays one substitutable seam, which is what
    the tests drive to exercise the gate/report paths without spending GPU minutes)."""
    # phase2_degradation is imported lazily: it pulls in the dataset loader, matplotlib and the
    # training loops, none of which a measurement worker should ever have in its address space.
    from phase2_degradation import (load_data, prep, train_mlp, pretrain_sgmae,
                                    finetune_proposed, train_hcs, NUM_CLASSES)
    cube, gt = load_data()
    Xtr, ytr, Xte, yte = prep(cube, gt, block=10, offset=args.seed)
    C = Xtr.shape[1]
    groups = contiguous_groups(C, args.groups)
    cwl = group_center_wavelengths(AVIRIS_WL_NM, groups)
    meta.update(n_train_px=int(Xtr.shape[0]), n_test_px=int(Xte.shape[0]), n_bands=int(C))
    print(f"cube {cube.shape} | train {Xtr.shape[0]} test {Xte.shape[0]} px | "
          f"{C} bands, {args.groups} groups")

    print("training MLP baseline + Proposed + B4 + B6 (1 seed) ...")
    m_mlp = train_mlp(Xtr, ytr, groups, args.seed, group_dropout=False, epochs=args.epochs)
    m_prop = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES)
    pretrain_sgmae(m_prop, Xtr, groups, args.seed, epochs=max(1, args.epochs // 2))
    finetune_proposed(m_prop, Xtr, ytr, groups, args.seed, epochs=args.epochs)
    m_b4 = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES, pe_type="learned")
    train_hcs(m_b4, Xtr, ytr, groups, args.seed, epochs=args.epochs)
    m_b6 = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES, pe_type="learned")
    pretrain_sgmae(m_b6, Xtr, groups, args.seed, epochs=max(1, args.epochs // 2))
    finetune_proposed(m_b6, Xtr, ytr, groups, args.seed, epochs=args.epochs, group_dropout=False)

    specs_src = [("MLP (B1)", m_mlp, "mlp", "sinusoidal"),
                 ("B4 attn", m_b4, "attn", "learned"),
                 ("B6 attn", m_b6, "attn", "learned"),
                 ("Proposed", m_prop, "attn", "sinusoidal")]
    # The strict gate checks the produced rows against EXPECTED_MODELS. If someone adds a model
    # here and not there, the gate would reject every future run as "unexpected row"; if someone
    # removes one, the gate would reject it as "missing row". Either way the failure should
    # surface HERE, at the one line that knows both, rather than as a mystery in the artefact.
    assert tuple(n for n, _, _, _ in specs_src) == EXPECTED_MODELS, (
        f"the trained roster {tuple(n for n, _, _, _ in specs_src)} does not match "
        f"EXPECTED_MODELS {EXPECTED_MODELS}")

    # ---- hand the workers everything they need, then drop the models from THIS process ----
    data_npz = os.path.join(workdir, "test_data.npz")
    np.savez(data_npz, Xte=Xte, yte=yte)
    groups_json = os.path.join(workdir, "groups.json")
    with open(groups_json, "w", encoding="utf-8") as f:
        json.dump({"groups": [np.asarray(g).tolist() for g in groups],
                   "cwl": np.asarray(cwl).tolist()}, f)

    specs = []
    for i, (name, model, kind, pe) in enumerate(specs_src):
        sd_path = os.path.join(workdir, f"model_{i}.pt")
        torch.save(model.to("cpu").state_dict(), sd_path)
        specs.append({
            "index": i, "name": name, "kind": kind, "pe_type": pe,
            "state_dict": sd_path, "data_npz": data_npz, "groups_json": groups_json,
            "out": os.path.join(workdir, f"result_{i}.json"),
            "device": dev.type, "seed": args.seed, "num_classes": NUM_CLASSES,
            "n_bands": int(C), "bs": int(args.bs),
            "bench_threads": int(args.bench_threads), "repeats": int(args.repeats),
            "warmup": int(args.warmup), "min_run_time": float(args.min_run_time),
            "max_run_time": float(args.max_run_time),
            "target_rel_iqr": float(args.target_rel_iqr),
            "max_missing": int(args.max_missing),
            # None => exhaustive when C(groups, max_missing) fits under the cap. Passed through
            # as None rather than resolved here so the worker's `missing_subsets` makes the one
            # decision, and the coverage string it returns describes what it actually did.
            "n_missing_subsets": (None if args.missing_subsets is None
                                  else int(args.missing_subsets)),
            "subset_exhaustive_max": int(args.subset_exhaustive_max),
            "subset_seed": int(args.subset_seed),
            "single_n": int(args.single_requests),
        })
    # Free the driver's GPU memory: the workers get a clean allocator of their own, but a driver
    # still holding four models resident would be occupying the same physical GPU they measure on.
    del m_mlp, m_prop, m_b4, m_b6, specs_src
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- one subprocess per model ----
    rows, hard_failures = [], []
    for spec in specs:
        spec_path = os.path.join(workdir, f"spec_{spec['index']}.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        # The thread pin is applied through the ENVIRONMENT as well as through torch, because the
        # BLAS libraries read it at load time — setting torch's thread count after OpenMP has
        # already sized its pool does not undo an oversubscribed pool.
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS=str(args.bench_threads),
                   MKL_NUM_THREADS=str(args.bench_threads),
                   OPENBLAS_NUM_THREADS=str(args.bench_threads),
                   NUMEXPR_NUM_THREADS=str(args.bench_threads),
                   BANDSIM_THREADS=str(args.bench_threads),
                   BANDSIM_DEVICE=dev.type,
                   CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   PYTHONHASHSEED=str(args.seed))
        print(f"[worker] {spec['name']} -> subprocess "
              f"(threads={args.bench_threads}, repeats={args.repeats})")
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker", spec_path],
                env=env, capture_output=True, text=True, timeout=args.worker_timeout)
        except subprocess.TimeoutExpired as e:
            # A timeout used to propagate out of here, past main()'s try (which had no except),
            # and into the finally that deletes the workdir -- so the ONE failure mode that
            # costs the most to reproduce (30 minutes of training plus a hung measurement)
            # destroyed the trained weights, every other worker's result JSON, and the stderr
            # that would say what hung. It is a hard failure like any other now: the remaining
            # models still get measured and the diagnostic table still gets written.
            tail = "\n".join(((e.stderr or "") if isinstance(e.stderr, str)
                              else (e.stderr or b"").decode(errors="replace")
                              ).strip().splitlines()[-25:])
            hard_failures.append(f"{spec['name']}: worker exceeded --worker-timeout "
                                 f"{args.worker_timeout:.0f}s and was killed\n{tail}")
            continue
        if proc.returncode != 0 or not os.path.exists(spec["out"]):
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
            hard_failures.append(f"{spec['name']}: worker exited {proc.returncode} "
                                 f"without a usable result\n{tail}")
            continue
        with open(spec["out"], encoding="utf-8") as f:
            res = json.load(f)
        res["timing_validity"] = validity
        row = result_to_row(res, spec, os.getpid())
        meta["effective_batch_size"] = row.get("batch_size")
        rows.append(row)
        print(f"  {row['name']:10s} params={row['params']/1e3:5.1f}k "
              f"size={row['size_mb']:.2f}MB "
              f"GPU {_fmt(row['lat_gpu_ms'], '.3f')}±{_fmt(row['lat_gpu_iqr_ms'], '.3f')}ms "
              f"CPU {_fmt(row['lat_cpu_ms'], '.3f')}±{_fmt(row['lat_cpu_iqr_ms'], '.3f')}ms "
              f"| int8 {_fmt(row['dynint8_size_mb'], '.2f')}MB "
              f"({row['quant_n_quantized']} mod) | mIoU clean "
              f"{_fmt(row['miou_fp32_clean'], '.1f')} miss{args.max_missing} "
              f"{_fmt(row['miou_fp32_miss_mean'], '.1f')}±"
              f"{_fmt(row['miou_fp32_miss_std'], '.1f')} "
              f"over {row['n_missing_subsets']} subsets | pid {row['worker_pid']}")
    return rows, hard_failures


# --------------------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=256, help="batch size for latency/memory bench")
    ap.add_argument("--max-missing", type=int, default=6)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                    help="device for training + GPU-labelled bench (default: auto; also BANDSIM_DEVICE)")
    # --- measurement budget (item 1) ---
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent adaptive_autorange measurements pooled per latency cell")
    ap.add_argument("--warmup", type=int, default=50,
                    help="untimed forward passes before each latency cell")
    ap.add_argument("--min-run-time", type=float, default=0.5,
                    help="seconds of sampling per repeat (torch.utils.benchmark)")
    ap.add_argument("--max-run-time", type=float, default=10.0,
                    help="hard cap per repeat when the IQR target is not reached")
    ap.add_argument("--target-rel-iqr", type=float, default=0.05,
                    help="adaptive_autorange stops once iqr/median falls under this")
    ap.add_argument("--max-rel-iqr", type=float, default=0.10,
                    help="strict mode rejects any steady-state latency noisier than this")
    ap.add_argument("--max-rel-iqr-single", type=float, default=0.25,
                    help="the same gate for SINGLE-REQUEST latency, which has no block averaging "
                         "and so a genuinely wider natural spread")
    ap.add_argument("--min-replicates", type=int, default=8,
                    help="strict mode rejects a latency pooled from fewer replicates than this")
    ap.add_argument("--single-requests", type=int, default=200,
                    help="synchronized one-forward-per-interval samples per single-request cell")
    # --- pinned environment (item 3) ---
    ap.add_argument("--bench-threads", type=int, default=None,
                    help="CPU threads pinned for EVERY timing measurement (recorded in the CSV). "
                         "Default: usable cores minus 2 -- pinning the whole cgroup quota leaves "
                         "nothing for the main thread or the CUDA driver and shows up as CFS "
                         "throttling in the middle of a measurement")
    # --- missing-band subsets (item 4) ---
    ap.add_argument("--missing-subsets", type=int, default=None,
                    help="how many distinct missing-group subsets to average the drop over "
                         "(default: ALL of them when C(groups, max-missing) <= "
                         "--subset-exhaustive-max, which at the defaults means all 210)")
    ap.add_argument("--subset-exhaustive-max", type=int, default=512,
                    help="enumerate every missing-group subset when there are no more than this")
    ap.add_argument("--subset-seed", type=int, default=0)
    # --- failure policy (item 5) ---
    ap.add_argument("--strict", dest="strict", action="store_true", default=True,
                    help="(default) exit nonzero and write NO paper artefact if a required "
                         "measurement is missing/noisy")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="write the artefacts anyway, with every failure recorded in `status`")
    # --- contended machine ---
    ap.add_argument("--allow-contended", action="store_true",
                    help="benchmark even though the box is busy; every row is stamped CONTENDED")
    ap.add_argument("--allow-unvalidated-quant", action="store_true",
                    help="accept INT8 numbers from a torch other than QUANT_VALIDATED_TORCH")
    ap.add_argument("--gpu-util-max", type=float, default=15.0)
    ap.add_argument("--load-per-core-max", type=float, default=0.6,
                    help="refuse when the 1-minute load average exceeds this per HOST core")
    ap.add_argument("--max-throttle-frac", type=float, default=0.01,
                    help="strict mode rejects the run when more than this fraction of the "
                         "cgroup's CPU scheduling periods were throttled DURING it")
    # --- plumbing ---
    ap.add_argument("--workdir", default=None,
                    help="where trained weights + per-worker JSON go (default: a temp dir)")
    ap.add_argument("--keep-workdir", action="store_true")
    ap.add_argument("--worker-timeout", type=float, default=1800.0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny/quick machinery check: few epochs, small batch, minimal "
                         "measurement budget (numbers are NOT for the paper)")
    return ap


def validate_args(args):
    """Reasons this configuration cannot produce a meaningful table. Empty list == usable.

    argparse checks TYPES, not meanings, and none of these crashed — each one produced a
    complete-looking artefact describing a run that did not happen:
      * `--bs -1` reaches `min(-1, len(Xte))` = -1, so the bench slices `Xte[:-1]` (the whole
        test set bar one row) while `batch_size` records -1 and `thr_cpu`, being batch/median,
        comes out NEGATIVE.
      * `--repeats -10` is clamped by a `max(1, ...)` inside the bench loop, so exactly one
        measurement is taken while the CSV and the .tex both claim -10 repeats.
      * `--max-missing -1` yields the single empty subset, so `miou_fp32_miss_mean` is the CLEAN
        accuracy under a missing-band column name, in a row labelled `missing_k=-1`.
      * `--bench-threads 0` asks torch for zero threads.
    A number is only as good as the provenance beside it; a provenance that describes a
    different run is worse than none, because it invites arithmetic."""
    bad = []
    if args.bs <= 0:
        bad.append(f"--bs must be >= 1 (got {args.bs}); a negative batch slices the test set "
                   f"from the end and makes throughput negative")
    if args.groups <= 0:
        bad.append(f"--groups must be >= 1 (got {args.groups})")
    if args.epochs < 0:
        bad.append(f"--epochs must be >= 0 (got {args.epochs})")
    if not (0 <= args.max_missing <= args.groups):
        bad.append(f"--max-missing must be in [0, --groups={args.groups}] (got "
                   f"{args.max_missing}); outside it the missing-band columns report the clean "
                   f"accuracy or nothing at all")
    if args.repeats < 1:
        bad.append(f"--repeats must be >= 1 (got {args.repeats}); the bench clamps it to 1 and "
                   f"the artefact would still claim {args.repeats}")
    if args.warmup < 0:
        bad.append(f"--warmup must be >= 0 (got {args.warmup})")
    if args.single_requests < 2:
        bad.append(f"--single-requests must be >= 2 (got {args.single_requests}); one sample has "
                   f"no dispersion")
    if args.min_run_time <= 0:
        bad.append(f"--min-run-time must be > 0 (got {args.min_run_time})")
    if args.max_run_time < args.min_run_time:
        bad.append(f"--max-run-time ({args.max_run_time}) is below --min-run-time "
                   f"({args.min_run_time})")
    if args.target_rel_iqr <= 0:
        bad.append(f"--target-rel-iqr must be > 0 (got {args.target_rel_iqr})")
    if args.max_rel_iqr <= 0:
        bad.append(f"--max-rel-iqr must be > 0 (got {args.max_rel_iqr})")
    if args.bench_threads is not None and args.bench_threads < 1:
        bad.append(f"--bench-threads must be >= 1 (got {args.bench_threads})")
    if args.missing_subsets is not None and args.missing_subsets < 1:
        bad.append(f"--missing-subsets must be >= 1 or omitted for exhaustive (got "
                   f"{args.missing_subsets})")
    if args.min_replicates < 1:
        bad.append(f"--min-replicates must be >= 1 (got {args.min_replicates})")
    if args.worker_timeout <= 0:
        bad.append(f"--worker-timeout must be > 0 (got {args.worker_timeout})")
    return bad


def stale_deliverables(sfx):
    """Deliverables from an EARLIER run still sitting in paper/, with their age.

    A strict failure writes nothing, which is correct — and is also exactly how the PREVIOUS
    run's table stays on disk, quotable, with nothing to say it is not this run's. That is not
    hypothetical here: paper/ held a phase-7 CSV in a 15-column schema this script stopped
    emitting four rewrites earlier, with no provenance sidecar, and the only symptom was an
    absence. Naming them on the way out is the cheap half; the standing check that a deliverable
    must not predate its own generator lives in scripts/doctor.py."""
    out = []
    for name in (f"results_phase7_efficiency{sfx}.csv",
                 f"results_phase7_efficiency{sfx}.tex",
                 f"results_phase7_efficiency{sfx}.csv.provenance.json"):
        p = P(name)
        try:
            if os.path.exists(p):
                out.append(f"{p} (last written {(time.time() - os.path.getmtime(p)) / 3600:.1f} h "
                           f"ago, by an EARLIER run)")
        except OSError:
            continue
    return out


def render_tex(prop, args, validity):
    """The \\prop* macro block for the Proposed row, as a string.

    Rendered rather than streamed so that nothing which can raise happens between writing the
    CSV and writing this: the bundle in paper/ used to be able to end up part this run and part
    the last one. Macro NAMES are kept stable for the paper (the ICAIMS one is frozen under
    archive/icaims2026_submission/); only the VALUES are made honest.

    \\propMacs is PER-SAMPLE MACs and 'n/a' (not 0) when fvcore cannot COMPLETELY count the model
    — the case here, since it cannot count scaled_dot_product_attention. The superseded artefact
    this replaces wrote 174.6M into that macro: the fvcore total over the WHOLE batch of 256,
    i.e. 256x the per-sample figure, and an undercount besides. Both errors were invisible in a
    file that just said `\\newcommand{\\propMacs}{174.6M}`.

    The latency macros are the STEADY-STATE medians and ship with an IQR macro each; the
    single-request macros are separate and are the ones a deployment claim wants."""
    g = prop.get
    mps = g("macs_per_sample")
    macs_tex = f"{mps / 1e6:.1f}M" if not _missing(mps) else "n/a"
    params = g("params")
    params_tex = "n/a" if _missing(params) else f"{params / 1e3:.1f}k"
    L = []
    L.append(f"% Phase 7 efficiency — Indian Pines, seed {args.seed}, batch "
             f"{g('batch_size')}, accel={g('accel_device')}, "
             f"{g('bench_threads')} pinned CPU threads\n")
    L.append(f"% timing_validity={validity} | \\propLatGpu/\\propLatCpu are STEADY-STATE medians "
             f"over {args.repeats} pooled torch.utils.benchmark repeats "
             f"(number_per_run={g('lat_gpu_number_per_run')}/{g('lat_cpu_number_per_run')}), "
             f"per-model subprocess-isolated\n")
    L.append("% \\propLatGpuSingle/\\propLatCpuSingle are SINGLE-REQUEST latencies: one forward "
             "per timed interval, synchronized both sides. Quote these for a deployment or "
             "edge-latency claim, not the steady-state pair.\n")
    if validity != "ok":
        L.append("% WARNING: these numbers were NOT measured on a quiet, "
                 "publication-grade machine — do not quote them as such.\n")
    L.append("% propMacs = PER-SAMPLE MACs (fvcore, one fused multiply-add = 1, i.e. "
             "MAC not 2xMAC FLOP), reported only when fvcore's count is COMPLETE.\n")
    L.append(f"% propMiouMiss* are over {g('missing_subset_coverage')} missing-group subsets at "
             f"ONE seed, so \\propMiouMissStd is subset variability, NOT seed variability. Do not "
             f"write it as a seed error bar; the multi-seed accuracy claim belongs to phase 1/4R.\n")
    L.append(f"% propSizeInt / propMiouInt are PARTIAL dynamic-int8. Verified scope: "
             f"{g('quant_scope')}\n")
    L.append(f"% quantisation API: {g('quant_api')}, engine {g('quant_engine')}, validated "
             f"against torch {g('quant_validated_torch')}, run on torch "
             f"{g('quant_torch_version')}\n")
    L.append(f"% peak memory is ALLOCATOR-tracked ({g('peak_mem_mb')} MB); this process occupies "
             f"{g('device_mem_used_mb')} MB on the card once the CUDA context is counted.\n")
    for macro, val in (
            ("propParams", params_tex),
            ("propMacs", macs_tex),
            ("propLatGpu", _fmt(g("lat_gpu_ms"), ".3f")),
            ("propLatGpuIqr", _fmt(g("lat_gpu_iqr_ms"), ".3f")),
            ("propLatGpuSingle", _fmt(g("lat_gpu_single_ms"), ".3f")),
            ("propLatCpu", _fmt(g("lat_cpu_ms"), ".3f")),
            ("propLatCpuIqr", _fmt(g("lat_cpu_iqr_ms"), ".3f")),
            ("propLatCpuSingle", _fmt(g("lat_cpu_single_ms"), ".3f")),
            ("propSizeFp", _fmt(g("size_mb"), ".2f")),
            ("propSizeInt", _fmt(g("dynint8_size_mb"), ".2f")),
            ("propPeakMemAlloc", _fmt(g("peak_mem_mb"), ".1f")),
            ("propMiouFp", _fmt(g("miou_fp32_clean"), ".1f")),
            ("propMiouInt", _fmt(g("miou_dynint8_clean"), ".1f")),
            ("propMiouMissMean", _fmt(g("miou_fp32_miss_mean"), ".1f")),
            ("propMiouMissStd", _fmt(g("miou_fp32_miss_std"), ".1f")),
            ("propMiouMissMedian", _fmt(g("miou_fp32_miss_median"), ".1f")),
            ("propMiouMissPfive", _fmt(g("miou_fp32_miss_p05"), ".1f")),
            ("propMissSubsets", g("n_missing_subsets")),
            ("propMissCoverage", g("missing_subset_coverage")),
            ("propBenchThreads", g("bench_threads")),
            ("propBenchRepeats", args.repeats),
            ("propQuantEngine", g("quant_engine")),
            ("propTimingValidity", validity)):
        # A macro must never expand to the string 'None'. The numeric ones already go through
        # _fmt, but the plain pass-throughs above are `.get()` results, and a paper containing
        # the literal word None where a thread count should be is the same failure as a
        # fabricated 0 -- it just looks like a typo instead of a measurement.
        L.append(f"\\newcommand{{\\{macro}}}{{{'n/a' if val is None else val}}}\n")
    return "".join(L)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    # Resolve the adaptive default BEFORE validating and before recording anything: every
    # artefact records `bench_threads`, and recording None (or re-deriving it later) would make
    # the CSV disagree with the configuration that produced it.
    if args.bench_threads is None:
        args.bench_threads = max(1, hw.available_cores() - 2)
        print(f"[threads] --bench-threads -> {args.bench_threads} "
              f"({hw.available_cores()} usable cores, 2 left for the main thread and the CUDA "
              f"driver: pinning the entire cgroup quota makes the benchmark contend with itself "
              f"and surfaces as CFS throttling mid-measurement)")
    bad = validate_args(args)
    if bad:
        print("REFUSING to run: this configuration cannot produce a meaningful table:",
              file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 2
    # --smoke is especially treacherous here: it changes --bs AND the measurement budget, and
    # every latency/throughput/peak-mem number is a function of both. A smoke run writing the real
    # CSV would not look obviously wrong, just quietly measured at bs=64 with 1 repeat. Give it its
    # own filenames.
    sfx = ""
    if args.smoke:
        args.epochs = 4
        args.bs = min(args.bs, 64)
        args.repeats = min(args.repeats, 2)
        args.warmup = min(args.warmup, 3)
        args.min_run_time = min(args.min_run_time, 0.02)
        args.max_run_time = min(args.max_run_time, 0.2)
        args.missing_subsets = 3 if args.missing_subsets is None else min(args.missing_subsets, 3)
        args.single_requests = min(args.single_requests, 5)
        sfx = "_smoke"
        print("[smoke] 4 epochs / bs<=64 / minimal timing budget — writing *_smoke artefacts, "
              "NOT the real deliverables; timings are machinery checks, not measurements")

    # ---- contention gate: BEFORE touching CUDA, so any compute process we see is someone else's
    want_cuda = (args.device or os.environ.get("BANDSIM_DEVICE") or "auto").lower() != "cpu"
    pre = probe_contention()
    reasons = contention_reasons(pre, want_cuda and hw.n_gpus() > 0,
                                 args.gpu_util_max, args.load_per_core_max,
                                 expected_own_compute_apps=0)
    unknowns = contention_unknowns(pre, want_cuda and hw.n_gpus() > 0)
    validity = "ok"
    if unknowns and not reasons:
        # Not a refusal: we cannot call a box busy on evidence we do not have. But the verdict
        # stops being `ok`, because `ok` asserts a check that did not happen.
        print("[contention] could NOT check some signals; this run will be stamped "
              "timing_validity=ok-unverified:")
        for u in unknowns:
            print(f"  - {u}")
    if reasons:
        print("[contention] this machine is BUSY:")
        for r in reasons:
            print(f"  - {r}")
        for s in stale_deliverables(sfx):
            print(f"[stale] STILL ON DISK from an earlier run: {s}")
        if args.smoke:
            validity = "smoke-on-contended-box"
            print("[contention] --smoke: continuing (machinery check only, output is suffixed)")
        elif args.allow_contended:
            validity = "CONTENDED"
            print("[contention] --allow-contended given: continuing, but every row will be "
                  "stamped timing_validity=CONTENDED and must NOT be quoted as a clean number")
        else:
            print("\nREFUSING to benchmark: a latency measured on a contended box is noise, not a\n"
                  "result. Wait for the box to be idle, or pass --allow-contended to record\n"
                  "explicitly-contaminated numbers, or use --smoke to validate the machinery.",
                  file=sys.stderr)
            return 3
    elif args.smoke:
        validity = "smoke"
    elif unknowns:
        validity = "ok-unverified"

    dev = hw.setup(seed=args.seed, deterministic=True, prefer=args.device)
    print("HW:", hw.info())
    if dev.type != "cuda":
        print("NOTE: running without CUDA -> GPU-labelled columns (lat_gpu_ms, thr_gpu, "
              "peak_mem_mb) are reported as n/a; CPU columns are still measured.")

    workdir = args.workdir or tempfile.mkdtemp(prefix="phase7_efficiency_")
    os.makedirs(workdir, exist_ok=True)
    keep = args.keep_workdir
    meta = {}                                  # filled in by _run_models, read by the stamp

    try:
        rows, hard_failures = _run_models(args, dev, workdir, validity, meta)
        if hard_failures:
            keep = True

        # ---- strict gate ----
        failures = list(hard_failures)
        failures += strict_failures(rows, dev.type, args.max_rel_iqr,
                                    check_dispersion=not args.smoke,
                                    min_replicates=args.min_replicates,
                                    # The roster applies to --smoke too: smoke runs the SAME
                                    # _run_models and trains the same four models, so exempting
                                    # it would only hide a worker that died. (The version pin is
                                    # exempted, because an unvalidated torch is a policy failure
                                    # rather than a machinery one, and --smoke exists to test the
                                    # machinery on whatever stack is present. The mismatch still
                                    # travels in the quant_torch_version column.)
                                    expected_models=EXPECTED_MODELS,
                                    allow_unvalidated_quant=(args.allow_unvalidated_quant
                                                             or args.smoke),
                                    max_rel_iqr_single=(None if args.smoke
                                                        else args.max_rel_iqr_single))
        # Re-probe AFTER the measurements: the pre-flight gate can only see the load that existed
        # when we started, and a job that arrived halfway through contaminates the rows just as
        # thoroughly as one that was already running.
        post = probe_contention()
        if contention_reasons(post, dev.type == "cuda", args.gpu_util_max,
                              args.load_per_core_max, expected_own_compute_apps=1):
            if validity in ("ok", "ok-unverified"):
                validity = "CONTENDED-DURING-RUN"
                failures.append("the box became contended DURING the run — the latencies above "
                                "are not a clean measurement (re-run on an idle box)")
        # Did WE get the CPU we pinned? The load average cannot answer that; the cgroup's own
        # throttle counter can, and it is the only signal here measured on this process rather
        # than on the machine. A run that spent part of its time descheduled by the CFS quota
        # measured the quota, not the model.
        t0, t1 = pre.get("cgroup_throttle"), post.get("cgroup_throttle")
        if t0 and t1:
            d_periods, d_throttled = t1[0] - t0[0], t1[1] - t0[1]
            frac = (d_throttled / d_periods) if d_periods > 0 else 0.0
            if frac > args.max_throttle_frac:
                if validity in ("ok", "ok-unverified"):
                    validity = "CPU-THROTTLED"
                failures.append(
                    f"the cgroup throttled this process in {d_throttled}/{d_periods} "
                    f"({100 * frac:.1f}%) of CPU scheduling periods DURING the run > "
                    f"{100 * args.max_throttle_frac:.1f}% — the CPU latencies measure the quota, "
                    f"not the model (lower --bench-threads)")
        # `validity` is stamped onto every row HERE and nowhere else. It is decided across three
        # separate points (pre-flight gate, --smoke, post-run re-probe), so writing it at each of
        # them is how a row ends up carrying a verdict that was later revised — i.e. a
        # contaminated measurement labelled clean.
        for r in rows:
            r["timing_validity"] = validity

        if failures and args.strict:
            # The diagnostic goes to the WORKDIR, never to paper/: a run that failed its own
            # checks must not leave a quotable table behind, but it must still leave enough to
            # debug with. Nothing here is a deliverable.
            keep = True
            diag = os.path.join(workdir, f"results_phase7_efficiency{sfx}.FAILED.csv")
            with open(diag, "w", newline="") as f:
                write_rows(f, rows)
            print("\nSTRICT MODE: refusing to write paper artefacts. "
                  f"{len(failures)} problem(s):", file=sys.stderr)
            for fmsg in failures:
                print(f"  - {fmsg}", file=sys.stderr)
            print(f"\ndiagnostic table (NOT a deliverable): {diag}", file=sys.stderr)
            # Refusing to write is only half of "leaves no quotable table behind": whatever the
            # LAST successful run wrote is still there, still looks like a result, and now has a
            # newer failed run standing invisibly behind it. Name it.
            left = stale_deliverables(sfx)
            if left:
                print("\nWARNING: this run wrote nothing, but EARLIER artefacts remain in paper/ "
                      "and are NOT this run's output:", file=sys.stderr)
                for s in left:
                    print(f"  - {s}", file=sys.stderr)
            return 2
        if failures:
            print(f"\nWARNING (--no-strict): {len(failures)} problem(s) recorded in the CSV:")
            for fmsg in failures:
                print(f"  - {fmsg}")

        # ---- RENDER every artefact before WRITING any of them ----
        # The order used to be: write CSV, then maybe write the .tex, then stamp. Each step could
        # fail after the previous one had landed, so paper/ could end up part this run and part
        # the last one — and that is not hypothetical, it is the state this repo was found in: a
        # CSV and a .tex with no provenance beside them. Rendering first means the only thing
        # left at write time is I/O, and the three files stand or fall together.
        buf = io.StringIO()
        write_rows(buf, rows)
        csv_text = buf.getvalue()
        prop = next((r for r in rows if r.get("name") == "Proposed"), None)
        tex_text = render_tex(prop, args, validity) if prop is not None else None

        with open(P(f"results_phase7_efficiency{sfx}.csv"), "w", newline="") as f:
            f.write(csv_text)

        # ---- tex macros (Proposed row) ----
        if tex_text is not None:
            with open(P(f"results_phase7_efficiency{sfx}.tex"), "w") as f:
                f.write(tex_text)
        else:
            # No Proposed row means no macros — but the PREVIOUS run's .tex is still on disk, and
            # its \propLatGpu would be read as this run's number by any document that \input's it.
            # Overwriting with an invalidation block (rather than deleting) makes every macro
            # expand to a visible marker, so the failure surfaces in the rendered PDF instead of
            # as a stale number nobody re-checked.
            with open(P(f"results_phase7_efficiency{sfx}.tex"), "w") as f:
                f.write("% INVALIDATED: this phase-7 run produced no 'Proposed' row, so it has no\n"
                        "% macros to offer. The previous run's values were HERE and have been\n"
                        "% overwritten deliberately -- quoting them as this run's output is the\n"
                        "% exact failure this file is preventing.\n")
                for macro in ("propParams", "propMacs", "propLatGpu", "propLatGpuIqr",
                              "propLatGpuSingle", "propLatCpu", "propLatCpuIqr",
                              "propLatCpuSingle", "propSizeFp", "propSizeInt", "propPeakMemAlloc",
                              "propMiouFp", "propMiouInt", "propMiouMissMean", "propMiouMissStd",
                              "propMiouMissMedian", "propMiouMissPfive", "propMissSubsets",
                              "propMissCoverage", "propBenchThreads", "propBenchRepeats",
                              "propQuantEngine", "propTimingValidity"):
                    f.write(f"\\newcommand{{\\{macro}}}{{INVALID-NO-PROPOSED-ROW}}\n")
            print("[tex] no 'Proposed' row in this run -> the macro file was overwritten with "
                  "INVALID markers so no stale value can be quoted")

        # An efficiency table is meaningless detached from the machine, the batch size and the
        # thread pin that produced it: the same model is milliseconds apart on V100 vs CPU, `bs`
        # scales throughput and peak memory directly, and the CPU columns are only comparable at a
        # fixed thread count. Record all of it, plus how busy the box was before and after.
        prov = stamp(P(f"results_phase7_efficiency{sfx}.csv"), args,
              extra={"accel_device": dev.type,
                     # BOTH batch sizes. `bs` is clamped to len(Xte) in the worker, so on a small
                     # test set the requested and the measured batch differ — and every latency,
                     # throughput and peak-memory number is a function of the one that was
                     # actually used, which is the one that used not to be recorded.
                     "requested_batch_size": int(args.bs),
                     "effective_batch_size": meta.get("effective_batch_size"),
                     "n_train_px": meta.get("n_train_px"), "n_test_px": meta.get("n_test_px"),
                     "n_bands": meta.get("n_bands"),
                     "timing_validity": validity,
                     "contention_unknowns": unknowns,
                     "isolation": "one subprocess per model",
                     "bench_threads": int(args.bench_threads),
                     "repeats": int(args.repeats), "warmup": int(args.warmup),
                     "single_requests": int(args.single_requests),
                     "min_run_time_s": float(args.min_run_time),
                     "target_rel_iqr": float(args.target_rel_iqr),
                     "max_rel_iqr": float(args.max_rel_iqr),
                     "max_rel_iqr_single": float(args.max_rel_iqr_single),
                     "min_replicates": int(args.min_replicates),
                     "max_throttle_frac": float(args.max_throttle_frac),
                     "strict": bool(args.strict),
                     "contention_pre": pre, "contention_post": post,
                     "allow_contended": bool(args.allow_contended),
                     "allow_unvalidated_quant": bool(args.allow_unvalidated_quant),
                     "quant_api": QUANT_API,
                     "quant_validated_torch": QUANT_VALIDATED_TORCH,
                     "quant_engine_by_model": {r.get("name"): r.get("quant_engine") for r in rows},
                     "quant_scope_by_model": {r.get("name"): r.get("quant_scope") for r in rows},
                     "gpu_state_by_model": {r.get("name"): {"clock_sm_mhz": r.get("gpu_clock_sm_mhz"),
                                                            "temp_c": r.get("gpu_temp_c")}
                                            for r in rows},
                     "macs_complete_by_model": {r.get("name"): not _missing(r.get("macs_per_sample"))
                                                for r in rows},
                     "missing_subset_coverage": (rows[0].get("missing_subset_coverage")
                                                 if rows else None),
                     "missing_subsets": (rows[0].get("missing_subsets") if rows else None)})
        # stamp() never raises by design — it would be worse to lose a finished experiment than to
        # lose its sidecar. But nobody looked at what it returned, so a silently failed stamp was
        # indistinguishable from a phase that never stamped, which is exactly how an unprovenanced
        # deliverable gets shipped looking like every other one.
        if prov is None:
            print("\nWARNING: the provenance sidecar could NOT be written. The CSV above is "
                  "UNPROVENANCED and scripts/doctor.py will (correctly) fail on it — do not cite "
                  "it until it is re-stamped.", file=sys.stderr)
        print(f"\nwrote: {P(f'results_phase7_efficiency{sfx}.csv')}")
        print(f"       {P(f'results_phase7_efficiency{sfx}.tex')}"
              f"{'' if tex_text is not None else '  (INVALIDATED - no Proposed row)'}")
        if prov:
            print(f"       {prov}")
        print(f"       timing_validity={validity}")
        return 0
    except BaseException:
        # Any crash — a timeout that escaped, a KeyboardInterrupt, an OOM — must not take the
        # evidence with it. `finally` deletes the workdir when keep is False, and the trained
        # weights and every worker's JSON live in there.
        keep = True
        raise
    finally:
        if not keep and not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        elif keep:
            print(f"[workdir kept] {workdir}")


if __name__ == "__main__":
    # `--worker <spec.json>` is the isolated per-model measurement process. It is dispatched
    # before argparse so the worker protocol stays independent of the driver's CLI surface.
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        sys.exit(worker_main(sys.argv[2]))
    sys.exit(main())
