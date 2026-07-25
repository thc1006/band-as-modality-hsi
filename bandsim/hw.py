"""Hardware configuration — adaptive device selection, reproducibility, and thread tuning.

One entry point `setup(seed, deterministic, ...)` used by every experiment so that:
  * training runs on the GPU when a working CUDA device is present, else CPU;
  * CPU threading is tuned to the ACTUAL machine (physical/affinity core count), not a
    hardcoded constant — so the same code squeezes an 8-core laptop and a 36-core, 2-GPU box;
  * numpy/torch/CUDA RNGs are seeded and (optionally) deterministic algorithms are enabled
    so results are as reproducible as the backend allows.

This is a single device-agnostic code path: the same scripts accelerate on whatever hardware
is present, with no code changes. It also underpins `bandsim.parallel`, which fans many
independent per-seed jobs across all GPUs + CPU cores.

Environment overrides (all optional)
------------------------------------
  BANDSIM_DEVICE   cpu | cuda | cuda:N | auto   force the device (default: auto)
  BANDSIM_THREADS  int                          intra-op / BLAS threads for THIS process
                                                (parallel workers set this to their budget)

The `CUBLAS_WORKSPACE_CONFIG` env is set at import (before the CUDA context initializes) so
deterministic cuBLAS is available if requested.
"""
from __future__ import annotations
import os
# must be set BEFORE the CUDA context initializes -> set at import, before torch CUDA use.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
import numpy as np
import torch


# ---------------------------------------------------------------------------------------
# hardware detection (no side effects)
# ---------------------------------------------------------------------------------------
def available_cores() -> int:
    """Number of CPUs this process may actually use.

    Honours cgroup / CPU-affinity limits (important inside containers and schedulers) via
    os.sched_getaffinity when available, falling back to os.cpu_count().
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))          # respects taskset / cgroup
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def n_gpus() -> int:
    """Number of CUDA devices torch can see (0 if CUDA unavailable)."""
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _env_threads(default: int) -> int:
    v = os.environ.get("BANDSIM_THREADS")
    if v:
        try:
            return max(1, int(v))
        except ValueError:
            pass
    return default


# ---------------------------------------------------------------------------------------
# device selection
# ---------------------------------------------------------------------------------------
def device(prefer: str | None = None) -> torch.device:
    """Return the torch device to use.

    prefer / BANDSIM_DEVICE:  "cpu" | "cuda" | "cuda:N" | "auto" (default auto = cuda->cpu).
    Inside a parallel worker that pinned a single GPU via CUDA_VISIBLE_DEVICES, "cuda"
    resolves to that GPU (which torch sees as cuda:0).
    """
    prefer = (prefer or os.environ.get("BANDSIM_DEVICE") or "auto").lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer.startswith("cuda"):
        if n_gpus() == 0:
            return torch.device("cpu")        # no GPU visible (CPU worker / no-GPU host) -> cpu
        if ":" in prefer:                     # EXPLICIT cuda:N -> validate the index, fail closed
            tail = prefer.split(":", 1)[1]
            idx = int(tail) if tail.isdigit() else -1
            if not (0 <= idx < n_gpus()):
                raise ValueError(f"requested device '{prefer}' but only {n_gpus()} GPU(s) visible; "
                                 f"an explicit cuda:N must be a valid index (use 'cuda'/'auto' to fall back)")
            return torch.device(prefer)
        return torch.device("cuda")           # bare 'cuda' -> the (pinned) visible GPU
    # auto
    return torch.device("cuda" if n_gpus() > 0 else "cpu")


# ---------------------------------------------------------------------------------------
# setup: threads + seeds + determinism
# ---------------------------------------------------------------------------------------
def setup(seed=0, deterministic=True, threads=None, prefer=None):
    """Configure device, threads and RNGs. Returns the selected torch.device.

    Parameters
    ----------
    seed          : int    seeds numpy + torch (+ cuda) RNGs.
    deterministic : bool   enable bit-reproducible algorithms where the backend supports it
                           (small speed cost). Set False for the fastest non-deterministic runs.
    threads       : int|None  intra-op / BLAS threads for this process. None -> auto:
                           BANDSIM_THREADS if set, else all available cores (single-process),
                           which a parallel worker overrides with its per-worker budget.
    prefer        : str|None  device preference; see device().
    """
    if threads is None:
        threads = _env_threads(available_cores())
    threads = max(1, int(threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))    # numpy/BLAS (best-effort)
    torch.set_num_threads(threads)                            # torch intra-op (CPU tensors too)
    # INTER-op too. set_num_threads covers only the intra-op pool; the inter-op pool keeps its own
    # default of "one thread per core as torch sees it", and torch sees the HOST's cores, not this
    # container's cgroup quota. Measured here: available_cores() correctly returns 8 and intra-op was
    # set to 7, while inter-op stayed at 36 -- so a single process opened ~31 threads on an 8-core
    # box, and the campaign's 4 workers would have opened 4 x 36 of them. That is pure scheduler
    # thrash: a smoke run that takes about a minute at one thread burned 4620 s of CPU over 22
    # wall-clock minutes. It changes no number, only wall time -- but phase7 MEASURES wall time.
    #
    # PyTorch allows this exactly once, and only before any parallel work has begun, so a second
    # setup() in the same process (or one after a forward pass) raises. Swallowing that is correct:
    # the first call already applied the budget, and refusing to start an experiment because the
    # thread pool was already sized would be worse than the thrash this prevents.
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass

    # Propagate an EXPLICIT device preference to the process env so later ARGLESS hw.device()
    # calls (inside pretrain_sgmae / finetune_proposed / recon_error_matrix etc.) resolve to the
    # SAME device setup chose. Without this, `--device cpu` on a GPU box is silently ignored by
    # those helpers (they call hw.device() with no arg -> auto -> cuda). Guard on `is not None`
    # so auto callers and parallel workers (prefer=None; device pinned via CUDA_VISIBLE_DEVICES)
    # are unaffected. Mirrors the workaround parallel.run_jobs already does for its serial-cpu path.
    if prefer is not None:
        os.environ["BANDSIM_DEVICE"] = str(prefer).lower()

    dev = device(prefer)

    # seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus() > 0:
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            # warn_only: some ops (e.g. GPU multi-head-attention backward) have no
            # deterministic kernel; we warn rather than crash so training still runs.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = False
        # Force the deterministic MATH scaled-dot-product-attention kernel: the flash /
        # mem-efficient kernels have NON-deterministic backward passes (they warn under
        # deterministic algorithms and break bit-reproducibility). Our attention models are
        # tiny (few group tokens), so the math kernel costs nothing here.
        for _name, _val in [("enable_flash_sdp", False),
                            ("enable_mem_efficient_sdp", False),
                            ("enable_math_sdp", True)]:
            _fn = getattr(torch.backends.cuda, _name, None)
            if callable(_fn):
                try:
                    _fn(_val)
                except Exception:
                    pass
    else:
        # deterministic=False must UNDO a prior deterministic setup in the same process, else the
        # state is sticky (deterministic algorithms stay ON and flash/mem-efficient SDP stay OFF).
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True                 # autotune convs when free to
        for _name, _val in [("enable_flash_sdp", True),
                            ("enable_mem_efficient_sdp", True),
                            ("enable_math_sdp", True)]:
            _fn = getattr(torch.backends.cuda, _name, None)
            if callable(_fn):
                try:
                    _fn(_val)
                except Exception:
                    pass
    return dev


# ---------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------
def gpu_summaries():
    """List of 'name, mem GB' strings, one per visible CUDA device (empty if none)."""
    out = []
    for i in range(n_gpus()):
        p = torch.cuda.get_device_properties(i)
        out.append(f"{p.name} {p.total_memory / 1e9:.0f}GB")
    return out


def p2p_links():
    """GPU index pairs (i, j) with peer (NVLink/PCIe) access. Empty unless >1 GPU visible.

    The current parallel design pins one GPU per worker, so P2P is NOT used for the many-tiny-
    jobs workload (independent jobs beat DDP there). It is reported for awareness and matters
    only if a single large job is ever split across GPUs (DDP/model-parallel), where gradient
    all-reduce would ride these links.
    """
    ng = n_gpus()
    pairs = []
    for i in range(ng):
        for j in range(i + 1, ng):
            try:
                if torch.cuda.can_device_access_peer(i, j):
                    pairs.append((i, j))
            except Exception:
                pass
    return pairs


def info() -> str:
    """Human-readable hardware summary (ACTUAL selected device + all GPUs + cores + threads).

    Reports the device that will actually be used — i.e. it honours BANDSIM_DEVICE / a forced
    prefer=cpu — instead of merely reporting GPU presence (which previously mislabelled cpu-forced
    runs as 'device=cuda' in reproducibility logs)."""
    ng = n_gpus()
    cores = available_cores()
    thr = torch.get_num_threads()
    sel = device()                                            # respects BANDSIM_DEVICE / auto
    if sel.type == "cuda" and ng > 0:
        gpus = gpu_summaries()
        head = gpus[0] if len(set(gpus)) == 1 else ", ".join(gpus)
        gtxt = f"{ng}x {head}" if len(set(gpus)) == 1 else head
        link = " | P2P peer-access" if (ng > 1 and p2p_links()) else ""
        return (f"device=cuda ({gtxt}){link} | {cores} cores, {thr} threads/proc | "
                f"torch {torch.__version__}")
    gpu_note = f" ({ng} GPU(s) present but device forced to cpu)" if ng > 0 else ""
    return f"device=cpu{gpu_note} ({cores} cores, {thr} threads) | torch {torch.__version__}"


def summary() -> dict:
    """Structured hardware snapshot (used by the parallel planner and for logging)."""
    return {
        "n_gpus": n_gpus(),
        "gpus": gpu_summaries(),
        "p2p_links": p2p_links(),
        "cores": available_cores(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
