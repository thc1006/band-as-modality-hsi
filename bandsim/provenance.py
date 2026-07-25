"""Record how a result file was produced, next to the result file.

Why this exists: every artefact in `paper/` is a bare table of numbers. Nothing in
`results_phase4R_reliability.csv` said whether it came from `--seeds 0 1 2 3 4 --epochs 60` or from a
15-second `--smoke`, and at one point the committed numbers were in fact smoke output being quoted as
converged. Worse, a smoke run used to overwrite the very files the paper's LaTeX tables read. Suffixing
smoke output stops the overwrite; this module answers the other half -- given a file, what produced it.

Usage from an experiment, right after writing its CSV:

    from bandsim.provenance import stamp
    stamp(P("results_phaseX.csv"), args, extra={"lut_sha256": ..., "n_granules": ...})

writes `results_phaseX.csv.provenance.json`. Reading one tells you the exact command, the resolved
arguments (so a default that changed later is still visible), the git commit and whether the tree was
dirty, library versions, and any experiment-specific hashes the caller passes in.

Deliberately dependency-free beyond numpy/torch introspection, and every field is best-effort: a
provenance stamp must never be the reason a finished experiment fails to save its results.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _git(*args, cwd=None):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# Paths whose modification can change a NUMBER. `dirty` is scoped to these on purpose -- see
# git_state.
_CODE_PREFIXES = ("bandsim/", "experiments/", "scripts/", "configs/", "tests/")
_CODE_FILES = ("reproduce.sh", "pyproject.toml", "requirements-lock.txt")


def _status_path(line):
    """Path out of one `git status --porcelain` line, taking the destination of a rename.

    Deliberately NOT `line[3:]`. Porcelain's prefix is a fixed-width 'XY ', but `_git` strips its
    output, so the FIRST line loses its leading space when X is blank: ' M bandsim/x.py' arrives as
    'M bandsim/x.py' and a fixed offset slices into the path itself ('andsim/x.py'), which then
    matches no prefix. That mis-classified exactly one file per call -- always the first -- so a
    lone code edit at the top of the list read as a clean tree."""
    body = line.strip().split(" ", 1)
    body = body[1].strip() if len(body) > 1 else ""
    if " -> " in body:
        body = body.split(" -> ", 1)[1]
    return body.strip().strip('"')


def _is_code(path):
    return path.startswith(_CODE_PREFIXES) or path in _CODE_FILES


def git_state(repo_dir=None):
    """Commit, branch and dirtiness. `dirty` matters as much as the commit: a result produced from an
    edited working tree is not reproducible from that commit alone, and saying so is the point.

    `dirty` covers CODE only (bandsim/, experiments/, scripts/, configs/, tests/, and the root build
    files). It used to be `bool(git status --porcelain)` over the whole tree, which made it true by
    construction during any experiment campaign: results are tracked in git, so the first phase to
    write paper/ dirtied the tree and every phase after it was stamped irreproducible while the code
    was pristine. A flag that is always set says nothing at the moment it matters most.

    Changes under paper/ and docs/ are still reported, as `outputs_dirty`, because "someone hand-
    edited a result" is worth seeing -- it is just not the same claim as "the code moved"."""
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = _git("rev-parse", "HEAD", cwd=repo_dir)
    status = _git("status", "--porcelain", cwd=repo_dir)
    if status is None:
        return {"commit": commit, "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_dir),
                "dirty": None, "dirty_files": 0, "outputs_dirty": None}
    paths = [_status_path(ln) for ln in status.splitlines() if ln.strip()]
    code = sorted(p for p in paths if _is_code(p))
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_dir),
        "dirty": bool(code),
        "dirty_files": len(code),
        "dirty_code_paths": code[:20],          # actionable, and capped so a stamp stays readable
        "outputs_dirty": bool(len(paths) - len(code)),
    }


def file_sha256(path, chunk=1 << 20):
    """Full content hash of an input file, or None if unreadable. Used for inputs small enough to
    hash outright (a wavelength axis, a transmittance table) -- not for multi-GB scene data."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    # pyspectral decides WHICH measured RSR a sensor simulation used and matplotlib decides what the
    # figure looks like; both change numbers or artefacts across versions, and neither was recorded.
    # (An RSR-store content change is caught more directly by the per-run W hash the SRF experiments
    # stamp -- this pins the code that read it.)
    for mod in ("numpy", "scipy", "torch", "h5py", "pyspectral", "matplotlib"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    try:
        import torch
        v["cuda"] = torch.version.cuda
        v["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return v


def record(args=None, extra=None):
    """Build the provenance dict without writing it (useful for embedding elsewhere)."""
    rec = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "git": git_state(),
        "versions": versions(),
    }
    if args is not None:
        # `smoke` is set FIRST and outside the try. It is the single field this whole module exists
        # for, and if it were set after a statement that can raise, a failure while serialising the
        # arguments would leave the key ABSENT -- whereupon a reader doing `rec.get("smoke")` gets
        # None, which is falsy, and a smoke run reads as a real one. Absent must never be mistakable
        # for False here, so the flag is recorded before anything else can go wrong.
        rec["smoke"] = bool(getattr(args, "smoke", False))
        # vars() on argparse.Namespace captures RESOLVED values, so a default that is edited later
        # does not silently rewrite the history of an old run.
        try:
            rec["args"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
        except Exception as e:
            rec["args"] = None
            rec["args_error"] = str(e)     # say WHY it is missing rather than leaving a bare null
    if extra:
        rec["extra"] = extra
    return rec


def stamp(result_path, args=None, extra=None):
    """Write `<result_path>.provenance.json`. Returns the path, or None if it could not be written.

    Never raises: losing a finished experiment's results because the provenance write failed would be
    a strictly worse outcome than an unstamped result."""
    try:
        out = f"{result_path}.provenance.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(record(args, extra), f, indent=2, sort_keys=True, default=str)
        return out
    except Exception as e:                                    # pragma: no cover - best effort by design
        print(f"[provenance] could not stamp {result_path}: {e}")
        return None
