#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repo and machine health check — every invariant this project has learned the hard way.

WHY THIS EXISTS. Over one working session the following were all found REACTIVELY, each after
someone noticed something looked off, and each after a status report had already said things were
fine: three orphaned worker processes holding 13.8 GB of VRAM for 73 minutes after their parent was
killed; a process opening 36 inter-op threads on an 8-core box; GPU0 at 92.9% VRAM with 2.3 GB of
headroom under an unattended overnight run; deliverables with no provenance; a reproduction script
that aborted on its first experiment.

None of them raised an error. That is the point: checks written to catch crashes do not catch waste,
risk, or drift. Each check below is one defect that actually happened, turned into an assertion so it
is found by running this rather than by someone asking the right question.

It cannot catch a class nobody has met yet. What it does is stop the KNOWN classes from needing a
human to notice them.

    python scripts/doctor.py              # everything
    python scripts/doctor.py --runtime    # machine state only (fast, safe during a campaign)
    python scripts/doctor.py --repo       # static repo checks only

Exit code is the number of FAILed checks, so it is usable as a gate.
"""
import argparse
import ast
import csv
import glob
import os
import re
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
PAPER = os.path.join(ROOT, "paper")

RESULTS, FAILED = [], 0


def check(name, ok, detail="", warn_only=False):
    global FAILED
    if ok:
        tag = "PASS"
    elif warn_only:
        tag = "WARN"
    else:
        tag = "FAIL"; FAILED += 1
    RESULTS.append((tag, name, detail))


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


# ------------------------------------------------------------------------------ runtime: processes
def orphan_workers():
    """ProcessPoolExecutor workers reparented to init.

    They survive a SIGTERM to the parent because Python's default handler skips atexit and
    ProcessPoolExecutor.__exit__. They are invisible to any search that greps for the experiment
    script's path, because `spawn` gives them the cmdline
    `python -c from multiprocessing.spawn import spawn_main...` instead. Three of them once ran for
    73 minutes for a dead parent, holding 6.9 GB of VRAM each and a core apiece."""
    out = []
    for p in glob.glob("/proc/[0-9]*"):
        pid = os.path.basename(p)
        try:
            with open(f"{p}/comm") as f:
                if not f.read().strip().startswith("python"):
                    continue
            ppid = open(f"{p}/stat").read().split()[3]
            cmd = open(f"{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except Exception:
            continue
        if ppid == "1" and ("multiprocessing.spawn" in cmd or "multiprocessing.resource_tracker" in cmd):
            out.append(pid)
    return out


def check_runtime():
    orph = orphan_workers()
    check("no orphaned pool workers", not orph,
          f"pids={orph} -- reparented to init, holding GPU memory and cores for a dead parent"
          if orph else "")

    # threads: intra-op AND inter-op must respect the cgroup, not the host
    try:
        sys.path.insert(0, ROOT)
        from bandsim import hw
        cores = hw.available_cores()
        code = ("import torch,sys;sys.path.insert(0,%r);from bandsim import hw;"
                "hw.setup(deterministic=True);"
                "print(torch.get_num_threads(),torch.get_num_interop_threads())" % ROOT)
        got = sh(os.path.join(ROOT, ".venv/bin/python"), "-c", code).split()
        intra, inter = (int(got[0]), int(got[1])) if len(got) == 2 else (-1, -1)
        check("intra-op threads <= cores", 0 < intra <= cores, f"intra={intra} cores={cores}")
        check("inter-op threads <= cores", 0 < inter <= cores,
              f"inter={inter} cores={cores} -- torch defaults this to the HOST's core count, "
              f"ignoring the cgroup quota")
    except Exception as e:
        check("thread budget", False, f"probe failed: {e}")

    # load
    try:
        load = float(open("/proc/loadavg").read().split()[0])
        # Normalise by HOST cores, not the cgroup quota. /proc/loadavg is host-wide and not
        # namespaced, so a container reads the whole machine's runnable count; dividing that by
        # available_cores() (8, the cgroup quota) is the same unit error phase7 carried -- it read
        # this shared box's steady ~28/36 = 0.8 host-load as "28 > 8*1.5, too busy" and warned on
        # every run while our own pinned cores sat near idle. host_cores catches a genuinely
        # saturated machine (load > 54) without crying wolf over other tenants on cores we cannot
        # use. (phase7 goes one better and samples the pinned cpuset; doctor only needs the
        # coarse machine-health signal.)
        host_cores = os.cpu_count() or cores
        check("load average sane", load <= host_cores * 1.5,
              f"load={load} host_cores={host_cores} (cgroup quota {cores})", warn_only=True)
    except Exception:
        pass

    # GPU: headroom, and symmetric allocation across cards
    apps = [l for l in sh("nvidia-smi", "--query-compute-apps=gpu_uuid,used_memory",
                          "--format=csv,noheader").splitlines() if l.strip()]
    per_gpu = {}
    for line in apps:
        uuid = line.split(",")[0].strip()
        per_gpu[uuid] = per_gpu.get(uuid, 0) + 1
    mem = [l.split(",") for l in sh("nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                    "--format=csv,noheader,nounits").splitlines() if l.strip()]
    for idx, used, total in mem:
        pct = 100.0 * int(used) / int(total)
        check(f"GPU{idx.strip()} VRAM headroom", pct < 85.0,
              f"{pct:.1f}% used -- an unattended run needs more than a couple of GB of margin",
              warn_only=pct < 92.0)
    if len(per_gpu) > 1:
        counts = sorted(per_gpu.values())
        check("GPU worker allocation symmetric", counts[-1] - counts[0] <= 1,
              f"processes per GPU: {counts}", warn_only=True)


# ------------------------------------------------------------------- repo: artefact-vs-code drift
def _paper_csv_names(tree):
    """Basenames of the paper/ CSVs a script writes, with any --smoke suffix stripped.

    `P(f"results_x{sfx}.csv")` -> `results_x.csv`. A set, because a script writing several
    different tables cannot have any one of them attributed to its single CSV_COLS."""
    names = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "P" and len(node.args) == 1):
            continue
        lit = re.sub(r"\{[^}]*\}", "", ast.unparse(node.args[0])).strip("f'\" ")
        if lit.endswith(".csv"):
            names.add(lit)
    return names


def _declared_columns():
    """{deliverable basename -> the column list its generator declares in CSV_COLS}.

    Only for scripts that declare a module-level CSV_COLS AND write exactly ONE paper/ CSV.
    phase8_cloudsen12 writes curve/perclass/scenarios from one module; attributing a single
    CSV_COLS to all three would manufacture failures, and a check that cries wolf gets muted."""
    out = {}
    for path in sorted(glob.glob(f"{EXP}/*.py")):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        cols = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "CSV_COLS" for t in node.targets):
                try:
                    cols = list(ast.literal_eval(node.value))
                except Exception:
                    cols = None
        names = _paper_csv_names(tree)
        if cols and len(names) == 1:
            out[next(iter(names))] = cols
    return out


def schema_drift(declared, paper_dir=None):
    """Deliverables whose header disagrees with their generator's CSV_COLS.

    Returns ``(drift, n_checked, n_total)``.

    BOTH directions are drift, and only one of them caught the defect that motivated this check.
    The stale phase-7 table was found because five columns had been RENAMED, so they no longer
    existed in CSV_COLS. Had the schema only GROWN -- the ordinary case, and exactly what phase 7
    itself did going 62 -> 89 columns -- an old header would be a strict SUBSET of the new one,
    every column in it would still be producible, and this check would have passed a file missing
    eighty-one columns. Verified by simulation before being believed. `write_rows` emits
    `csv.DictWriter(fieldnames=CSV_COLS).writeheader()`, so a current file's header is EXACTLY
    CSV_COLS and anything else is drift.

    `n_checked`/`n_total` come back so the caller can state what a PASS covered. Only a script
    declaring a module-level CSV_COLS and writing exactly one paper/ CSV can be attributed, which
    today is one deliverable out of twenty-five -- and a green check that verified nothing must not
    read like one that verified everything. That is the same failure this whole exercise was about,
    one level up, in the tool that is supposed to catch it."""
    paper_dir = paper_dir or PAPER
    files = [f for f in sorted(glob.glob(f"{paper_dir}/results_*.csv"))
             if "_smoke" not in os.path.basename(f)]
    drift, checked = [], 0
    for f in files:
        base = os.path.basename(f)
        if base not in declared:
            continue
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), [])
        except Exception:
            continue
        checked += 1
        want = declared[base]
        gone = [c for c in header if c not in want]
        absent = [c for c in want if c not in header]
        bits = []
        if gone:
            bits.append(f"{len(gone)} column(s) its generator no longer writes "
                        f"({', '.join(gone[:3])}{'...' if len(gone) > 3 else ''})")
        if absent:
            bits.append(f"{len(absent)} column(s) its generator writes but this file lacks "
                        f"({', '.join(absent[:3])}{'...' if len(absent) > 3 else ''})")
        if bits:
            drift.append(f"{base}: " + "; ".join(bits))
    return drift, checked, len(files)


# --------------------------------------------------------------------------------- repo: integrity
def check_repo():
    # every script reproduce.sh calls must exist; every phase must be covered
    sh_txt = open(os.path.join(ROOT, "reproduce.sh")).read()
    called = set()
    for line in sh_txt.splitlines():
        called.update(re.findall(r"(?:experiments|scripts(?:/\w+)*)/[\w]+\.py", line.split("#", 1)[0]))
    missing = sorted(p for p in called if not os.path.exists(os.path.join(ROOT, p)))
    check("reproduce.sh scripts all exist", not missing, f"missing: {missing}")

    # intra-repo imports resolve (archiving a module that something imports breaks collection)
    unresolved = []
    for path in glob.glob(f"{EXP}/*.py") + glob.glob(f"{ROOT}/bandsim/*.py") + \
            glob.glob(f"{ROOT}/scripts/**/*.py", recursive=True):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                    and node.level == 0 else [])
            for m in mods:
                top = m.split(".")[0]
                if (top.startswith("phase") or top.startswith("experiment_")) \
                        and not os.path.exists(os.path.join(EXP, top + ".py")):
                    unresolved.append(f"{os.path.basename(path)} -> {m}")
    check("intra-repo imports resolve", not unresolved, f"{unresolved}")

    # undefined names (the NameError class py_compile cannot see)
    try:
        from pyflakes import api as pf_api, reporter as pf_rep
        import io
        out = io.StringIO()
        rep = pf_rep.Reporter(out, io.StringIO())
        for path in glob.glob(f"{EXP}/*.py") + glob.glob(f"{ROOT}/bandsim/*.py") + \
                glob.glob(f"{ROOT}/scripts/**/*.py", recursive=True):
            pf_api.checkPath(path, rep)
        und = [l for l in out.getvalue().splitlines() if "undefined name" in l]
        check("no undefined names", not und, "\n      ".join(und))
    except ImportError:
        check("no undefined names", True, "pyflakes not installed -- SKIPPED", warn_only=True)

    # a sensor simulation built straight from pyspectral_srf runs on the RSR STORE's band list, not
    # on the sensor's surface-reflectance PRODUCT: 13 bands for Sentinel-2 (B10 cirrus included) and
    # 9 for Landsat-8 OLI (15 m panchromatic AND cirrus included, and the store's 'B6' is the 1373 nm
    # cirrus band while USGS 'B6' is 1609 nm SWIR-1, so matching by NAME pairs the wrong wavelength).
    # On the Indian Pines axis the cirrus band additionally lands on the removed water-vapour gap and
    # is synthesized from a surviving tail, then renormalised back to row-sum 1 -- finite, plausible,
    # measured centre 1374 nm, and wrong. bandsim.srf.sensor_bandset pins one canonical band set
    # matched by CENTRE WAVELENGTH and fails closed on any band the axis cannot resolve; a module
    # calling pyspectral_srf directly has opted out of that guarantee.
    raw_srf = []
    for path in sorted(glob.glob(f"{EXP}/*.py") + glob.glob(f"{ROOT}/bandsim/*.py")):
        if os.path.basename(path) == "srf.py":
            continue                                    # this is where they are defined
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if "pyspectral_srf" in called and not called & {"sensor_bandset", "select_canonical_bandset"}:
            raw_srf.append(os.path.basename(path))
    check("sensor band-sets go through the canonical contract", not raw_srf,
          f"{raw_srf} call pyspectral_srf directly, so they synthesize the RSR store's 13/9-band "
          f"list (S2 B10 cirrus; OLI pan + cirrus) instead of the 12/7 surface-reflectance product, "
          f"and on a gappy axis the cirrus band is integrated across missing data. Fix: replace the "
          f"srf_source branch with bandsim.srf.sensor_bandset(wl, sensor, source=srf_source)")

    # ONE definition of the macro estimand. `common_class_set` decides WHICH classes the headline
    # mIoU/AA average over, and `per_class_recall` is the per-class form of AA; both were hand-copied
    # into experiments/ before bandsim.metrics had them, and the two common_class_set copies already
    # differ in signature. WARN rather than FAIL on purpose: the copies still agree numerically, so
    # this is drift RISK, not a wrong number today -- the same grading as the dirty-tree check. The
    # precedent for why it matters is `block_grid`, whose hand-copy into phase4R/phase9 silently
    # moved the conformal units.
    homes = {}
    for path in sorted(glob.glob(f"{EXP}/*.py") + glob.glob(f"{ROOT}/bandsim/*.py")):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("common_class_set",
                                                                   "per_class_recall"):
                homes.setdefault(node.name, []).append(os.path.relpath(path, ROOT))
    copied = {k: v for k, v in homes.items() if len(v) > 1}
    check("one definition of the macro-estimand helpers", not copied,
          f"{copied} -- import from bandsim.metrics rather than redefining; the mIoU everything is "
          f"compared on is exactly the wrong thing to keep two of", warn_only=True)

    # every non-smoke deliverable carries provenance
    unstamped = [os.path.basename(f) for f in sorted(glob.glob(f"{PAPER}/results_*.csv"))
                 if "_smoke" not in f and not os.path.exists(f + ".provenance.json")]
    check("every deliverable is provenance-stamped", not unstamped,
          f"{len(unstamped)} unstamped: {unstamped[:6]}{' ...' if len(unstamped) > 6 else ''}")

    # a stamped result whose sidecar says the code was dirty is not reproducible from its commit
    import json
    dirty = []
    for f in sorted(glob.glob(f"{PAPER}/results_*.csv.provenance.json")):
        if "_smoke" in f:
            continue
        try:
            g = json.load(open(f)).get("git", {})
            if g.get("dirty"):
                dirty.append(os.path.basename(f).replace(".csv.provenance.json", ""))
        except Exception:
            pass
    check("no deliverable was produced from a dirty tree", not dirty,
          f"{dirty} -- rebuild from a clean commit before citing", warn_only=True)

    # A deliverable whose COLUMNS its generator can no longer produce was written by code that no
    # longer exists. This is the defect the two checks above cannot see: the stamp can be present,
    # clean and internally consistent while the table came from a different generation of the
    # script. paper/results_phase7_efficiency.csv sat in a 15-column schema for four rewrites after
    # its generator moved to 89 columns, and the only visible symptom was a MISSING sidecar --
    # indistinguishable from the eighteen other artefacts that were merely not stamped yet. FAIL,
    # not warn: this is not "re-run me soon", it is proof the file did not come from this code.
    drift, checked, total = schema_drift(_declared_columns())
    # The coverage is reported on PASS as well as on FAIL, on purpose. This check can only attribute
    # a deliverable to a generator that declares a module-level CSV_COLS and writes exactly one
    # paper/ CSV -- one file in twenty-five today. Named "no deliverable has drifted" and reported
    # bare, a green tick would assert far more than it looked at.
    cover = (f"covers {checked}/{total} deliverable(s) -- the rest build their header inline and "
             f"cannot be attributed to a declared schema")
    check("no attributable deliverable has drifted from its generator's schema", not drift,
          ("; ".join(drift) + " -- re-run it | " if drift else "") + cover)

    # A deliverable older than the code that produces it. Weaker evidence than the schema check
    # (a whitespace commit to the generator makes every artefact 'stale'), so it WARNS -- it is a
    # re-run worklist, not a verdict. Scoped to the script NAMED IN THE STAMP's own command line,
    # so it answers "is this table older than the code that made it", not "is anything newer".
    stale = []
    for f in sorted(glob.glob(f"{PAPER}/results_*.csv.provenance.json")):
        if "_smoke" in f:
            continue
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        gen = rec.get("generated_utc")
        scripts = re.findall(r"(?:experiments|scripts)/[\w/]+\.py", rec.get("command") or "")
        if not (gen and scripts):
            continue
        code = sh("git", "-C", ROOT, "log", "-1", "--format=%cI", "--", scripts[0]).strip()
        if not code:
            continue
        try:
            if datetime.fromisoformat(gen) < datetime.fromisoformat(code):
                stale.append(f"{os.path.basename(f).replace('.csv.provenance.json', '')} "
                             f"(built {gen[:16]}, {os.path.basename(scripts[0])} changed "
                             f"{code[:16]})")
        except (ValueError, TypeError):
            continue
    check("no deliverable predates its own generator", not stale,
          f"{len(stale)} stale: {'; '.join(stale[:3])}{' ...' if len(stale) > 3 else ''}",
          warn_only=True)


def check_return_arity():
    """Every fixed-arity tuple unpack of a repo function must match a return the function makes.

    The failure this catches produced two real breaks on 2026-07-20 and NEITHER was visible to the
    test suite: phase2_degradation.run_seed grew a third return value, and both
    phase2_group_ablation and adversarial_verify still unpacked two -- so they raised
    `too many values to unpack` AFTER training their models. A cross-PR arity change is invisible to
    a unit test of the callee and to any test that does not drive the caller's main().

    Starred targets (`a, *_ = f()`) are the sanctioned way to be arity-tolerant and are skipped.
    """
    import ast, collections
    rets = collections.defaultdict(set)
    for path in sorted(glob.glob(f"{ROOT}/bandsim/*.py") + glob.glob(f"{EXP}/*.py")):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for r in ast.walk(node):
                    if isinstance(r, ast.Return) and isinstance(r.value, ast.Tuple):
                        rets[node.name].add(len(r.value.elts))
    bad = []
    for path in sorted(glob.glob(f"{EXP}/*.py") + glob.glob(f"{ROOT}/scripts/*.py")):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                    and isinstance(node.value, ast.Call)):
                continue
            elts = node.targets[0].elts
            if any(isinstance(e, ast.Starred) for e in elts):
                continue
            fn = node.value.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in rets and len(elts) not in rets[name]:
                bad.append(f"{os.path.basename(path)}:{node.lineno} unpacks {len(elts)} from "
                           f"{name}() which returns {sorted(rets[name])}")
    check("every tuple unpack matches its callee's return arity", not bad,
          f"{bad} -- a cross-PR arity change no unit test can see; use `a, *_ = f()` where the "
          f"tail is deliberately ignored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", action="store_true", help="machine state only")
    ap.add_argument("--repo", action="store_true", help="static repo checks only")
    a = ap.parse_args()
    do_rt = a.runtime or not a.repo
    do_rp = a.repo or not a.runtime
    if do_rt:
        check_runtime()
    if do_rp:
        check_repo()
        check_return_arity()
    width = max(len(n) for _, n, _ in RESULTS)
    print("=" * (width + 30))
    for tag, name, detail in RESULTS:
        print(f"[{tag}] {name.ljust(width)}  {detail}")
    print("=" * (width + 30))
    n_warn = sum(1 for t, _, _ in RESULTS if t == "WARN")
    print(f"{len(RESULTS)} checks, {FAILED} FAIL, {n_warn} WARN")
    return FAILED


if __name__ == "__main__":
    sys.exit(main())
