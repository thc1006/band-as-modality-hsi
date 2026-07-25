"""Every entry point the documentation promises must actually resolve.

Written from three failures that happened on 2026-07-20, each of which passed `py_compile` and every
existing test, and each of which broke something a reader or a reviewer would hit first:

  1. `reproduce.sh` — the documented one-command reproduction — called
     `experiments/experiment_synthetic_multiseed.py` after that file was archived during the repo
     reorg. With `set -e` the entire reproduction aborted at its FIRST experiment, before a single
     paper number was produced. `integrity_check.py` had been updated for the same archive; this
     file had not, and nothing checked that they agreed.

  2. `experiments/phase8F_emit.py` was archived on the strength of having no deliverable in `paper/`
     and no reference in `paper/` or `docs/`. Neither check looks at imports, and
     `phase8F_multi.py` does `import phase8F_emit as F` to reuse `recon_error_matrix`. The whole
     test suite stopped collecting with ModuleNotFoundError.

  3. `scripts/emit/preflight_emit.py` called `mask_band_indices(...)` without importing it. A pure
     NameError, invisible to `py_compile`, which survived because the script was never executed
     after the edit that introduced it — so the committed `preflight_quality_table.txt` predates it.

All three are static properties. These tests cost no GPU and no data, so they can run everywhere.
"""
import ast
import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_ROOT, "experiments")


def _py_files(*dirs):
    out = []
    for d in dirs:
        for root, _, files in os.walk(os.path.join(_ROOT, d)):
            if "__pycache__" in root:
                continue
            out.extend(os.path.join(root, f) for f in files if f.endswith(".py"))
    return sorted(out)


# --------------------------------------------------------------------- 1. reproduce.sh resolves
def test_every_script_reproduce_sh_calls_exists():
    """`set -e` turns a missing script into a full abort, and it aborts at the FIRST one, so a stale
    path near the top hides everything below it."""
    sh = open(os.path.join(_ROOT, "reproduce.sh"), encoding="utf-8").read()
    # only real invocations, not the paths named inside explanatory comments
    called = set()
    for line in sh.splitlines():
        code = line.split("#", 1)[0]
        called.update(re.findall(r"(?:experiments|scripts(?:/\w+)*)/[\w]+\.py", code))
    assert called, "parsed no script invocations out of reproduce.sh -- the regex has drifted"
    missing = sorted(p for p in called if not os.path.exists(os.path.join(_ROOT, p)))
    assert not missing, f"reproduce.sh calls scripts that do not exist: {missing}"


# A phase file kept only because another experiment imports it for a helper. Listing it here is
# deliberate and must stay short: the exemption says "this is a library, not a result", and every
# entry is a phase whose numbers NOBODY reproduces.
_LIBRARY_ONLY = {
    # phase8F_multi.py does `import phase8F_emit as F` to reuse recon_error_matrix. As an EXPERIMENT
    # it is superseded -- phase8F_multi's own header says it "fixes every issue an adversarial review
    # raised on the single-granule phase8F" -- so reproduce.sh deliberately does not run it, while
    # deleting it would break the import. Moving recon_error_matrix into bandsim/ would retire this.
    "phase8F_emit.py",
}


def _imported_by_siblings(name):
    mod = name[:-3]
    for f in os.listdir(_EXP):
        if not f.endswith(".py") or f == name:
            continue
        src = open(os.path.join(_EXP, f), encoding="utf-8").read()
        if re.search(rf"^\s*(?:import {mod}\b|from {mod} import)", src, re.M):
            return f
    return None


def test_reproduce_sh_covers_every_phase_script():
    """A phase that exists but is never reproduced is a result nobody else can regenerate. This
    caught reproduce.sh omitting NINE phases, including both flagships (CloudSEN12 and phase8R)."""
    sh = open(os.path.join(_ROOT, "reproduce.sh"), encoding="utf-8").read()
    phases = [f for f in os.listdir(_EXP) if f.startswith("phase") and f.endswith(".py")]
    missing = sorted(p for p in phases if p not in sh and p not in _LIBRARY_ONLY)
    assert not missing, f"phase scripts never run by reproduce.sh: {missing}"


@pytest.mark.parametrize("name", sorted(_LIBRARY_ONLY))
def test_library_only_exemptions_are_still_earned(name):
    """An exemption that stops being true is worse than no exemption: it hides a phase whose results
    nobody regenerates behind a comment claiming someone depends on it."""
    assert os.path.exists(os.path.join(_EXP, name)), f"{name} is exempted but does not exist"
    importer = _imported_by_siblings(name)
    assert importer, (f"{name} is exempted from reproduce.sh coverage as a library, but no sibling "
                      f"experiment imports it any more -- either run it or archive it")


# ------------------------------------------------------- 2. intra-repo imports actually resolve
@pytest.mark.parametrize("path", _py_files("experiments", "scripts", "bandsim"),
                         ids=lambda p: os.path.relpath(p, _ROOT))
def test_sibling_and_package_imports_resolve(path):
    """`import phase8F_emit` / `from bandsim.x import y` must point at a file that exists.

    Archiving a module is not a documentation change: something may import it for a helper even when
    it produces no deliverable of its own and is named nowhere in the paper."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    unresolved = []
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods = [node.module]
        for m in mods:
            top = m.split(".")[0]
            if top == "bandsim":
                target = os.path.join(_ROOT, *m.split(".")) + ".py"
                if not os.path.exists(target) and not os.path.isdir(os.path.dirname(target)):
                    unresolved.append(m)
            elif os.path.exists(os.path.join(_EXP, top + ".py")) or top.startswith("phase") \
                    or top.startswith("experiment_"):
                # a sibling experiment module, imported via the sys.path insert these scripts do
                if not os.path.exists(os.path.join(_EXP, top + ".py")):
                    unresolved.append(m)
    assert not unresolved, f"{os.path.relpath(path, _ROOT)} imports missing modules: {unresolved}"


# ------------------------------------------------------------------ 3. no undefined names at all
def test_no_undefined_names_anywhere():
    """The NameError class that `py_compile` cannot see. pyflakes does the scope analysis; skipping
    when it is absent is honest, but it is in requirements for exactly this reason."""
    pyflakes = pytest.importorskip(
        "pyflakes.api", reason="pyflakes not installed; `uv pip install --no-deps pyflakes`")
    from pyflakes import reporter as pyflakes_reporter
    import io

    out, err = io.StringIO(), io.StringIO()
    rep = pyflakes_reporter.Reporter(out, err)
    for p in _py_files("experiments", "scripts", "bandsim"):
        pyflakes.checkPath(p, rep)
    undefined = [ln for ln in out.getvalue().splitlines() if "undefined name" in ln]
    assert not undefined, "undefined names (would raise NameError at runtime):\n" + "\n".join(undefined)
