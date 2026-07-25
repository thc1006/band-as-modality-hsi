"""A --smoke run must not be able to write an unsuffixed deliverable.

The failure this locks out left no trace. `--smoke` reduced seeds/epochs and then wrote to the SAME
paths as a full run, so a 15-second sanity check silently replaced the CSVs the paper's LaTeX reads
(make_paper_tables.py loads paper/results_phase2_curve.csv straight into a table) with nothing in
the file to say it was a smoke. It was only visible from file mtimes, and only if someone looked.

The check is STATIC on purpose: it costs no GPU time, it covers the scripts that need real data this
box may not have, and it fails at the point the protection is dropped rather than after a run has
already overwritten something. For every experiment script that accepts `--smoke`, every path handed
to a WRITE call must be an expression that varies with `args.smoke` -- not a bare string literal.

Two ways to defeat a weaker version of this test, both closed below:
  * interpolating a variable that has nothing to do with --smoke -> the name must be assigned under
    an `if args.smoke:` (or from an expression mentioning args.smoke), and
  * `sfx = ""` inside the smoke branch, which is smoke-conditional and still a no-op -> that
    assignment must actually carry a "smoke" string literal.

Scope is DISCOVERED, not listed: add `--smoke` to another experiment and it is covered here from
that moment. Scripts with no `--smoke` flag cannot be clobbered by this vector and are not checked.
"""
import ast
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_ROOT, "experiments")

# Calls whose path argument is WRITTEN. `open` is handled separately (mode-dependent); the rest take
# a destination path and are grouped by the attribute/function name actually used in this codebase,
# plus the obvious neighbours so a future `np.savez(P(...))` is covered the day it is written.
#
# The second group MUTATES a path without ever opening it, which is how a script can evade this
# check while looking like it complies. Two shapes were about to be written and are covered in
# advance: an atomic write (`open(P(f"...{sfx}.csv.tmp"), "w")` then `os.replace(tmp, final)` -- the
# open is suffixed and passes, while the destination of the replace is where an unsuffixed
# deliverable would actually be clobbered), and removing a stale artefact (`os.remove(P(...))`),
# where a --smoke run would DELETE the real deliverable rather than overwrite it. Verified against
# every experiment script at the time of writing: none of these names is currently used with a
# paper/ path, so adding them costs nothing today and closes the hole before it is dug.
_WRITE_CALLS = {
    "savefig", "stamp", "to_csv", "to_json", "to_latex", "to_parquet",
    "save", "savez", "savez_compressed", "savetxt", "imwrite", "imsave",
    "write_text", "write_bytes", "dump",
    "replace", "rename", "move", "copy", "copy2", "copyfile", "remove", "unlink", "rmtree",
}

# Scripts required to stamp their primary result CSV with bandsim.provenance. Still an explicit list
# rather than an inferred "every script" rule: the point of the list is to FAIL when a script that
# used to stamp stops doing so, and a rule derived from the scripts themselves cannot notice that.
#
# phase2_degradation, phase4_ablation and phase8F_multi were excluded here while they were
# suffix-protected but unstamped. Every experiment script now stamps (2026-07-20), so they are in.
# phase8F_emit.py stays: phase8F_multi.py does `import phase8F_emit as F` to reuse
# recon_error_matrix, so it is a live module dependency, not merely a superseded experiment. It was
# briefly archived on the strength of having no deliverable and no paper reference -- neither check
# looks at imports, and the whole test suite failed to collect.
_MUST_STAMP = {
    "experiment_synthetic_multiseed.py",
    "phase1_indian_pines.py", "phase2_cross_sensor.py", "phase2_degradation.py",
    "phase2_group_ablation.py", "phase3_atmosphere.py", "phase4R_reliability.py",
    "phase4_ablation.py", "phase5_ab_flagship.py", "phase6_second_dataset.py",
    "phase7_efficiency.py", "phase8D_difficulty.py", "phase8E_dofa.py",
    "phase8F_emit.py", "phase8F_multi.py", "phase8G_emit_reliability.py",
    "phase9_physics_vs_random.py",
    "phase8R_reliability.py", "phase8_cloudsen12.py",
}


# --------------------------------------------------------------------------------------- helpers
def _callee(call):
    """Name of the thing being called: `open` -> 'open', `fig.savefig` -> 'savefig'."""
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _mentions_smoke(node):
    """True if the expression reads `<something>.smoke` (i.e. args.smoke)."""
    return any(isinstance(n, ast.Attribute) and n.attr == "smoke" for n in ast.walk(node))


def _has_smoke_literal(node):
    """True if a string literal containing 'smoke' appears in the expression. This is what stops
    `sfx = ""` in the smoke branch from counting as protection: it is smoke-conditional and changes
    nothing, which is precisely the bug in a different costume."""
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and "smoke" in n.value
               for n in ast.walk(node))


def _accepts_smoke(tree):
    """True if the script declares a `--smoke` argparse flag."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and _callee(node) == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--smoke"):
            return True
    return False


def _smoke_suffix_names(tree):
    """Names whose value both depends on --smoke AND carries a 'smoke' literal.

    Two accepted shapes, because the repo legitimately uses both:
      `if args.smoke: ...; sfx = "_smoke"`                       (phase3/4R/7/8D/8E/8F/8G/8R/8)
      `sfx = args.tag + ("_smoke" if args.smoke else "")`        (phase8F_multi, composing --tag)
    """
    names = set()
    for node in ast.walk(tree):
        # shape 1: assigned inside a branch guarded by args.smoke
        if isinstance(node, ast.If) and _mentions_smoke(node.test):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Assign) and _has_smoke_literal(sub.value):
                        names.update(t.id for t in sub.targets if isinstance(t, ast.Name))
        # shape 2: assigned from an expression that itself reads args.smoke
        if isinstance(node, ast.Assign) and _mentions_smoke(node.value) and _has_smoke_literal(node.value):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _is_paper_path(node):
    """`P("...")` -- the helper every experiment uses to build a path under paper/."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "P" and len(node.args) == 1)


def _opens_for_writing(call):
    mode = None
    if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
        mode = call.args[1].value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value
    if mode is None:
        return False                                  # bare open(p) is a read
    return isinstance(mode, str) and any(c in mode for c in "wax+")


def _paper_path_vars(tree):
    """{name: [path_expression, ...]} for every local assigned DIRECTLY from `P(...)`.

    Without this, every check in this module has the same blind spot: it only recognises a literal
    `P(...)` written inside the write call itself, so a script that names its artefact once and
    reuses the variable -- which is the right way to keep the write, the stamp and the sidecar
    pointing at ONE path instead of three copies of an expression that can drift -- becomes
    INVISIBLE here. The stamp test then reports "never calls stamp()" for a script that does, and,
    far worse, a --smoke script could lose its suffix protection with nothing failing, because
    `unsuffixed_writes` iterates the same (empty) site list and concludes there is nothing to
    protect. Resolving one level of assignment closes that.

    A name assigned several different P() paths contributes ALL of them, so the check stays
    conservative: a variable that is suffix-protected on one branch and not on another still fails.
    """
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_paper_path(node.value):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, []).append(node.value.args[0])
    return out


def write_sites(tree):
    """[(lineno, callee, path_expression_node)] for every write of a paper/ path in this module."""
    sites = []
    pvars = _paper_path_vars(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node)
        if name == "open":
            if not _opens_for_writing(node):
                continue
        elif name not in _WRITE_CALLS:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if _is_paper_path(arg):
                sites.append((node.lineno, name, arg.args[0]))
            elif isinstance(arg, ast.Name) and arg.id in pvars:
                for expr in pvars[arg.id]:
                    sites.append((node.lineno, name, expr))
    return sites


def _interpolated_names(path_expr):
    """Names the path expression depends on. A bare string literal depends on nothing -> empty set,
    which is exactly the condition that fails the assertion below."""
    if isinstance(path_expr, ast.Constant):
        return set()
    return {n.id for n in ast.walk(path_expr) if isinstance(n, ast.Name)}


def unsuffixed_writes(source):
    """The whole check in one function, over source TEXT so the negative control can exercise the
    same code path the real scripts go through. Returns [] when the module is protected."""
    tree = ast.parse(source)
    if not _accepts_smoke(tree):
        return []                                     # no --smoke -> no smoke-clobber vector
    suffixes = _smoke_suffix_names(tree)
    bad = []
    for lineno, callee, path_expr in write_sites(tree):
        if not (_interpolated_names(path_expr) & suffixes):
            bad.append((lineno, callee, ast.unparse(path_expr)))
    return bad


def _scripts():
    """Experiment scripts that both accept --smoke and write a paper/ artefact -- i.e. exactly the
    ones a smoke run could clobber."""
    out = []
    for fn in sorted(os.listdir(_EXP)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_EXP, fn), encoding="utf-8").read())
        if _accepts_smoke(tree) and write_sites(tree):
            out.append(fn)
    return out


_SMOKE_SCRIPTS = _scripts()


# ----------------------------------------------------------------------------------------- tests
def test_discovery_actually_finds_the_scripts_it_is_meant_to_guard():
    """A parametrized test over an empty list is a green tick that checked nothing. Pin the scripts
    known to write paper/ deliverables from a --smoke path so discovery cannot silently collapse."""
    assert len(_SMOKE_SCRIPTS) >= 12, f"discovery found only {_SMOKE_SCRIPTS}"
    for required in ["phase2_degradation.py", "phase3_atmosphere.py", "phase4R_reliability.py",
                     "phase4_ablation.py", "phase7_efficiency.py", "phase8D_difficulty.py",
                     "phase8E_dofa.py", "phase8F_emit.py", "phase8F_multi.py",
                     "phase8G_emit_reliability.py", "phase8R_reliability.py",
                     "phase8_cloudsen12.py"]:
        assert required in _SMOKE_SCRIPTS, f"{required} dropped out of the smoke-isolation scope"


@pytest.mark.parametrize("script", _SMOKE_SCRIPTS)
def test_smoke_cannot_write_an_unsuffixed_deliverable(script):
    """THE lock. Every CSV, .tex and figure an experiment writes must carry the smoke suffix, not
    just the headline CSV: one unsuffixed artefact is still a clobbered deliverable, and a figure
    silently replaced by a 1-seed version is no better than a table."""
    bad = unsuffixed_writes(open(os.path.join(_EXP, script), encoding="utf-8").read())
    assert not bad, (
        f"{script} writes {len(bad)} paper/ artefact(s) whose path does not vary with --smoke:\n"
        + "\n".join(f"  line {ln}: {callee}(P({expr}))" for ln, callee, expr in bad)
        + "\n-> a --smoke run of this script overwrites the real deliverable(s).")


@pytest.mark.parametrize("script", sorted(_MUST_STAMP))
def test_primary_result_csv_is_provenance_stamped(script):
    """Suffixing answers 'is this file smoke output'; the stamp answers 'what produced this file'.
    Both were needed: the committed phase4R numbers were once a 1-seed 12-epoch smoke quoted as
    converged, and the CSV itself could not say otherwise."""
    tree = ast.parse(open(os.path.join(_EXP, script), encoding="utf-8").read())
    stamped = [expr for _, callee, expr in
               [(ln, c, ast.unparse(p)) for ln, c, p in write_sites(tree)] if callee == "stamp"]
    assert stamped, f"{script} writes a result CSV but never calls bandsim.provenance.stamp()"
    assert any(".csv" in s for s in stamped), f"{script} stamps something that is not its result CSV"


def test_the_check_fails_on_a_reverted_script():
    """Negative control: the assertion above is only worth anything if it FAILS on the pattern it
    exists to catch. This is the pre-fix shape of every one of these scripts."""
    reverted = '''
import argparse
def P(rel): return rel
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = ""
    if args.smoke:
        args.epochs = 1
        sfx = "_smoke"
    with open(P("results_phaseX.csv"), "w") as f:            # REVERTED: no suffix
        f.write("x")
    fig.savefig(P(f"figs/fig_x{sfx}.pdf"))                   # protected
'''
    bad = unsuffixed_writes(reverted)
    assert len(bad) == 1, f"expected exactly the unsuffixed CSV to be flagged, got {bad}"
    assert bad[0][1] == "open" and "results_phaseX.csv" in bad[0][2]


def test_a_path_held_in_a_variable_is_still_seen():
    """Negative control for the blind spot `_paper_path_vars` closes.

    Naming an artefact once and reusing the variable is the RIGHT thing to do -- it keeps the write,
    the stamp and the provenance sidecar pointing at ONE path instead of three copies of an
    expression that can drift apart. It also used to make a script invisible to every check in this
    module, because both checks iterate `write_sites` and an unrecognised path yields no site at all:
    zero sites reads as "nothing to protect" AND as "never stamps", both silently wrong. Here the
    protected and unprotected variables must be told apart, and the stamp must be found."""
    named = '''
import argparse
def P(rel): return rel
def stamp(p, a): pass
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    good = P(f"results_phaseX{sfx}.csv")
    leaky = P("results_phaseY.csv")                          # REVERTED: no suffix
    with open(good, "w") as f:
        f.write("x")
    with open(leaky, "w") as f:
        f.write("x")
    stamp(good, args)
'''
    sites = write_sites(ast.parse(named))
    assert len(sites) == 3, f"variable-held paths must be visible as write sites: {sites}"
    assert any(callee == "stamp" for _, callee, _ in sites), "a stamp() on a named path must be seen"
    bad = unsuffixed_writes(named)
    assert len(bad) == 1 and "results_phaseY.csv" in bad[0][2], \
        f"the unsuffixed variable-held write must still be caught: {bad}"


def test_an_empty_suffix_in_the_smoke_branch_is_not_accepted():
    """`sfx = ""` under `if args.smoke:` is smoke-conditional and changes no filename. It must not
    count as protection, or the guard can be defeated while looking exactly like the fix."""
    fake_fix = '''
import argparse
def P(rel): return rel
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = ""
    if args.smoke:
        sfx = ""
    with open(P(f"results_phaseX{sfx}.csv"), "w") as f:
        f.write("x")
'''
    assert unsuffixed_writes(fake_fix), "an empty smoke suffix must not satisfy the check"


def test_a_suffix_unrelated_to_smoke_is_not_accepted():
    """Interpolating *a* variable is not the property being tested; interpolating one that varies
    with --smoke is. A plain `--tag` with no smoke composition leaves smoke output on the real
    paths, which is exactly how phase8F_multi was still exposed while looking parameterised."""
    tag_only = '''
import argparse
def P(rel): return rel
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
    sfx = args.tag
    with open(P(f"results_phaseX{sfx}.csv"), "w") as f:
        f.write("x")
'''
    assert unsuffixed_writes(tag_only), "a --tag that ignores --smoke must not satisfy the check"


def test_composing_tag_with_smoke_is_accepted():
    """The shape phase8F_multi now uses: keep --tag working, and still isolate smoke output."""
    composed = '''
import argparse
def P(rel): return rel
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = args.tag + ("_smoke" if args.smoke else "")
    with open(P(f"results_phaseX{sfx}.csv"), "w") as f:
        f.write("x")
'''
    assert not unsuffixed_writes(composed)


def test_a_script_without_a_smoke_flag_is_out_of_scope():
    """phase1/phase5/phase6 write paper/ deliverables but have no --smoke, so they cannot be
    clobbered by this vector. Checking them would demand a suffix that nothing ever sets."""
    no_smoke = '''
def P(rel): return rel
def main():
    with open(P("results_phaseX.csv"), "w") as f:
        f.write("x")
'''
    assert not unsuffixed_writes(no_smoke)


def test_reads_are_not_mistaken_for_writes():
    """`open(p)` and `open(p, "rb")` are reads -- phase8E hashes its DOFA checkpoint that way. A
    checker that flagged reads would push people to suffix paths that are inputs."""
    reader = '''
import argparse
def P(rel): return rel
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    with open(P("lookup_table.npz"), "rb") as f:
        pass
    with open(P("results_phaseX" + sfx + ".csv"), "w") as f:
        f.write("x")
'''
    assert not unsuffixed_writes(reader)
