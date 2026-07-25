"""scripts/doctor.py's own assertions need assertions.

doctor.py exists to turn defects found reactively into standing checks. A check that cannot itself
fail on the pattern it targets is a green tick that verified nothing -- which is the very failure
doctor.py was written to catch, one level up, in the tool that is supposed to catch it.

The schema-drift check is tested here rather than only observed working once, because its first
version had a real blind spot that a single successful run did not reveal: it compared the header
against CSV_COLS in ONE direction. It caught the stale phase-7 table only because five columns had
been RENAMED. Had that schema merely GROWN -- the ordinary case, and exactly what phase 7 did going
62 -> 89 columns -- the old header would have been a strict subset, every column still producible,
and a table missing eighty-one columns would have passed. That case is now the second test below.
"""
import ast
import csv
import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("doctor", os.path.join(_ROOT, "scripts", "doctor.py"))
doctor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doctor)                                           # no side effects at import


def _paper(tmp_path, name, header):
    d = tmp_path / "paper"
    d.mkdir(exist_ok=True)
    with open(d / name, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerow(["x"] * len(header))
    return str(d)


# ------------------------------------------------------------------- schema drift, both directions
def test_a_renamed_column_is_drift(tmp_path):
    """The shape that was actually found in paper/: results_phase7_efficiency.csv carried
    int8_size_mb long after the generator had renamed it dynint8_size_mb."""
    p = _paper(tmp_path, "results_x.csv", ["name", "int8_size_mb"])
    drift, checked, total = doctor.schema_drift({"results_x.csv": ["name", "dynint8_size_mb"]}, p)
    assert checked == 1 and total == 1
    assert drift and "int8_size_mb" in drift[0]
    assert "no longer writes" in drift[0]


def test_a_header_that_is_a_strict_subset_is_drift(tmp_path):
    """THE blind spot. Every column in this file is still producible -- nothing was renamed, the
    schema only grew. One-directional comparison passes it while it is missing 3 of 5 columns."""
    p = _paper(tmp_path, "results_x.csv", ["name", "params"])
    drift, _, _ = doctor.schema_drift(
        {"results_x.csv": ["name", "params", "lat_cpu_ms", "thr_cpu", "quant_scope"]}, p)
    assert drift, "a header missing 3 of 5 columns passed as current"
    assert "this file lacks" in drift[0]


def test_an_exact_header_is_not_drift(tmp_path):
    """Negative control: the check must be passable, or the two above prove nothing."""
    cols = ["name", "params", "lat_cpu_ms"]
    p = _paper(tmp_path, "results_x.csv", cols)
    drift, checked, total = doctor.schema_drift({"results_x.csv": cols}, p)
    assert drift == [] and checked == 1 and total == 1


def test_column_ORDER_alone_is_not_drift(tmp_path):
    """Readers address columns by name. Flagging a reordering would be a false positive, and a
    check that cries wolf gets muted -- which costs more than the reorder it caught."""
    p = _paper(tmp_path, "results_x.csv", ["params", "name"])
    drift, _, _ = doctor.schema_drift({"results_x.csv": ["name", "params"]}, p)
    assert drift == []


# ------------------------------------------------------------------------------- honest coverage
def test_unattributable_deliverables_are_counted_not_silently_skipped(tmp_path):
    """A PASS must say what it looked at. Three deliverables, one attributable: reporting a bare
    green tick would assert the other two were verified when they were never opened."""
    d = tmp_path / "paper"
    d.mkdir()
    for n in ("results_a.csv", "results_b.csv", "results_c.csv"):
        _paper(tmp_path, n, ["name"])
    drift, checked, total = doctor.schema_drift({"results_a.csv": ["name"]}, str(d))
    assert drift == []
    assert (checked, total) == (1, 3), "coverage must distinguish 'checked' from 'present'"


def test_smoke_output_is_out_of_scope(tmp_path):
    """Smoke tables are machinery evidence, not deliverables, and are already suffixed away."""
    d = tmp_path / "paper"
    d.mkdir()
    _paper(tmp_path, "results_x_smoke.csv", ["stale_column"])
    drift, checked, total = doctor.schema_drift({"results_x_smoke.csv": ["name"]}, str(d))
    assert (drift, checked, total) == ([], 0, 0)


# --------------------------------------------------------------------------- attribution is narrow
def test_the_smoke_suffix_is_stripped_when_naming_the_deliverable():
    names = doctor._paper_csv_names(ast.parse(
        'def P(r): return r\n'
        'def main(args):\n'
        '    sfx = "_smoke" if args.smoke else ""\n'
        '    open(P(f"results_x{sfx}.csv"), "w")\n'))
    assert names == {"results_x.csv"}


def test_a_script_writing_several_tables_cannot_be_attributed():
    """phase8_cloudsen12 writes curve/perclass/scenarios from one module. Charging all three
    against its single CSV_COLS would manufacture failures on two of them."""
    names = doctor._paper_csv_names(ast.parse(
        'def P(r): return r\n'
        'def main(args):\n'
        '    sfx = "_smoke" if args.smoke else ""\n'
        '    open(P(f"results_x_curve{sfx}.csv"), "w")\n'
        '    open(P(f"results_x_perclass{sfx}.csv"), "w")\n'))
    assert len(names) == 2, "attribution must refuse a multi-table script (len != 1 -> skipped)"


def test_non_csv_paper_paths_are_ignored():
    names = doctor._paper_csv_names(ast.parse(
        'def P(r): return r\n'
        'def main(args):\n'
        '    sfx = "_smoke" if args.smoke else ""\n'
        '    fig.savefig(P(f"figs/fig_x{sfx}.pdf"))\n'
        '    open(P(f"results_x{sfx}.tex"), "w")\n'))
    assert names == set()


def test_declared_columns_attributes_the_real_phase7_deliverable():
    """Ties the synthetic tests above to the repo: phase7_efficiency.py is the script that
    declares CSV_COLS and writes exactly one paper/ CSV, so it must resolve."""
    declared = doctor._declared_columns()
    assert "results_phase7_efficiency.csv" in declared, \
        "the one attributable deliverable stopped being attributable"
    cols = declared["results_phase7_efficiency.csv"]
    assert "name" in cols and "dynint8_size_mb" in cols and len(cols) > 60
    assert "int8_size_mb" not in cols, "the pre-rewrite column name is back in CSV_COLS"


@pytest.mark.parametrize("mutator", ["replace", "rename", "remove", "unlink", "copy", "move"])
def test_the_smoke_guard_sees_path_mutations_not_just_opens(mutator):
    """A path can be MUTATED without being opened -- os.replace() finishing an atomic write, or
    os.remove() clearing a stale artefact. Both were about to be written into phase 7, and a
    --smoke run doing either to an unsuffixed path clobbers the real deliverable."""
    import sys
    sys.path.insert(0, os.path.join(_ROOT, "tests"))
    import test_smoke_isolation as guard
    src = (
        'import argparse, os\n'
        'def P(r): return r\n'
        'def main():\n'
        '    ap = argparse.ArgumentParser()\n'
        '    ap.add_argument("--smoke", action="store_true")\n'
        '    args = ap.parse_args()\n'
        '    sfx = ""\n'
        '    if args.smoke:\n'
        '        sfx = "_smoke"\n'
        f'    os.{mutator}(P("results_phaseX.csv"))\n')
    assert guard.unsuffixed_writes(src), f"os.{mutator}() on a paper/ path evaded the smoke guard"
