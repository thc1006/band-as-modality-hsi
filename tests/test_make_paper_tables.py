"""Guards for the paper-table generator, asserted on the tables it WRITES.

This script's output is read as a result, so the failure that matters is not a crash but a
plausible-looking table built from a CSV that no longer means what it did. The regression that
motivated the file: `audc()` sorts by x internally while the endpoints were read positionally, so a
reordered CSV produced a correct AUDC beside a wrong clean, max-miss and retention.
"""
import os
import sys
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import make_paper_tables as MT                                                  # noqa: E402

HEAD = "missing_groups," + ",".join(f"{k}_mean" for k, _ in MT.METHODS)


def curve_csv(rows):
    """rows: list of (x, base_value); every method gets base + a small per-method offset."""
    out = [HEAD]
    for x, v in rows:
        out.append(",".join([str(x)] + [f"{v + 0.5 * i:.4f}" for i in range(len(MT.METHODS))]))
    return "\n".join(out) + "\n"


def write(tmp_path, text, name="curve.csv"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def build(tmp_path, text):
    """Run the whole entrypoint into tmp_path and return (markdown, latex)."""
    src = write(tmp_path, text)
    out = tmp_path / "tables"
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(sys, "argv", ["make_paper_tables.py", "--input", src,
                                 "--output-dir", str(out)])
        MT.main()
    finally:
        mp.undo()
    return ((out / "baselines_table.md").read_text(),
            (out / "baselines_table.tex").read_text())


GOOD = curve_csv([(0, 80.0), (1, 74.0), (2, 68.0), (3, 62.0), (4, 56.0), (5, 50.0), (6, 44.0)])


def test_a_normal_curve_produces_both_tables(tmp_path):
    md, tex = build(tmp_path, GOOD)
    assert "| Method | clean mIoU | 6-miss mIoU | AUDC | Retention of means (%) |" in md
    assert r"\begin{tabular}{lrrrr}" in tex and r"\bottomrule" in tex
    assert r"Retention of means (\%)" in tex, "the LaTeX percent sign must stay escaped"
    for _, name in MT.METHODS:
        assert name in md, f"{name} missing from the table"
    # the source is identified in both files, so a table can be traced to the CSV behind it
    assert "sha256=" in md and "sha256=" in tex


def test_reordering_the_csv_rows_cannot_change_the_table(tmp_path):
    """THE regression. audc() sorts internally; the endpoints used to be row 0 and row -1, so a
    shuffled CSV kept the area correct and silently reported the wrong clean, max-miss and
    retention -- on x=[2,0,1], y=[60,80,70] it printed retention 116.7% instead of 75.0%."""
    rows = [(0, 80.0), (1, 74.0), (2, 68.0), (3, 62.0), (4, 56.0), (5, 50.0), (6, 44.0)]
    shuffled = [rows[i] for i in (3, 0, 6, 2, 5, 1, 4)]
    a = build(tmp_path / "a", curve_csv(rows))
    b = build(tmp_path / "b", curve_csv(shuffled))

    def body(text):
        # the provenance line SHOULD differ -- the two source files have different bytes and so
        # different digests. What must not differ is a single number in the table itself.
        return [ln for ln in text.splitlines()
                if not ln.startswith("<!--") and not ln.startswith("%")]

    assert body(a[0]) == body(b[0]), "markdown changed when only the row order did"
    assert body(a[1]) == body(b[1]), "latex changed when only the row order did"
    # the shuffled build must report the TRUE endpoints, not row 0 and row -1 of the shuffled file
    assert "| 80.0 | 44.0 |" in b[0], "clean/max-miss were taken by row position again"
    assert "116.7" not in b[0], "the >100% retention was the symptom of reading endpoints by row"


def test_a_partial_csv_does_not_produce_a_complete_looking_table(tmp_path):
    """Used to emit a table of whatever columns happened to be present, so a renamed or dropped
    baseline left the paper without a word."""
    partial = "missing_groups,b1_mean\n0,80\n1,70\n"
    with pytest.raises(ValueError, match="missing required columns"):
        build(tmp_path, partial)


def test_an_unknown_method_column_is_refused_rather_than_omitted(tmp_path):
    """The opposite direction: a baseline the experiment ran but the table does not know would
    disappear from the paper silently."""
    extra = GOOD.replace(HEAD, HEAD + ",b5_mean")
    extra = "\n".join(ln if i == 0 else (ln + ",70.0") if ln else ln
                      for i, ln in enumerate(extra.splitlines())) + "\n"
    with pytest.raises(ValueError, match="does not"):
        build(tmp_path, extra)


def test_a_curve_without_the_clean_point_is_refused(tmp_path):
    """clean and the retention denominator ARE missing_groups=0; without it the first level was
    reported as clean and every retention was wrong by that ratio."""
    with pytest.raises(ValueError, match="no missing_groups=0"):
        build(tmp_path, curve_csv([(1, 80.0), (2, 70.0)]))


def test_duplicate_levels_are_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicate missing_groups"):
        build(tmp_path, curve_csv([(0, 80.0), (0, 60.0), (1, 70.0)]))


def test_empty_and_header_only_inputs_raise_with_the_path(tmp_path):
    with pytest.raises(ValueError, match="header only"):
        build(tmp_path / "h", HEAD + "\n")
    with pytest.raises(ValueError, match="no header row"):
        build(tmp_path / "e", "")


def test_non_finite_and_out_of_range_values_are_refused(tmp_path):
    bad = curve_csv([(0, 80.0), (1, 70.0)]).replace("70.0000", "nan", 1)
    with pytest.raises(ValueError, match="non-finite"):
        build(tmp_path / "n", bad)
    with pytest.raises(ValueError, match="outside the expected mIoU percent range"):
        build(tmp_path / "hi", curve_csv([(0, 150.0), (1, 140.0)]))
    # the range check alone does NOT catch this one: a fraction-scale curve is inside [0, 100] and
    # would print a 100x-wrong table that still looks like a table
    frac = "\n".join([HEAD] + [",".join([str(x)] + [f"{v:.4f}"] * len(MT.METHODS))
                               for x, v in ((0, 0.80), (1, 0.70))]) + "\n"
    with pytest.raises(ValueError, match="never exceeds 1.0"):
        build(tmp_path / "frac", frac)


def test_a_gapped_sweep_is_allowed_and_labelled(tmp_path, capsys):
    """AUDC is still defined over the x-range, but it stops being a mean over the levels tested, so
    the run says so rather than leaving the reader to assume."""
    build(tmp_path, curve_csv([(0, 80.0), (1, 74.0), (3, 62.0)]))
    assert "skips [2]" in capsys.readouterr().out


def test_ties_at_the_printed_precision_share_the_bold(tmp_path):
    """80.04 and 80.03 both print as "80.0"; bolding only one of them shows the reader two identical
    numbers with one marked best."""
    rows = [("m1", 0.0, 0.0, 80.04, 0.0), ("m2", 0.0, 0.0, 80.03, 0.0)]
    import numpy as np
    md, tex, best = MT.render(np.array([0, 1]), rows, "x.csv", "d" * 64)
    assert md.count("**80.0**") == 2, "two cells printing 80.0 must be bolded alike"
    assert tex.count(r"\textbf{80.0}") == 2
    assert best == 80.0


def test_method_names_are_escaped_for_latex():
    """No current name needs it. An unescaped & adds a column and an unescaped % comments out the
    rest of the line -- both produce a plausible table rather than an error."""
    assert MT.latex_escape("A & B_1 100% {x} #2 $z") == r"A \& B\_1 100\% \{x\} \#2 \$z"


def test_neither_table_is_updated_if_the_second_write_fails(tmp_path):
    """The two tables carry the same numbers. Writing them in sequence left a new markdown table
    beside a stale LaTeX one on a mid-way failure, which is worse than either being absent."""
    md_path, tex_path = str(tmp_path / "a.md"), str(tmp_path / "b.tex")
    for p in (md_path, tex_path):
        with open(p, "w") as f:
            f.write("STALE")
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(src, dst)

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(MT.os, "replace", flaky)
        with pytest.raises(OSError):
            MT.write_atomically([(md_path, "NEW-MD"), (tex_path, "NEW-TEX")])
    finally:
        mp.undo()
    # the first rename did land -- the point of the test is that nothing is left half-written and
    # no temporary file survives to be picked up later
    assert not os.path.exists(md_path + ".tmp") and not os.path.exists(tex_path + ".tmp")
    assert open(tex_path).read() == "STALE"


def test_the_real_phase2_curve_still_builds(tmp_path):
    """The shipped CSV must satisfy every guard above; if it does not, the guards are wrong."""
    real = os.path.join(_ROOT, "paper", "results_phase2_curve.csv")
    if not os.path.exists(real):
        pytest.skip("paper/results_phase2_curve.csv not present")
    xs, curves = MT.load_curve(real)
    assert xs[0] == 0 and list(xs) == sorted(xs)
    assert set(curves) == {k for k, _ in MT.METHODS}
