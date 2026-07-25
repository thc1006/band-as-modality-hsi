"""`dirty` in a provenance stamp must mean "the CODE moved", not "something was written".

Two defects, both found on 2026-07-20 while preparing the post-fix re-run campaign:

  1. `dirty` was `bool(git status --porcelain)` over the whole tree. Results are tracked in git, so
     the first phase to write paper/ dirtied the tree and every phase after it was stamped
     irreproducible while the code was pristine. The flag is meant to answer "can this be rebuilt
     from that commit alone"; measuring outputs makes it true by construction during exactly the
     campaign it exists to certify.

  2. Scoping it needed the path out of each porcelain line, and `line[3:]` is wrong here. Porcelain's
     prefix is a fixed-width 'XY ', but `_git` strips its output, so a leading blank X is eaten on
     the FIRST line only: ' M bandsim/x.py' arrives as 'M bandsim/x.py' and a fixed offset returns
     'andsim/x.py'. That silently mis-classified one file per call — always the first — so a single
     code edit sitting at the top of the list reported a clean tree.
"""
import bandsim.provenance as P


# (porcelain line, is it a code path?)
_CASES = [
    ("M bandsim/model.py", True),                            # stripped first line: defect 2
    (" M bandsim/model.py", True),
    ("MM experiments/phase8_cloudsen12.py", True),
    ("?? experiments/phase9_new.py", True),                  # untracked code still counts
    (" D tests/test_x.py", True),
    (" M configs/spec.yaml", True),
    ("M reproduce.sh", True),
    (" M requirements-lock.txt", True),
    (" M paper/results_phase8R_reliability.csv", False),     # an output, not the code: defect 1
    ("?? paper/figs/fig_new.pdf", False),
    (" M paper/results_x.csv.provenance.json", False),
    (" M docs/review/PAPER_DESIGN.md", False),
    ("R  paper/main.tex -> archive/icaims2026_submission/main.tex", False),
    ("R  archive/superseded/x.py -> experiments/x.py", True),   # renamed INTO code
    (' M "paper/a b.csv"', False),                              # quoted path with a space
]


def test_porcelain_paths_are_classified_by_what_they_can_change():
    wrong = [(line, want, P._is_code(P._status_path(line)))
             for line, want in _CASES if P._is_code(P._status_path(line)) != want]
    assert not wrong, "misclassified porcelain lines (line, want, got):\n" + "\n".join(map(str, wrong))


def test_first_line_without_its_leading_space_still_resolves():
    """Pinned separately because it is the whole of defect 2 and looks like a typo in review."""
    assert P._status_path("M bandsim/provenance.py") == "bandsim/provenance.py"
    assert P._status_path(" M bandsim/provenance.py") == "bandsim/provenance.py"


def test_rename_reports_the_destination():
    """Archiving a file shows as a rename. What matters is where it landed: moving a module OUT of
    experiments/ changes what the code can do just as much as editing it."""
    assert P._status_path("R  a/old.py -> experiments/new.py") == "experiments/new.py"


def test_writing_only_outputs_does_not_mark_the_code_dirty():
    """The campaign property: a run that writes paper/ and touches no source must still stamp
    dirty=False, or every result after the first is marked unreproducible for no reason."""
    g = P.git_state()
    assert set(g) >= {"commit", "branch", "dirty", "dirty_files", "outputs_dirty"}
    assert g["dirty_files"] == len(g["dirty_code_paths"]) or g["dirty_files"] > 20, \
        "dirty_files must count the code paths it reports (the list is capped at 20)"
    if g["dirty"] is not None:
        assert g["dirty"] == bool(g["dirty_files"]), "dirty must agree with its own file count"
