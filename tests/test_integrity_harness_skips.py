"""The integrity harness must SKIP, not fail, a phase whose input data are absent -- and must still RUN
every phase whose data are there.

reproduce.sh runs experiments/integrity_check.py under `set -e` before any tier. A phase that crashed
for want of its data therefore stopped the whole reproduction -- including the tiers whose data WERE
present (a clone holding only CloudSEN12 never reached the flagship). This pins the skips for the 6S
table, the CloudSEN12 dataset, the Indian Pines cube and the EMIT inputs, in both directions: a guard
that skipped a phase whose data are present would make the gate pass vacuously. The expected skip sets
are derived from what each invocation reads (the cube by file name, through phase2's load_data or
phase1's load_indian_pines; the 6S table by file name; CloudSEN12 by its data path or through phase8's
DATA or load_split), so a new phase of that kind cannot be added unguarded, and with no data at all only
the two data-free phases may run. `run_phase` is stubbed, and the existence checks (and the CloudSEN12
directory check, in the tests that pass `s2`) are faked for exactly the paths the harness should look at
-- a guard pointed anywhere else sees the real filesystem and the tests notice. The glob is faked too:
only the harness's exact EMIT pattern can find granules (when a test says they exist), and every other
pattern finds nothing, which also keeps main()'s backup and restore of paper/ empty, so these tests
rewrite no result file. A launch through subprocess (any Popen, however it was imported) or through
os.system (called as os.system, or bound at import to a module-level name of integrity_check) fails the
test, so an edit that bypassed the stub that way could not start the real phases from a unit test.
"""
import glob
import os
import re
import sys
from pathlib import Path

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_ROOT, "experiments")
sys.path.insert(0, _EXP)
sys.path.insert(0, _ROOT)
import integrity_check as IC                                                     # noqa: E402

_IP_FILES = ("Indian_pines_corrected.mat", "Indian_pines_gt.mat")
_IP_PATHS = {os.path.join(IC.ROOT, "data", "indian_pines", f): f for f in _IP_FILES}
_LUT = os.path.join(IC.ROOT, "data", "srf_cache", "T_6s_grid.npz")
_EMIT_SAMPLE = os.path.join(IC.ROOT, "data", "emit", "emit_sample.npz")
_EMIT_GLOB = os.path.join(IC.ROOT, "data", "emit*", "*RFL*.nc")
_S2_DIR = os.path.join(IC.ROOT, "data", "cloudsen12")


def _source(argv):
    return Path(_EXP, argv.split()[0]).read_text(encoding="utf-8")


def _reads_indian_pines(argv):
    """Static: does this invocation load the Indian Pines cube (by file name, or via phase2's load_data or
    phase1's load_indian_pines)? A --dataset option decides it outright (phase6 runs `--dataset synthetic`)."""
    script, *rest = argv.split()
    if "--dataset" in rest:
        return rest[rest.index("--dataset") + 1] == "indian_pines"
    return bool(re.search(r"Indian_pines_corrected|\bload_data\(|\bload_indian_pines\(", _source(script)))


_READS_IP = {label for label, argv, _ in IC.PHASES if _reads_indian_pines(argv)}


_READS_6S = {label for label, argv, _ in IC.PHASES if "T_6s_grid.npz" in _source(argv)}
_S2_RE = re.compile(r'\bP8\.DATA\b|\bload_split\(|"data", "cloudsen12"')
_READS_S2 = {label for label, argv, _ in IC.PHASES if _S2_RE.search(_source(argv))}
_ALL = {label for label, _, _ in IC.PHASES}
_DATA_FREE = {"Phase0 synthetic PoC", "Phase6 synthetic"}


def _run(monkeypatch, capsys, ip_present=(), emit_sample=False, granules=False, lut=True, s2=None):
    """IC.main() with run_phase stubbed; of the exact paths the harness should check, the Indian Pines
    files in `ip_present` exist, the 6S table if `lut`, the EMIT extract and granules as given, and the
    CloudSEN12 directory if `s2` (None: the real one); every other path's existence is the real
    filesystem's, and every other glob finds nothing (so main() backs up and restores no paper/ file).
    Returns (ran labels, out)."""
    ran = []
    monkeypatch.setattr(IC, "run_phase", lambda label, argv, outputs: (ran.append(label), (True, "stub"))[1])

    def launch(*a, **k):
        pytest.fail("the harness tried to launch a phase for real")
    bound = [n for n, v in vars(IC).items() if v is os.system]
    assert not bound, f"os.system bound at import ({bound}) would escape the patch below"
    monkeypatch.setattr(IC.subprocess, "run", launch)
    monkeypatch.setattr(IC.subprocess.Popen, "_execute_child", launch)    # every Popen, however imported
    monkeypatch.setattr(IC.os, "system", launch)
    real_exists = os.path.exists

    def exists(p):
        p = str(p)
        if p in _IP_PATHS:
            return _IP_PATHS[p] in ip_present
        if p == _LUT:
            return lut
        if p == _EMIT_SAMPLE:
            return emit_sample
        return real_exists(p)
    monkeypatch.setattr(IC.os.path, "exists", exists)
    if s2 is not None:
        real_isdir = os.path.isdir
        monkeypatch.setattr(IC.os.path, "isdir", lambda p: s2 if str(p) == _S2_DIR else real_isdir(p))
    granule = os.path.join(_ROOT, "data", "emit_x", "EMIT_L2A_RFL_001_x.nc")
    monkeypatch.setattr(glob, "glob", lambda pat, *a, **k: [granule] if (granules and pat == _EMIT_GLOB) else [])
    with pytest.raises(SystemExit) as stop:
        IC.main()
    assert stop.value.code == 0
    return set(ran), capsys.readouterr().out


def _skipped_for(reason, out):
    return {label for label in _ALL if re.search(rf"\[SKIP\] {re.escape(label)}\s+{re.escape(reason)}", out)}


def _skipped_for_ip(out):
    return _skipped_for("needs Indian Pines", out)


def test_the_static_check_finds_the_hsi_phases():
    assert len(_READS_IP) >= 9, f"found only {sorted(_READS_IP)} reading Indian Pines -- the check has drifted"
    assert not _DATA_FREE & _READS_IP
    # phase8H (not in the harness) reaches the cube only through phase1's load_indian_pines
    assert _reads_indian_pines("phase8H_hsi6s_shift.py")


@pytest.mark.parametrize("present", [(), ("Indian_pines_corrected.mat",), ("Indian_pines_gt.mat",)])
def test_hsi_phases_are_skipped_without_the_whole_cube(monkeypatch, capsys, present):
    """Absent, or half present (load_data reads both files): skipped, never run."""
    ran, out = _run(monkeypatch, capsys, ip_present=present)
    assert not _READS_IP & ran, f"ran without the cube: {sorted(_READS_IP & ran)}"
    assert _skipped_for_ip(out) == _READS_IP, "exactly the phases that read the cube are skipped for it"
    assert _DATA_FREE <= ran


def test_hsi_phases_run_when_the_cube_is_present(monkeypatch, capsys):
    ran, out = _run(monkeypatch, capsys, ip_present=_IP_FILES)
    assert _READS_IP <= ran, f"skipped although the cube is present: {sorted(_READS_IP - ran)}"
    assert not _skipped_for_ip(out)


@pytest.mark.parametrize("granules", [False, True])
@pytest.mark.parametrize("emit_sample", [False, True])
def test_emit_phases_follow_the_input_each_reads(monkeypatch, capsys, granules, emit_sample):
    """phase8F_multi reads the granules; phase8F_emit and phase8G read only the extract
    data/emit/emit_sample.npz, which no script here builds. Each runs exactly when its input is there."""
    ran, out = _run(monkeypatch, capsys, ip_present=_IP_FILES, emit_sample=emit_sample, granules=granules)
    label = {argv.split()[0]: lab for lab, argv, _ in IC.PHASES}
    assert (label["phase8F_multi.py"] in ran) is granules
    for script in ("phase8F_emit.py", "phase8G_emit_reliability.py"):
        assert (label[script] in ran) is emit_sample, script
        if not emit_sample:
            assert re.search(rf"\[SKIP\] {re.escape(label[script])}\s+needs the EMIT extract", out), script


def test_the_static_checks_find_the_6s_and_cloudsen12_phases():
    assert _READS_6S == {"Phase3 atmosphere", "Phase5 A+B flagship"}, sorted(_READS_6S)
    assert _READS_S2 == {"Phase8 CloudSEN12", "Phase8R reliability", "Phase8D difficulty", "Phase8E DOFA"}, \
        sorted(_READS_S2)
    for line in ('root = os.path.join(P8.DATA, split)', 'X, y = P8.load_split("test", "L1C")',
                 'DATA = os.path.join(ROOT, "data", "cloudsen12")'):
        assert _S2_RE.search(line), f"a CloudSEN12 pattern no longer matches what it is for: {line}"


def test_with_no_data_at_all_only_the_data_free_phases_run(monkeypatch, capsys):
    """With every input absent, a phase none of whose inputs is guarded still runs, and fails this."""
    ran, _ = _run(monkeypatch, capsys, ip_present=(), lut=False, s2=False)
    assert ran == _DATA_FREE, sorted(ran ^ _DATA_FREE)


@pytest.mark.parametrize("lut", [False, True])
def test_6s_phases_follow_the_shipped_table(monkeypatch, capsys, lut):
    """phase3 and phase5 read the 6S table on top of the cube: with the cube present they run exactly when
    the table is there, and are skipped for it, with that reason, when it is not; every other phase, all
    its inputs present, runs either way."""
    ran, out = _run(monkeypatch, capsys, ip_present=_IP_FILES, lut=lut, s2=True, emit_sample=True, granules=True)
    assert _skipped_for("needs 6S table", out) == (set() if lut else _READS_6S)
    assert ran == (_ALL if lut else _ALL - _READS_6S), sorted(ran ^ _ALL)


@pytest.mark.parametrize("s2", [False, True])
def test_cloudsen12_phases_follow_the_dataset(monkeypatch, capsys, s2):
    """The four CloudSEN12 phases run exactly when its directory is there, and are skipped for it, with that
    reason, when it is not; every other phase, all its inputs present, runs either way."""
    ran, out = _run(monkeypatch, capsys, ip_present=_IP_FILES, s2=s2, emit_sample=True, granules=True)
    assert _skipped_for("needs CloudSEN12", out) == (set() if s2 else _READS_S2)
    assert ran == (_ALL if s2 else _ALL - _READS_S2), sorted(ran ^ _ALL)
