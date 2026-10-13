"""The integrity harness must SKIP, not fail, a phase whose input data are absent -- and must still RUN
every phase whose data are there.

reproduce.sh runs experiments/integrity_check.py under `set -e` before any tier. A phase that crashed
for want of its data therefore stopped the whole reproduction -- including the tiers whose data WERE
present (a clone holding only CloudSEN12 never reached the flagship). The harness already skipped the
6S and CloudSEN12 phases; this pins the same for the Indian Pines cube and for the EMIT inputs, in both
directions: a guard that skipped a phase whose data are present would make the gate pass vacuously.
The expected skip set is derived from what each invocation reads (a phase that loads the cube by file
name, through phase2's load_data or through phase1's load_indian_pines), so a new phase of that kind
cannot be added unguarded. `run_phase` is stubbed and the existence checks and glob are faked for
exactly the paths the harness should look at -- a guard pointed anywhere else sees the real filesystem
and the tests notice; a `subprocess.run` launch fails the test, so an edit that bypassed the stub could
not start the real phases from a unit test.
"""
import glob
import os
import re
import sys

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


def _reads_indian_pines(argv):
    """Static: does this invocation load the Indian Pines cube (by file name, or via phase2's load_data or
    phase1's load_indian_pines)? A --dataset option decides it outright (phase6 runs `--dataset synthetic`)."""
    script, *rest = argv.split()
    if "--dataset" in rest:
        return rest[rest.index("--dataset") + 1] == "indian_pines"
    src = open(os.path.join(_EXP, script), encoding="utf-8").read()
    return bool(re.search(r"Indian_pines_corrected|\bload_data\(|\bload_indian_pines\(", src))


_READS_IP = {label for label, argv, _ in IC.PHASES if _reads_indian_pines(argv)}


def _run(monkeypatch, capsys, ip_present=(), emit_sample=False, granules=False, lut=True):
    """IC.main() with run_phase stubbed; of the exact paths the harness should check, the Indian Pines
    files in `ip_present` exist, the 6S table if `lut`, the EMIT extract and granules as given; every other
    path is the real filesystem's. Returns (ran labels, out)."""
    ran = []
    monkeypatch.setattr(IC, "run_phase", lambda label, argv, outputs: (ran.append(label), (True, "stub"))[1])
    monkeypatch.setattr(IC.subprocess, "run",
                        lambda *a, **k: pytest.fail("the harness tried to launch a phase for real"))
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
    granule = os.path.join(_ROOT, "data", "emit_x", "EMIT_L2A_RFL_001_x.nc")
    monkeypatch.setattr(glob, "glob", lambda pat, *a, **k: [granule] if (granules and pat == _EMIT_GLOB) else [])
    with pytest.raises(SystemExit) as stop:
        IC.main()
    assert stop.value.code == 0
    return set(ran), capsys.readouterr().out


def _skipped_for_ip(out):
    return {label for label in (lab for lab, _, _ in IC.PHASES)
            if re.search(rf"\[SKIP\] {re.escape(label)}\s+needs Indian Pines", out)}


def test_the_static_check_finds_the_hsi_phases():
    assert len(_READS_IP) >= 9, f"found only {sorted(_READS_IP)} reading Indian Pines -- the check has drifted"
    assert "Phase6 synthetic" not in _READS_IP and "Phase0 synthetic PoC" not in _READS_IP


@pytest.mark.parametrize("present", [(), ("Indian_pines_corrected.mat",), ("Indian_pines_gt.mat",)])
def test_hsi_phases_are_skipped_without_the_whole_cube(monkeypatch, capsys, present):
    """Absent, or half present (load_data reads both files): skipped, never run."""
    ran, out = _run(monkeypatch, capsys, ip_present=present)
    assert not _READS_IP & ran, f"ran without the cube: {sorted(_READS_IP & ran)}"
    assert _skipped_for_ip(out) == _READS_IP, "exactly the phases that read the cube are skipped for it"
    assert {"Phase0 synthetic PoC", "Phase6 synthetic"} <= ran


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
