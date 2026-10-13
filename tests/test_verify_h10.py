"""scripts/verify_h10.py is the tool behind experiments/H10_TRAIN_ONCE_VERIFICATION.md, so its own checks
must not pass vacuously: ab-compare has to refuse to compare a run with itself or two runs of the same
side or of different scripts, find a single differing field (NaN equal to NaN, a type change a
difference), and exit non-zero when it compared nothing; every count argument has to be positive."""
import importlib.util
import json
import os
import pickle
import shutil
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def vh():
    spec = importlib.util.spec_from_file_location("_verify_h10_under_test",
                                                  os.path.join(_ROOT, "scripts", "verify_h10.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ROWS = [{"method": "b2", "state": "clean", "arm": "naive", "acc": 81.25, "temperature": 1.0625,
          "plugin_cond_risk": float("nan"), "crc_feasible": True}]


def _run(tmp_path, name, mode, rows=_ROWS, script="phase8R"):
    path = tmp_path / name
    run = {"mode": mode, "script": script, "repo": "/somewhere", "time_utc": f"t-{name}",
           "items": [(0, 3)], "results": [[dict(r) for r in rows]]}
    if mode is None:
        del run["mode"]
    with open(path, "wb") as f:
        pickle.dump(run, f)
    return str(path)


def _compare(vh, monkeypatch, a, b):
    monkeypatch.setattr(sys, "argv", ["verify_h10.py", "ab-compare", a, b])
    return vh.main()


def test_equal_runs_pass_and_one_differing_field_fails(vh, monkeypatch, tmp_path, capsys):
    old = _run(tmp_path, "old.pkl", "old")
    assert _compare(vh, monkeypatch, old, _run(tmp_path, "new.pkl", "new")) == 0
    for field, value in (("temperature", 1.0625 + 2 ** -40), ("crc_feasible", 1)):   # a value, a type
        changed = [dict(_ROWS[0], **{field: value})]
        assert _compare(vh, monkeypatch, old, _run(tmp_path, f"new_{field}.pkl", "new", changed)) == 1, field
    assert '"differing_at_full_precision": 1' in capsys.readouterr().out


def test_comparing_nothing_fails(vh, monkeypatch, tmp_path):
    assert _compare(vh, monkeypatch, _run(tmp_path, "old.pkl", "old", rows=[]),
                    _run(tmp_path, "new.pkl", "new", rows=[])) == 1


def test_a_run_is_never_compared_with_itself(vh, monkeypatch, tmp_path):
    old = _run(tmp_path, "old.pkl", "old")
    shutil.copy(old, tmp_path / "copy.pkl")
    for b in (old, str(tmp_path / "copy.pkl")):                           # the same file, the same bytes
        with pytest.raises(SystemExit, match="same run"):
            _compare(vh, monkeypatch, old, b)


def test_one_old_and_one_new_run_of_the_same_script(vh, monkeypatch, tmp_path):
    for side in ("new", None):                                            # two of one side; one unlabelled
        with pytest.raises(SystemExit, match="one `old` run with one `new` run"):
            _compare(vh, monkeypatch, _run(tmp_path, "a.pkl", side), _run(tmp_path, "b.pkl", "new"))
    with pytest.raises(SystemExit, match="different scripts"):
        _compare(vh, monkeypatch, _run(tmp_path, "old.pkl", "old", script="phase8R2"),
                 _run(tmp_path, "new.pkl", "new"))
    # a pickle from before ab-run recorded its script (pass 1's old run) still compares
    assert _compare(vh, monkeypatch, _run(tmp_path, "old0.pkl", "old", script=None),
                    _run(tmp_path, "new0.pkl", "new")) == 0


_COUNTS = ([[cmd, "--synthetic", "0"] for cmd in ("train-hash", "rows", "chunking")]
           + [[cmd, "--synthetic", "9", flag, "0"] for cmd in ("train-hash", "rows", "chunking")
              for flag in ("--epochs", "--threads")]
           + [["train-hash", "--synthetic", "9", "--repeat", "0"],
              ["chunking", "--synthetic", "9", "--chunks", "0"],
              ["cpu-repeat", "--threads", "0"], ["cpu-repeat", "--repeat", "-1"],
              ["cpu-thread-logits", "--thread-counts", "1", "0"], ["cpu-thread-logits", "--rows", "0"]])


@pytest.mark.parametrize("argv", _COUNTS, ids=[" ".join(a) for a in _COUNTS])
def test_count_arguments_must_be_positive(vh, monkeypatch, argv):
    """All fifteen count arguments, each given a non-positive value."""
    assert len(_COUNTS) == 15
    monkeypatch.setattr(sys, "argv", ["verify_h10.py", *argv])
    with pytest.raises(SystemExit) as e:
        vh.main()
    assert e.value.code == 2


def test_train_hash_compares_only_its_in_process_repeats(vh, monkeypatch, capsys):
    """One repeat compares nothing (repeats_identical: null, exit 0 -- the hashes are for comparing
    across processes); two equal trainings say true and exit 0; two different ones say false and exit 1.
    Training is stubbed: what is pinned is train-hash's own bookkeeping."""
    import torch
    monkeypatch.syspath_prepend(os.path.join(_ROOT, "experiments"))
    import phase8R_reliability as P8R
    calls = []

    def train_models(*a, **k):
        calls.append(1)
        with torch.random.fork_rng(devices=[]):        # Linear's init draws from torch's global generator
            m = torch.nn.Linear(2, 2)
        with torch.no_grad():
            m.weight.fill_(drift[0] * len(calls))
            m.bias.zero_()
        return {"proposed": m}
    monkeypatch.setattr(P8R, "train_models", train_models)
    monkeypatch.setattr(vh, "_setup", lambda threads: ([], [], {"device": "stub"}))
    monkeypatch.setattr(vh, "_inputs", lambda inputs, synthetic: (None,) * 5 + ("stub",))
    for drift, repeat, identical, rc in (([0.0], "1", None, 0), ([0.0], "2", True, 0), ([1.0], "2", False, 1)):
        calls.clear()
        monkeypatch.setattr(sys, "argv",
                            ["verify_h10.py", "train-hash", "--synthetic", "9", "--repeat", repeat])
        assert vh.main() == rc
        out = json.loads(capsys.readouterr().out)
        assert out["repeats_identical"] is identical and out["repeats_in_process"] == int(repeat)
