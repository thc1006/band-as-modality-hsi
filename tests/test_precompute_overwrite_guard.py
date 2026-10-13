"""The 6S tables ship with the repository (Indian Pines: the one the paper's phase3/5/9 runs used, by the
sha256 their sidecars record). Regenerating one in place would move every later run off it -- and the
embedded generated_utc changes the file's hash even when the values agree -- so both precompute scripts
refuse to overwrite an existing table without --force, before computing anything. Both directions are
pinned: a new --out must still be written (the documented audit command), and --force must overwrite.
6S itself is stubbed out; only the scripts' own handling of the output path is under test."""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)

CASES = [("precompute_6s_table", []), ("precompute_6s_dataset", ["--dataset", "salinas"])]


def _main(monkeypatch, module, argv, compute=True):
    mod = __import__(module)
    stub = ((lambda wl, cwv, aod, **k: np.full(np.asarray(wl).shape, 0.5)) if compute
            else (lambda *a, **k: pytest.fail("computed before refusing")))
    monkeypatch.setattr(mod, "run_6s_transmittance", stub)
    monkeypatch.setattr(sys, "argv", [f"{module}.py", *argv])
    return mod.main()


@pytest.mark.parametrize("module,argv", CASES)
def test_an_existing_table_is_not_overwritten_without_force(monkeypatch, tmp_path, module, argv):
    target = tmp_path / "T_existing.npz"
    target.write_bytes(b"shipped")
    for out in (str(target), str(target)[:-len(".npz")]):     # numpy appends .npz to a bare name
        with pytest.raises(SystemExit, match="exists"):
            _main(monkeypatch, module, argv + ["--out", out], compute=False)
    assert target.read_bytes() == b"shipped"
    assert sorted(os.listdir(tmp_path)) == ["T_existing.npz"]


@pytest.mark.parametrize("module,argv", CASES)
def test_a_new_output_is_written_and_force_overwrites(monkeypatch, tmp_path, module, argv):
    new = tmp_path / "T_regen.npz"
    _main(monkeypatch, module, argv + ["--out", str(new)])
    with np.load(new) as z:
        assert {"cwv0.5", "cwv2.0", "cwv4.0", "wl_nm"} <= set(z.files)
    old = tmp_path / "T_old.npz"
    old.write_bytes(b"shipped")
    _main(monkeypatch, module, argv + ["--out", str(old), "--force"])
    with np.load(old) as z:
        assert "cwv4.0" in z.files


def test_the_indian_pines_default_is_the_shipped_table():
    import precompute_6s_table as PT
    assert PT.OUT == os.path.join(_ROOT, "data", "srf_cache", "T_6s_grid.npz")
