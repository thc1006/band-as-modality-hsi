"""phase8R2 --norm-stats: which train pixels set the standardisation statistics.

`train` (default; the paper's runs) uses the whole train draw, target-domain train pixels included;
`source` uses the source-domain train pixels only (experiments/PHASE8R2_NORM_STATS_CHECK.md). Pinned:
what each choice computes, that the default reproduces the pre-option arithmetic exactly, and that
main() actually routes the option into the standardisation.
"""
import ast
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)
import phase8R2_landcover_reliability as R2                                     # noqa: E402

_SRC = open(os.path.join(_ROOT, "experiments", "phase8R2_landcover_reliability.py"), encoding="utf-8").read()


def _draw():
    rng = np.random.default_rng(5)
    Xtr = rng.normal(size=(600, 13)) * rng.uniform(0.5, 2.0, 13) + rng.uniform(-1, 1, 13)
    is_tgt = rng.random(600) < 0.3
    Xtr[is_tgt] += 3.0                                   # the target domain sits elsewhere
    return Xtr.astype(np.float32), Xtr[~is_tgt].astype(np.float32)


def test_train_is_the_whole_draw_and_bit_identical_to_the_original_arithmetic():
    Xtr, Xtr_src = _draw()
    mu, sd = R2.standardisation_stats(Xtr, Xtr_src, "train")
    mu0 = Xtr.mean(0); sd0 = Xtr.std(0) + 1e-8          # the line the option replaced, verbatim
    assert np.array_equal(mu, mu0) and np.array_equal(sd, sd0)


def test_source_uses_only_the_source_pixels():
    Xtr, Xtr_src = _draw()
    mu, sd = R2.standardisation_stats(Xtr, Xtr_src, "source")
    assert np.array_equal(mu, Xtr_src.mean(0)) and np.array_equal(sd, Xtr_src.std(0) + 1e-8)
    mu_all, _ = R2.standardisation_stats(Xtr, Xtr_src, "train")
    assert np.abs(mu - mu_all).max() > 0.5, "the draw must make the two choices differ, or this proves nothing"


def test_an_unknown_choice_is_refused():
    Xtr, Xtr_src = _draw()
    with pytest.raises(ValueError, match="norm-stats"):
        R2.standardisation_stats(Xtr, Xtr_src, "target")


def test_main_routes_the_option_into_the_standardisation_and_defaults_to_train():
    main = next(n for n in ast.parse(_SRC).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    opt = next(c for c in ast.walk(main) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "add_argument"
               and c.args and getattr(c.args[0], "value", None) == "--norm-stats")
    kw = {k.arg: ast.literal_eval(k.value) for k in opt.keywords if k.arg in ("default", "choices")}
    assert kw == {"default": "train", "choices": ["train", "source"]}
    calls = [c for c in ast.walk(main) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "standardisation_stats"]
    assert len(calls) == 1 and [ast.unparse(a) for a in calls[0].args] == ["Xtr", "Xtr_src", "args.norm_stats"]
    # and nothing else in main() computes per-band statistics of the train pixels behind its back
    stray = [ast.unparse(c) for c in ast.walk(main) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", "") in ("mean", "std") and ast.unparse(c.func.value) in ("Xtr", "Xtr_src")]
    assert not stray, stray
