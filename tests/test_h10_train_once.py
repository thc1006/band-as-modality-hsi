"""H-10 (train once per model seed) and the P100 memory fixes, pinned by EXACT-equality tests.

H-10 is only a refactor if the two-stage campaign (train_job once per model seed -> eval_job once per
(split, model) cell) reproduces the per-cell path (run_seed: train + evaluate in one call) EXACTLY.
These tests compare every field of every row with ==, no tolerance, on small synthetic problems on the
CPU -- in-process, and through the real process pools main() uses (spawned workers, a low open-files
limit). They also pin run_two_stage's wiring (one thread budget for both stages, seed -> model
association, result order, the per-stage VRAM budget, the plans it records), what both main()s hand
to it (on synthetic data), the device moves in train_job / eval_job, --alpha reaching every CRC call of
phase8R2, the planner's per-worker VRAM budget, and the job-boundary CUDA-cache release (success and
error paths, pooled and serial, only where CUDA is initialised).

WHY ONE THREAD WHEREVER ROWS ARE COMPARED WITH ==. At two intra-op threads per worker (the only count
above one measured), CPU results were not bitwise repeatable even at a fixed thread count: 14 of 160
repetitions of a 6-cell campaign differed (0 to 7 per batch) -- in b2 rows wherever the fields were
recorded: temperatures, CRC thresholds and, once on real pixels, AURC/AUROC at temperature 1, i.e. the
logits (measured 2026-10-13; mechanism not isolated, oneMKL's CNR mode did not remove it -- see
bandsim.parallel, "Determinism"). Single-threaded, the same comparison was exact in 120 of 120
repetitions, so exact (==) comparisons are made at one thread -- and they are what can catch a
refactor that changes WHAT is computed.
"""
import json
import math
import os
import pickle
import sys
import types

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(_HERE, ".."))

# Process-wide knobs that would change the plan under test (e.g. BANDSIM_WORKERS=1 turns every pool
# into a serial run in the pytest process). Cleared for the tests that plan or spawn.
_PLAN_ENV = ("BANDSIM_WORKERS", "BANDSIM_THREADS", "BANDSIM_GPU_OVERSUB", "BANDSIM_PER_WORKER_GB",
             "BANDSIM_RESERVE_CORES")


@pytest.fixture
def cpu(monkeypatch):
    """CPU device (hw.device() reads BANDSIM_DEVICE on every call; spawned workers inherit it), no plan
    overrides from the environment, and ONE intra-op thread in this process for the test's duration --
    see the module docstring."""
    monkeypatch.setenv("BANDSIM_DEVICE", "cpu")
    for k in _PLAN_ENV:
        monkeypatch.delenv(k, raising=False)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(autouse=True)
def _restore_global_rngs():
    """train_models reseeds the process-global RNGs (hw.seed_model); hand the CPU generators (random,
    numpy, torch) back as they were, so no later test runs on a state these tests left behind. The CUDA
    generators, which torch.manual_seed also seeds when CUDA is available, are not restored: restoring
    them would initialise CUDA, and these tests draw nothing from them."""
    import random
    saved = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    yield
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    torch.random.set_rng_state(saved[2])


@pytest.fixture
def job_modules(monkeypatch):
    """parallel._import_by_path caches the job modules it loads (in _MODULE_CACHE and sys.modules); give
    the test a private cache and drop whatever it imported afterwards."""
    from bandsim import parallel
    monkeypatch.setattr(parallel, "_MODULE_CACHE", {})
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("_bandsim_job_"):
            del sys.modules[name]


def _problem(seed=0, n_tr=1200, units=24, per_unit=100):
    import phase8_cloudsen12 as P8
    from bandsim.grouping import group_center_wavelengths
    rng = np.random.default_rng(seed)
    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    W = rng.standard_normal((13, 4))

    def draw(n):
        X = rng.standard_normal((n, 13)).astype(np.float32)
        y = (X @ W + 0.8 * rng.standard_normal((n, 4))).argmax(1)
        return X, y
    Xtr, ytr = draw(n_tr)
    X1, yte = draw(units * per_unit)
    X2 = (X1 * 1.15 + 0.3).astype(np.float32)          # a crude "L2A-like" product shift
    unit = np.repeat(np.arange(units), per_unit)
    return dict(Xtr=Xtr, ytr=ytr, Xte={"l1c": X1, "l2a": X2}, yte=yte, unit=unit,
                groups=groups, cwl=cwl)


def _assert_rows_identical(a, b):
    assert len(a) == len(b) and len(a) > 0
    for ra, rb in zip(a, b):
        assert ra.keys() == rb.keys()
        for k in ra:
            va, vb = ra[k], rb[k]
            if isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb):
                continue
            assert va == vb, f"field {k!r}: {va!r} != {vb!r}"


def _assert_cells_identical(per_cell, two_stage):
    assert len(per_cell) == len(two_stage) and len(per_cell) > 0
    for a, b in zip(per_cell, two_stage):
        _assert_rows_identical(a, b)


def _square(x):                                        # module level: a run_jobs target
    return x * x


# ------------------------------------------------------------------------- in-process equivalence
def test_phase8R_two_stage_equals_per_cell(cpu):
    import phase8R_reliability as P8R
    p = _problem()
    cells = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    per_cell = [P8R.run_seed(job, p["Xtr"], p["ytr"], p["Xte"], p["yte"], p["unit"], p["groups"],
                             p["cwl"], 0.8, 1, target=0.1) for job in cells]
    trained = {ms: P8R.train_job(ms, p["Xtr"], p["ytr"], p["groups"], p["cwl"], 0.8, 1) for ms in (0, 1)}
    for blob in trained.values():                     # stage-1 output: immutable bytes of CPU, eval-mode models
        assert isinstance(blob, bytes)
        for m in pickle.loads(blob).values():
            assert not m.training
            assert all(t.device.type == "cpu" for t in m.state_dict().values())
    before = dict(trained)
    two_stage = [P8R.eval_job(job, trained, p["Xte"], p["yte"], p["unit"], p["groups"], target=0.1)
                 for job in cells]
    _assert_cells_identical(per_cell, two_stage)
    assert trained == before                          # no evaluation may swap a seed's model


def test_phase8R2_two_stage_equals_per_cell(cpu):
    import phase8R_reliability as P8R
    import phase8R2_landcover_reliability as R2
    p = _problem(seed=1)
    is_target = p["unit"] < 12                         # 12 target units, 12 source units
    X, y = p["Xte"]["l1c"], p["yte"]
    cells = [(0, 0), (1, 0), (0, 1)]
    per_cell = [R2.run_seed(job, p["Xtr"], p["ytr"], X, y, p["unit"], is_target, p["groups"],
                            p["cwl"], 0.8, 1, 0.1) for job in cells]
    trained = {ms: P8R.train_job(ms, p["Xtr"], p["ytr"], p["groups"], p["cwl"], 0.8, 1) for ms in (0, 1)}
    two_stage = [R2.eval_job(job, trained, X, y, p["unit"], is_target, p["groups"], 0.1)
                 for job in cells]
    _assert_cells_identical(per_cell, two_stage)


def test_phase8R2_target_reaches_every_crc_call(cpu, monkeypatch):
    """phase8R2's half of the P0-1 guard: whatever target enters R2.run_seed or R2.eval_job is the alpha
    of EVERY conformal call of its domain-gap arms (a hard-coded alpha=0.10 there used to pass the suite)."""
    import phase8R_reliability as P8R
    import phase8R2_landcover_reliability as R2
    seen = []
    real = R2.conformal_risk_control

    def crc(*a, **k):
        seen.append(k.get("alpha"))
        return real(*a, **k)
    monkeypatch.setattr(R2, "conformal_risk_control", crc)
    p = _problem(seed=2)
    is_target = p["unit"] < 12
    X, y = p["Xte"]["l1c"], p["yte"]
    target = 0.37                                      # anything but the old 0.10 default
    R2.run_seed((0, 0), p["Xtr"], p["ytr"], X, y, p["unit"], is_target, p["groups"], p["cwl"], 0.8, 1, target)
    blob = P8R.train_job(0, p["Xtr"], p["ytr"], p["groups"], p["cwl"], 0.8, 1)
    R2.eval_job((1, 0), {0: blob}, X, y, p["unit"], is_target, p["groups"], target)
    assert seen and all(v == target for v in seen), f"alphas seen: {sorted(set(map(str, seen)))}"


class _Stop(Exception):
    """Raised by the run_two_stage spy: by then main() has decided everything these tests pin."""


def _fake_loader(calls):
    """A synthetic stand-in for P8.load_split covering the parts of its contract the two main()s use: the
    SAME pixels for L1C and L2A of a split with the same seed (L2A with B10 zeroed, as the real loader
    leaves it), pixels_per_patch rows per patch, and patch ids from patch_ids, else n_patches of them
    (range(n_patches); the real loader samples them), else every patch of the split. Records each call."""
    import pandas as pd
    import phase8_cloudsen12 as P8

    def load_split(split, product="L1C", pixels_per_patch=400, n_patches=None, seed=0, patch_ids=None,
                   return_patch_id=False, return_roi_id=False, return_pixel_index=False):
        if return_roi_id or return_pixel_index:
            raise NotImplementedError("the synthetic loader does not model roi_id / pixel_index")
        if patch_ids is not None:
            ids = np.asarray(patch_ids)
        elif n_patches is not None:
            ids = np.arange(n_patches)
        else:
            ids = np.arange(len(pd.read_csv(os.path.join(P8.DATA, split, "metadata.csv"))))
        pid = np.repeat(ids, pixels_per_patch)
        rng = np.random.default_rng([seed, pid.size])
        X = rng.standard_normal((pid.size, 13)).astype(np.float32)
        y = (np.abs(X[:, 0]) * 3).astype(np.int64) % 4
        if product == "L2A":
            X = (X * 1.1 + 0.2).astype(np.float32)
            X[:, P8.B10_IDX] = 0.0
        calls.append(dict(split=split, product=product, X=X, y=y, pid=pid, seed=seed))
        return (X, y, pid) if return_patch_id else (X, y)
    return load_split


def _run_main_until_two_stage(monkeypatch, tmp_path, module, argv):
    """Run module.main() on the synthetic loader up to its run_two_stage call; return what main passed
    there and the loader calls. Needs the CloudSEN12 metadata (patch -> ROI / scene / land cover)."""
    import phase8_cloudsen12 as P8
    import phase8R_reliability as P8R
    from bandsim import hw
    if not os.path.exists(os.path.join(P8.DATA, "test", "metadata.csv")):
        pytest.skip("needs the CloudSEN12 metadata")
    calls, seen = [], {}
    monkeypatch.setattr(P8, "load_split", _fake_loader(calls))
    monkeypatch.setattr(hw, "setup", lambda *a, **k: None)          # main's process setup is not under test
    monkeypatch.setattr(P8R, "PAPER_DIR", str(tmp_path))

    def spy(eval_fn, jobs, **kw):
        seen.update(kw, eval_fn=eval_fn, jobs=list(jobs))
        raise _Stop
    monkeypatch.setattr(P8R, "run_two_stage", spy)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_Stop):
        module.main()
    return seen, calls


def _normalised(Xtr, *others):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    return [((A - mu) / sd).astype(np.float32) for A in (Xtr, *others)]


def _assert_groups_and_wavelengths(train_shared, eval_shared):
    import phase8_cloudsen12 as P8
    from bandsim.grouping import group_center_wavelengths
    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    for shared in (train_shared, eval_shared):
        assert len(shared["groups"]) == len(groups)
        assert all(np.array_equal(a, b) for a, b in zip(shared["groups"], groups))
    assert np.array_equal(train_shared["cwl"], cwl)


@pytest.mark.parametrize("nondeterministic", [False, True])
def test_phase8R_main_hands_the_two_stage_driver_the_right_arguments(cpu, monkeypatch, tmp_path,
                                                                     nondeterministic):
    """Behavioural, on synthetic pixels: main() trains on the normalised TRAIN pixels, evaluates both
    products with the TRAIN normalisation, groups test pixels by SCENE COMPONENT (P0-2), hands both stages
    the spectral groups (and training their centre wavelengths), and passes --alpha, --jobs, --device and
    --nondeterministic through -- and the flagship gets no special VRAM budget."""
    import phase8R_reliability as P8R
    seen, calls = _run_main_until_two_stage(monkeypatch, tmp_path, P8R, [
        "phase8R_reliability.py", "--split-seeds", "0", "1", "--model-seeds", "3", "5",
        "--patches-train", "40", "--px-train", "2", "--patches-test", "60", "--px-test", "2",
        "--alpha", "0.2", "--jobs", "3", "--device", "cpu", "--out-tag", "_pytest"]
        + (["--nondeterministic"] if nondeterministic else []))
    tr_call = next(c for c in calls if c["split"] == "train")
    l1c = next(c for c in calls if c["split"] == "test" and c["product"] == "L1C")
    l2a = next(c for c in calls if c["split"] == "test" and c["product"] == "L2A")
    Xtr, Xl1c, Xl2a = _normalised(tr_call["X"], l1c["X"], l2a["X"])
    assert seen["eval_fn"] is P8R.eval_job and seen["jobs"] == [(0, 3), (0, 5), (1, 3), (1, 5)]
    tr, ev = seen["train_shared"], seen["eval_shared"]
    assert np.array_equal(tr["Xtr"], Xtr) and np.array_equal(tr["ytr"], tr_call["y"])
    assert tr["subsample_frac"] == 0.8 and tr["epochs"] == 40
    assert np.array_equal(ev["Xte"]["l1c"], Xl1c) and np.array_equal(ev["Xte"]["l2a"], Xl2a)
    assert np.array_equal(ev["yte"], l1c["y"])
    assert np.array_equal(ev["rid_te"], P8R.scene_component_ids("test")[l1c["pid"]])
    assert ev["target"] == 0.2
    _assert_groups_and_wavelengths(tr, ev)
    assert seen["n_jobs"] == 3 and seen["prefer"] == "cpu"
    assert seen["deterministic"] is (not nondeterministic)
    assert seen.get("eval_per_worker_gb") is None


@pytest.mark.parametrize("extra,expected", [([], 54321), (["--test-pixel-seed", "7"], 7)])
def test_phase8R_main_draws_the_test_pixels_with_the_test_pixel_seed(cpu, monkeypatch, tmp_path, extra, expected):
    """--test-pixel-seed (default 54321, the paper's draw) reaches every test-split read, L1C and L2A alike
    (they must stay the same pixels), and only those: the train draw keeps its own seed."""
    import phase8R_reliability as P8R
    seen, calls = _run_main_until_two_stage(monkeypatch, tmp_path, P8R, [
        "phase8R_reliability.py", "--split-seeds", "0", "1", "--model-seeds", "3",
        "--patches-train", "40", "--px-train", "2", "--patches-test", "60", "--px-test", "2",
        "--device", "cpu", "--out-tag", "_pytest"] + extra)
    test_calls = [c for c in calls if c["split"] == "test"]
    assert {c["product"] for c in test_calls} == {"L1C", "L2A"}
    assert all(c["seed"] == expected for c in test_calls)
    train_calls = [c for c in calls if c["split"] == "train"]
    assert train_calls and all(c["seed"] == 12345 for c in train_calls)


def test_phase8R_stamps_the_test_pixel_seed_it_used():
    """The provenance estimand names the pixel-sampling seed; it must be the one the run used, not a
    hard-coded 54321 that a --test-pixel-seed run would then misreport."""
    import ast
    src = open(os.path.join(_HERE, "..", "experiments", "phase8R_reliability.py"), encoding="utf-8").read()
    parts = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.JoinedStr) and "shared across all runs" in ast.unparse(n)]
    assert len(parts) == 1 and "args.test_pixel_seed" in ast.unparse(parts[0])
    assert "54321 shared across all runs" not in src


@pytest.mark.parametrize("norm_stats", [None, "source"])
def test_phase8R2_main_hands_the_two_stage_driver_the_right_arguments(cpu, monkeypatch, tmp_path, norm_stats):
    """Behavioural, on synthetic pixels: phase8R2's main() trains on the SOURCE-domain train pixels ONLY
    (normalised by default, as it always was, with the statistics of the whole train draw; with
    --norm-stats source, with those of the source-domain train pixels), evaluates every leak-guarded test
    pixel with its scene component and target-domain flag, and gives the evaluation stage --alpha and its
    measured VRAM budget."""
    import pandas as pd
    import phase8_cloudsen12 as P8
    import phase8R2_landcover_reliability as R2
    seen, calls = _run_main_until_two_stage(monkeypatch, tmp_path, R2, [
        "phase8R2_landcover_reliability.py", "--split-seeds", "0", "1", "--model-seeds", "3", "5",
        "--patches-train", "40", "--px-train", "2", "--px-test", "2", "--alpha", "0.2",
        "--jobs", "3", "--device", "cpu", "--axis", "surface", "--out-tag", "_pytest"]
        + (["--norm-stats", norm_stats] if norm_stats else []))
    tr_call = next(c for c in calls if c["split"] == "train")
    te_call = next(c for c in calls if c["split"] == "test")
    src = ~R2._target_mask("surface", "train", tr_call["pid"])
    train_products = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    keep = ~np.isin(te_call["pid"], meta.index[meta["s2_id"].isin(train_products)].to_numpy())
    pid = te_call["pid"][keep]
    stats_px = tr_call["X"][src] if norm_stats == "source" else tr_call["X"]
    mu = stats_px.mean(0); sd = stats_px.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    assert 0 < src.sum() < src.size, "the synthetic train draw must hold both domains"
    assert seen["eval_fn"] is R2.eval_job and seen["jobs"] == [(0, 3), (0, 5), (1, 3), (1, 5)]
    tr, ev = seen["train_shared"], seen["eval_shared"]
    assert np.array_equal(tr["Xtr"], norm(tr_call["X"][src])), "phase8R2 must train on the source domain only"
    assert np.array_equal(tr["ytr"], tr_call["y"][src])
    assert np.array_equal(ev["X"], norm(te_call["X"][keep])) and np.array_equal(ev["y"], te_call["y"][keep])
    assert np.array_equal(ev["unit"], P8.scene_component_ids("test")[pid])
    assert np.array_equal(ev["is_target"], R2._target_mask("surface", "test", pid))
    assert ev["target"] == 0.2
    _assert_groups_and_wavelengths(tr, ev)
    assert seen["eval_per_worker_gb"] == R2.EVAL_VRAM_GB
    assert seen["n_jobs"] == 3 and seen["prefer"] == "cpu" and seen["deterministic"] is True


@pytest.mark.parametrize("module_name", ["phase8R_reliability", "phase8R2_landcover_reliability"])
@pytest.mark.parametrize("flag", ["--split-seeds", "--model-seeds"])
def test_main_rejects_duplicate_seeds_before_touching_any_data(monkeypatch, tmp_path, module_name, flag):
    """A repeated seed is the SAME draw; counted twice it inflates n in the two-way SE (and the two-stage
    driver would train it once but evaluate its cells twice). Both mains refuse it up front.

    Hermetic, because this test runs the real main(): should the check ever go, main() must stop at the
    data loader rather than run a campaign. Without these stubs a mutant that dropped the check ran a
    real 4-cell GPU campaign and wrote results_phase8R2_landcover*.csv into its copy's paper/ -- the
    names of committed deliverables in the repository itself."""
    import importlib
    import phase8_cloudsen12 as P8
    import phase8R_reliability as P8R
    from bandsim import hw
    mod = importlib.import_module(module_name)

    def touched(*a, **k):
        raise AssertionError("main() went past its seed checks and touched the data or the run")
    monkeypatch.setattr(hw, "setup", lambda *a, **k: None)
    monkeypatch.setattr(P8, "load_split", touched)
    monkeypatch.setattr(P8R, "run_two_stage", touched)
    monkeypatch.setattr(P8R, "PAPER_DIR", str(tmp_path))
    other = "--model-seeds" if flag == "--split-seeds" else "--split-seeds"
    monkeypatch.setattr(sys, "argv", [f"{module_name}.py", flag, "4", "4", other, "0", "1",
                                      "--out-tag", "_pytest"])
    with pytest.raises(ValueError, match="has duplicates"):
        mod.main()


def _run_main_end_to_end(monkeypatch, tmp_path, module, argv):
    """Run module.main() to the end on the synthetic loader (tiny, serial, CPU) with the REAL run_two_stage,
    wrapped to keep the compute record it returned; return that record and the provenance sidecars main()
    wrote (into tmp_path, its PAPER_DIR here)."""
    import phase8_cloudsen12 as P8
    import phase8R_reliability as P8R
    from bandsim import hw
    if not os.path.exists(os.path.join(P8.DATA, "test", "metadata.csv")):
        pytest.skip("needs the CloudSEN12 metadata")
    os.makedirs(tmp_path / "figs")
    monkeypatch.setattr(P8, "load_split", _fake_loader([]))
    monkeypatch.setattr(hw, "setup", lambda *a, **k: None)          # main's process setup is not under test
    monkeypatch.setattr(P8R, "PAPER_DIR", str(tmp_path))
    seen, real = {}, P8R.run_two_stage

    def keep(*a, **k):
        results, compute = real(*a, **k)
        seen["compute"] = compute
        return results, compute
    monkeypatch.setattr(P8R, "run_two_stage", keep)
    monkeypatch.setattr(sys, "argv", argv)
    import matplotlib
    with matplotlib.rc_context():                     # main() restyles the figures it draws
        module.main()
    sidecars = {q.name: json.loads(q.read_text(encoding="utf-8")) for q in tmp_path.glob("*.provenance.json")}
    return seen["compute"], sidecars


@pytest.mark.parametrize("module_name, results", [
    ("phase8R_reliability", ("results_phase8R_reliability_pytest.csv", "results_phase8R_raw_pytest.csv")),
    ("phase8R2_landcover_reliability", ("results_phase8R2_landcover_pytest.csv",
                                        "results_phase8R2_landcover_raw_pytest.csv"))])
def test_main_stamps_the_compute_it_ran_and_the_train_protocol(cpu, monkeypatch, tmp_path, module_name,
                                                               results):
    """End to end on synthetic pixels (one epoch, serial): every result file's provenance sidecar carries
    the `compute` record run_two_stage returned -- the plans that ran -- and the train_protocol saying how
    the models were trained. Nothing else reads them back, so a main() that dropped either would pass
    every other test."""
    import importlib
    module = importlib.import_module(module_name)
    argv = [f"{module_name}.py", "--split-seeds", "0", "1", "--model-seeds", "3", "--patches-train", "40",
            "--px-train", "2", "--px-test", "2", "--epochs", "1", "--jobs", "1", "--device", "cpu",
            "--out-tag", "_pytest"] + (["--patches-test", "60"] if module_name == "phase8R_reliability" else [])
    compute, sidecars = _run_main_end_to_end(monkeypatch, tmp_path, module, argv)
    assert sorted(sidecars) == sorted(f"{r}.provenance.json" for r in results)
    expected = json.loads(json.dumps(compute, default=str))
    for name, side in sidecars.items():
        assert side["extra"]["compute"] == expected, name
        assert side["extra"]["train_protocol"].startswith("H-10 two-stage"), name
    assert compute["threads_per_worker"] == compute["train_stage"]["threads_per_worker"] \
        == compute["eval_stage"]["threads_per_worker"]


# ------------------------------------------------------------------------- run_two_stage wiring
def test_run_two_stage_wiring(monkeypatch):
    """Without spawning anything: both stages get ONE thread budget -- the per-cell protocol's for this
    campaign (plan_jobs over the CELLS) -- and the caller's worker count, device and determinism flag;
    the training stage gets each model seed once, in first-appearance order; every evaluation job gets
    all seeds' bytes plus eval_shared; the evaluation stage alone gets the VRAM budget; results come back
    in `jobs` order; `compute` records the plans the two run_jobs calls REPORTED (not a re-plan). The core
    count is pinned so the check means the same on every host: there, left to plan_jobs, the training
    stage (2 jobs) would get 3 threads per worker against the cells' 2. threads=0 means "not given", so
    it gets the derived budget too (not one per stage); a negative count is refused."""
    import phase8R_reliability as P8R
    from bandsim import hw, parallel
    monkeypatch.setenv("BANDSIM_DEVICE", "cpu")
    for k in _PLAN_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(hw, "available_cores", lambda: 8)
    calls = []
    marker = {P8R.train_job: parallel.Plan(77, "cuda", 4, 2, 9, budget_gb=8.0),
              P8R.eval_job: parallel.Plan(88, "cuda", 4, 1, 9, budget_gb=11.0)}

    def fake_run_jobs(target, items, shared=None, prefer=None, jobs=None, threads=None,
                      deterministic=True, log=print, label="job", per_worker_gb=None, on_plan=None):
        calls.append(dict(target=target, items=list(items), shared=shared, threads=threads, jobs=jobs,
                          prefer=prefer, deterministic=deterministic, label=label,
                          per_worker_gb=per_worker_gb))
        on_plan(marker[target])                       # a plan no re-plan could reproduce
        if target is P8R.train_job:
            return [f"blob{s}".encode() for s in items]
        return [(job, shared["trained"][job[1]], shared["extra"], sorted(shared["trained"])) for job in items]
    monkeypatch.setattr(parallel, "run_jobs", fake_run_jobs)

    cells = [(ss, ms) for ss in (0, 1, 2) for ms in (5, 3)]
    out, compute = P8R.run_two_stage(P8R.eval_job, cells, train_shared={"k": 1}, eval_shared={"extra": 7},
                                     prefer="cpu", n_jobs=3, deterministic=False, label="t", log=None,
                                     eval_per_worker_gb=11.0)
    want = parallel.plan_jobs(len(cells), prefer="cpu", jobs=3).threads_per_worker
    assert want != parallel.plan_jobs(2, prefer="cpu", jobs=3).threads_per_worker, \
        "on the pinned 8 cores the unpinned training plan must differ, or this test proves nothing"
    train, ev = calls
    assert train["target"] is P8R.train_job and train["items"] == [5, 3] and train["shared"] == {"k": 1}
    assert ev["target"] is P8R.eval_job and ev["items"] == cells
    assert train["threads"] == ev["threads"] == want
    assert compute["threads_per_worker"] == marker[P8R.eval_job].threads_per_worker   # what the plan ran
    assert train["jobs"] == ev["jobs"] == 3 and train["prefer"] == ev["prefer"] == "cpu"
    assert not train["deterministic"] and not ev["deterministic"]
    assert train["per_worker_gb"] is None and ev["per_worker_gb"] == 11.0
    assert out == [(job, f"blob{job[1]}".encode(), 7, [3, 5]) for job in cells]
    for key, target in (("train_stage", P8R.train_job), ("eval_stage", P8R.eval_job)):
        plan, rec = marker[target], compute[key]
        assert rec["plan"] == plan.describe()
        assert (rec["workers"], rec["max_workers_per_gpu"], rec["gpu_oversub"], rec["vram_budget_gb"]) == \
            (plan.workers, plan.per_gpu, plan.oversub, plan.budget_gb)
        assert rec["device"] == "cuda" and rec["workers_forced"] is False and rec["gpu_oversub_forced"] is False
    assert compute["device"] == "cuda" and compute["deterministic_flag"] is False
    assert compute["eval_vram_budget_gb_requested"] == 11.0

    calls.clear()                                      # an explicit budget overrides the derived one
    P8R.run_two_stage(P8R.eval_job, cells, train_shared={}, eval_shared={"extra": 0}, threads=4, log=None)
    assert [c["threads"] for c in calls] == [4, 4]
    calls.clear()                                      # 0 means "auto": the derived budget, for BOTH stages
    P8R.run_two_stage(P8R.eval_job, cells, train_shared={}, eval_shared={"extra": 0}, prefer="cpu", n_jobs=3,
                      threads=0, log=None)
    assert [c["threads"] for c in calls] == [want, want]
    for bad in (-1, -2):
        with pytest.raises(ValueError, match="threads must be >= 0"):
            P8R.run_two_stage(P8R.eval_job, cells, train_shared={}, eval_shared={"extra": 0}, threads=bad,
                              log=None)
    marker[P8R.eval_job] = parallel.Plan(88, "cuda", 4, 1, 9, budget_gb=float("inf"))   # an inf budget is
    _, compute = P8R.run_two_stage(P8R.eval_job, cells, train_shared={}, eval_shared={"extra": 0},  # recorded
                                   log=None, eval_per_worker_gb=float("inf"))                      # as "inf"
    assert compute["eval_stage"]["vram_budget_gb"] == compute["eval_vram_budget_gb_requested"] == "inf"
    json.dumps(compute, allow_nan=False)               # JSON has no Infinity: this would raise
    calls.clear()                                      # a bad budget is refused before anything trains
    for bad in (float("nan"), 0.0, -1.0):
        with pytest.raises(ValueError):
            P8R.run_two_stage(P8R.eval_job, cells, train_shared={}, eval_shared={"extra": 0}, log=None,
                              eval_per_worker_gb=bad)
    assert calls == []
    _, compute = P8R.run_two_stage(P8R.eval_job, cells, train_shared={}, eval_shared={"extra": 0}, log=None,
                                   eval_per_worker_gb="11")
    assert calls[-1]["per_worker_gb"] == compute["eval_vram_budget_gb_requested"] == 11.0


def test_per_worker_vram_budget_sets_workers_per_gpu(monkeypatch):
    """The size-aware planner in decimal GB (what hw.gpu_mem_gb reports): a 16 GiB P100 is 17.07 GB, so
    the default 8 GB budget gives it 2 workers per GPU and phase8R2's EVAL_VRAM_GB gives it 1; a 24 GiB
    card keeps 2 with it. The SMALLEST card decides. BANDSIM_PER_WORKER_GB, the user's override, replaces
    a caller's budget -- unless it is malformed, NaN or non-positive, when it is ignored (it must not fail
    open into 3 workers per card); a caller passing such a budget is a bug. A larger budget never gives
    more workers: inf, from either, gives one per GPU. BANDSIM_GPU_OVERSUB beats every budget, and the
    plan says which rule set it."""
    import phase8R2_landcover_reliability as R2
    from bandsim import hw, parallel
    for k in _PLAN_ENV:
        monkeypatch.delenv(k, raising=False)
    gpus = lambda *gb: monkeypatch.setattr(hw, "gpu_mem_gb", lambda: list(gb))
    gpus(17.07, 17.07, 17.07, 17.07)
    assert hw.gpu_oversub_default() == 2
    assert hw.gpu_oversub_default(per_worker_gb=R2.EVAL_VRAM_GB) == 1
    gpus(25.77, 25.77)
    assert hw.gpu_oversub_default(per_worker_gb=R2.EVAL_VRAM_GB) == 2
    gpus(12.0, 34.36)                                  # heterogeneous: the 12 GB card decides
    assert hw.gpu_oversub_default() == 1
    gpus(17.07, 17.07)
    monkeypatch.setenv("BANDSIM_PER_WORKER_GB", "8")
    assert hw.gpu_oversub_default(per_worker_gb=11.0) == 2
    for bad in ("abc", "11GB", "nan", "-1", "0"):
        monkeypatch.setenv("BANDSIM_PER_WORKER_GB", bad)
        assert hw.gpu_oversub_default(per_worker_gb=11.0) == 1, f"override {bad!r} must be ignored"
        assert hw.gpu_oversub_default() == 2, f"override {bad!r} must be ignored"
    for big in ("inf", "1e9"):
        monkeypatch.setenv("BANDSIM_PER_WORKER_GB", big)
        assert hw.gpu_oversub_default() == hw.gpu_oversub_default(per_worker_gb=4.0) == 1, big
    monkeypatch.delenv("BANDSIM_PER_WORKER_GB")
    assert hw.gpu_oversub_default(per_worker_gb=float("inf")) == 1
    for bad in (float("nan"), 0.0, -3.0):
        with pytest.raises(ValueError):
            hw.gpu_oversub_default(per_worker_gb=bad)

    monkeypatch.setattr(hw, "n_gpus", lambda: 4)
    monkeypatch.setattr(hw, "available_cores", lambda: 8)
    gpus(17.07, 17.07, 17.07, 17.07)
    p = parallel.plan_jobs(100, prefer="cuda")
    assert p.workers == 8 and "size-aware ~2/GPU at 8GB/worker" in p.describe()
    p = parallel.plan_jobs(100, prefer="cuda", per_worker_gb=11.0)
    assert p.workers == 4 and "size-aware ~1/GPU at 11GB/worker" in p.describe()
    monkeypatch.setenv("BANDSIM_GPU_OVERSUB", "3")
    p = parallel.plan_jobs(100, prefer="cuda", per_worker_gb=11.0)
    assert p.workers == 12 and "forced 3/GPU by BANDSIM_GPU_OVERSUB" in p.describe()
    monkeypatch.delenv("BANDSIM_GPU_OVERSUB")
    with pytest.raises(ValueError):
        parallel.plan_jobs(100, prefer="cpu", per_worker_gb=float("nan"))


def test_a_forced_worker_count_above_the_budget_is_reported_and_warned(monkeypatch):
    """--jobs / BANDSIM_WORKERS can put more workers on a GPU than the VRAM budget allows (16 workers on
    4 P100s = 4 per card against phase8R2's 1): the plan must say so -- in describe(), which goes into
    provenance -- and run_jobs must warn, instead of recording "size-aware ~1/GPU" for a plan that ran 4."""
    from bandsim import hw, parallel
    for k in _PLAN_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(hw, "gpu_mem_gb", lambda: [17.07] * 4)
    monkeypatch.setattr(hw, "n_gpus", lambda: 4)
    monkeypatch.setattr(hw, "available_cores", lambda: 8)
    p = parallel.plan_jobs(100, prefer="cuda", jobs=16, per_worker_gb=11.0)
    assert (p.workers, p.per_gpu, p.oversub, p.workers_forced, p.over_budget) == (16, 4, 1, True, True)
    assert "forced by jobs=/BANDSIM_WORKERS: up to 4/GPU, MORE than the 1/GPU its 11GB/worker budget allows" \
        in p.describe()
    ok = parallel.plan_jobs(100, prefer="cuda", jobs=4, per_worker_gb=11.0)
    assert ok.workers_forced and not ok.over_budget
    assert not parallel.plan_jobs(100, prefer="cuda", per_worker_gb=11.0).workers_forced
    odd = parallel.plan_jobs(100, prefer="cuda", jobs=5, per_worker_gb=11.0)   # round-robin: GPU 0 gets 2
    assert (odd.per_gpu, odd.over_budget) == (2, True)
    assert "up to 2/GPU, MORE than the 1/GPU its 11GB/worker budget allows" in odd.describe()
    monkeypatch.setenv("BANDSIM_GPU_OVERSUB", "1")     # a forced worker count overrides a forced oversub too
    both = parallel.plan_jobs(100, prefer="cuda", jobs=8)
    assert (both.workers, both.per_gpu, both.oversub_forced, both.over_budget) == (8, 2, True, True)
    assert "forced 1/GPU by BANDSIM_GPU_OVERSUB" in both.describe()
    assert "up to 2/GPU, MORE than the 1/GPU BANDSIM_GPU_OVERSUB sets" in both.describe()
    monkeypatch.setenv("BANDSIM_GPU_OVERSUB", "2")
    assert not parallel.plan_jobs(100, prefer="cuda", jobs=8).over_budget
    monkeypatch.delenv("BANDSIM_GPU_OVERSUB")
    monkeypatch.setattr(hw, "gpu_mem_gb", lambda: [])  # memory unreadable: 1/GPU, and not from a budget
    blind = parallel.plan_jobs(100, prefer="cuda", per_worker_gb=11.0)
    assert (blind.oversub, blind.budget_gb) == (1, None) and "1/GPU, GPU memory unknown" in blind.describe()
    assert "size-aware" not in blind.describe() and "budget" not in blind.describe()
    blind = parallel.plan_jobs(100, prefer="cuda", jobs=8, per_worker_gb=11.0)
    assert blind.over_budget and blind.budget_gb is None
    assert "MORE than the 1/GPU used when GPU memory is unknown" in blind.describe()

    logged = []
    monkeypatch.setenv("BANDSIM_DEVICE", "cpu")        # the serial run_jobs below writes this key; restore it
    monkeypatch.setattr(hw, "setup", lambda **kw: None)
    over = parallel.Plan(1, "cuda", 1, 0, 1, gpu_gb=[17.07], budget_gb=11.0, workers_forced=True)   # serial:
    monkeypatch.setattr(parallel, "plan_jobs", lambda *a, **k: over)                               # nothing
    for label, limit in (("x", "its 11GB/worker budget allows"), ("y", "BANDSIM_GPU_OVERSUB sets"),  # spawned
                         ("z", "used when GPU memory is unknown")):
        if label == "y":
            over.oversub_forced = True
        if label == "z":
            over.oversub_forced, over.gpu_gb, over.budget_gb = False, [], None
        logged.clear()
        assert parallel.run_jobs(_square, [3], log=logged.append, label=label) == [9]
        assert any(m.startswith(f"[parallel] WARNING {label}:") and f"above the 0/GPU {limit} -- this can run "
                   f"out of GPU memory" in m for m in logged), (label, logged)


def test_run_jobs_forwards_the_budget_and_reports_the_plan_it_ran(monkeypatch):
    """run_jobs hands per_worker_gb to the planner and calls on_plan once, before any job, with the very
    Plan it then runs (run_two_stage records that one, not a re-plan)."""
    from bandsim import hw, parallel
    for k in _PLAN_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BANDSIM_DEVICE", "cpu")        # the serial run_jobs below writes this key; restore it
    monkeypatch.setattr(hw, "setup", lambda **kw: None)
    asked, used = [], []
    real_plan = parallel.plan_jobs

    def plan(*a, **k):
        asked.append(k.get("per_worker_gb"))
        return real_plan(1, prefer="cpu")
    monkeypatch.setattr(parallel, "plan_jobs", plan)
    _RAN.clear()
    out = parallel.run_jobs(_logged_square, [2, 3], per_worker_gb=11.0, log=None,
                            on_plan=lambda p: used.append((p, list(_RAN))))
    assert asked == [11.0]
    assert len(used) == 1 and used[0][0].serial, "on_plan must be called exactly once, with the plan run"
    assert used[0][1] == [], "on_plan must be called before any job runs"
    assert out == [4, 9] and _RAN == [2, 3]


_RAN = []                                              # jobs _logged_square has run (serial path: this process)


def _logged_square(x):
    _RAN.append(x)
    return x * x


def test_train_job_moves_models_to_cpu_and_eval_job_to_the_workers_device(cpu, monkeypatch):
    """The two device moves of the transport: train_job copies every model to the CPU before pickling
    (so the bytes load on any worker), and eval_job moves every unpickled model to hw.device(). On a CPU
    host both are no-ops, so they are pinned with spies, and hw.device() is made to answer a device
    distinguishable from the default."""
    import phase8R_reliability as P8R
    from bandsim import hw
    p = _problem(n_tr=300, units=24, per_unit=10)
    moves = []
    real_cpu, real_to = torch.nn.Module.cpu, torch.nn.Module.to

    def spy_cpu(self):
        moves.append(("cpu", type(self).__name__))
        return real_cpu(self)

    def spy_to(self, *a, **k):
        moves.append(("to", type(self).__name__, a[0] if a else k.get("device")))
        return real_to(self, *a, **k)
    monkeypatch.setattr(torch.nn.Module, "cpu", spy_cpu)
    monkeypatch.setattr(torch.nn.Module, "to", spy_to)
    blob = P8R.train_job(0, p["Xtr"], p["ytr"], p["groups"], p["cwl"], 0.8, 1)
    assert {m[1] for m in moves if m[0] == "cpu"} >= {"GroupedCrossBandAttention", "MLPBaseline"}
    moves.clear()
    target_dev = torch.device("cpu", 0)               # != torch.device("cpu"), yet runnable here
    monkeypatch.setattr(hw, "device", lambda *a, **k: target_dev)
    P8R.eval_job((0, 0), {0: blob}, p["Xte"], p["yte"], p["unit"], p["groups"], target=0.1)
    moved = {m[1] for m in moves if m[0] == "to" and m[2] == target_dev}
    assert moved >= {"GroupedCrossBandAttention", "MLPBaseline"}, f"moves seen: {moves}"


# ------------------------------------------------------------- through the real process pools
def test_two_stage_through_real_pools_survives_a_low_open_files_limit(cpu):
    """main()'s actual path -- both stages through parallel.run_jobs in SPAWNED workers -- against the
    per-cell protocol through the same pool machinery, with the two-stage run held to 48 spare file
    descriptors. Stage 2 hands every evaluation job ALL model seeds' models; handing nn.Module objects to
    the pool makes torch's multiprocessing reducer move each tensor to shared memory and pass one file
    descriptor per tensor (40 per seed: 400 per job at 10 seeds), which crashed the pool with "Too many
    open files" under the common `ulimit -n 1024` (measured 2026-10-13). train_job's pickled BYTES need
    none. One thread per worker on both sides (module docstring), so the rows must match exactly."""
    resource = pytest.importorskip("resource")
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("needs /proc/self/fd to size the limit")
    import phase8R_reliability as P8R
    from bandsim import parallel
    p = _problem(n_tr=600, units=24, per_unit=50)
    cells = [(ss, ms) for ss in (0, 1) for ms in range(10)]
    tr = dict(Xtr=p["Xtr"], ytr=p["ytr"], groups=p["groups"], cwl=p["cwl"], subsample_frac=0.8, epochs=1)
    ev = dict(Xte=p["Xte"], yte=p["yte"], rid_te=p["unit"], groups=p["groups"], target=0.1)
    per_cell = parallel.run_jobs(P8R.run_seed, cells, shared=dict(tr, **{k: v for k, v in ev.items()
                                                                       if k != "groups"}),
                                 jobs=2, threads=1, log=None)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = max(int(fd) for fd in os.listdir("/proc/self/fd")) + 1 + 48
    if hard != resource.RLIM_INFINITY and limit > hard:
        pytest.skip("hard open-files limit too low to run the pool at all")
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    try:
        two_stage, compute = P8R.run_two_stage(P8R.eval_job, cells, train_shared=tr, eval_shared=ev,
                                               n_jobs=2, threads=1, log=None)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert compute["train_stage"]["workers"] > 1 and compute["eval_stage"]["workers"] > 1, "must use pools"
    _assert_cells_identical(per_cell, two_stage)
    assert compute["threads_per_worker"] == 1


# ------------------------------------------------------------------- job-boundary cache release
def test_job_boundary_cache_release_runs_on_success_and_on_error(tmp_path, monkeypatch, job_modules):
    from bandsim import parallel
    calls = []
    monkeypatch.setattr(parallel, "_release_cuda_cache", lambda: calls.append(1))
    mod = tmp_path / "h10_jobmod.py"
    mod.write_text("def ok(item, k=0):\n    return item + k\n\n"
                   "def boom(item):\n    raise RuntimeError('job failed')\n")
    assert parallel._run_one(str(mod), "ok", 2, {"k": 3}) == 5
    with pytest.raises(RuntimeError, match="job failed"):
        parallel._run_one(str(mod), "boom", 1, {})
    assert len(calls) == 2, f"cache released {len(calls)}x; expected 2 (success AND error path)"


def test_release_error_never_masks_the_jobs_own_error(tmp_path, monkeypatch, job_modules):
    """A failing release after a FAILED job must not replace the job's error (it rides along as a
    note); after a SUCCESSFUL job it must propagate -- the device is then in a bad state."""
    from bandsim import parallel

    def failing_release():
        raise RuntimeError("release failed")
    monkeypatch.setattr(parallel, "_release_cuda_cache", failing_release)
    mod = tmp_path / "h10_jobmod2.py"
    mod.write_text("def ok(item):\n    return item\n\n"
                   "def boom(item):\n    raise ValueError('job failed')\n")
    with pytest.raises(ValueError, match="job failed") as ei:
        parallel._run_one(str(mod), "boom", 1, {})
    assert any("release failed" in n for n in getattr(ei.value, "__notes__", []))
    with pytest.raises(RuntimeError, match="release failed"):
        parallel._run_one(str(mod), "ok", 1, {})


def test_serial_path_releases_the_cache_after_every_job(monkeypatch):
    """With one job (one model seed, --smoke, --jobs 1) run_jobs runs in the PARENT; it must release
    there too, or the parent keeps its training cache on the GPU while the next stage's workers run."""
    from bandsim import hw, parallel
    calls = []
    monkeypatch.setattr(parallel, "_release_cuda_cache", lambda: calls.append(1))
    monkeypatch.setattr(hw, "setup", lambda **kw: None)    # keep the pytest process's threads/RNGs as they are
    monkeypatch.setenv("BANDSIM_DEVICE", "cpu")
    for k in _PLAN_ENV:                                    # BANDSIM_WORKERS=4 would turn this into a pool
        monkeypatch.delenv(k, raising=False)
    assert parallel.run_jobs(_square, [1, 2, 3], jobs=1, log=None) == [1, 4, 9]
    assert len(calls) == 3


def test_cache_release_only_when_cuda_is_initialised(monkeypatch):
    """torch.cuda.empty_cache() is called exactly when this process has initialised CUDA, and torch is
    never imported by the release itself."""
    from bandsim import parallel
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    parallel._release_cuda_cache()
    assert calls == []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    parallel._release_cuda_cache()
    assert calls == [1]
    monkeypatch.setattr(parallel, "sys", types.SimpleNamespace(modules={}))   # a process without torch
    parallel._release_cuda_cache()
    assert calls == [1]
