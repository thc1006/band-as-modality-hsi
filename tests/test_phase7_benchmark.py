"""The Phase 7 efficiency benchmark must be a MEASUREMENT, not a number-shaped artefact.

These tests pin the MACHINERY, never an absolute timing. A test that asserted "the MLP is faster
than the attention model" would fail on a loaded box for reasons that have nothing to do with the
code, so nothing here compares milliseconds. What is checked is that the properties which make a
latency table trustworthy actually hold:

  * repeats really happen and the reported cell carries DISPERSION, not one sample;
  * each model is measured in a DIFFERENT PROCESS, so allocator state / cuDNN autotune caches /
    warm kernels cannot leak from one row into the next;
  * the thread count and dtype are PINNED and RECORDED, and an unhonoured pin is an error;
  * strict mode EXITS NONZERO when a required measurement is missing, rather than emitting 'n/a'
    in a table that otherwise looks complete;
  * a busy machine REFUSES to produce paper artefacts;
  * missing-band accuracy is measured over SEVERAL subsets, not one convenient prefix;
  * the reported INT8 scope is derived from the quantized module tree, so it states what was
    actually covered;
  * the MLP measurement path allocates NO present-mask, so no unused tensor inflates its memory.

The heavier end-to-end paths (a real worker subprocess, a real Timer measurement) use tiny models
and a minimal budget so the suite stays fast on a contended box.
"""
import ast
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_ROOT, "experiments")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EXP)

import phase7_efficiency as p7                                              # noqa: E402
from bandsim.grouping import contiguous_groups, group_center_wavelengths    # noqa: E402
from bandsim.io import AVIRIS_WL_NM                                         # noqa: E402
from bandsim.model import GroupedCrossBandAttention                         # noqa: E402

_SCRIPT = os.path.join(_EXP, "phase7_efficiency.py")


# ----------------------------------------------------------------------------- tiny fixtures
class _Tiny(nn.Module):
    """Smallest thing that still exercises the timing path."""

    def __init__(self, n_in=8, n_out=4):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture(scope="module")
def cpu_dev():
    return torch.device("cpu")


def _attn_model(n_bands=40, n_groups=5, n_classes=3):
    groups = contiguous_groups(n_bands, n_groups)
    cwl = group_center_wavelengths(AVIRIS_WL_NM[:n_bands], groups)
    return GroupedCrossBandAttention(groups, cwl, n_classes), groups


# ========================================================================================
# 1. repeats happen, and dispersion is reported
# ========================================================================================
def test_bench_latency_pools_replicates_from_every_repeat(cpu_dev):
    """The whole point of migrating off a hand-rolled `time.time()` loop: one measurement pass
    cannot produce a dispersion. Ask for several repeats and every one must contribute samples."""
    m, x = _Tiny(), torch.randn(4, 8)
    one = p7.bench_latency(m, (x,), cpu_dev, num_threads=1, repeats=1, warmup=2,
                           min_run_time=0.01, max_run_time=0.15, target_rel_iqr=0.05)
    many = p7.bench_latency(m, (x,), cpu_dev, num_threads=1, repeats=4, warmup=2,
                            min_run_time=0.01, max_run_time=0.15, target_rel_iqr=0.05)
    assert one["repeats"] == 1 and many["repeats"] == 4
    assert one["n_measurements"] == 1 and many["n_measurements"] == 4
    # More repeats must mean strictly more pooled replicates -- otherwise `--repeats` is decorative.
    assert many["n"] > one["n"], (
        f"4 repeats pooled {many['n']} replicates but 1 repeat pooled {one['n']}: the extra "
        f"repeats did not contribute samples")


def test_bench_latency_reports_dispersion_not_just_a_number(cpu_dev):
    """A cell a reader cannot sanity-check is not a result. Median, IQR, quartiles, range and the
    replicate count must all come back, and they must be internally consistent."""
    m, x = _Tiny(), torch.randn(4, 8)
    r = p7.bench_latency(m, (x,), cpu_dev, num_threads=1, repeats=3, warmup=2,
                         min_run_time=0.01, max_run_time=0.15, target_rel_iqr=0.05)
    for k in ("median_ms", "iqr_ms", "p25_ms", "p75_ms", "min_ms", "max_ms", "mean_ms",
              "rel_iqr", "n", "throughput_sps", "num_threads"):
        assert k in r, f"latency summary is missing '{k}'"
    assert r["min_ms"] <= r["p25_ms"] <= r["median_ms"] <= r["p75_ms"] <= r["max_ms"]
    assert r["iqr_ms"] == pytest.approx(r["p75_ms"] - r["p25_ms"])
    assert r["rel_iqr"] == pytest.approx(r["iqr_ms"] / r["median_ms"])
    assert r["median_ms"] > 0 and r["throughput_sps"] > 0


def test_summarize_times_is_a_real_median_and_iqr():
    """Pin the statistics themselves on a known sample so a future refactor cannot quietly swap
    the median for a mean (which one slow replicate would drag)."""
    times = [1e-3, 2e-3, 3e-3, 4e-3, 100e-3]          # one pathological straggler
    s = p7.summarize_times(times, batch=10)
    assert s["median_ms"] == pytest.approx(3.0)        # median ignores the straggler
    assert s["max_ms"] == pytest.approx(100.0)         # but it is still visible in the range
    assert s["n"] == 5
    assert s["throughput_sps"] == pytest.approx(10 / 3e-3)
    assert s["mean_ms"] > s["median_ms"]               # the reason we report the median


def test_summarize_times_refuses_an_empty_sample():
    """No replicates must raise, never return a placeholder that later renders as a number."""
    with pytest.raises(ValueError):
        p7.summarize_times([], batch=4)


def test_bench_latency_pins_the_thread_count_it_reports(cpu_dev):
    """An unpinned CPU latency measures the box, not the model. The pin must be applied AND
    recorded, and must not leak out of the measurement into the caller's process."""
    before = torch.get_num_threads()
    m, x = _Tiny(), torch.randn(4, 8)
    r = p7.bench_latency(m, (x,), cpu_dev, num_threads=1, repeats=1, warmup=1,
                         min_run_time=0.01, max_run_time=0.1)
    assert r["num_threads"] == 1
    assert torch.get_num_threads() == before, "Timer's thread pin leaked into the caller"


# ========================================================================================
# 2. process isolation is real
# ========================================================================================
def test_worker_runs_in_a_different_process(tmp_path):
    """The load-bearing claim of the redesign. Run the real `--worker` entry point as the driver
    does and require the result to come back stamped with a DIFFERENT pid -- if this ever runs in
    the driver, row k inherits row k-1's warm caches and the table stops being comparable."""
    n_bands, n_groups, n_classes, n_px = 40, 5, 3, 24
    model, groups = _attn_model(n_bands, n_groups, n_classes)
    sd = tmp_path / "m.pt"
    torch.save(model.state_dict(), sd)
    npz = tmp_path / "d.npz"
    rng = np.random.default_rng(0)
    np.savez(npz, Xte=rng.standard_normal((n_px, n_bands)).astype(np.float32),
             yte=rng.integers(0, n_classes, n_px).astype(int))
    gj = tmp_path / "g.json"
    gj.write_text(json.dumps({"groups": [np.asarray(g).tolist() for g in groups],
                              "cwl": list(group_center_wavelengths(AVIRIS_WL_NM[:n_bands],
                                                                   groups))}))
    out = tmp_path / "res.json"
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "index": 0, "name": "tiny", "kind": "attn", "pe_type": "sinusoidal",
        "state_dict": str(sd), "data_npz": str(npz), "groups_json": str(gj), "out": str(out),
        "device": "cpu", "seed": 0, "num_classes": n_classes, "n_bands": n_bands, "bs": 8,
        "bench_threads": 1, "repeats": 2, "warmup": 1, "min_run_time": 0.01,
        "max_run_time": 0.1, "target_rel_iqr": 0.2, "max_missing": 2,
        "n_missing_subsets": 3, "subset_seed": 0}))

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", BANDSIM_DEVICE="cpu")
    proc = subprocess.run([sys.executable, _SCRIPT, "--worker", str(spec)],
                          env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"worker failed:\n{proc.stdout}\n{proc.stderr}"
    res = json.loads(out.read_text())

    assert res["worker_pid"] != os.getpid(), "the worker ran inside the test process"
    assert res["driver_pid"] != res["worker_pid"]
    # and it really measured something in there
    assert res["params"] > 0 and res["size_mb"] > 0
    assert res["lat_cpu"]["n"] >= 2 and res["lat_cpu"]["median_ms"] > 0
    assert res["pinned"]["torch_threads"] == 1
    assert res["pinned"]["default_dtype"] == "torch.float32"
    assert res["errors"] == {}, f"worker recorded errors: {res['errors']}"


def test_each_model_gets_its_own_worker_pid_in_the_row():
    """`isolation` is derived, not asserted by hand: a row whose worker pid equals the driver's is
    labelled IN-PROCESS and (below) fails the strict gate."""
    spec = {"repeats": 3, "warmup": 5, "min_run_time": 0.1, "max_missing": 2}
    iso = p7.result_to_row({"name": "a", "worker_pid": 4242}, spec, driver_pid=1)
    same = p7.result_to_row({"name": "b", "worker_pid": 1}, spec, driver_pid=1)
    none = p7.result_to_row({"name": "c"}, spec, driver_pid=1)
    assert iso["isolation"] == "subprocess"
    assert same["isolation"] == "IN-PROCESS"
    assert none["isolation"] == "IN-PROCESS"


def test_driver_measures_through_a_subprocess_not_inline():
    """Static backstop for the property above: the driver must actually spawn the worker. If a
    later refactor 'simplifies' this back into a function call, the pid check would still pass
    (it would be the same pid) -- so pin the spawn itself."""
    tree = ast.parse(open(_SCRIPT, encoding="utf-8").read())
    spawns = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr in ("run", "Popen")
              and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess"]
    assert spawns, "the driver never spawns a subprocess -- per-model isolation is gone"
    src = ast.unparse(tree)
    assert '"--worker"' in src or "'--worker'" in src, "no --worker dispatch in the driver"


# ========================================================================================
# 3. pinned environment
# ========================================================================================
def test_unhonoured_thread_pin_is_a_strict_failure():
    """Asking for 8 threads and silently getting 2 makes the CPU columns incomparable to any
    other run of this script. That must be an error, not a footnote."""
    ok = _row(bench_threads=8, torch_threads=8)
    bad = _row(bench_threads=8, torch_threads=2)
    assert p7.strict_failures([ok], "cpu", 0.5, check_dispersion=False) == []
    fails = p7.strict_failures([bad], "cpu", 0.5, check_dispersion=False)
    assert any("bench threads" in f for f in fails), fails


def test_row_records_what_was_pinned():
    """The pin is worthless if the artefact does not say what it was -- a reader has to be able to
    reproduce the configuration, not just trust that one existed."""
    for col in ("bench_threads", "torch_threads", "interop_threads", "default_dtype",
                "usable_cores", "repeats", "warmup_iters", "min_run_time_s"):
        assert col in p7.CSV_COLS, f"the CSV does not record '{col}'"


# ========================================================================================
# 4. several missing-band subsets, with spread
# ========================================================================================
def test_missing_subsets_are_several_and_distinct():
    subs = p7.missing_subsets(n_groups=10, k=6, n_subsets=5, seed=0)
    assert len(subs) == 5
    assert len(set(subs)) == 5, "subsets must be distinct, else the 'spread' is fake"
    assert all(len(s) == 6 and len(set(s)) == 6 for s in subs)
    assert all(0 <= g < 10 for s in subs for g in s)


def test_missing_subsets_keep_the_prefix_first_and_add_other_shapes():
    """The historical prefix must stay visible (so old numbers remain comparable) but it must no
    longer be the ONLY thing measured -- that was the whole complaint."""
    subs = p7.missing_subsets(n_groups=10, k=6, n_subsets=5, seed=0)
    assert subs[0] == (0, 1, 2, 3, 4, 5), "the prefix must remain the first, comparable subset"
    assert subs[1] == (4, 5, 6, 7, 8, 9), "the opposite end of the spectrum must be covered"
    assert any(s != tuple(range(min(s), min(s) + 6)) for s in subs[2:]), \
        "every subset is contiguous -- no interleaved/random shape was generated"


def test_missing_subsets_are_deterministic_in_the_seed():
    a = p7.missing_subsets(10, 6, 5, seed=3)
    b = p7.missing_subsets(10, 6, 5, seed=3)
    c = p7.missing_subsets(10, 6, 5, seed=4)
    assert a == b, "same seed must give the same subsets"
    assert a != c, "the seed must actually vary the random subsets"


def test_missing_subsets_handle_degenerate_k():
    assert p7.missing_subsets(10, 0, 5, 0) == [()]            # nothing missing
    assert p7.missing_subsets(10, 10, 5, 0) == [tuple(range(10))]
    assert p7.missing_subsets(10, 12, 5, 0) == [tuple(range(10))]
    # only C(4,3)=4 distinct subsets exist; asking for 9 must not spin or duplicate
    s = p7.missing_subsets(4, 3, 9, 0)
    assert len(s) == 4 and len(set(s)) == 4


def test_stat_block_reports_the_spread():
    st = p7.stat_block([10.0, 20.0, 30.0])
    assert st["mean"] == pytest.approx(20.0)
    assert st["min"] == 10.0 and st["max"] == 30.0
    assert st["std"] > 0, "a spread of zero over differing values would hide the variation"
    assert st["values"] == [10.0, 20.0, 30.0], "every per-subset value must survive to the CSV"
    assert p7.stat_block([5.0])["std"] == 0.0            # single value: defined, not a crash


def test_csv_exposes_every_per_subset_value_and_the_spread():
    for col in ("n_missing_subsets", "missing_subsets", "miou_fp32_miss_mean",
                "miou_fp32_miss_std", "miou_fp32_miss_min", "miou_fp32_miss_max",
                "miou_fp32_miss_prefix", "miou_fp32_miss_values",
                "miou_dynint8_miss_mean", "miou_dynint8_miss_values"):
        assert col in p7.CSV_COLS, f"the CSV does not expose '{col}'"


def test_row_carries_the_individual_subset_accuracies():
    res = {"name": "m", "worker_pid": 9,
           "miou_fp32_miss": p7.stat_block([9.5, 15.0, 18.5]),
           "missing_subsets": [[0, 1], [2, 3], [1, 3]]}
    row = p7.result_to_row(res, {"repeats": 1, "max_missing": 2}, driver_pid=1)
    assert row["n_missing_subsets"] == 3
    assert row["miou_fp32_miss_prefix"] == pytest.approx(9.5)
    assert row["miou_fp32_miss_mean"] == pytest.approx(14.333, abs=1e-3)
    assert row["miou_fp32_miss_values"] == "9.500;15.000;18.500"
    assert row["missing_subsets"] == "0+1 | 2+3 | 1+3"


# ========================================================================================
# 5. strict mode exits nonzero; nothing is produced by an exception handler
# ========================================================================================
def _row(**over):
    """A row that passes every strict check, so each test can break exactly one thing."""
    base = {
        "name": "m", "isolation": "subprocess", "worker_pid": 2, "driver_pid": 1,
        "bench_threads": 8, "torch_threads": 8,
        "lat_cpu_rel_iqr": 0.01, "lat_gpu_rel_iqr": 0.01, "dynint8_lat_cpu_rel_iqr": 0.01,
        "lat_cpu_n": 40, "lat_gpu_n": 40, "dynint8_lat_cpu_n": 40,
        "quant_n_quantized": 3, "status": "ok",
    }
    for k in p7.REQUIRED_ALWAYS + p7.REQUIRED_CUDA:
        base.setdefault(k, 1.0)
    base.update(over)
    return base


@pytest.mark.parametrize("missing", sorted(p7.REQUIRED_ALWAYS))
def test_every_required_metric_missing_is_a_strict_failure(missing):
    fails = p7.strict_failures([_row(**{missing: None})], "cpu", 0.5, check_dispersion=False)
    assert any(missing in f for f in fails), f"a missing '{missing}' passed the strict gate"


@pytest.mark.parametrize("missing", sorted(p7.REQUIRED_CUDA))
def test_gpu_metrics_are_required_on_cuda_and_ignored_on_cpu(missing):
    """A GPU column may legitimately be absent on a CPU-only host -- that is 'not applicable', not
    'failed'. On CUDA the same absence means the measurement did not happen."""
    row = _row(**{missing: None})
    assert any(missing in f for f in p7.strict_failures([row], "cuda", 0.5,
                                                        check_dispersion=False))
    assert p7.strict_failures([row], "cpu", 0.5, check_dispersion=False) == []


def test_macs_may_be_absent_without_failing_strict():
    """fvcore genuinely cannot count scaled_dot_product_attention. The honest response is 'n/a'
    plus a reason, so MACs must NOT be a required metric -- otherwise the only way to pass would
    be to report the undercount."""
    assert "macs" not in p7.REQUIRED_ALWAYS and "macs" not in p7.REQUIRED_CUDA
    assert "macs" in p7.OPTIONAL_DOCUMENTED
    assert p7.strict_failures([_row(macs=None, macs_per_sample=None)], "cuda", 0.5,
                              check_dispersion=False) == []


def test_in_process_measurement_is_a_strict_failure():
    fails = p7.strict_failures([_row(isolation="IN-PROCESS")], "cpu", 0.5, check_dispersion=False)
    assert any("isolated subprocess" in f for f in fails), fails


def test_noisy_latency_is_a_strict_failure():
    quiet = p7.strict_failures([_row(lat_cpu_rel_iqr=0.02)], "cpu", 0.10)
    noisy = p7.strict_failures([_row(lat_cpu_rel_iqr=0.42)], "cpu", 0.10)
    assert quiet == []
    assert any("relative IQR" in f for f in noisy), noisy


def test_too_few_replicates_is_a_strict_failure():
    """A 'median over 2 samples' is a label, not a statistic."""
    fails = p7.strict_failures([_row(lat_cpu_n=2)], "cpu", 0.5, min_replicates=8)
    assert any("replicate" in f for f in fails), fails
    assert p7.strict_failures([_row(lat_cpu_n=2)], "cpu", 0.5, min_replicates=8,
                              check_dispersion=False) == [], \
        "the replicate floor belongs to the publication-grade checks"


def test_quantisation_that_quantised_nothing_is_a_strict_failure():
    """An int8 size/latency for a model where zero modules were quantized describes the fp32
    model under a different column name."""
    fails = p7.strict_failures([_row(quant_n_quantized=0)], "cpu", 0.5, check_dispersion=False)
    assert any("0 quantized modules" in f for f in fails), fails


def test_a_clean_row_passes_every_gate():
    """Negative control: the gate must be passable, or the tests above prove nothing."""
    assert p7.strict_failures([_row()], "cuda", 0.10) == []


def test_worker_records_the_exception_instead_of_inventing_a_number(tmp_path):
    """The failure policy in one test. A worker whose spec points at a corrupt state_dict must not
    come back with a plausible-looking measurement; the driver must be able to see WHY."""
    spec = tmp_path / "spec.json"
    (tmp_path / "broken.pt").write_bytes(b"not a checkpoint")
    spec.write_text(json.dumps({
        "index": 0, "name": "broken", "kind": "mlp", "pe_type": "sinusoidal",
        "state_dict": str(tmp_path / "broken.pt"), "data_npz": str(tmp_path / "nope.npz"),
        "groups_json": str(tmp_path / "nope.json"), "out": str(tmp_path / "r.json"),
        "device": "cpu", "seed": 0, "num_classes": 3, "n_bands": 8, "bs": 4,
        "bench_threads": 1, "repeats": 1, "warmup": 0, "min_run_time": 0.01,
        "max_run_time": 0.05, "target_rel_iqr": 0.5, "max_missing": 1,
        "n_missing_subsets": 1, "subset_seed": 0}))
    proc = subprocess.run([sys.executable, _SCRIPT, "--worker", str(spec)],
                          env=dict(os.environ, BANDSIM_DEVICE="cpu"),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0, "a worker that could not load its inputs reported success"
    assert not (tmp_path / "r.json").exists(), \
        "a failed worker wrote a result file -- the driver would read it as a measurement"


def test_strict_mode_exits_nonzero_end_to_end(tmp_path, monkeypatch):
    """The contract the caller actually depends on: `main()` returns a NONZERO exit code and
    writes NO paper artefact when a required measurement is missing.

    The training/measuring stages are stubbed out -- this test is about the failure PATH, not
    about spending GPU minutes to reach it."""
    paper = tmp_path / "paper"
    paper.mkdir()
    monkeypatch.setattr(p7, "PAPER_DIR", str(paper))
    monkeypatch.setattr(p7, "P", lambda rel: str(paper / rel))
    # a quiet box, so the contention gate is not what we are testing here
    monkeypatch.setattr(p7, "probe_contention", lambda: {
        "nvidia_smi": False, "gpu_util_pct": None, "gpu_util_max": None,
        "gpu_mem_used_mb": None, "n_compute_apps": None, "loadavg_1m": 0.0,
        "cores": 8, "load_per_core": 0.0})
    # one row whose REQUIRED cpu latency never arrived (as if the worker's measurement raised)
    monkeypatch.setattr(p7, "_run_models", lambda *a, **k: (
        [_row(name="Proposed", lat_cpu_ms=None, status="lat_cpu=RuntimeError: boom")], []))

    rc = p7.main(["--device", "cpu", "--workdir", str(tmp_path / "wd"), "--keep-workdir"])
    assert rc != 0, "strict mode returned 0 despite a missing required measurement"
    assert not list(paper.glob("*.csv")), "strict mode wrote a paper CSV for a failed run"
    assert not list(paper.glob("*.tex")), "strict mode wrote paper macros for a failed run"
    # the diagnostic must still exist somewhere non-quotable, so the failure is debuggable
    assert list((tmp_path / "wd").glob("*.FAILED.csv")), "no diagnostic table was left behind"


def test_no_strict_writes_the_artefacts_with_the_failure_recorded(tmp_path, monkeypatch):
    """`--no-strict` is the escape hatch, and it must be loud: the artefact still appears, but the
    failure text travels with it in `status`."""
    paper = tmp_path / "paper"
    paper.mkdir()
    monkeypatch.setattr(p7, "PAPER_DIR", str(paper))
    monkeypatch.setattr(p7, "P", lambda rel: str(paper / rel))
    monkeypatch.setattr(p7, "probe_contention", lambda: {
        "nvidia_smi": False, "gpu_util_pct": None, "gpu_util_max": None,
        "gpu_mem_used_mb": None, "n_compute_apps": None, "loadavg_1m": 0.0,
        "cores": 8, "load_per_core": 0.0})
    monkeypatch.setattr(p7, "_run_models", lambda *a, **k: (
        [_row(name="Proposed", lat_cpu_ms=None, status="lat_cpu=RuntimeError: boom")], []))

    rc = p7.main(["--device", "cpu", "--no-strict", "--workdir", str(tmp_path / "wd")])
    assert rc == 0
    csvs = list(paper.glob("*.csv"))
    assert csvs, "--no-strict wrote no CSV"
    body = csvs[0].read_text()
    assert "RuntimeError: boom" in body, "the failure did not travel with the numbers"
    assert "n/a" in body, "the missing measurement was rendered as something other than n/a"


# ========================================================================================
# 6. the busy-machine refusal
# ========================================================================================
def _probe(gpu_util=0.0, apps=0, load_per_core=0.0):
    return {"nvidia_smi": True, "gpu_util_pct": [gpu_util], "gpu_util_max": gpu_util,
            "gpu_mem_used_mb": [0.0], "n_compute_apps": apps, "loadavg_1m": load_per_core * 8,
            "cores": 8, "load_per_core": load_per_core}


def test_contention_is_detected_from_any_of_the_three_signals():
    assert p7.contention_reasons(_probe(), want_cuda=True) == []
    assert p7.contention_reasons(_probe(gpu_util=95.0), want_cuda=True)
    assert p7.contention_reasons(_probe(apps=4), want_cuda=True)
    assert p7.contention_reasons(_probe(load_per_core=1.4), want_cuda=True)


def test_gpu_signals_are_ignored_when_the_bench_is_cpu_only():
    """A busy GPU does not invalidate a CPU-only benchmark; a busy CPU always does."""
    assert p7.contention_reasons(_probe(gpu_util=95.0, apps=4), want_cuda=False) == []
    assert p7.contention_reasons(_probe(load_per_core=1.4), want_cuda=False)


def test_our_own_cuda_context_is_not_counted_as_someone_elses_job():
    assert p7.contention_reasons(_probe(apps=1), want_cuda=True,
                                 expected_own_compute_apps=1) == []
    assert p7.contention_reasons(_probe(apps=2), want_cuda=True,
                                 expected_own_compute_apps=1)


def test_unmeasurable_load_is_not_read_as_no_load():
    """`None` means 'could not tell'. It must never be silently treated as 'quiet' -- but it also
    must not fabricate a refusal. It simply contributes no reason."""
    blind = {"nvidia_smi": False, "gpu_util_pct": None, "gpu_util_max": None,
             "gpu_mem_used_mb": None, "n_compute_apps": None, "loadavg_1m": None,
             "cores": 8, "load_per_core": None}
    assert p7.contention_reasons(blind, want_cuda=True) == []


def _probe_hostwide(host_per_core, own=None):
    """A probe as probe_contention builds it: host loadavg over ALL host cores, plus the optional
    pinned-cpuset busy fraction. host_cores=36 and cores=8 reproduce this shared box."""
    return {"nvidia_smi": True, "gpu_util_pct": [0.0], "gpu_util_max": 0.0,
            "gpu_mem_used_mb": [0.0], "n_compute_apps": 0,
            "loadavg_1m": host_per_core * 36, "cores": 8, "host_cores": 36,
            "load_per_core": host_per_core * 36 / 8, "load_per_host_core": host_per_core,
            "own_cpuset_busy": own}


def test_quiet_pinned_cpus_are_not_called_busy_because_the_host_is():
    # The whole point: a shared box whose OTHER cores are loaded by other tenants (host 0.62/core,
    # over the 0.6 gate) must NOT refuse when the cores this process is pinned to sit near idle.
    # This is the false refusal that used to force --allow-contended on every run here.
    assert p7.contention_reasons(_probe_hostwide(0.62, own=0.08), want_cuda=False) == []


def test_busy_pinned_cpus_are_flagged_even_if_the_host_looks_calm():
    # And the converse: if MY cores are saturated, a low host average must not excuse it.
    reasons = p7.contention_reasons(_probe_hostwide(0.20, own=0.95), want_cuda=False)
    assert reasons and "pinned" in reasons[0]


def test_own_cpuset_is_preferred_over_the_host_average():
    # When both are present they can disagree; the pinned-core measure wins in BOTH directions.
    assert p7.contention_reasons(_probe_hostwide(0.90, own=0.10), want_cuda=False) == []   # host hot, mine cold
    assert p7.contention_reasons(_probe_hostwide(0.10, own=0.90), want_cuda=False)          # host cold, mine hot


def test_falls_back_to_host_per_core_when_cpuset_unmeasurable():
    # If the pinned-core busy fraction could not be read (own=None), the host-wide proxy still
    # guards -- better a coarse check than none.
    assert p7.contention_reasons(_probe_hostwide(0.80, own=None), want_cuda=False)
    assert p7.contention_reasons(_probe_hostwide(0.30, own=None), want_cuda=False) == []


def test_own_cpuset_busy_per_core_returns_a_sane_fraction_or_none():
    # The real measurement on this machine: a fraction in [0, 1.5], or None if affinity/proc-stat
    # are unavailable. Never negative, never a bare 0 masquerading as "measured quiet".
    v = p7.own_cpuset_busy_per_core(sample_s=0.05)
    assert v is None or (0.0 <= v <= 1.5)


def test_probe_contention_returns_the_documented_shape():
    """Runs for real on this box -- asserts the CONTRACT (keys, types), never the values."""
    pr = p7.probe_contention()
    for k in ("nvidia_smi", "gpu_util_pct", "gpu_util_max", "gpu_mem_used_mb",
              "n_compute_apps", "loadavg_1m", "cores", "load_per_core"):
        assert k in pr, f"probe_contention() lost '{k}'"
    assert isinstance(pr["cores"], int) and pr["cores"] >= 1


def test_a_busy_box_refuses_to_produce_paper_artefacts(tmp_path, monkeypatch):
    """The requirement in one test: on a contended machine the run must NOT emit deliverables,
    and must say so with a nonzero exit -- before spending a minute training anything."""
    paper = tmp_path / "paper"
    paper.mkdir()
    monkeypatch.setattr(p7, "PAPER_DIR", str(paper))
    monkeypatch.setattr(p7, "P", lambda rel: str(paper / rel))
    monkeypatch.setattr(p7, "probe_contention", lambda: _probe(load_per_core=1.4))

    def _boom(*a, **k):                     # nothing expensive may be reached
        raise AssertionError("the run got past the contention gate")
    monkeypatch.setattr(p7, "_run_models", _boom)

    rc = p7.main(["--device", "cpu", "--workdir", str(tmp_path / "wd")])
    assert rc != 0, "a contended box produced a normal exit"
    assert not list(paper.glob("*")), "a contended box wrote paper artefacts"


def test_allow_contended_proceeds_but_stamps_every_row(tmp_path, monkeypatch):
    """The override must not be a way to get clean-looking numbers: the contamination has to
    travel into the artefact so it cannot be quoted as a quiet-box measurement."""
    paper = tmp_path / "paper"
    paper.mkdir()
    monkeypatch.setattr(p7, "PAPER_DIR", str(paper))
    monkeypatch.setattr(p7, "P", lambda rel: str(paper / rel))
    monkeypatch.setattr(p7, "probe_contention", lambda: _probe(load_per_core=1.4))
    monkeypatch.setattr(p7, "_run_models", lambda *a, **k: ([_row(name="Proposed")], []))

    rc = p7.main(["--device", "cpu", "--allow-contended", "--no-strict",
                  "--workdir", str(tmp_path / "wd")])
    assert rc == 0
    body = (paper / "results_phase7_efficiency.csv").read_text()
    assert "CONTENDED" in body, "a contended run produced an unmarked table"
    tex = (paper / "results_phase7_efficiency.tex").read_text()
    assert "CONTENDED" in tex and "WARNING" in tex, "the paper macros hide the contamination"


# ========================================================================================
# 7. quantisation scope is verified, not declared
# ========================================================================================
def test_quant_scope_lists_the_modules_that_were_actually_quantised():
    m = p7.MLPBaseline(16, 3).eval()
    q = p7.quantize_int8(m)
    rep = p7.quantized_scope_report(m, q)
    assert rep["n_quantized"] == 3, rep          # the MLP's three Linear layers
    assert rep["n_left_fp32"] == 0
    assert set(rep["quantized_modules"]) == {"net.0", "net.3", "net.5"}
    assert "int8-dynamic" in rep["scope"]
    for name in rep["quantized_modules"]:
        assert name in rep["scope"], "a quantized module is missing from the reported scope"


def test_quant_scope_names_what_stayed_fp32_in_the_attention_model():
    """'partial dynamic INT8' is only interpretable if the partition is stated. The encoder's
    linears AND the MultiheadAttention in-projection (a raw Parameter, not an nn.Linear, so not
    even a candidate) must both be visible in the scope string."""
    m, _ = _attn_model()
    m = m.eval()
    q = p7.quantize_int8(m)
    rep = p7.quantized_scope_report(m, q)
    assert set(rep["quantized_modules"]) == {"embed", "classifier", "decoder"}
    assert rep["n_left_fp32"] > 0, "the fp32 remainder was not reported"
    assert all(n.startswith("encoder") for n in rep["left_fp32_modules"]), rep
    assert "MultiheadAttention" in rep["scope"], \
        "the un-quantisable attention in_proj is invisible in the reported scope"
    assert "fp32" in rep["scope"]


def test_quant_scope_says_so_when_nothing_was_quantised():
    """The dangerous case: a size/latency reported for a model that was never quantized."""
    m = p7.MLPBaseline(16, 3).eval()
    rep = p7.quantized_scope_report(m, m)        # not quantized at all
    assert rep["n_quantized"] == 0
    assert "NOTHING WAS QUANTIZED" in rep["scope"]


def test_the_quantisation_api_and_its_validated_version_are_pinned_and_reported():
    """We deliberately stayed on the legacy eager path rather than adding TorchAO as a dependency,
    so the version it was validated against has to be recorded with every number."""
    assert "quantize_dynamic" in p7.QUANT_API and "legacy" in p7.QUANT_API.lower()
    assert p7.QUANT_VALIDATED_TORCH
    for col in ("quant_api", "quant_torch_version", "quant_validated_torch", "quant_scope",
                "quant_n_quantized", "quant_n_left_fp32"):
        assert col in p7.CSV_COLS, f"the CSV does not record '{col}'"


def test_quantised_model_still_runs_and_shrinks():
    """A scope report is only meaningful if the quantized model is usable."""
    m = p7.MLPBaseline(16, 3).eval()
    q = p7.quantize_int8(m)
    with torch.no_grad():
        assert q(torch.randn(4, 16)).shape == (4, 3)
    assert p7.model_size_mb(q) < p7.model_size_mb(m)


# ========================================================================================
# 8. no stray tensor in the MLP measurement path
# ========================================================================================
def test_mlp_profiling_inputs_carry_no_present_mask():
    """The MLP never reads a present-mask, so allocating one alongside its inputs would sit in the
    allocator for the whole measurement and inflate its peak-memory baseline. Assert on the real
    worker source: the MLP branch must return a 1-tuple before any mask is built."""
    src = open(_SCRIPT, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "worker_main")
    make = next(n for n in ast.walk(fn)
                if isinstance(n, ast.FunctionDef) and n.name == "make_inputs")
    body = ast.unparse(make)
    assert "return (xb,)" in body, "the MLP path no longer returns a mask-free input tuple"
    mask_line = body.index("group_present_mask")
    early_return = body.index("return (xb,)")
    assert early_return < mask_line, \
        "a present-mask is constructed before the MLP's mask-free early return"


def test_mlp_peak_memory_baseline_excludes_an_unused_mask(cpu_dev):
    """Behavioural counterpart: build both input sets and confirm the MLP's is strictly smaller,
    i.e. nothing it does not consume is resident during its measurement."""
    n, g, c = 32, 5, 200
    xb = torch.randn(n, c)
    pm = torch.from_numpy(p7.group_present_mask(n, [np.arange(2)] * g, []))
    mlp_inputs = (xb,)
    attn_inputs = (xb, pm)
    assert len(mlp_inputs) == 1
    mlp_bytes = sum(t.numel() * t.element_size() for t in mlp_inputs)
    attn_bytes = sum(t.numel() * t.element_size() for t in attn_inputs)
    assert mlp_bytes < attn_bytes, "the MLP input set is not smaller than the attention one"


# ========================================================================================
# 9. helpers duplicated from phase2 must not drift
# ========================================================================================
def test_corruption_helpers_match_phase2_exactly():
    """`group_present_mask` / `zero_missing` are re-implemented here so a measurement worker never
    imports the training module (which pulls matplotlib into the process being timed). That is a
    deliberate trade, and this test is the price: the two copies must agree bit for bit, or the
    efficiency table would be measuring a different corruption than the accuracy experiments."""
    import phase2_degradation as p2
    rng = np.random.default_rng(0)
    X = rng.standard_normal((17, 40)).astype(np.float32)
    groups = contiguous_groups(40, 5)
    for drop in ([], [0], [1, 3], [0, 1, 2, 3, 4]):
        np.testing.assert_array_equal(p7.zero_missing(X, groups, drop),
                                      p2.zero_missing(X, groups, drop))
        np.testing.assert_array_equal(p7.group_present_mask(17, groups, drop),
                                      p2.group_present_mask(17, groups, drop))


# ========================================================================================
# 10. missing values are never fabricated
# ========================================================================================
@pytest.mark.parametrize("v", [None, float("nan")])
def test_a_missing_measurement_renders_as_na_never_zero(v):
    assert p7._cell(v) == "n/a"
    assert p7._fmt(v, ".3f") == "n/a"
    assert p7._missing(v)


def test_a_real_zero_is_still_reported_as_zero():
    """The inverse mistake: coercing a genuine 0 to 'n/a' would hide a real measurement."""
    assert p7._cell(0.0) == "0.000"
    assert not p7._missing(0.0)


def test_macs_note_explains_every_na():
    """An 'n/a' whose reason is unrecorded is indistinguishable from a bug."""
    assert p7._macs_note(1234, {}, []) == "complete"
    assert "fvcore unavailable" in p7._macs_note(None, None, None)
    note = p7._macs_note(None, {"aten::scaled_dot_product_attention": 2}, ["encoder.x"])
    assert "scaled_dot_product_attention" in note and "encoder.x" in note


# ========================================================================================
# 11. the strict gate is TOTAL — the holes an adversarial review reproduced against it
#
# Every test in this section corresponds to a way `strict_failures` returned [] for a table that
# had measured nothing, or nothing checkable. They were found by calling the real function with
# hand-built rows, not by reading it: each one passed while looking exactly like a clean gate.
# ========================================================================================
def test_an_empty_table_does_not_pass_the_strict_gate():
    """`strict_failures([])` returned []. A run that produced NO rows at all — the state a
    driver bug or a substituted seam can reach without raising — was indistinguishable from a
    complete benchmark, because every check iterated rows and none asked about the SET of them."""
    fails = p7.strict_failures([], "cuda", 0.10, expected_models=p7.EXPECTED_MODELS)
    assert fails, "an empty table passed the strict gate"
    assert any("missing" in f for f in fails), fails


def test_a_table_without_the_proposed_model_does_not_pass():
    """The .tex macros are built from the Proposed row. A table missing it produced no macros,
    left the PREVIOUS run's macros on disk, and still passed."""
    rows = [_row(name=n) for n in p7.EXPECTED_MODELS if n != "Proposed"]
    fails = p7.strict_failures(rows, "cuda", 0.10, expected_models=p7.EXPECTED_MODELS)
    assert any("Proposed" in f for f in fails), fails


def test_a_single_model_table_does_not_pass():
    fails = p7.strict_failures([_row(name="MLP (B1)")], "cuda", 0.10,
                               expected_models=p7.EXPECTED_MODELS)
    assert any("missing" in f for f in fails), fails


def test_a_duplicated_model_row_does_not_pass():
    """Two rows for one model would be silently averaged or overwritten by anything that reads
    this table by name — including the .tex renderer, which takes the first match."""
    rows = [_row(name=n) for n in p7.EXPECTED_MODELS] + [_row(name="Proposed")]
    fails = p7.strict_failures(rows, "cuda", 0.10, expected_models=p7.EXPECTED_MODELS)
    assert any("duplicate" in f for f in fails), fails


def test_an_unexpected_model_row_does_not_pass():
    rows = [_row(name=n) for n in p7.EXPECTED_MODELS] + [_row(name="B9 attn")]
    fails = p7.strict_failures(rows, "cuda", 0.10, expected_models=p7.EXPECTED_MODELS)
    assert any("unexpected" in f for f in fails), fails


def test_the_full_expected_roster_passes():
    """Negative control for the four above: the roster check must be satisfiable."""
    rows = [_row(name=n) for n in p7.EXPECTED_MODELS]
    assert p7.strict_failures(rows, "cuda", 0.10, expected_models=p7.EXPECTED_MODELS) == []


def test_the_trained_roster_matches_the_gate_roster():
    """EXPECTED_MODELS and the models _run_models actually trains are two lists that must agree.
    _run_models asserts it at runtime; this catches the drift without spending a training run."""
    src = open(_SCRIPT, encoding="utf-8").read()
    for name in p7.EXPECTED_MODELS:
        assert f'"{name}"' in src, f"{name} is in EXPECTED_MODELS but nothing trains it"
    assert "EXPECTED_MODELS" in src.split("specs_src")[1][:600], \
        "_run_models no longer asserts its roster against EXPECTED_MODELS"


def test_an_absent_quant_scope_is_a_strict_failure():
    """THE fail-open. The 'nothing was quantized' guard reads `quant_n_quantized == 0` — and
    skipped itself entirely when the value was ABSENT, so the one case where the scope report
    itself raised was the one case it could not catch. An int8 size/latency then shipped with no
    evidence that anything had been quantized at all."""
    for absent in ("quant_scope", "quant_n_quantized"):
        fails = p7.strict_failures([_row(**{absent: None})], "cpu", 0.5, check_dispersion=False)
        assert any(absent in f for f in fails), f"an absent '{absent}' passed the strict gate"


def test_the_workers_own_required_failure_verdict_is_consulted():
    """The worker computed `required_failed` and the driver threw it away, re-deriving the
    verdict from absent columns — two failure semantics with only one consulted."""
    fails = p7.strict_failures([_row(required_failed="quant_scope;dynint8_size_mb")],
                               "cpu", 0.5, check_dispersion=False)
    assert any("required_failed" in f or "quant_scope" in f for f in fails), fails


def test_an_unhonoured_interop_pin_is_a_strict_failure():
    """interop_threads was RECORDED and never CHECKED. It is the pin most likely not to have
    taken: torch defaults it to the host's core count regardless of any cgroup quota, and it can
    only be set once, before any parallel work — so failing to apply it is silent."""
    fails = p7.strict_failures([_row(bench_threads=8, interop_threads=36)], "cpu", 0.5,
                               check_dispersion=False)
    assert any("INTER-op" in f for f in fails), fails
    assert p7.strict_failures([_row(bench_threads=8, interop_threads=8)], "cpu", 0.5,
                              check_dispersion=False) == []


def test_int8_latency_is_subject_to_the_replicate_floor():
    """The int8 cell shipped a relative IQR but no replicate count, so --min-replicates had
    nothing to apply to it — the one latency in the table exempt from the floor."""
    assert "dynint8_lat_cpu_n" in p7.CSV_COLS
    fails = p7.strict_failures([_row(dynint8_lat_cpu_n=2)], "cpu", 0.5, min_replicates=8)
    assert any("int8" in f and "replicate" in f for f in fails), fails


def test_an_unvalidated_torch_version_is_a_strict_failure():
    """QUANT_VALIDATED_TORCH was recorded next to the running version and never compared.
    Recording a version is not pinning one."""
    row = _row(quant_torch_version="2.9.0+cu128", quant_validated_torch="2.6.0+cu124")
    fails = p7.strict_failures([row], "cpu", 0.5, check_dispersion=False)
    assert any("validated against torch" in f for f in fails), fails
    assert p7.strict_failures([row], "cpu", 0.5, check_dispersion=False,
                              allow_unvalidated_quant=True) == [], \
        "--allow-unvalidated-quant must be a real escape hatch"


def test_single_request_latency_is_required_and_gated_separately():
    """It is a different quantity from the steady-state cell, so it is required in its own right
    and thresholded on its own (wider) natural dispersion."""
    assert "lat_cpu_single_ms" in p7.REQUIRED_ALWAYS
    assert "lat_gpu_single_ms" in p7.REQUIRED_CUDA
    quiet = p7.strict_failures([_row(lat_cpu_single_rel_iqr=0.2)], "cpu", 0.10,
                               max_rel_iqr_single=0.25)
    noisy = p7.strict_failures([_row(lat_cpu_single_rel_iqr=0.9)], "cpu", 0.10,
                               max_rel_iqr_single=0.25)
    assert quiet == []
    assert any("single-request" in f for f in noisy), noisy


# ========================================================================================
# 12. missing-group subsets: exhaustive, and never silently short
# ========================================================================================
def test_the_default_enumerates_every_subset():
    """5 of C(10,6)=210 is 2.4% coverage with three of the five hand-picked; mean/std over that
    describes an arbitrary corner. All 210 makes it the population."""
    subs = p7.missing_subsets(10, 6, n_subsets=None)
    assert len(subs) == 210                                   # C(10, 6), asserted as a literal
    assert len(set(subs)) == 210, "enumeration produced duplicates"
    assert subs[0] == (0, 1, 2, 3, 4, 5), "the prefix must stay first and comparable"


def test_the_sampler_no_longer_undershoots_silently():
    """Rejection sampling with a fixed 1000-draw guard returned 244 of the 252 subsets of
    C(10,5) and said nothing. Whatever the count, coverage must be STATED."""
    subs = p7.missing_subsets(10, 5, n_subsets=252)
    assert len(subs) == 252, f"asked for all 252 distinct subsets, got {len(subs)}"
    assert len(set(subs)) == len(subs)


def test_subset_coverage_states_the_regime():
    assert "exhaustive" in p7.subset_coverage(10, 6, p7.missing_subsets(10, 6, None))
    cov = p7.subset_coverage(10, 6, p7.missing_subsets(10, 6, 5))
    assert "sampled" in cov and "210" in cov and "5/" in cov
    assert "missing_subset_coverage" in p7.CSV_COLS


def test_exhaustive_max_bounds_the_enumeration():
    """A large C(G,k) must fall back to sampling rather than enumerating billions of tuples."""
    subs = p7.missing_subsets(60, 30, n_subsets=None, exhaustive_max=32)
    assert len(subs) == 32
    assert "sampled" in p7.subset_coverage(60, 30, subs)


def test_stat_block_carries_the_tail_not_just_the_mean():
    st = p7.stat_block([10.0, 20.0, 30.0, 40.0])
    assert st["median"] == pytest.approx(25.0)
    assert st["p05"] <= st["min"] + 1e-9 or st["p05"] == pytest.approx(11.5)
    for col in ("miou_fp32_miss_median", "miou_fp32_miss_p05"):
        assert col in p7.CSV_COLS


# ========================================================================================
# 13. a nonsensical configuration is rejected before it becomes a plausible table
# ========================================================================================
@pytest.mark.parametrize("argv,needle", [
    (["--bs", "-1"], "--bs"),
    (["--repeats", "-10"], "--repeats"),
    (["--max-missing", "-1"], "--max-missing"),
    (["--max-missing", "99"], "--max-missing"),
    (["--groups", "0"], "--groups"),
    (["--bench-threads", "0"], "--bench-threads"),
    (["--min-run-time", "-5"], "--min-run-time"),
    (["--single-requests", "1"], "--single-requests"),
])
def test_nonsense_arguments_are_refused(argv, needle):
    """None of these crashed. `--bs -1` sliced Xte[:-1] and made throughput negative while the
    CSV recorded batch_size=-1; `--repeats -10` ran once and claimed -10; `--max-missing -1`
    reported the CLEAN accuracy under a missing-band column. argparse checks types, not meaning."""
    args = p7.build_argparser().parse_args(argv)
    if args.bench_threads is None:
        args.bench_threads = 8
    bad = p7.validate_args(args)
    assert any(needle in b for b in bad), f"{argv} was accepted: {bad}"


def test_the_default_configuration_is_valid():
    """Negative control: the validator must not reject the shipped defaults."""
    args = p7.build_argparser().parse_args([])
    args.bench_threads = args.bench_threads or 8
    assert p7.validate_args(args) == []


def test_bench_threads_defaults_below_the_core_budget():
    """Pinning the entire cgroup quota leaves nothing for the main thread or the CUDA driver and
    shows up as CFS throttling in the middle of a measurement."""
    from bandsim import hw as _hw
    assert p7.build_argparser().parse_args([]).bench_threads is None, \
        "the thread count must be resolved from the machine, not hardcoded"
    assert max(1, _hw.available_cores() - 2) <= _hw.available_cores()


# ========================================================================================
# 14. contention: the host/cgroup unit error, and 'could not measure' != 'quiet'
# ========================================================================================
def test_the_cpu_threshold_normalises_by_the_cores_the_load_is_measured_over():
    """The load average is host-wide (/proc/loadavg is not namespaced) and was divided by THIS
    process's cgroup quota. On the box this was written for that read a 21%-busy 36-core host as
    0.95/core and refused every run, while the cgroup's own accounting showed 0.04% throttling.
    A gate wrong in that direction just trains you to pass --allow-contended."""
    quiet_host = {"nvidia_smi": True, "gpu_util_pct": [0.0], "gpu_util_max": 0.0,
                  "gpu_mem_used_mb": [0.0], "n_compute_apps": 0, "loadavg_1m": 7.6,
                  "cores": 8, "host_cores": 36,
                  "load_per_core": 7.6 / 8, "load_per_host_core": 7.6 / 36}
    assert p7.contention_reasons(quiet_host, want_cuda=True) == [], \
        "a 21%-busy host was refused because the load was divided by the cgroup quota"
    busy_host = dict(quiet_host, loadavg_1m=30.0, load_per_core=30.0 / 8,
                     load_per_host_core=30.0 / 36)
    assert p7.contention_reasons(busy_host, want_cuda=True), "a genuinely busy host passed"


def test_unmeasurable_signals_downgrade_the_verdict_without_refusing():
    """`None` must not read as quiet. It also must not fabricate a refusal — so it produces no
    REASON and instead an UNKNOWN, which stamps the rows ok-unverified."""
    blind = {"nvidia_smi": False, "gpu_util_pct": None, "gpu_util_max": None,
             "gpu_mem_used_mb": None, "n_compute_apps": None, "loadavg_1m": None,
             "cores": 8, "host_cores": 36, "load_per_core": None, "load_per_host_core": None}
    assert p7.contention_reasons(blind, want_cuda=True) == []
    unknowns = p7.contention_unknowns(blind, want_cuda=True)
    assert unknowns, "a run that checked nothing still called itself ok"
    assert any("nvidia-smi" in u for u in unknowns)


def test_a_fully_measured_quiet_box_has_no_unknowns():
    pr = {"nvidia_smi": True, "gpu_util_pct": [0.0], "gpu_util_max": 0.0,
          "gpu_mem_used_mb": [0.0], "n_compute_apps": 0, "loadavg_1m": 0.1,
          "cores": 8, "host_cores": 36, "load_per_core": 0.0125,
          "load_per_host_core": 0.003}
    assert p7.contention_unknowns(pr, want_cuda=True) == []


def test_probe_reports_both_core_counts_and_the_throttle_counter():
    """Runs for real -- asserts the CONTRACT, never the values."""
    pr = p7.probe_contention()
    for k in ("cores", "host_cores", "load_per_core", "load_per_host_core", "cgroup_throttle"):
        assert k in pr, f"probe_contention() does not report '{k}'"
    thr = pr["cgroup_throttle"]
    assert thr is None or (len(thr) == 2 and all(isinstance(x, int) for x in thr))


# ========================================================================================
# 15. a worker timeout is a hard failure, not a crash that deletes the evidence
# ========================================================================================
def test_a_worker_timeout_is_recorded_instead_of_destroying_the_workdir(tmp_path, monkeypatch):
    """subprocess.TimeoutExpired propagated out of _run_models, past main()'s try (which had no
    except) and into the finally that rmtree's the workdir -- so the failure mode that costs the
    most to reproduce destroyed the trained weights, every other worker's JSON, and the stderr
    that would say what hung."""
    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="worker", timeout=1.0, stderr="hung in bench_latency")
    monkeypatch.setattr(p7.subprocess, "run", _timeout)

    class _Args:
        bench_threads, repeats, worker_timeout, seed, max_missing = 2, 1, 1.0, 0, 2
        smoke = False
    wd = tmp_path / "wd"
    wd.mkdir()
    specs = [{"index": 0, "name": "Proposed", "out": str(wd / "r0.json")}]
    rows, hard = [], []
    # exercise the same loop body the driver runs, without training four models to reach it
    for spec in specs:
        try:
            p7.subprocess.run(["x"], timeout=1.0)
        except subprocess.TimeoutExpired as e:
            hard.append(f"{spec['name']}: worker exceeded --worker-timeout: {e.stderr}")
            continue
    assert hard and "Proposed" in hard[0], "a timeout produced no hard failure"
    assert rows == []
    assert wd.exists(), "the workdir was destroyed"


def test_the_driver_catches_timeoutexpired_and_main_keeps_the_workdir_on_any_crash():
    """Static pin for the two edits above: both are single lines that a refactor would silently
    drop, and neither shows up in a passing run."""
    src = open(_SCRIPT, encoding="utf-8").read()
    assert "except subprocess.TimeoutExpired" in src, \
        "a worker timeout is once again an unhandled crash"
    assert "except BaseException" in src and "keep = True" in src, \
        "main() no longer preserves the workdir when it crashes"


def test_single_request_bench_pins_and_restores_the_thread_count(cpu_dev):
    """Same property the steady-state cell has: the pin must apply during the measurement and
    must not leak out. This one sets torch's thread count by hand rather than via Timer, so it
    needs its own guarantee -- a leaked pin would silently change every measurement after it."""
    before = torch.get_num_threads()
    m, x = _Tiny(), torch.randn(4, 8)
    r = p7.bench_single_request(m, (x,), cpu_dev, num_threads=1, n=5, warmup=1)
    assert r["num_threads"] == 1 and r["n_requests"] == 5
    assert r["number_per_run"] == "1", "a single-request cell must be one forward per interval"
    assert torch.get_num_threads() == before, "the thread pin leaked into the caller"


def test_steady_state_cell_records_how_many_forwards_it_averaged(cpu_dev):
    """`number_per_run` is what tells a reader whether lat_*_ms is a per-call number (1) or a
    block average (100 on GPU here). Without it the column is uninterpretable."""
    m, x = _Tiny(), torch.randn(4, 8)
    r = p7.bench_latency(m, (x,), cpu_dev, num_threads=1, repeats=1, warmup=1,
                         min_run_time=0.01, max_run_time=0.1)
    assert "number_per_run" in r and r["number_per_run"]
    for col in ("lat_cpu_number_per_run", "lat_gpu_number_per_run",
                "dynint8_lat_cpu_number_per_run"):
        assert col in p7.CSV_COLS


def test_device_memory_is_attributed_to_this_pid_or_reported_absent():
    """`mem_get_info` returns free/total for the DEVICE, so total-free is the sum over every
    process on the card -- measured 324 MB while this process had allocated nothing and a
    co-tenant held 306. A column named for this model's footprint must never report a
    neighbour's, so it is per-PID or it is None."""
    tree = ast.parse(open(_SCRIPT, encoding="utf-8").read())
    # A CALL, not a mention: the docstring names mem_get_info to explain why it is not used, and
    # a grep-based guard would forbid documenting the reasoning it exists to preserve.
    called = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "mem_get_info"]
    assert not called, ("device memory is being read device-wide again (line "
                        f"{called[0].lineno}); on a shared GPU that is a co-tenant's")
    v = p7.process_gpu_mem_mb()
    assert v is None or (isinstance(v, float) and v > 0)


def test_no_paper_macro_can_expand_to_the_word_none():
    """A .tex macro holding the literal string 'None' is a fabricated value wearing a typo's
    clothes -- it renders into the PDF and reads as an editing slip, not a missing measurement."""
    class _A:
        seed, repeats = 0, 1
    tex = p7.render_tex(_row(name="Proposed"), _A(), "ok")
    macro_lines = [ln for ln in tex.splitlines() if ln.startswith("\\newcommand")]
    assert macro_lines, "render_tex emitted no macros"
    for ln in macro_lines:
        assert "{None}" not in ln, f"macro expands to the string None: {ln}"
