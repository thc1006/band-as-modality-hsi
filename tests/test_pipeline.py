"""Orchestrator (pipeline.simulate) composition tests — guards the config-driven B->C->A->D chain
and the fail-closed YAML front-end, not the individual designs (those have their own tests).
Roadmap docs/guide/03 §3.

NOTHING HERE SKIPS. The 6S table is gitignored, so every test that needs a transmittance LUT builds
a synthetic one in tmp_path instead of gating on the real file. The previous version guarded its
atmosphere tests with `skipif(not os.path.exists(_TABLE))`, which meant Design B was silently
untested on any machine without the precomputed table — including CI — while the suite still
reported green. `_synthetic_table` gives Design B a LUT unconditionally; the one test that asserts
something about the SHIPPED table is the only one allowed to skip, and only when it is truly absent.

Wavelength-axis identity of the LUT (wrong axis / reversed / same-length-but-different) is covered
in depth by tests/test_transmittance_axis_guard.py; this file keeps one end-to-end axis rejection so
a regression in the pipeline's threading of that guard fails here too.
"""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandsim.pipeline import simulate
from bandsim.srf import (gaussian_srf, SENTINEL2_MSI_CENTERS_NM, LANDSAT8_OLI_CENTERS_NM)
from bandsim.io import AVIRIS_WL_NM

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TABLE = os.path.join(_REPO, "data", "srf_cache", "T_6s_grid.npz")
_CFG_DIR = os.path.join(_REPO, "configs")

# The key contract after the AOD axis was dropped: CWV alone, float()-normalised. "cwv2.0_aod0.1"
# was the pre-2026-07 form and must not reappear.
_KEY = "cwv2.0"


def _srf_cfg(centers=SENTINEL2_MSI_CENTERS_NM):
    """Gaussian SRF with the REAL band names attached (B8A/B9/B11/B12 survive, not auto B1..BN)."""
    return gaussian_srf(AVIRIS_WL_NM, list(centers.values()), fwhm_nm=30.0, names=list(centers.keys()))


def _synthetic_table(tmp_path, wl_nm=AVIRIS_WL_NM, key=_KEY, name="T.npz"):
    """A stand-in 6S LUT: physical T in [0,1] on a stated axis. Lets Design B run everywhere.

    T dips hard around 1400 nm so an atmosphere-on run is unmistakably darker than atmosphere-off,
    without pretending to be 6S output."""
    wl = np.asarray(wl_nm, float)
    T = np.clip(1.0 - 0.8 * np.exp(-0.5 * ((wl - 1400.0) / 120.0) ** 2), 0.0, 1.0)
    path = os.path.join(str(tmp_path), name)
    np.savez_compressed(path, wl_nm=wl, **{key: T})
    return path, T


def test_design_A_maps_hsi_to_sensor_bands():
    cube = np.random.default_rng(0).random((6, 5, 200))
    cfg = {"seed": 0, "A": {"enable": True, "srf": _srf_cfg()}}
    out, info = simulate(cube, AVIRIS_WL_NM, cfg)
    assert out.shape == (6, 5, 12)                 # HSI 200 -> S2 12 bands
    assert len(info["band_names"]) == 12
    assert np.isfinite(out).all()


def test_cirrus_stage_composes_before_srf():
    cube = np.ones((4, 4, 200))
    base = simulate(cube, AVIRIS_WL_NM, {"A": {"enable": True, "srf": _srf_cfg()}})[0]
    withc = simulate(cube, AVIRIS_WL_NM,
                     {"A": {"enable": True, "srf": _srf_cfg()}, "C": {"enable": True, "tau": 0.6}})[0]
    assert withc.shape == base.shape               # still S2 bands after composition
    assert not np.allclose(withc, base)            # cirrus changed the result


def test_pipeline_is_seeded_reproducible():
    cube = np.random.default_rng(1).random((5, 5, 200))
    cfg = {"seed": 7, "A": {"enable": True, "srf": _srf_cfg()},
           "D": {"enable": True, "stripe_eps": 0.05, "dead_col_frac": 0.02}}
    a = simulate(cube, AVIRIS_WL_NM, cfg)[0]
    b = simulate(cube, AVIRIS_WL_NM, cfg)[0]       # same seed -> identical
    assert np.array_equal(a, b)


# ===================== Design B: always exercised, never skipped =====================

def test_atmosphere_stage_suppresses_signal_before_srf(tmp_path):
    table, _ = _synthetic_table(tmp_path)
    cube = np.ones((4, 4, 200))
    plain = simulate(cube, AVIRIS_WL_NM, {"A": {"enable": True, "srf": _srf_cfg()}})[0]
    atmos = simulate(cube, AVIRIS_WL_NM, {
        "A": {"enable": True, "srf": _srf_cfg()},
        "B": {"enable": True, "hard_mask_cores": True,
              "cache_npz": table, "cache_key": _KEY}})[0]
    # atmosphere (T<=1) + hard-mask can only reduce integrated band reflectance
    assert (atmos <= plain + 1e-6).all()
    assert atmos.mean() < plain.mean()


def test_full_BCAD_runs_in_documented_order(tmp_path):
    """All four stages, and the ORDER is asserted rather than assumed.

    B and C act on the hyperspectral axis, A collapses it to sensor bands, D then acts on those
    bands. The order is pinned by reconstructing it stagewise and requiring bit-equality: any
    reshuffle (e.g. SRF before atmosphere) changes the numbers, because SRF is a weighted average
    and does not commute with a wavelength-dependent transmittance."""
    from bandsim import atmosphere as _at, cirrus as _ci, noise as _no
    from bandsim.srf import build_resampling_matrix, apply_srf

    table, T = _synthetic_table(tmp_path)
    cube = np.random.default_rng(2).random((6, 6, 200))
    cfg = {"seed": 3,
           "A": {"enable": True, "srf": _srf_cfg(LANDSAT8_OLI_CENTERS_NM)},
           "B": {"enable": True, "hard_mask_cores": True, "cache_npz": table, "cache_key": _KEY},
           "C": {"enable": True, "tau": 0.3},
           "D": {"enable": True, "stripe_eps": 0.02, "dead_col_frac": 0.5}}
    out, info = simulate(cube, AVIRIS_WL_NM, cfg)
    assert out.shape == (6, 6, 7) and np.isfinite(out).all()          # OLI band count survives D
    assert info["band_names"] == list(LANDSAT8_OLI_CENTERS_NM)        # exact OLI identity, B1..B7

    wl = np.asarray(AVIRIS_WL_NM, float)
    rng = np.random.default_rng(3)                                    # same seed the pipeline uses
    x = _at.apply_atmosphere(cube, T, keep_mask=_at.hard_mask_absorption_cores(wl))   # B
    x = _ci.apply_cirrus(x, wl, tau=0.3)                                              # C
    W, _n = build_resampling_matrix(wl, _srf_cfg(LANDSAT8_OLI_CENTERS_NM))
    x = apply_srf(x, W)                                                               # A
    x = _no.add_band_noise(x, _no.hyperion_like_snr(W @ wl), rng)                     # D
    x = _no.add_striping(x, rng, 0.02, 0.5)
    assert np.array_equal(out, x), "pipeline no longer runs B -> C -> A -> D"


def test_design_B_records_the_cwv_only_cache_key(tmp_path):
    # Contract: the LUT is keyed by CWV alone (6S gaseous transmittance is aerosol-independent, so
    # the old "cwv2.0_aod0.1" form encoded an inert factor). A spec asking for cwv2.0 must hit
    # exactly that array, and the retired composite key must not resolve.
    from bandsim.config_runner import build_cfg
    table, T = _synthetic_table(tmp_path, key=_KEY)
    spec = {"input": {"wavelength_axis": "aviris"},
            "designs": {"B": {"enable": True, "cwv_g_cm2": 2.0, "table": table}}}
    cfg = build_cfg(spec, AVIRIS_WL_NM)
    assert cfg["B"]["cache_key"] == _KEY == "cwv2.0"
    assert "aod" not in cfg["B"]["cache_key"]
    # integer 2 and float 2.0 must normalise to the SAME key, not "cwv2"
    spec["designs"]["B"]["cwv_g_cm2"] = 2
    assert build_cfg(spec, AVIRIS_WL_NM)["B"]["cache_key"] == _KEY

    cube = np.ones((3, 3, 200))
    out, _ = simulate(cube, AVIRIS_WL_NM, dict(cfg, A={"enable": True, "srf": _srf_cfg()}))
    assert np.isfinite(out).all()
    with pytest.raises(KeyError, match="cwv2.0_aod0.1"):
        simulate(cube, AVIRIS_WL_NM, {"B": dict(cfg["B"], cache_key="cwv2.0_aod0.1")})


def test_wrong_axis_lut_is_rejected_end_to_end(tmp_path):
    # Same length, different axis (gapless linspace vs the real gapped AVIRIS axis) -> must raise.
    # Depth coverage lives in tests/test_transmittance_axis_guard.py; this pins the pipeline wiring.
    bad, _ = _synthetic_table(tmp_path, wl_nm=np.linspace(400.0, 2500.0, 200), name="bad.npz")
    assert len(np.linspace(400.0, 2500.0, 200)) == len(AVIRIS_WL_NM)
    cfg = {"A": {"enable": True, "srf": _srf_cfg()},
           "B": {"enable": True, "cache_npz": bad, "cache_key": _KEY}}
    with pytest.raises(ValueError, match="wavelength-axis MISMATCH"):
        simulate(np.ones((3, 3, 200)), AVIRIS_WL_NM, cfg)


def test_reversed_axis_lut_is_rejected_end_to_end(tmp_path):
    rev, _ = _synthetic_table(tmp_path, wl_nm=np.asarray(AVIRIS_WL_NM, float)[::-1], name="rev.npz")
    cfg = {"A": {"enable": True, "srf": _srf_cfg()},
           "B": {"enable": True, "cache_npz": rev, "cache_key": _KEY}}
    with pytest.raises(ValueError, match="wavelength-axis MISMATCH"):
        simulate(np.ones((3, 3, 200)), AVIRIS_WL_NM, cfg)


# ===================== band identity survives the whole chain =====================

def test_exact_sentinel2_and_oli_band_names_survive(tmp_path):
    # The band SET is the physical content of Design A, so the names must be the real ones, not
    # auto-numbered: auto-naming maps S2 B8A->B9 and B9->B10 and silently relabels the sensor.
    table, _ = _synthetic_table(tmp_path)
    for centers, expect in [(SENTINEL2_MSI_CENTERS_NM,
                             ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]),
                            (LANDSAT8_OLI_CENTERS_NM,
                             ["B1", "B2", "B3", "B4", "B5", "B6", "B7"])]:
        cfg = {"seed": 0, "A": {"enable": True, "srf": _srf_cfg(centers)},
               "B": {"enable": True, "cache_npz": table, "cache_key": _KEY}}
        _, info = simulate(np.ones((3, 3, 200)), AVIRIS_WL_NM, cfg)
        assert info["band_names"] == expect
    # OLI genuinely lacks S2's red-edge and narrow water-vapour bands (the "missing band" premise)
    assert {"B8A"} <= set(SENTINEL2_MSI_CENTERS_NM) and "B8A" not in LANDSAT8_OLI_CENTERS_NM


def test_striping_records_realised_dead_columns():
    # dead_col_frac is an EXPECTED fraction (Bernoulli per column), so a run must report what
    # actually happened -- at 145 columns and frac=0.01 about 23% of seeds produce NO dead column.
    cfg = {"seed": 5, "A": {"enable": True, "srf": _srf_cfg()},
           "D": {"enable": True, "stripe_eps": 0.02, "dead_col_frac": 1.0}}
    out, info = simulate(np.ones((4, 9, 200)), AVIRIS_WL_NM, cfg)
    s = info["striping"]
    assert s["n_cols"] == 9 and s["dead_col_count"] == 9          # frac=1.0 -> every column dead
    assert s["dead_col_indices"] == list(range(9)) and s["dead_col_frac_realised"] == 1.0
    assert np.allclose(out, 0.0)

    _, info0 = simulate(np.ones((4, 9, 200)), AVIRIS_WL_NM,
                        dict(cfg, D={"enable": True, "stripe_eps": 0.02, "dead_col_frac": 0.0}))
    assert info0["striping"]["dead_col_count"] == 0 and info0["striping"]["dead_col_indices"] == []


# ===================== YAML front-end: fail-closed =====================

def test_config_driven_sim_a_runs_without_table():
    # sim_A_s2_vs_oli.yaml is Design-A-only (B/C/D disabled) so it runs with NO 6S table.
    from bandsim.config_runner import load_spec, run_spec
    cube = np.random.default_rng(9).random((5, 5, 200))
    spec = load_spec(os.path.join(_CFG_DIR, "sim_A_s2_vs_oli.yaml"))
    out, info = run_spec(cube, spec)
    assert out.shape == (5, 5, 12) and np.isfinite(out).all()
    assert "B8A" in info["band_names"] and "B9" in info["band_names"]   # real S2 identity preserved


def test_shipped_yaml_specs_run_end_to_end():
    """The shipped specs drive the simulator (load_spec -> build_cfg -> simulate).

    Skips only if the real 6S table is genuinely absent (it is gitignored); a PRESENT table that
    fails is a hard failure, never a skip."""
    from bandsim.config_runner import load_spec, run_spec
    if not os.path.exists(_TABLE):
        pytest.skip("6S table not present on this machine")
    cube = np.random.default_rng(4).random((5, 5, 200))
    for name, nbands in [("sim_AB_atmos.yaml", 12), ("sim_ABCD_full.yaml", 7)]:
        spec = load_spec(os.path.join(_CFG_DIR, name))
        out, info = run_spec(cube, spec)
        assert out.shape == (5, 5, nbands) and np.isfinite(out).all()
        assert len(info["band_names"]) == nbands


def test_yaml_table_path_resolves_against_the_spec_not_the_cwd(tmp_path, monkeypatch):
    # Resolving `table:` against the process CWD made the same spec mean different things depending
    # on where python was launched, and a stray file under the CWD would be read INSTEAD of the
    # repo's table -- a silent atmosphere swap. The path must follow the YAML file.
    from bandsim.config_runner import load_spec
    monkeypatch.chdir(tmp_path)
    spec = load_spec(os.path.join(_CFG_DIR, "sim_AB_atmos.yaml"))
    table = spec["designs"]["B"]["table"]
    assert os.path.isabs(table)
    assert os.path.normpath(table) == os.path.normpath(_TABLE)


def test_unknown_yaml_key_is_rejected_not_ignored():
    # THE regression: `hard_mask_core` (missing 's') used to be dropped in silence while Design B
    # ran with the DEFAULT hard_mask_cores=True -- the author believed the config had taken effect.
    from bandsim.config_runner import build_cfg, validate_spec
    base = {"input": {"wavelength_axis": "aviris"},
            "designs": {"B": {"enable": True, "cwv_g_cm2": 2.0, "table": "t.npz",
                              "hard_mask_core": False}}}
    with pytest.raises(ValueError, match="unknown key.*hard_mask_core"):
        build_cfg(base, AVIRIS_WL_NM)
    for spec, pat in [
        ({"input": {"wavelength_axis": "aviris"}, "desgins": {}}, "unknown key.*desgins"),
        ({"input": {"wavelength_axis": "aviris"}, "seeeed": 1}, "unknown key.*seeeed"),
        ({"input": {"wavelength_axis": "aviris", "wavelength_axes": "aviris"}}, "spec.input"),
        ({"input": {"wavelength_axis": "aviris"},
          "designs": {"A": {"enable": True, "target_sensor": "sentinel2", "fwhm": 30.0}}}, "fwhm"),
        ({"input": {"wavelength_axis": "aviris"}, "eval": {"seedz": 5}}, "spec.eval"),
    ]:
        with pytest.raises(ValueError, match=pat):
            validate_spec(spec, require_provenance=False)


def test_out_of_range_and_non_finite_parameters_are_rejected():
    from bandsim.config_runner import validate_spec
    bad = [
        ({"D": {"enable": True, "stripe_eps": -0.2}}, "stripe_eps"),        # a negative std
        ({"D": {"enable": True, "stripe_eps": float("nan")}}, "finite"),
        ({"D": {"enable": True, "stripe_eps": float("inf")}}, "finite"),
        ({"D": {"enable": True, "dead_col_frac": 1.5}}, "dead_col_frac"),
        ({"D": {"enable": True, "dead_col_frac": -0.01}}, "dead_col_frac"),
        ({"C": {"enable": True, "tau": -1.0}}, "tau"),
        ({"B": {"enable": True, "cwv_g_cm2": -1.0, "table": "t.npz"}}, "cwv_g_cm2"),
        ({"A": {"enable": True, "target_sensor": "sentinel2", "fwhm_nm": 0.0}}, "fwhm_nm"),
    ]
    for designs, pat in bad:
        with pytest.raises(ValueError, match=pat):
            validate_spec({"input": {"wavelength_axis": "aviris"}, "designs": designs},
                          require_provenance=False)
    # wrong TYPES must not be coerced: a string is truthy, so "false" would enable the stage.
    # B carries cwv_g_cm2 here because it is REQUIRED once B is enabled -- without it the spec fails
    # on the missing field first and this case would silently stop testing the string-bool at all.
    for designs in [{"B": {"enable": True, "cwv_g_cm2": 2.0, "hard_mask_cores": "false",
                           "table": "t.npz"}},
                    {"A": {"enable": "yes", "target_sensor": "sentinel2"}}]:
        with pytest.raises(TypeError):
            validate_spec({"input": {"wavelength_axis": "aviris"}, "designs": designs},
                          require_provenance=False)


def test_unknown_enum_values_are_rejected():
    from bandsim.config_runner import validate_spec
    for designs, pat in [
        ({"A": {"enable": True, "target_sensor": "sentinel3"}}, "target_sensor"),
        # only gaussian is wired into the YAML path; 'pyspectral' must fail, not be downgraded
        ({"A": {"enable": True, "target_sensor": "sentinel2", "srf_source": "pyspectral"}}, "srf_source"),
        ({"C": {"enable": True, "validation_status": "totally_validated"}}, "validation_status"),
    ]:
        with pytest.raises(ValueError, match=pat):
            validate_spec({"input": {"wavelength_axis": "aviris"}, "designs": designs},
                          require_provenance=False)
    with pytest.raises(ValueError, match="wavelength_axis"):
        validate_spec({"input": {"wavelength_axis": "hyperion"}}, require_provenance=False)


def test_yaml_files_must_declare_schema_version_and_claim_scope(tmp_path):
    # A spec read off disk is a durable artefact that gets cited, so it must state which schema it
    # was written against and what it is allowed to support. A bare `yaml.safe_load` asserted neither.
    import yaml
    from bandsim.config_runner import load_spec, SCHEMA_VERSION
    full = {"schema_version": SCHEMA_VERSION, "claim_scope": "band_set_geometry", "seed": 0,
            "input": {"wavelength_axis": "aviris"},
            "designs": {"A": {"enable": True, "target_sensor": "sentinel2"}}}

    def _write(d, name):
        p = os.path.join(str(tmp_path), name)
        with open(p, "w") as f:
            yaml.safe_dump(d, f)
        return p

    assert load_spec(_write(full, "ok.yaml"))["claim_scope"] == "band_set_geometry"
    for drop, pat in [("schema_version", "schema_version"), ("claim_scope", "claim_scope")]:
        d = {k: v for k, v in full.items() if k != drop}
        with pytest.raises(ValueError, match=pat):
            load_spec(_write(d, f"no_{drop}.yaml"))
    with pytest.raises(ValueError, match="schema_version"):
        load_spec(_write(dict(full, schema_version=SCHEMA_VERSION + 1), "future.yaml"))
    with pytest.raises(ValueError, match="claim_scope"):
        load_spec(_write(dict(full, claim_scope="peer_reviewed_physics"), "overclaim.yaml"))


def test_validation_status_travels_with_the_numbers():
    # The honesty label used to live only in a YAML comment, so it vanished the moment a caller held
    # the array. simulate() must return it alongside the cube.
    from bandsim.config_runner import load_spec, run_spec
    if not os.path.exists(_TABLE):
        pytest.skip("6S table not present on this machine")
    spec = load_spec(os.path.join(_CFG_DIR, "sim_ABCD_full.yaml"))
    _, info = run_spec(np.random.default_rng(0).random((4, 4, 200)), spec)
    assert info["claim_scope"] == "robustness_stress_test"
    assert info["validation_status"] == {"A": "first_order_approximation",
                                         "B": "physics_6s_gaseous",
                                         "C": "schematic_illustrative_only",
                                         "D": "schematic_uncalibrated"}
