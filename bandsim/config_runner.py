"""Config-driven front-end for the physical SIMULATOR: YAML spec -> simulate() cfg -> run.

SCOPE (read this before trusting a YAML): this front-end is a band-loss SIMULATION driver, NOT a
full experiment runner. `run_spec(cube, spec)` consumes EXACTLY three things from the YAML:
  * `seed`                     -> simulate() RNG seed
  * `input.wavelength_axis`    -> the named sensor axis (bandsim.io.AVIRIS_WL_NM / ROSIS_WL_NM)
  * `designs.{A,B,C,D}`        -> the SRF / 6S / cirrus / noise stages
Everything ELSE that appears in the example YAMLs is DECLARATION / PROVENANCE and is NOT executed
here: `input.dataset/path/gt_path` document which cube the CALLER must load and pass in (run_spec
takes the cube as an argument -- it does not read the file), and the whole `eval` block (seeds /
split / metrics) is run by the experiment scripts (experiments/phaseX.py), not by this module.
`validate_scope()` warns if a spec carries those unconsumed keys so the config can never silently
imply it drives data loading or evaluation when it does not.

FAIL-CLOSED. Every key is checked against an explicit whitelist at every level and every value is
range/type/finiteness checked, because the previous `.get()`-everywhere front-end failed OPEN: a
one-character typo silently reverted the stage to its default while the author believed the config
had taken effect. `hard_mask_core: false` (singular) was accepted in silence and Design B ran with
hard_mask_cores=True -- an unnoticed no-op edit, not an error. Unknown keys are now rejected rather
than ignored, so a config either means what it says or refuses to run.
"""
from __future__ import annotations
import os
import warnings
import numpy as np

from .srf import gaussian_srf, SENTINEL2_MSI_CENTERS_NM, LANDSAT8_OLI_CENTERS_NM
from .io import AVIRIS_WL_NM, ROSIS_WL_NM
from .pipeline import simulate

_SENSORS = {"sentinel2": SENTINEL2_MSI_CENTERS_NM, "landsat_oli": LANDSAT8_OLI_CENTERS_NM}
_AXES = {"aviris": AVIRIS_WL_NM, "rosis": ROSIS_WL_NM}

# Keys run_spec() actually consumes (see module docstring). Anything else is provenance/declaration.
_CONSUMED_INPUT = {"wavelength_axis"}
_UNCONSUMED_INPUT = {"dataset", "path", "gt_path"}

# Bumped whenever the accepted key set or a value's meaning changes, so an old YAML fails loudly
# instead of being reinterpreted under new semantics.
SCHEMA_VERSION = 1

# What a spec is allowed to claim. Deliberately a closed vocabulary: a free-text scope would let a
# schematic run label itself anything, which is the failure mode this field exists to prevent.
CLAIM_SCOPES = {
    "band_set_geometry":       "which bands EXIST cross-sensor (Design A band set); SRF SHAPE is approximate",
    "gaseous_absorption":      "+ real 6S gaseous transmittance (Design B) on the cube's own axis",
    "robustness_stress_test":  "includes SCHEMATIC stages (C cirrus / D noise): robustness ablation only",
    "demo_only":               "illustrative; not evidence for any claim in the paper",
}
# Per-design honesty label, propagated into simulate()'s info so it travels with the numbers.
VALIDATION_STATUSES = {
    "measured_srf",                    # pyspectral / ESA / USGS measured response
    "first_order_approximation",       # e.g. a single Gaussian FWHM standing in for a measured SRF
    "physics_6s_gaseous",              # real 6S transmittance_global_gas on the cube's axis
    "schematic_illustrative_only",     # hand-crafted shape, no physical validation (Design C)
    "schematic_uncalibrated",          # standard-form model, not calibrated to a sensor (Design D)
}


def derived_validation_status(design, block):
    """The ONLY honest status for this design AS IT WILL ACTUALLY EXECUTE.

    A closed vocabulary stops free text but not a lie: it was possible to enable only the schematic
    Design D, label it `physics_6s_gaseous`, and have that label ride into `info` beside the numbers.
    The whole point of propagating the label is that a reader can trust it, so it is DERIVED from the
    execution path and any value written in the YAML is only allowed to AGREE with it."""
    if design == "A":
        return "measured_srf" if block.get("srf_source") == "pyspectral" else "first_order_approximation"
    return {"B": "physics_6s_gaseous",
            "C": "schematic_illustrative_only",
            "D": "schematic_uncalibrated"}[design]


def allowed_claim_scopes(designs):
    """Scopes the ENABLED stage combination can actually support.

    Under-claiming is always allowed (`demo_only` is legal for anything); over-claiming is not. Once
    a schematic stage runs, no amount of real physics elsewhere makes the RESULT physics-grounded,
    so C or D being on collapses the permissible scopes to the stress-test/demo pair."""
    on = {k for k, v in (designs or {}).items() if (v or {}).get("enable")}
    if on & {"C", "D"}:
        return {"robustness_stress_test", "demo_only"}
    if "B" in on:
        return {"gaseous_absorption", "band_set_geometry", "demo_only"}
    if "A" in on:
        return {"band_set_geometry", "demo_only"}
    return {"demo_only"}

_ALLOWED_TOP = {"schema_version", "claim_scope", "seed", "input", "designs", "eval"}
_ALLOWED_INPUT = _CONSUMED_INPUT | _UNCONSUMED_INPUT
_ALLOWED_EVAL = {"seeds", "split", "metrics"}
_ALLOWED_DESIGN = {
    "A": {"enable", "target_sensor", "srf_source", "fwhm_nm", "validation_status"},
    "B": {"enable", "cwv_g_cm2", "hard_mask_cores", "table", "validation_status"},
    "C": {"enable", "tau", "validation_status"},
    "D": {"enable", "stripe_eps", "dead_col_frac", "validation_status"},
}
# Only 'gaussian' is wired into build_cfg below; 'pyspectral' is reachable from Python
# (bandsim.srf.pyspectral_srf) but NOT from a YAML, so the YAML must not advertise it.
_SRF_SOURCES = {"gaussian"}


def _reject_unknown(got, allowed, where):
    """Unknown keys are errors, never ignored -- see the module docstring's typo case."""
    if not isinstance(got, dict):
        raise TypeError(f"{where} must be a mapping, got {type(got).__name__}")
    stray = set(got) - set(allowed)
    if stray:
        raise ValueError(f"unknown key(s) {sorted(stray)} in {where}; allowed: {sorted(allowed)}. "
                         f"A typo here would otherwise be ignored and the DEFAULT silently used.")


def _num(d, key, where, lo=None, hi=None, allow_none=True):
    """Finite real in [lo, hi]. bool is excluded explicitly: it is an int subclass, so `enable: true`
    landing in a numeric slot would otherwise validate as the number 1."""
    if key not in d:
        if allow_none:
            return None
        raise ValueError(f"{where}.{key} is required")
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{where}.{key} must be a number, got {type(v).__name__} ({v!r})")
    v = float(v)
    if not np.isfinite(v):
        raise ValueError(f"{where}.{key} must be finite, got {v!r}")
    if lo is not None and v < lo:
        raise ValueError(f"{where}.{key} must be >= {lo}, got {v}")
    if hi is not None and v > hi:
        raise ValueError(f"{where}.{key} must be <= {hi}, got {v}")
    return v


def _flag(d, key, where, default=None):
    """Strict bool: `hard_mask_cores: "false"` (a string) is truthy in Python and would enable the
    very thing the author was switching off."""
    if key not in d:
        return default
    v = d[key]
    if not isinstance(v, bool):
        raise TypeError(f"{where}.{key} must be true/false, got {type(v).__name__} ({v!r})")
    return v


def _choice(d, key, where, options, required=False, default=None):
    if key not in d:
        if required:
            raise ValueError(f"{where}.{key} is required (one of {sorted(options)})")
        return default
    v = d[key]
    if v not in options:
        raise ValueError(f"{where}.{key} = {v!r} is not one of {sorted(options)}")
    return v


def validate_spec(spec, require_provenance=True):
    """Fail-closed structural + value validation of a spec dict. Raises on anything unrecognised.

    `require_provenance` demands `schema_version` and `claim_scope`. It is True for YAML files
    (load_spec) because a file is a durable artefact that will be cited, and False for dicts built
    inline in Python (build_cfg's own entry check), where the calling code IS the provenance and
    demanding a version header would only add ceremony. Key/type/range checking is identical in
    both modes -- the typo and out-of-range defences are never relaxed."""
    _reject_unknown(spec, _ALLOWED_TOP, "spec")

    if require_provenance:
        if "schema_version" not in spec:
            raise ValueError(f"spec.schema_version is required (this build accepts {SCHEMA_VERSION})")
        # `is True` is not `== 1`, but Python says otherwise: bool subclasses int, so a stray
        # `schema_version: true` compares equal to version 1 and sails through.
        if isinstance(spec["schema_version"], bool) or not isinstance(spec["schema_version"], int):
            raise ValueError(f"spec.schema_version must be an int, got "
                             f"{type(spec['schema_version']).__name__} {spec['schema_version']!r}")
        if spec["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"spec.schema_version = {spec['schema_version']!r} but this build of "
                             f"config_runner accepts {SCHEMA_VERSION}; the accepted key set or a "
                             f"value's meaning has changed -- port the config, do not reinterpret it.")
        _choice(spec, "claim_scope", "spec", CLAIM_SCOPES, required=True)
    else:
        _choice(spec, "claim_scope", "spec", CLAIM_SCOPES)

    # Integer, not "any finite real": `seed: 1.9` used to validate and then be truncated to 1 by
    # int() later, so the file said one thing and the RNG did another.
    if "seed" in spec:
        s = spec["seed"]
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise ValueError(f"spec.seed must be a non-negative int, got "
                             f"{type(s).__name__} {s!r} (a float would be silently truncated)")
    # The scope may not exceed what the enabled stages can support (under-claiming is fine).
    scope = spec.get("claim_scope")
    if scope is not None:
        ok = allowed_claim_scopes(spec.get("designs"))
        if scope not in ok:
            on = sorted(k for k, v in (spec.get("designs") or {}).items() if (v or {}).get("enable"))
            raise ValueError(
                f"spec.claim_scope={scope!r} overstates what stages {on or ['(none)']} can support. "
                f"Permitted here: {sorted(ok)}. A schematic stage anywhere in the chain caps what the "
                f"RESULT may be called, however real the other stages are.")

    inp = spec.get("input") or {}
    _reject_unknown(inp, _ALLOWED_INPUT, "spec.input")
    _choice(inp, "wavelength_axis", "spec.input", _AXES, required=True)

    if "eval" in spec and spec["eval"]:
        # Declaration only (experiments/phaseX.py runs it), but a typo here still misleads a reader.
        _reject_unknown(spec["eval"], _ALLOWED_EVAL, "spec.eval")

    designs = spec.get("designs") or {}
    _reject_unknown(designs, _ALLOWED_DESIGN, "spec.designs")
    for name, block in designs.items():
        if block is None:
            continue
        where = f"spec.designs.{name}"
        _reject_unknown(block, _ALLOWED_DESIGN[name], where)
        _flag(block, "enable", where)
        _choice(block, "validation_status", where, VALIDATION_STATUSES)
        # Vocabulary membership is not enough: the label must match what this design will DO.
        stated, want = block.get("validation_status"), derived_validation_status(name, block)
        if stated is not None and stated != want:
            raise ValueError(
                f"{where}.validation_status is {stated!r} but this design as configured is "
                f"{want!r}. The status travels into the results, so it is derived from the execution "
                f"path and the file may only agree with it -- not assert something else.")
        if name == "A":
            _choice(block, "target_sensor", where, _SENSORS, required=bool(block.get("enable")))
            _choice(block, "srf_source", where, _SRF_SOURCES)
            _num(block, "fwhm_nm", where, lo=1e-9)
        elif name == "B":
            # Required once B is on: it used to pass validation absent and then fail deep in
            # build_cfg with a bare KeyError, which reads like a crash rather than a bad config.
            _num(block, "cwv_g_cm2", where, lo=0.0, allow_none=not block.get("enable"))
            _flag(block, "hard_mask_cores", where)
            if block.get("enable") and not block.get("table"):
                raise ValueError(f"{where}.table is required when Design B is enabled")
            if "table" in block and not isinstance(block["table"], str):
                raise TypeError(f"{where}.table must be a path string, got {type(block['table']).__name__}")
        elif name == "C":
            _num(block, "tau", where, lo=0.0)
        elif name == "D":
            _num(block, "stripe_eps", where, lo=0.0)      # std of a gain: a negative std is meaningless
            _num(block, "dead_col_frac", where, lo=0.0, hi=1.0)
    return spec


def validate_scope(spec):
    """Warn about YAML keys that LOOK executable but are NOT consumed by run_spec (data loading /
    evaluation). Keeps the config honest: a `path:` or `eval:` block here is provenance only."""
    inp = spec.get("input", {}) or {}
    stray_in = _UNCONSUMED_INPUT.intersection(inp)
    if stray_in:
        warnings.warn(f"config_runner does NOT load data: input.{sorted(stray_in)} are provenance only "
                      f"-- run_spec(cube, spec) uses the cube you pass, not these paths.")
    if spec.get("eval"):
        warnings.warn("config_runner does NOT run evaluation: the `eval` block is a declaration executed "
                      "by experiments/phaseX.py, not by run_spec (which only returns the simulated cube).")


def wavelength_axis(spec):
    return np.array(_AXES[spec["input"]["wavelength_axis"]], float)   # copy: never hand out the module-global axis


def build_cfg(spec, wavelengths_nm):
    """Translate a YAML spec dict into a pipeline.simulate() cfg dict.

    Validates first (fail-closed) so an unknown key or an out-of-range value cannot reach simulate()
    disguised as a default. Provenance headers are not demanded here so a hand-built dict in a test
    or experiment stays ergonomic; load_spec demands them for anything read off disk."""
    validate_spec(spec, require_provenance=False)
    d = spec.get("designs") or {}
    cfg = {"seed": int(spec.get("seed", 0))}
    if spec.get("claim_scope"):
        cfg["claim_scope"] = spec["claim_scope"]

    a = d.get("A") or {}
    if a.get("enable"):
        centers = _SENSORS[a["target_sensor"]]
        srf_source = a.get("srf_source", "gaussian")            # validated against _SRF_SOURCES above
        fwhm = float(a.get("fwhm_nm", 30.0))
        srf = gaussian_srf(wavelengths_nm, list(centers.values()), fwhm_nm=fwhm,
                           names=list(centers.keys()))   # preserve REAL names (B8A/B9/B11/B12), not auto B1..BN
        # One Gaussian FWHM for every band is a first-order bandpass, not a measured SRF, so Design A
        # is labelled as such unless the YAML says otherwise (see srf.gaussian_srf's docstring).
        cfg["A"] = {"enable": True, "srf": srf, "srf_source": srf_source, "fwhm_nm": fwhm,
                    "validation_status": a.get("validation_status", "first_order_approximation")}

    b = d.get("B") or {}
    if b.get("enable"):
        # Gas-only transmittance stress-test: keyed by CWV alone. AOD is INERT here (6S
        # transmittance_global_gas is aerosol-independent; empirically aod0.1==aod0.4 exactly), so it
        # is NOT part of the key. float() normalises 2 and 2.0 to the same key "cwv2.0".
        cfg["B"] = {"enable": True, "hard_mask_cores": b.get("hard_mask_cores", True),
                    "cache_npz": b["table"], "cache_key": f"cwv{float(b['cwv_g_cm2'])}",
                    "validation_status": b.get("validation_status", "physics_6s_gaseous")}

    c = d.get("C") or {}
    if c.get("enable"):
        cfg["C"] = {"enable": True, "tau": float(c.get("tau", 0.3)),
                    "validation_status": c.get("validation_status", "schematic_illustrative_only")}

    dd = d.get("D") or {}
    if dd.get("enable"):
        cfg["D"] = {"enable": True, "stripe_eps": float(dd.get("stripe_eps", 0.02)),
                    "dead_col_frac": float(dd.get("dead_col_frac", 0.01)),
                    "validation_status": dd.get("validation_status", "schematic_uncalibrated")}
    return cfg


def load_spec(path):
    """Read + fail-closed-validate a YAML spec, resolving `designs.B.table` against the YAML's OWN
    directory.

    The table path used to be resolved against the process CWD, so the same spec meant different
    things depending on where python was launched from: `cd /tmp && python /repo/script.py` raised
    FileNotFoundError, and -- worse -- a stray `data/srf_cache/T_6s_grid.npz` under the CWD was read
    INSTEAD of the repo's, silently swapping the atmosphere (a T=1 decoy shifted a run's mean from
    0.4435 to 0.4979 with no error). A config's relative paths belong to the config, not the shell."""
    import yaml
    with open(path) as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(spec).__name__}")
    validate_spec(spec, require_provenance=True)
    base = os.path.dirname(os.path.abspath(path))
    b = (spec.get("designs") or {}).get("B") or {}
    if isinstance(b.get("table"), str) and not os.path.isabs(b["table"]):
        b["table"] = os.path.normpath(os.path.join(base, b["table"]))
    return spec


def run_spec(cube, spec):
    """Run the config-driven SIMULATION on a caller-provided cube. Returns (out_cube, info).

    NOTE: does not load `input.path` and does not run the `eval` block (see module docstring);
    `validate_scope` warns if the spec carries those unconsumed keys."""
    validate_scope(spec)
    wl = wavelength_axis(spec)
    return simulate(cube, wl, build_cfg(spec, wl))
