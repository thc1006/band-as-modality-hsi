"""Every `conformal_at_risk` call must be sized from the unit its split was built on.

The defect this pins produced no crash and no warning. `conformal_at_risk`'s finite-sample margin is
`z*sqrt(p(1-p)/n)`, so `n` decides whether `conservative=True` means anything. phase8R handed its ROI
ids to `conformal_risk_control` and, on the adjacent lines, called `conformal_at_risk` without them --
counting 194,000 correlated pixels as independent evidence instead of the 97 ROIs the calibration/
evaluation split is actually made on. That is a ~45x understatement of the margin at the raw counts,
which turns the "conservative" flag into a no-op: the threshold comes out too aggressive and the
achieved risk sits above target with nothing to announce it.

phase8E is the more exposed of the two: it has NO CRC arm, so the plug-in operating point IS its
reliability claim, and there was nothing to cushion the understated margin.

Static assertions rather than end-to-end ones, because reproducing either script needs the CloudSEN12
download and hours of GPU. Static is enough here: the failure is a missing argument at a call site.
"""
import ast
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_ROOT, "experiments")
sys.path.insert(0, _ROOT)

from bandsim.reliability import conformal_at_risk  # noqa: E402

# Scripts whose calibration rows are CLUSTERED (spatial ROIs / patches), so every plug-in call in
# them must pass a group. Listed explicitly rather than discovered: the point of the list is to fail
# when a script that used to group stops doing so, and a rule derived from the scripts cannot.
_MUST_GROUP = ["phase8R_reliability.py", "phase8E_dofa.py"]


def _calls_to(tree, fname):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name == fname:
                yield node


@pytest.mark.parametrize("script", _MUST_GROUP)
def test_every_plugin_call_passes_a_calibration_group(script):
    path = os.path.join(_EXP, script)
    tree = ast.parse(open(path, encoding="utf-8").read())
    calls = list(_calls_to(tree, "conformal_at_risk"))
    assert calls, f"{script} no longer calls conformal_at_risk -- update this test's premise"
    ungrouped = [c.lineno for c in calls
                 if not any(kw.arg == "calib_group" for kw in c.keywords)]
    # A deliberately ungrouped call is allowed only where it is the ROW-SIZED CONTRAST that exists
    # to be compared against the grouped one; those are assigned to a *_rows name.
    src_lines = open(path, encoding="utf-8").read().splitlines()
    unexplained = [ln for ln in ungrouped if "_rows" not in src_lines[ln - 1]]
    assert not unexplained, (
        f"{script} calls conformal_at_risk WITHOUT calib_group at line(s) {unexplained}. "
        f"Its calibration rows are clustered by ROI, so the margin would be sized from correlated "
        f"pixels and `conservative=True` would be a no-op.")


@pytest.mark.parametrize("script", _MUST_GROUP)
def test_the_group_it_passes_is_the_one_crc_uses(script):
    """Not just 'a group' -- the SAME unit the split and (where present) CRC are built on. A plug-in
    margin sized from patches while CRC certifies over ROIs would be two different claims sharing a
    row in one CSV."""
    src = open(os.path.join(_EXP, script), encoding="utf-8").read()
    # Per-file, because the two scripts name their exchangeable-unit vector differently: phase8R
    # threads `unit_cal` (its rebuild's vocabulary) and phase8E threads `roi_cal` (PR #20's). What
    # is pinned is the identity of the computed variable, not one shared spelling.
    expected = {"phase8R_reliability.py": "calib_group=unit_cal",
                "phase8E_dofa.py": "calib_group=roi_cal"}[script]
    assert expected in src, (
        f"{script} must size the plug-in margin from its ROI-id vector ({expected}), the same "
        f"unit its calib/eval split is made on")


def test_group_and_no_group_actually_differ_at_phase8R_shape():
    """The guard above is only worth having if the argument changes the answer. At phase8R's real
    shape -- 194,000 calibration pixels over 97 ROIs -- it changes it by a lot."""
    n_roi, per = 97, 2000
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(n_roi), per)
    roi_effect = rng.normal(0, 1.0, n_roi)
    p = np.clip(0.85 + 0.08 * roi_effect[g], 0.02, 0.99)
    corr = (rng.random(g.size) < p).astype(float)
    conf = np.clip(p + rng.normal(0, 0.03, g.size), 0.01, 0.99)

    rows = conformal_at_risk(corr, conf, corr, conf, target_risk=0.10)
    grouped = conformal_at_risk(corr, conf, corr, conf, target_risk=0.10, calib_group=g)

    assert rows["n_calib_units"] == g.size, "ungrouped must still count rows"
    assert grouped["n_calib_units"] < rows["n_calib_units"], "grouping must shrink the effective n"
    assert grouped["margin"] > rows["margin"] * 3, (
        "the corrected margin must be materially larger, not a rounding change")
    # more conservative in the only direction that matters
    assert grouped["threshold"] >= rows["threshold"]
    assert grouped["coverage"] <= rows["coverage"] + 1e-12


def test_phase8E_derives_its_unit_from_the_flat_pixel_index():
    """phase8E samples `px_per_patch` pixels from each 224x224 patch and concatenates them, so a
    sampled pixel's flat index // IMG**2 is its patch position and roi_all maps that to a LOCATION.
    Pinned because the whole unit id rests on that arithmetic staying true."""
    IMG, n_patches, k = 224, 5, 7
    rs = np.random.default_rng(0)
    idx = np.concatenate([rs.choice(IMG * IMG, size=k, replace=False) + p * IMG * IMG
                          for p in range(n_patches)])
    assert np.array_equal(idx // (IMG * IMG), np.repeat(np.arange(n_patches), k))
    roi_all = np.array([10, 10, 11, 11, 12])          # five patches over three locations
    unit = roi_all[idx // (IMG * IMG)]
    assert unit.size == idx.size
    assert np.unique(unit).size == 3, "distinct ROIs, not distinct patches, is the unit count"
