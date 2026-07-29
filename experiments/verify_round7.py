#!/usr/bin/env python
"""COMPREHENSIVE functional verification of every round-7 code block, with edge cases and explicit
regression tests for every issue the adversarial reviews found. Merge gate: all checks must PASS."""
import os, sys, traceback
import numpy as np
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
os.environ.setdefault("OMP_NUM_THREADS", "1")

P = F = 0
def check(name, fn):
    global P, F
    try:
        fn(); print(f"  PASS  {name}"); P += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}"); traceback.print_exc(); F += 1

# ==================== R10 normalization_control ====================
def r10_band_stats():
    from phase8R10_normalization_control import band_stats
    x = np.random.default_rng(0).normal(3, 2, (200, 6))
    mu, sd = band_stats("t", x, [0, 1, 2, 3, 4, 5])
    assert np.allclose(mu, x.mean(0)) and np.allclose(sd, x.std(0))
    xc = x.copy(); xc[:, 2] = 5.0                                 # near-constant retained band must RAISE
    try:
        band_stats("t", xc, [0, 1, 2]); raise AssertionError("no raise on constant band")
    except ValueError:
        pass
check("R10 band_stats: mean/std + near-constant guard", r10_band_stats)

def r10_transports():
    from phase8R10_normalization_control import quantile_match, robust_transport
    rng = np.random.default_rng(1); nb = 5
    Xsrc = rng.normal(2, 1, (600, nb)); Xcal = rng.normal(5, 2, (400, nb)); Xeval = rng.normal(5, 2, (300, nb))
    mu, sd = Xsrc.mean(0), Xsrc.std(0); keep = [0, 1, 2, 3]
    for fn in (quantile_match, robust_transport):
        X = fn(Xeval, Xcal, Xsrc, mu, sd, keep)
        assert np.isfinite(X).all()
        assert np.allclose(X[:, 4], (Xeval[:, 4] - mu[4]) / sd[4])           # dropped band -> train z-score
        for b in keep:
            assert abs(np.median(X[:, b])) < 0.35                            # transported median ~ source median (=0)
    Xq = quantile_match(Xeval, Xcal, Xsrc, mu, sd, keep)
    for b in keep:                                                          # monotone transport
        o = np.argsort(Xeval[:, b]); assert np.all(np.diff(Xq[o, b]) >= -1e-6)
    # clipping-tie robustness (agent P2-1): heavily-tied calib band still gives bounded finite output
    Xcal_t = Xcal.copy(); Xcal_t[:, 0] = np.clip(Xcal_t[:, 0], 4.9, 5.1)
    Xqt = quantile_match(Xeval, Xcal_t, Xsrc, mu, sd, keep); assert np.isfinite(Xqt).all()
    # robust near-constant guard raises
    Xcb = Xcal.copy(); Xcb[:, 0] = 3.0
    try:
        robust_transport(Xeval, Xcb, Xsrc, mu, sd, keep); raise AssertionError("robust no raise")
    except ValueError:
        pass
check("R10 quantile/robust transport: finite, dropped-band, median-align, monotone, clip-tie, guard", r10_transports)

def r10_helpers():
    from phase8R10_normalization_control import comp_equal_acc, paired_delta
    corr = np.array([1, 1, 0, 0, 1, 1]); comp = np.array([0, 0, 0, 1, 1, 1])   # comp0 acc 2/3, comp1 2/3 -> 66.7
    assert abs(comp_equal_acc(corr, comp) - (2 / 3 + 2 / 3) / 2 * 100) < 1e-6
    ra = [(0, 0, 12.0), (0, 1, 11.0), (1, 0, 13.0), (1, 1, 10.0)]
    rb = [(0, 0, 10.0), (0, 1, 10.0), (1, 0, 10.0), (1, 1, 10.0)]
    m, se, lo, hi = paired_delta(ra, rb, [0, 1], [0, 1])
    assert abs(m - 1.5) < 1e-6 and lo < m < hi
check("R10 comp_equal_acc + paired_delta", r10_helpers)

# ==================== R15 receptive_field ====================
def r15_convnet():
    import torch
    from phase8R15_receptive_field import ConvNet, n_params, ARMS
    rf = {1: 1, 3: 9, 5: 17, 7: 25}
    for k, r in rf.items():
        assert ConvNet(9, 4, 32, k=k, depth=4).rf == r
    pc = {lb: n_params(ConvNet(9, 4, w, k=k)) for lb, k, w in ARMS}
    assert pc["k1"] == 3876 and pc["k3"] == 30756 and pc["k5"] == 84516 and pc["k7"] == 165156
    for kk in (3, 5, 7):                                                    # capacity-matched controls within 3%
        assert abs(pc[f"k1_w{kk}"] / pc[f"k{kk}"] - 1) < 0.03
    torch.manual_seed(0); m1 = ConvNet(9, 4, 32, k=1).eval()               # k=1 is per-pixel at inference
    x = torch.randn(1, 9, 24, 24)
    with torch.no_grad():
        y0 = m1(x); x2 = x.clone(); x2[0, :, 0, 0] += 7; y2 = m1(x2)
    assert torch.allclose(y0[0, :, 12, 12], y2[0, :, 12, 12], atol=1e-6)     # far pixel unchanged
    assert not torch.allclose(y0[0, :, 0, 0], y2[0, :, 0, 0], atol=1e-3)     # same pixel responds
    m3 = ConvNet(9, 4, 32, k=3).eval()                                      # k=3 local rf
    with torch.no_grad():
        y0 = m3(x); x2 = x.clone(); x2[0, :, 0, 0] += 7; y2 = m3(x2)
    assert torch.allclose(y0[0, :, 12, 12], y2[0, :, 12, 12], atol=1e-6)
    assert not torch.allclose(y0[0, :, 2, 2], y2[0, :, 2, 2], atol=1e-3)
check("R15 ConvNet: rf, params(7 arms), capacity-match, per-pixel(k1) vs local(k3)", r15_convnet)

def r15_train_determinism():
    from phase8R15_receptive_field import train_convnet
    X = np.random.default_rng(0).normal(0, 1, (4, 9, 8, 8)).astype(np.float32)
    Y = np.random.default_rng(1).integers(0, 4, (4, 8, 8)).astype(np.int64)
    m1, s1 = train_convnet(X, Y, 9, 3, 32, 0, "cpu", 1, 4)
    m2, s2 = train_convnet(X, Y, 9, 3, 32, 0, "cpu", 1, 4)                  # same seed -> identical
    w1 = next(m1.parameters()).detach().numpy(); w2 = next(m2.parameters()).detach().numpy()
    assert np.allclose(w1, w2) and s1 == s2
check("R15 train_convnet: deterministic (seed-before-constructor)", r15_train_determinism)

# ==================== R16 classwise_10x10 ====================
def r16_perclass_and_wjguard():
    from phase8R_classwise import perclass_seg, perclass_risk, NC
    y = np.array([0, 0, 1, 1, 2, 3]); pred = np.array([0, 0, 1, 2, 2, 3])
    seg = perclass_seg(y, pred); rk = perclass_risk(y, pred, np.full(6, .9), .5)
    assert abs(seg[0]["iou"] - 1) < 1e-9 and abs(seg[1]["iou"] - .5) < 1e-9
    assert abs(rk[1]["joint"] - .5) < 1e-9
    # perclass_risk on an ABSENT class -> NaN joint (the wj 0*NaN hazard)
    seg2 = perclass_seg(np.array([0, 0]), np.array([0, 0])); rk2 = perclass_risk(np.array([0, 0]), np.array([0, 0]), np.array([.9, .9]), .5)
    assert seg2[1]["support"] == 0 and not np.isfinite(rk2[1]["joint"])
    # R16 guard regression: `if tot and support:` skips the 0*NaN term -> wj finite
    wj = 0.0; tot = 2
    for c in range(NC):
        if tot and seg2[c]["support"]:
            wj += (seg2[c]["support"] / tot) * (rk2[c]["joint"] * 100)
    assert np.isfinite(wj)
check("R16 perclass_seg/risk + wj zero-support guard (0*NaN regression)", r16_perclass_and_wjguard)

def r16_logits_cache_equiv():
    import phase8_cloudsen12 as P8, phase8R_reliability as P8R
    from bandsim.model import GroupedCrossBandAttention
    from bandsim.grouping import group_center_wavelengths
    from bandsim import hw
    groups = P8.s2_physical_groups(); cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    hw.seed_model(0); m = GroupedCrossBandAttention(groups, cwl, 4)
    X = np.random.RandomState(0).randn(400, len(P8.S2_WL_NM)).astype(np.float32); idx = np.array([1, 50, 399])
    for drop in ([], [g_b10]):                                              # cache-then-index == per-mask (bit-identical)
        full = P8R.logits_at("proposed", m, X, groups, drop)
        assert float(np.abs(full[idx] - P8R.logits_at("proposed", m, X[idx], groups, drop)).max()) == 0.0
check("R16 logits caching bit-identical to per-mask (no batch dependence)", r16_logits_cache_equiv)

# ==================== R11 weighted_crc_formal ====================
def r11_estweight():
    from phase8R11_weighted_crc_formal import estimated_weight_positive_control, ess_frac
    assert abs(ess_frac(np.ones(50)) - 1) < 1e-9
    w = np.ones(50); w[0] = 1e4; assert ess_frac(w) < 0.1
    r = estimated_weight_positive_control(0.10, 4, 2., 4., 4., 2., ncomp=150, m=200)   # moderate: recover+track
    assert r["naive"] > 11 and r["tru"] <= 11 and abs(r["est"] - r["tru"]) < 3 and 0.75 <= r["auroc"] <= 0.99
    rs = estimated_weight_positive_control(0.10, 4, 2., 14., 14., 2., ncomp=150, m=200)  # separable: collapse
    assert rs["tcov"] < 25 and rs["auroc"] > 0.95
    rn = estimated_weight_positive_control(0.10, 4, 2., 4., 2., 4., ncomp=150, m=200)    # NO shift: est~true, AUROC~0.5
    assert abs(rn["est"] - rn["tru"]) < 2 and rn["auroc"] < 0.65
check("R11 estimated_weight PC: moderate recover / separable collapse / no-shift null", r11_estweight)

def r11_crc_machinery():
    from phase8R11_weighted_crc_formal import weighted_crc_perunit, comp_loss, make_grid, ess_frac
    rng = np.random.default_rng(3); Nc, m = 40, 100
    conf = rng.uniform(.5, 1, (Nc, m)); wrong = (rng.uniform(0, 1, (Nc, m)) < 0.2).astype(float)
    comp = np.repeat(np.arange(Nc), m); grid = make_grid(conf.ravel())
    L, COV, ids = comp_loss(conf.ravel(), wrong.ravel(), comp, grid)
    assert np.all(np.diff(L, axis=1) <= 1e-9)                              # loss non-increasing in threshold
    assert np.all(np.diff(COV, axis=1) <= 1e-9)                            # coverage non-increasing in threshold
    cal, ev = np.arange(20), np.arange(20, 40)
    r_u, c_u, f_u = weighted_crc_perunit(L[cal], np.ones(20), grid, L[ev], COV[ev], np.ones(20), 0.10)
    assert np.isfinite(r_u).all() and (c_u >= 0).all() and (c_u <= 100.001).all()
    assert f_u.all()                                                       # uniform: target=alpha*W>=0 -> all FEASIBLE
    # infeasible path (P0-3): a huge test-point weight forces target<0 -> NO valid bound -> abstain-FALLBACK,
    # which must be FLAGGED infeasible (not recorded as a formal 0-risk).
    r_a, c_a, f_a = weighted_crc_perunit(L[cal], np.ones(20), grid, L[ev], COV[ev], np.full(20, 1e6), 0.10)
    assert r_a[0] == 0.0 and c_a[0] == 0.0 and not f_a.any()
check("R11 comp_loss monotone + weighted_crc_perunit feasibility flag (P0-3)", r11_crc_machinery)

# ==================== nested_boot ====================
def nb_ci_relation():
    from phase8R9_surface_nested_boot import ci_relation
    assert ci_relation(10.8, 13.0, 10)[0] == "entirely above"
    assert ci_relation(7.0, 9.0, 10)[0] == "entirely below"               # P0-5 REGRESSION: not "includes"
    assert ci_relation(9.0, 11.0, 10)[0] == "includes"
    assert "BELOW" in ci_relation(7, 9, 10)[1]
check("nested-boot ci_relation 3-way (P0-5 regression: [7,9] is BELOW not includes)", nb_ci_relation)

def nb_parse_seed():
    from phase8R9_surface_nested_boot import parse_seed
    assert parse_seed("/x/seed0.npz") == 0 and parse_seed("seed10.npz") == 10 and parse_seed("seed_7.npz") == 7
    try:
        parse_seed("nope.npz"); raise AssertionError("no raise")
    except ValueError:
        pass
check("nested-boot parse_seed (numeric, incl seed10) + fail-closed", nb_parse_seed)

def nb_core():
    import phase8R9_surface_nested_boot as NB
    rng = np.random.default_rng(0); nd, nb_, ppc = 20, 10, 40
    comp = np.repeat(np.arange(nd + nb_), ppc)
    lg = rng.normal(0, 1, (len(comp), 4)); y = rng.integers(0, 4, len(comp)).astype(np.int64)
    dd = NB._prep_arrays(lg.astype(np.float32), y, comp.astype(np.int64))
    assert len(dd["cidx"]) == nd + nb_ and len(dd["cidx"][0]) == ppc      # _prep_arrays cidx correct
    NB._DATA = [dd]; NB._DARK = np.arange(nd); NB._BRIGHT = np.arange(nd, nd + nb_)
    lgg, yg, gg = NB.gather(np.array([0, 0, 1]), dd["cidx"], lg, y)
    assert set(np.unique(gg)) == {0, 1, 2} and len(lgg) == ppc * 3         # gather: fresh group ids
    try:
        NB.gather(np.array([], dtype=int), dd["cidx"], lg, y); raise AssertionError("no raise on empty")
    except ValueError:
        pass
    ss = np.random.SeedSequence(1).spawn(2)
    r1 = NB.one_replicate((ss[0], 1)); r1b = NB.one_replicate((ss[0], 1))
    assert r1[0] == r1b[0] and np.isfinite(r1[0]) and 0 <= r1[0] <= 100    # one_replicate deterministic + valid
    assert 0 <= r1[2] <= 1                                                 # feasible-rate in [0,1]
    th = NB.point_estimate(2); assert np.isfinite(th) and 0 <= th <= 100   # point_estimate
check("nested-boot _prep_arrays/gather/one_replicate(determinism)/point_estimate", nb_core)

def nb_cell_feasible():
    import phase8R9_surface_nested_boot as NB
    rng = np.random.default_rng(5); nc, ppc = 30, 60
    comp = np.repeat(np.arange(nc), ppc)
    lg = rng.normal(0, 1, (len(comp), 4)).astype(np.float32); y = rng.integers(0, 4, len(comp)).astype(np.int64)
    dd = NB._prep_arrays(lg, y, comp.astype(np.int64))
    ids = np.arange(nc)
    j, cov, feas = NB._cell(dd, ids[:12], ids[12:24], ids[24:])            # temp/calib/eval cohorts
    assert isinstance(feas, bool) and 0 <= cov <= 100 and (np.isnan(j) or 0 <= j <= 100)
check("nested-boot _cell returns certified (joint, coverage, feasible) via CRC API", nb_cell_feasible)

# ==================== retained_bias ====================
def rb_cliffs():
    from phase8R9_acolite_retained_bias import cliffs_delta
    assert abs(cliffs_delta([1, 2, 3], [1, 2, 3])[0]) < 1e-9
    assert abs(cliffs_delta([4, 5, 6], [1, 2, 3])[0] - 1) < 1e-9
    assert abs(cliffs_delta([1, 2, 3], [4, 5, 6])[0] + 1) < 1e-9
    assert abs(cliffs_delta([2, 2], [1, 2, 3])[0]) < 1e-9                  # ties handled
    for bad in ([], [np.nan, 1]):
        try:
            cliffs_delta(bad, [1, 2]); raise AssertionError("no raise")
        except ValueError:
            pass
check("retained_bias cliffs_delta: value, ties, empty/non-finite fail-closed", rb_cliffs)

def rb_cramers():
    import pandas as pd
    from phase8R9_acolite_retained_bias import cramers_v
    s = pd.Series(["a", "a", "b", "b"]); ret = np.array([True, True, False, False])
    v, _ = cramers_v(s, ret, ~ret); assert v > 0.85                       # Yates-correction=False REGRESSION (was 0)
    s2 = pd.Series(["a", "b", "a", "b"]); v2, _ = cramers_v(s2, ret, ~ret); assert v2 < 0.35
    s3 = pd.Series(["a", "a", "a", "a"]); v3, m3 = cramers_v(s3, ret, ~ret); assert m3 == "n/a"  # single category
check("retained_bias cramers_v: perfect(no-Yates regression)/independent/degenerate", rb_cramers)

# ==================== coverage completion: remaining standalone functions ====================
def r11_section12():
    from phase8R11_weighted_crc_formal import state_arrays, domain_weights, synthetic_positive_control, make_grid
    rng = np.random.default_rng(7); n, C = 800, 4
    lg = rng.normal(0, 1, (n, C)); y = rng.integers(0, C, n); comp = np.repeat(np.arange(40), 20)
    grid = make_grid(rng.uniform(.4, 1, 200))
    L, COV, ids, feat, mc = state_arrays(lg, y, comp, np.ones(n, bool), 1.0, grid)
    assert L.shape[0] == 40 and np.all(np.diff(L, axis=1) <= 1e-9) and feat.shape == (40, C)  # per-component loss non-increasing
    F0 = rng.normal(0, 1, (200, 3)); F1 = rng.normal(2.5, 1, (200, 3))
    (w0, w1), auc = domain_weights(F0, F1, [F0, F1])
    assert auc > 0.7 and np.isfinite(w0).all() and (w0 >= 1e-3).all() and (w0 <= 1e3).all()  # separated -> high AUROC, clipped
    mpn, spn, mpw, spw, nc, wc = synthetic_positive_control(0.10, 8, 2.5)
    assert mpn > 10 and mpw <= 10                                          # known-weight PC: naive breaches, weighted recovers
check("R11 state_arrays / domain_weights / synthetic_positive_control (section 1-2)", r11_section12)

def nb_load_raw():
    import tempfile
    import phase8R9_surface_nested_boot as NB
    d = tempfile.mkdtemp(prefix="nbtest_")
    good = os.path.join(d, "seed0_good.npz")
    np.savez(good, logits=np.random.default_rng(0).normal(0, 1, (100, 4)).astype(np.float32),
             y=np.random.default_rng(1).integers(0, 4, 100).astype(np.int16),
             comp=np.repeat(np.arange(10), 10).astype(np.int32), is_target=(np.arange(100) >= 50))
    lg, y, comp, it = NB._load_raw(good)
    assert lg.shape == (100, 4) and it.dtype == bool and it.sum() == 50
    bad = os.path.join(d, "seed0_bad.npz")
    np.savez(bad, logits=np.zeros((100, 4), np.float32), y=np.zeros(100, np.int16),
             comp=np.repeat(np.arange(10), 10).astype(np.int32), is_target=np.full(100, 2, np.int16))
    try:
        NB._load_raw(bad); raise AssertionError("no raise on is_target=2")   # non-binary is_target must fail closed
    except ValueError:
        pass
check("nested-boot _load_raw: loads valid + rejects non-binary is_target", nb_load_raw)

def rb_ncomp():
    from phase8R9_acolite_retained_bias import _n_components
    comp = np.array(["a", "a", "b", "c", "c"]); mask = np.array([True, True, False, True, True])
    assert _n_components(comp, mask) == 2                                   # unique components under the mask: a, c
check("retained_bias _n_components", rb_ncomp)

print(f"\n===== round-7 COMPREHENSIVE verification: {P} PASS, {F} FAIL =====")
sys.exit(1 if F else 0)
