# phase8R10: the quantile-transport tie convention, on the same models (checked 2026-10-14)

## The question

Round 8e (a707065) changed `quantile_match` in `phase8R10_normalization_control.py` -- and so R17 and
R18, which import it -- from the earlier empirical CDF (`np.sort` with plotting positions and `np.interp`,
which gives a tied value the rank of its last duplicate) to the tie-aware mid-distribution CDF
(`_mid_cdf_knots`; adversarial-review P0-2). R19 already had its own tie-aware CDF (`empirical_cdf_knots`,
round 8d). Quantised reflectance is heavily tied, so the two conventions can differ. The committed
R10/R17/R18 outputs predate the change: their full-calibration quantile-transport arm, 9.546 ± 0.238, is
the number the paper prints as 9.6 (abstract; `paper/RESULT_MANIFEST.md`, the P0-2 row and D11). The change
was recorded as lowering that arm on a full re-run (R10 "9.6 -> 9.5", R17 "9.55 -> 9.35"), "within the
pipeline's GPU run-to-run non-determinism (attention backward has no deterministic kernel, ...)". The
committed outputs contradict that explanation: `hw.setup` forced the math attention kernel and
deterministic algorithms (warn-only) in all of these runs, and on the 2x V100 box R10 (0e6a77f), R17
(34c4052) and R18 (01cef0e) each trained their models afresh and gave this arm bit-identical joint risks
in all 100 cells. How much does the convention itself move the number?

## What was run

One change to `phase8R10_normalization_control.py` (the diff applies to commit 6c1e392): the earlier
function, verbatim from 0e6a77f (the commit that produced the committed `_percell.csv` and `_summary.json`;
the two committed R10 logs come from c59100a, before the quantile arm existed), added as an extra arm that
is evaluated on the same trained model, split, temperature and CRC threshold as the current one. The arm's
name, `L2A_prod_disj_quantile_firstrank`, is a misnomer kept in the committed data: the earlier convention
gives a tied value the rank of its last duplicate. The run used a copy of the working tree before this
round's changes were committed; what R10 imports differs from 6c1e392 only in comments and docstrings,
`group_center_wavelengths`' input checks, `label_histogram`'s cache directory and phase8R's new option, none
of which touches R10's computation.

```diff
--- a/experiments/phase8R10_normalization_control.py
+++ b/experiments/phase8R10_normalization_control.py
@@ -171,6 +171,23 @@
     return dict(identity_max_abs_err=id_err, monotone=mono, tie_example_F0=tie_u)
 
 
+def quantile_match_firstrank(Xeval, Xcal_l2a, Xsrc, mu_tr, sd_tr, keep_idx):
+    """VERBATIM the 0e6a77f (round-7) quantile_match: np.sort + plotting positions + np.interp (ties get a
+    first/last rank). Added in the scratch copy only, to measure the convention's own effect."""
+    out = ((Xeval - mu_tr) / sd_tr).astype(np.float32)       # safe finite default for every band
+    keep = {int(i) for i in keep_idx}
+    for b in range(Xeval.shape[1]):
+        if b not in keep:
+            continue
+        cal = np.sort(Xcal_l2a[:, b]); src = np.sort(Xsrc[:, b])
+        cdf_cal = (np.arange(len(cal)) + 0.5) / len(cal)     # plotting-position empirical CDF of calib L2A
+        cdf_src = (np.arange(len(src)) + 0.5) / len(src)
+        u = np.interp(Xeval[:, b], cal, cdf_cal)             # F_{L2A-cal}(x) in (0,1), monotone
+        v = np.interp(u, cdf_src, src)                       # Q_src(u): source quantile at that level
+        out[:, b] = (v - mu_tr[b]) / sd_tr[b]                # z-score on the model's own training scale
+    return out
+
+
 def comp_equal_acc(corr, comp):
     """Component-equal (not pixel-pooled) accuracy, to match the certified triple's weighting."""
     order = np.argsort(comp, kind="stable")
@@ -250,7 +267,7 @@
             "L2A_src":        (X_l2a, mu_tr, sd_tr, DROP_L2A),
             "L2A_pairedL1C":  (X_l2a, mu_1t, sd_1t, DROP_L2A),
             "L2A_prod":       (X_l2a, mu_2t, sd_2t, DROP_L2A)}
-    trk = list(ARMS) + ["L2A_prod_disj", "L2A_prod_disj_robust", "L2A_prod_disj_quantile"]
+    trk = list(ARMS) + ["L2A_prod_disj", "L2A_prod_disj_robust", "L2A_prod_disj_quantile", "L2A_prod_disj_quantile_firstrank"]
     print(f"  eval {len(y_te)} px / {len(np.unique(comp_all))} components; L1C-train vs L2A-test per-band mean "
           f"shift {np.abs(mu_tr - mu_2t)[keep_b].mean():.3f}; {len(seeds)} seeds x {len(splits)} splits", flush=True)
 
@@ -323,6 +340,8 @@
             Xqm = quantile_match(Xl2a_c[me], Xl2a_c[mc], Xtr_c, mu_tr, sd_tr, keep_b,
                                  diag=(qm_diag if not qm_diag else None))   # collect diag on the first cell
             record_normalized("L2A_prod_disj_quantile", Xqm, DROP_L2A, me, Tc, thr)
+            Xqm0 = quantile_match_firstrank(Xl2a_c[me], Xl2a_c[mc], Xtr_c, mu_tr, sd_tr, keep_b)
+            record_normalized("L2A_prod_disj_quantile_firstrank", Xqm0, DROP_L2A, me, Tc, thr)
         cur = {s: np.mean([r[2] for r in rows[s] if r[0] == seed]) for s in ("clean", "L2A_src", "L2A_prod")}
         print(f"  seed {seed}: clean {cur['clean']:.1f}  L2A_src {cur['L2A_src']:.1f}  "
               f"L2A_prod {cur['L2A_prod']:.1f}", flush=True)
@@ -352,7 +371,8 @@
         ("clean_selfnorm", "clean", "generic test-batch renorm on the in-domain case (expect ~0)"),
         ("L2A_src", "L2A_prod", "total fix (stale - product-aware transductive)"),
         ("L2A_prod_disj", "L2A_prod_disj_robust", "richer: robust 2-moment transport vs mean/std disjoint"),
-        ("L2A_prod_disj", "L2A_prod_disj_quantile", "richer: full quantile transport vs mean/std disjoint")]:
+        ("L2A_prod_disj", "L2A_prod_disj_quantile", "richer: full quantile transport vs mean/std disjoint"),
+        ("L2A_prod_disj_quantile", "L2A_prod_disj_quantile_firstrank", "CONVENTION: mid-CDF - first-rank (same models)")]:
         md, se, lo, hi = paired_delta(rows[a], rows[b], seeds, splits)
         z0 = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
         print(f"    d[{a} - {b}] = {md:+6.2f} +/- {se:.2f}  [{lo:+5.1f},{hi:+5.1f}]  ({z0})  {label}", flush=True)
```

`python experiments/phase8R10_normalization_control.py --out <prefix>` (all defaults: 10 model seeds x
10 splits) on one P100 of the 4x P100 box, 66 minutes.

## Result

Mean over the 100 cells (%), with R10's own standard error (CGM-ss, see `paper/RESULT_MANIFEST.md`):

| arm | joint risk | coverage | committed (2x V100) |
|---|---|---|---|
| quantile transport, tie-aware CDF (current code) | 9.610 ± 0.215 | 69.8 | -- |
| quantile transport, earlier CDF (the committed outputs' convention) | 9.649 ± 0.215 | 69.7 | 9.546 ± 0.238 |
| product-aware, evaluation-disjoint (the paper's 10.8) | 10.739 ± 0.337 | 74.2 | 10.748 ± 0.300 |
| stale normalisation (L2A_src) | 29.251 ± 0.626 | 93.3 | 28.432 ± 0.427 |

Paired over the same 100 cells, tie-aware minus earlier: joint risk -0.039 ± 0.018 (CGM-ss), 95 % t9
interval [-0.079, +0.002], p = 0.058 (with plain CGM, the flagship's estimator: [-0.077, -0.001],
p = 0.047); coverage +0.074 [+0.040, +0.107]; selective risk -0.070 [-0.129, -0.011]; pixel accuracy
+0.071 [+0.028, +0.115]. Per cell the joint-risk difference runs from -0.29 to +0.05 (70 of the 100
cells lower); all 10 split means and 8 of the 10 model-seed means are negative, the seed means running
from -0.18 to +0.02.

## What it means

The convention moves the quantile-transport joint risk by about -0.04 points: under a fifth of the arm's
standard error, consistent in sign, and at the edge of the 5 % level (p = 0.058 with R10's CGM-ss, 0.047
with plain CGM); coverage and accuracy move by under a tenth of a point. It does not account for the
shifts recorded earlier. R10's "9.6 -> 9.5" compares the paper's mis-rounded 9.546 with a one-decimal
value and is consistent with the -0.04 measured here. R17's "9.55 -> 9.35" is not; and since R17's full
arm is R10's quantile arm cell for cell, same-box re-runs could not have given both 9.5 and 9.35. Neither
re-run was committed, so the source of that 0.2 is unknown (9.35 is also the seed-7 mean printed in the
committed R17 and R18 logs).

If the P100 effect transfers to the paper's models (across model seeds it runs from -0.18 to +0.02), the
arm would read about 9.51 instead of 9.546 -- still 9.5 at one decimal (the paper's 9.6 is a rounding error
either way, D11) -- and its interval, [9.01, 10.08] printed as [9.0, 10.1], would become about
[8.97, 10.05], printed as [9.0, 10.0]. The paper's reading that the interval straddles 10 % would survive,
by 0.05 (by 0.005 at the far end of the effect's own interval). The product-aware arms, which give the
paper's headline 10.8, do not use this function.

## Scope

One machine (4x P100), the paper's full design. The per-cell results of every R10 arm, including the
extra one, are committed in `paper/p100_rerun_2026-10-14/results_phase8R10_quantile_convention_percell.csv`
(with the run's `_summary.json`); the console log is not committed. R17 and R18 compute the same
full-calibration arm (their committed value is the same 9.546) and their other arms use the same
function; they were not re-run.
