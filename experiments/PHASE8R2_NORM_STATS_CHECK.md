# phase8R2: which pixels set the standardisation statistics (checked 2026-10-13)

## The question

`phase8R2_landcover_reliability.py` trains on the SOURCE domain of one deployment axis (dark surfaces,
or the Northern Hemisphere) and evaluates on the unseen TARGET domain (bright surfaces, or the Southern
Hemisphere). It standardises every input with per-band statistics computed from the whole train draw --
which includes the target domain's train pixels. The model never trains on those pixels, but their
unlabelled statistics shape the input scale, so a little information about the target domain reaches the
pipeline before deployment. The stricter protocol computes the statistics from the source-domain train
pixels only. Does the choice move any reported number?

## What was run

`--norm-stats {train,source}` selects the statistics (`standardisation_stats` in the script); `train`
stays the default, so the paper's runs are unchanged. Both protocols, both axes, the paper's full design
(10 split seeds x 10 model seeds), on the 4x P100 box with the same data, splits and seeds:

```
python experiments/phase8R2_landcover_reliability.py --split-seeds $(seq 0 9) --model-seeds $(seq 0 9) \
    --axis surface --norm-stats source --out-tag _10seed_srcnorm
python experiments/phase8R2_landcover_reliability.py --split-seeds $(seq 0 9) --model-seeds $(seq 0 9) \
    --axis geography --norm-stats source --out-tag _10seed_srcnorm
```

against the `train`-statistics runs on the same machine (the H-10 acceptance runs, `--out-tag
_10seed_repro`; experiments/H10_TRAIN_ONCE_VERIFICATION.md). The two protocols ran from two copies of the
code that compute identically: the `train` runs from the H-10 acceptance copy, the `source` runs from a copy
of commit 6c1e392 with this option and phase8R's `--test-pixel-seed` added (their sidecars therefore record
no commit). The committed option has the same flag, default and arithmetic; it moves the choice into
`standardisation_stats`. Each cell's two models differ only in the statistics they were
standardised with, so the comparison is paired by (split seed, model seed); the standard error of the mean
paired difference is phase8R2's own two-way cluster-robust estimator (split and model seed; CGM, without
small-sample factors), with 9 degrees of freedom for the t test.

## Result

| axis | model | arm | metric | `train` | `source` | paired diff | two-way SE | p |
|---|---|---|---|---|---|---|---|---|
| surface | proposed | naive | joint risk | 11.338 | 11.387 | +0.049 | 0.308 | 0.88 |
| surface | proposed | naive | coverage | 74.305 | 74.879 | +0.575 | 0.446 | 0.23 |
| surface | proposed | Mondrian | joint risk | 2.403 | 2.462 | +0.059 | 0.062 | 0.36 |
| surface | proposed | Mondrian | coverage | 39.568 | 40.887 | +1.319 | 0.656 | 0.075 |
| surface | b2 | naive | joint risk | 12.042 | 11.946 | -0.096 | 0.170 | 0.59 |
| surface | b2 | Mondrian | joint risk | 2.554 | 2.495 | -0.059 | 0.041 | 0.19 |
| geography | proposed | naive | joint risk | 9.850 | 9.860 | +0.010 | 0.110 | 0.93 |
| geography | proposed | naive | coverage | 79.046 | 78.839 | -0.207 | 0.207 | 0.34 |
| geography | proposed | Mondrian | joint risk | 5.781 | 5.816 | +0.036 | 0.069 | 0.62 |
| geography | b2 | naive | joint risk | 10.196 | 10.244 | +0.047 | 0.057 | 0.42 |
| geography | b2 | Mondrian | joint risk | 5.829 | 5.829 | +0.000 | 0.052 | 1.00 |

(Joint risk and coverage in %. The full set of 44 comparisons -- every arm's joint risk, coverage,
selective risk and fitted temperature, and both models' target-domain accuracy, AURC and AUROC, on both
axes -- recomputes from the two protocols' committed raw CSVs; the comparison script and its output are
kept locally, see Scope.)

Two of the 44 comparisons have p < 0.05, both fitted temperatures moving by about 1 % (b2: surface naive
+0.014, p = 0.031; geography Mondrian -0.016, p = 0.041) -- about what 44 comparisons give by chance, and
neither survives a Holm or Bonferroni correction. Every joint risk, coverage, selective risk, accuracy,
AURC and AUROC comparison has p >= 0.075 (the smallest: Mondrian coverage, surface, proposed, +1.3
points), and the headline naive joint risks move by -0.10 to +0.05 points, well inside the standard
errors those numbers carry in the committed CSVs (0.23 to 0.42 points; the paper prints 0.45 and 0.26
for the proposed model). The comparison cannot exclude effects of several tenths of a point: the proposed
model's surface naive joint risk moves +0.05 points, 95 % interval [-0.65, +0.75], and its surface
Mondrian coverage +1.32 points, [-0.17, +2.80].
**Including the target domain's train pixels in the statistics has no detectable effect on the joint
risks, coverages, selective risks, accuracies or ranking quality reported for the two axes; the
published surface and geography numbers stand.** The stricter `source` protocol is available for anyone
who prefers it.

## Scope

Both protocols ran on the 4x P100 box, so the comparison holds the hardware fixed; the paper's numbers
come from 2x V100, whose models differ from these (see the H-10 record). Not re-checked: the surface
logit dumps (`phase8R2_surfacedump.py`, which feeds the surface bootstrap intervals) standardise the same
way and have no such option. Both campaigns' CSVs and sidecars are committed in
`paper/p100_rerun_2026-10-13/`; the run logs, the comparison script and its full output are kept
locally under `paper/logs/norm_stats_check_2026-10-13/` (`paper/logs/` is gitignored).
