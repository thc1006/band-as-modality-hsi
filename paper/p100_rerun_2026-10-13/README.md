# Re-runs on the 4x Tesla P100 box (2026-10-13)

These are NOT the paper's numbers -- those came from 2x V100 and are the `*_10seed.csv` files one level up.
They are the evidence for three records in `experiments/`; each run is the paper's full 10 split x 10
model-seed design on the same data, splits and seeds.

| files | what | record |
|---|---|---|
| `results_phase8R_{raw,reliability}_10seed_repro.csv` | the flagship as `reproduce.sh` runs it (test-pixel seed 54321) | `H10_TRAIN_ONCE_VERIFICATION.md` |
| `results_phase8R2_{landcover,geography}{,_raw}_10seed_repro.csv` | both deployment axes, standardised with the whole train draw (the paper's protocol) | `H10_TRAIN_ONCE_VERIFICATION.md`, `PHASE8R2_NORM_STATS_CHECK.md` |
| `results_phase8R2_{landcover,geography}{,_raw}_10seed_srcnorm.csv` | the same with `--norm-stats source` | `PHASE8R2_NORM_STATS_CHECK.md` |
| `results_phase8R_{raw,reliability}_pxseed{1,2,3,4}.csv` | the flagship with `--test-pixel-seed 1` ... `4` | `PHASE8R_TEST_PIXEL_SEED_CHECK.md` |

Every run was made from a copy of the code, so the `.provenance.json` sidecars record no git commit: the
`_10seed_repro` runs from the H-10 acceptance copy (`frozen_accept`; see its record), the `_srcnorm` and
`_pxseed*` runs from a copy of commit 6c1e392 with the `--norm-stats` and `--test-pixel-seed` options added
(`exp_next`): the same flags, defaults and arithmetic as committed later, where phase8R2 moves the choice
into `standardisation_stats`. Their `command` and `args` fields are complete; the `_10seed_repro` sidecars
predate both options and so record neither, which means the defaults (test-pixel seed 54321,
`--norm-stats train`). The copies were temporary directories that are not kept; after the runs, each
sidecar's `cwd` field, which held that directory's local path, was replaced by
`<temporary copy of the repository: NAME>` with the copy's name. No other field was edited.
