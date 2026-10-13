# phase8R: how much the test-pixel draw moves the flagship (checked 2026-10-13/14)

## The question

The flagship (`phase8R_reliability.py`) evaluates every test patch on a fixed random sample of its pixels,
drawn once with seed 54321 and shared by all 100 (split seed, model seed) cells. Its reported standard
error is two-way over the split and model seeds, so it leaves out the uncertainty of that pixel draw (the
H-8 caveat the script stamps into every provenance record). How large is the part it leaves out?

## What was run

`--test-pixel-seed` sets that draw's seed (default 54321, so the paper's runs are unchanged). The full
10 split x 10 model-seed flagship ran five times on the 4x P100 box -- seed 54321 (the H-10 acceptance
run, from the acceptance copy of the code) and seeds 1, 2, 3 and 4 (from a copy of commit 6c1e392
carrying this option; besides it, and phase8R2's `--norm-stats`, the two copies differ only in
docstrings and comments, an up-front check of the GPU memory budget and the formatting of one
provenance field):

```
python experiments/phase8R_reliability.py --split-seeds $(seq 0 9) --model-seeds $(seq 0 9) \
    --test-pixel-seed 1 --out-tag _pxseed1          # and 2, 3, 4
```

Only the test pixels change: the train draw keeps its own seed (12345), and training is bit-repeatable on
CUDA here (`H10_TRAIN_ONCE_VERIFICATION.md`), so the five campaigns share their trained models. Each seed
starts an independent generator (`np.random.default_rng(seed)` in `load_split`), so the five draws are
independent samples of each patch's pixels.

## Result

Mean over the 100 cells (%), for each draw, and the spread across draws:

| quantity | 54321 | 1 | 2 | 3 | 4 | SD across draws | two-way SE (per draw) |
|---|---|---|---|---|---|---|---|
| proposed, L2A, naive joint risk (headline) | 28.707 | 28.728 | 28.773 | 28.814 | 28.865 | 0.064 | 0.811 – 0.824 |
| proposed, L2A, naive coverage | 93.058 | 93.073 | 93.160 | 93.126 | 93.144 | 0.045 | 0.530 – 0.538 |
| proposed, L2A, Mondrian joint risk | 8.602 | 8.614 | 8.632 | 8.606 | 8.637 | 0.016 | 0.241 – 0.255 |
| proposed, L2A, Mondrian coverage | 47.678 | 47.621 | 47.727 | 47.617 | 47.439 | 0.109 | 1.292 – 1.335 |
| proposed, clean, naive joint risk | 8.489 | 8.497 | 8.513 | 8.487 | 8.497 | 0.010 | 0.186 – 0.212 |
| b2, L2A, naive joint risk | 34.403 | 34.488 | 34.493 | 34.528 | 34.538 | 0.053 | 0.497 – 0.510 |
| proposed, −B10, naive joint risk | 8.179 | 8.184 | 8.209 | 8.191 | 8.232 | 0.022 | 0.237 – 0.258 |
| proposed, −B1/B9/B10, naive joint risk | 8.669 | 8.717 | 8.761 | 8.746 | 8.788 | 0.046 | 0.260 – 0.278 |

The headline moves by 0.16 points across the five draws (SD 0.064), against a reported standard error of
about 0.82. Adding the pixel-draw variance to the two-way variance would raise that SE by about 0.3 %
(sqrt(0.815^2 + 0.064^2) = 0.818); for the other quantities above the rise is 0.1 % to 1.5 % (largest for
the -B1/B9/B10 joint risk). Five draws give the SD only 4 degrees of freedom, but even its 95 % upper
bound (0.18) would raise the headline SE by 2.5 %. Within one cell the headline varies by about 0.11
points between draws, against 2.76 points between cells. **The pixel-sampling uncertainty the reported SE
leaves out is negligible next to it; the H-8 omission does not change any interval or conclusion.** (The
headline happens to rise with the seed number from 1 to 4; for independent draws that order has a
two-sided chance of about 0.08, and both models share the same pixels, so their parallel trends are one
coincidence, not two.)

## Scope

All five draws ran on the 4x P100 box; the paper's numbers come from 2x V100, whose models differ (see
the H-10 record). The five campaigns' CSVs and sidecars are committed in `paper/p100_rerun_2026-10-13/`.
