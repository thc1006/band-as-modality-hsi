# H-10 (train each model seed once) — verification record

`phase8R_reliability.py` and `phase8R2_landcover_reliability.py` used to run every (split_seed, model_seed)
cell in one call (`run_seed`: train, then evaluate), so a 10 x 10 campaign trained each model seed ten times.
Training reads the model seed only, so H-10 (`run_two_stage`) trains every seed ONCE (`train_job`, pickled
bytes of the CPU models) and evaluates every cell against it (`eval_job`). `bandsim.parallel` also releases
the CUDA cache at every job boundary, and the planner takes a per-stage VRAM budget. This file records what
was compared to establish that the change computes the same numbers, and how to repeat it.

Machine: 4x Tesla P100-PCIE-16GB (sm_60), driver 535.230.02, 8 vCPU (Xeon E5-2696 v4), torch 2.6.0+cu124,
oneMKL 2024.2, Python 3.12.15; CloudSEN12-high restored from Hugging Face `csaybar/CloudSEN12-high` @
`d97728dd`. Date: 2026-10-13. "Pre-H-10" = commit `d99ab1f` in a separate worktree (the 16-job
one-worker-per-GPU memory run of section 4 used the main checkout before H-10, which differed from it in
`pyproject.toml`, `requirements-lock.txt` and `reproduce.sh` and held untracked files of other work -- none
of which phase8R imports or reads). Passes 1 and 2 are two GPU verification runs on frozen copies of the
working tree, taken at 11:38 and 13:01; the acceptance runs used a third copy, taken at 17:58. "Review N" is
the N-th independent adversarial review of this change (twelve in all). The changes after pass 2 came from
reviews 5 to 7 and touched the planner's descriptions, warnings and budget edge cases, run_two_stage's
argument checks and provenance record, provenance text, `scripts/verify_h10.py` and the tests -- in code,
nothing on the training or evaluation path; reviews 8 to 12 changed documentation only. After the acceptance
copy, the code changed only in run_two_stage's budget handling and the tests.

## 1. The protocols compute the same rows (CUDA)

| comparison | cells | precision | result |
|---|---|---|---|
| pre-H-10 per-cell vs two-stage (first, file-descriptor transport) | 10 x 10 | raw + aggregate CSVs (raw floats `:.6g`) | byte-identical (2401 + 25 lines) |
| pre-H-10 per-cell vs two-stage (pickled-bytes transport) | 10 x 10 | same | byte-identical |
| pre-H-10 per-cell vs two-stage (pass 1) | 10 x 10 | same | byte-identical |
| pre-H-10 per-cell vs two-stage, in-memory rows | 2 x 10 (all ten model seeds) | full float precision | 0 of 17,280 fields differ (480 rows) |
| same, pass 2 | 2 x 10 | full float precision | 0 of 17,280 fields differ |
| pre-H-10 per-cell vs two-stage, acceptance run (`reproduce.sh`'s Phase 8R line) | 10 x 10 | raw + aggregate CSVs | byte-identical |
| phase8R2 (surface axis), pre-H-10 vs two-stage, in-memory rows (pass 2) | 2 x 10 | full float precision | 0 of 1,440 fields differ (80 rows) |
| one flagship cell (split 3, model seed 7), `run_seed` vs `eval_job(train_job bytes)`, one P100 (pass 2) | 1 | full float precision | 0 of 864 fields differ (24 rows) |

In the per-cell runs each cell retrained its model seed on whichever of the four GPU workers picked the cell
up, and its rows still equal those of the seed's single two-stage training (section 2 checks training across
the GPUs directly). The headline (proposed, L2A_real, naive) on the P100s is 28.707 +/- 0.815 against the
paper's 27.849 +/- 0.786 (the seed-fixed 2x V100 run committed in c59100a). The per-cell code gives the same
CSVs on the P100s as the two-stage code, so that gap is not a consequence of H-10: the trained models are
re-drawn on the other machine -- clean accuracy differs in all 100 cells, for both models. The two machines
share the input data (the restored CloudSEN12-high: the test inputs compared value by value with inputs
banked on the V100 box, the training draw through its normalisation mean and standard deviation, the only
statistic of the training pixels banked there) and the numeric packages (torch 2.6.0+cu124, numpy 2.1.0,
scipy 1.18.0); they differ in GPU (V100 sm_70 against P100 sm_60), CPU (an AVX-512 Xeon Gold 6154 against
this box's AVX2 Xeon E5-2696 v4), NVIDIA driver patch (535.161.08 against 535.230.02) -- the V100 host's CPU
and driver from notes kept on that machine, as its provenance records neither -- Python patch release
(3.12.3 against 3.12.15), operating system and container. Which of these re-draws the models was not
isolated. The seeded draws are the same on both: both models draw their random initial weights only through
`uniform_`, for which torch 2.6.0 ships no AVX-512 kernel, so an AVX-512 CPU is dispatched to the AVX2
kernel this box uses by default (forcing ATen's `default` path here does change the weights). The proposed
model's construction does differ, in its fixed wavelength encoding, through `exp`: torch hands it to oneMKL,
whose result depends on its instruction-set path (restricting MKL to SSE4.2 here changes 13 of the
encoding's 448 elements), and the V100-built models banked in `paper/scenedump_rich` carry an encoding that
differs from this box's in 12 elements (at most 2.4e-7; the same in all ten, identical here at 1, 4 and 8
threads) -- reproduced exactly by raising two of the 32 `exp` outputs by one ulp. b2, whose construction
computes nothing but `uniform_`, differs in every cell all the same, so construction is not the only cause;
and the V100 run, made from the development repository's working tree, was never repeated there, so its own
run-to-run spread is unknown. phase8R2's 10 x 10 on the P100s (the acceptance runs, both axes) re-draws its
models too (accuracy differs for 399 of its 400 (cell, model) pairs), and every arm's mean joint risk and
coverage moves by at most 0.69 of the paper's standard error (largest: geography, proposed, naive coverage,
79.05 against 78.58 +/- 0.68; joint risks by at most 0.14 points). Compared cell by cell -- the two
campaigns share splits, seeds and data -- one land-cover estimate (b2, Mondrian, joint risk) is set aside
because its two-way variance nearly cancels (SE 0.0025 against an iid 0.038: 12.9 two-way standard errors,
0.84 iid ones). The other 15 shifts reach at most 2.33 two-way cluster-robust standard errors (geography,
proposed, naive coverage; all of geography's naive-arm shifts are upward; one further land-cover SE is the
iid floor, its two-way variance being negative), and none is significant once the 16 comparisons are taken
into account (two-sided p > 0.04 each at 9 degrees of freedom). So the hardware re-draw leaves at most a
hint of a systematic shift, small next to the reported uncertainty.

## 2. Training is bit-identical across processes, GPUs and thread counts (CUDA, deterministic=True)

| check | result |
|---|---|
| flagship model seed 0, three concurrent processes + a second training inside one of them (session probe) | identical b2 and proposed weight hashes |
| flagship model seed 0 on each of the four GPUs at 1 and at 4 intra-op threads (`train-hash`, pass 2) | one SHA-256 set over all 8 runs (proposed `34f88be1...`, b2 `e9c13dd0...`), the same as the session probe's |
| flagship model seed 0 trained twice in one process, at 8 intra-op threads (`train-hash --repeat 2`, pass 2) | identical, and the same set |

No log of the GPU runs in this record holds a PyTorch determinism warning: neither "... does not have a
deterministic implementation" nor the cuBLAS one about `CUBLAS_WORKSPACE_CONFIG` (`bandsim.hw` sets it at
import, before CUDA initializes). Under `warn_only=True` such a warning is what an op PyTorch knows to be
non-deterministic produces instead of an error.

## 3. The model transport

Returning `nn.Module`s from the pool and handing them to every evaluation job made torch's multiprocessing
reducer pass one file descriptor per tensor (40 per seed, 400 per evaluation job at 10 seeds): with
`ulimit -n 1024` the pool died with `OSError: [Errno 24] Too many open files` (this box's limit is 1048576,
which hid it). Plain pickle stores each storage by value: 0.57 MB per seed; 100 jobs under `ulimit -n 128`
ran with every state dict bit-exact. `tests/test_h10_train_once.py::test_two_stage_through_real_pools_survives_a_low_open_files_limit`
fails with "Too many open files" if the module transport is put back (in that test's spawned pool torch
raises it as `RuntimeError: unable to open shared memory object ...: Too many open files (24)`).

## 4. Memory (nvidia-smi)

| run | workers per GPU | peak |
|---|---|---|
| one flagship job alone (pre-H-10) | 1 | 5,478 MiB |
| pre-H-10, 16 jobs, no cache release | 1 | 8,526 MiB: two workers grew 5,478 -> 6,986 -> 8,526 over their first three jobs, the other two stayed at 4,526 |
| pre-H-10, 10 x 10, no cache release | 1 | 7,038 MiB per worker, held for most of the run |
| pre-H-10, 16 jobs, no cache release | 2 | 13,220 MiB on one card (the run completed) |
| two-stage with the job-boundary release, 10 x 10 (four runs, the acceptance run included) | 2 | at most 11,574 MiB per card |
| phase8R2 evaluation, two-stage, `EVAL_VRAM_GB` = 11 | 1 | 9,090 MiB per worker process, 9,092 MiB per card |
| phase8R2 10 x 10 acceptance runs (surface, geography), two-stage | 2 training, 1 evaluation | 9,092 MiB per card |

Without the release a worker keeps what its jobs reserved, so two per card can add up: two workers at the
8,526 MiB above would hold 17,052 MiB, more than a card's 16,384 -- a worst case, which the one 16-job run at
two per GPU did not reach; the per-cell campaigns of section 1 ran at one worker per GPU. The planner works in
decimal GB (`total_memory / 1e9`; a P100 is 17.07). At the default 8 GB per worker it gives a P100 two
workers, which the release keeps within memory for phase8R (the 10 x 10 row); phase8R2's whole-test-set
forward needs its 11 GB budget, one worker per P100 (two of its evaluation workers would hold 18,180 MiB).
Chunking that forward instead is not bitwise neutral on CUDA: with the phase8R flagship's models over its
388,400 test rows (`chunking`, pass 2) the proposed model matched in 65,536-, 32,768- and 16,384-row chunks,
but b2 differed in 16,384-row chunks by up to 7.6e-6 in absolute logit (it matched in the two larger ones).

## 5. On the CPU

* Weights depend on the intra-op thread count, which is why `run_two_stage` gives both stages the per-cell
  plan's thread budget (before it did, the first review measured a 2-seed x 3-split CPU campaign
  differing from the per-cell one in 257 of its fields).
* Logits moved with the thread count at the larger two of the four input sizes measured
  (`cpu-thread-logits`, models trained on the synthetic problem): none at 2,400 or 60,000 rows; up to
  2.4e-7 (proposed) at 145,650 rows; up to 2.4e-7 (proposed) and 1.2e-7 (b2) at 388,400 rows, in absolute
  logit.
* With two threads per worker even a fixed budget was not bitwise repeatable: the synthetic 6-cell per-cell
  vs two-stage campaign disagreed in 3 of 20, 7 of 40 and (with `cpu-repeat`) 2 of 40 repetitions, in 2 of
  40 with oneMKL's CNR mode (`MKL_CBWR=AUTO`), and in 0 of 20 when the fifth review repeated it -- 14 of 160
  in all; these loops logged only how many fields differed (24, 72 or 96). In two further runs of the same
  campaign the differing fields were recorded -- in full once, partly once -- and they were the b2 rows'
  fitted temperatures and the CRC thresholds derived from them: in the full record, temperatures by
  1e-9..4e-9 and thresholds by up to 2.6e-8, both relative, and nothing else. On real CloudSEN12 pixels (the
  fifth review: a 3-split x 2-seed phase8R campaign on 24,000 train and 24,000 test pixels, three workers x
  two threads -- two workers in the two-stage training stage -- run three times per protocol) one of the six
  runs, a pre-H-10 one, differed from the other five, which were identical at full precision: in 48 fields
  of one cell's b2 rows -- temperatures and CRC thresholds (up to 1.2e-8 and 1.2e-7 relative) and AURC/AUROC
  (up to 2.6e-7 relative), at temperature 1 too (up to 1.9e-7), which depends on the logits alone, so there
  the b2 logits themselves had moved. With one thread per worker the synthetic campaign agreed in 60 of 60,
  and again in 60 of 60 with `cpu-repeat`. The locus was not isolated (ruled out: input alignment, heap
  layout, FTZ/DAZ, hash seed, CPU-feature heterogeneity, the numpy temperature fit); exact CPU tests
  therefore run single-threaded.

## 6. Tests

`tests/test_h10_train_once.py`, `tests/test_phase8R_p0_fixes.py` and `tests/test_verify_h10.py`: 52 tests,
exact (==) comparisons, single-threaded where rows are compared; the tests that run a script's main() stub
its data loader (they read only the CloudSEN12 metadata) and write into a temporary directory. Source-level
mutation testing on a copy of the repository, with the GPUs hidden (each defect planted in
`bandsim/parallel.py`, `bandsim/hw.py`, one of the two scripts or `scripts/verify_h10.py` must fail a test;
workers re-import from disk, so the copy is mutated on disk): 72 of 72 mutants killed. The full suite's
failures are the 25 that failed before the change, and only those (in a copy without `.git`, before and
after alike, one more fails: a test that needs git).

Throughput on this box: 10 x 10 in 28-33 min two-stage at two workers per GPU (four runs), against 133.7 min
per-cell at one worker per GPU. At two per GPU the per-cell code risks the memory sum of section 4; its
16-cell run there took 17.8 min, about 1.9 h for 100 cells at that rate.

## How to repeat

```
python scripts/verify_h10.py prep INPUTS.npz
CUDA_VISIBLE_DEVICES=0 python scripts/verify_h10.py train-hash --inputs INPUTS.npz --threads 1   # each GPU, 1 and 4
python scripts/verify_h10.py train-hash --inputs INPUTS.npz --repeat 2
python scripts/verify_h10.py rows --inputs INPUTS.npz --split 3 --seed 7
BANDSIM_GPU_OVERSUB=1 python scripts/verify_h10.py ab-run old OLD_WORKTREE old.pkl     # pre-H-10 worktree
python scripts/verify_h10.py ab-run new . new.pkl
python scripts/verify_h10.py ab-compare old.pkl new.pkl
BANDSIM_GPU_OVERSUB=1 python scripts/verify_h10.py ab-run old OLD_WORKTREE r2old.pkl --script phase8R2   # + new, compare
python scripts/verify_h10.py chunking --inputs INPUTS.npz
python scripts/verify_h10.py cpu-repeat --threads 2 --repeat 40      # and --threads 1 --repeat 60
python scripts/verify_h10.py cpu-thread-logits --rows 388400
```

The 10 x 10 CSV comparisons are two runs of `phase8R_reliability.py --split-seeds $(seq 0 9) --model-seeds
$(seq 0 9)` (with different `--out-tag`s), one from a worktree of the pre-H-10 commit with
`BANDSIM_GPU_OVERSUB=1` (one worker per GPU, see section 4), compared with `cmp`. The pre-H-10 runs above
use `BANDSIM_GPU_OVERSUB=1` for the same reason. Session-only probes, not in the script: the data check
against the V100 banks, the initialisation and encoding probes and the P100-against-paper comparisons
(section 1), the three-process training probe and the scan of the run logs for determinism warnings (section
2), the nvidia-smi memory sampling and the benchmark timings (sections 4 and 6), the mutation harness
(section 6), the open-files reproductions (the test above covers a smaller case), the CPU repetition loops
other than `cpu-repeat` (3 of 20, 7 of 40, `MKL_CBWR=AUTO` 2 of 40, the fifth review's 0 of 20, the first
single-thread 60 of 60), the real-data repetitions, the probes behind the "ruled out" list and the two field
captures (section 5), and the first review's 257-field measurement.
