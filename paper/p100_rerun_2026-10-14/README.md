# Re-runs on the 4x Tesla P100 box (2026-10-14)

These are NOT the paper's numbers -- those came from 2x V100 and are the files one level up. They are the
evidence for two records in `experiments/`. Each run is the paper's full 10 model-seed x 10 split design
on the same data, splits and seeds, with one script changed to evaluate a second variant on the same
trained models; the change (a diff against commit 6c1e392) and the code the run used are described in the
record.

| files | what | record |
|---|---|---|
| `results_phase8R10_quantile_convention_{percell.csv,summary.json}` | `phase8R10_normalization_control.py` with one extra arm, `L2A_prod_disj_quantile_firstrank`: the quantile transport under the earlier empirical-CDF convention the committed R10/R17/R18 outputs used (despite the name, that convention gives a tied value the rank of its last duplicate) | `PHASE8R10_QUANTILE_CONVENTION_CHECK.md` |
| `results_phase8R5_secondbench_labelversion_raw.csv` | `phase8R5_secondbench.py` calibrating and scoring every cell twice: with the validation sample's own (tacofoundation) labels and with CloudSEN12-high's at the same pixels (column `labels`) | `CLOUDSEN12_VAL_LABEL_VERSION_CHECK.md` |

Neither script writes a `.provenance.json` sidecar. The second file joins the three parts the run was
split into (seeds 0-3, 4-6 and 7-9), unchanged.
