#!/usr/bin/env python
"""R6 (reviewer challenge H2): is the formal weighted CRC's abstention on L2A a GENUINE near-total covariate
shift, or an artefact of estimating the likelihood ratio from the compressed 4-class softmax representation?

We measure how separable clean and L2A scene-components are (a clean-vs-L2A logistic domain classifier's
5-fold cross-validated AUROC, on per-component mean covariates) under progressively richer / different
representations of the covariate X:
  (a) raw (clipped) reflectance                          -- the physical input, all bands;
  (b) source-normalized reflectance (stale L1C z-score)  -- the representation the model actually sees;
  (c) product-aware-normalized reflectance (the E4 fix)  -- L2A z-scored with its OWN statistics.
Reference: the softmax representation used in phase8R11 gives AUROC ~0.99.

Reading. If (a)/(b) are near-separable too, the abstention is NOT a softmax artefact -- the L1C->L2A shift
is a genuinely large covariate shift that source-only reweighting cannot span. If (c) collapses the
separability, that identifies the shift as the input-SCALE mismatch and shows that the only representation in
which the covariate distributions overlap is the product-aware-normalized one -- i.e. what a reweighting
would need to become feasible is exactly the re-normalization that already fixes the certificate outright.
This does NOT prove weighted conformal is impossible; it locates where and why source-only reweighting fails.

Run: python phase8R13_weighted_representation.py
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase8R_reliability as P8R
from bandsim import hw


def comp_means(X, comp):
    order = np.argsort(comp, kind="stable")
    cs, Xs = comp[order], X[order]
    _, starts = np.unique(cs, return_index=True)
    counts = np.diff(np.append(starts, len(cs)))[:, None]
    return np.add.reduceat(Xs, starts, axis=0) / counts


def cv_auroc(F0, F1, folds=5):
    """Group-aware CV domain-classifier AUROC. F0 (clean) and F1 (L2A) are PAIRED component means -- row i of
    each is the SAME scene-component -- so a component's two views must stay in the same fold, else the
    classifier can see a component's clean view in train and score its L2A view in test, inflating the AUROC
    (P1-4). We group by component and use StratifiedGroupKFold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X = np.vstack([F0, F1])
    y = np.r_[np.zeros(len(F0)), np.ones(len(F1))]
    groups = np.r_[np.arange(len(F0)), np.arange(len(F1))]       # component i's clean+L2A views share a group
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=StratifiedGroupKFold(n_splits=folds), groups=groups,
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    hw.setup(deterministic=True, prefer="cpu")
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    ids_px = np.flatnonzero(~meta["s2_id"].isin(train_prod).to_numpy())

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids_px, seed=54321,
                             return_patch_id=(prd == "L1C"))
    X_l1c, y_te, pid = load("L1C")
    X_l2a, _ = load("L2A")
    comp = P8R.scene_component_ids("test")[pid]
    Xtr, _ = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)

    clip = lambda A: np.clip(A, -0.1, 1.6)
    mu1, sd1 = clip(Xtr).mean(0), clip(Xtr).std(0) + 1e-8
    X2c = clip(X_l2a)
    mu2, sd2 = X2c.mean(0), X2c.std(0) + 1e-8
    n1 = lambda A: (clip(A) - mu1) / sd1
    n2 = lambda A: (clip(A) - mu2) / sd2
    keep = [i for i in range(len(mu1)) if i != P8.B10_IDX]
    ncomp = len(np.unique(comp))
    print(f"clean-vs-L2A component-mean domain-classifier AUROC (5-fold CV), {ncomp} scene-components, "
          f"{len(keep)} bands:", flush=True)

    reps = [("raw reflectance", clip(X_l1c), clip(X_l2a)),
            ("source-normalized (stale L1C z-score)", n1(X_l1c), n1(X_l2a)),
            ("product-aware normalized (E4 fix)", n1(X_l1c), n2(X_l2a))]
    aucs = {}
    for name, Xa, Xb in reps:
        F0 = comp_means(Xa, comp)[:, keep]
        F1 = comp_means(Xb, comp)[:, keep]
        aucs[name] = cv_auroc(F0, F1)
        print(f"    {name:40s} AUROC {aucs[name]:.3f}", flush=True)
    print("    softmax (as used in phase8R11)           AUROC 0.99  [reference]")

    raw = aucs["raw reflectance"]
    prod = aucs["product-aware normalized (E4 fix)"]
    print(f"\n  -> raw-input AUROC {raw:.2f}: {'clean and L2A are near-separable in the PHYSICAL input too, so the abstention is NOT a softmax artefact -- the covariate shift is genuinely large' if raw > 0.9 else 'clean and L2A are only moderately separable in the raw input, so the softmax representation OVERSTATES the shift -- the weighted result is representation-dependent and must be softened'}.")
    print(f"  -> product-aware-normalized AUROC {prod:.2f}: {'re-normalization collapses the separability -- the shift is the input-scale mismatch, and the only representation in which the distributions overlap is the re-normalized one (which already fixes the certificate outright)' if prod < raw - 0.15 else 'even product-aware normalization leaves the components separable -- the shift is not purely a global scale mismatch'}.")
    print("  NOTE: this locates WHERE source-only reweighting fails; it does not prove weighted conformal is "
          "impossible (Mondrian, with target labels, reaches 8.6% here).")


if __name__ == "__main__":
    main()
