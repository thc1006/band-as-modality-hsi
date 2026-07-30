#!/usr/bin/env python3
"""Scene-COMPONENT-level AURC on the CloudSEN12 flagship — the honest unit-matched check.

The pixel-level AURC (aurc_cloudsen12_core.py) treats each pixel independently, which can look OPTIMISTIC when
errors cluster spatially. The paper's exchangeable unit is the scene-connected-component, so this recomputes
AURC two unit-honest ways from the banked scenedump_rich (offline; AURC is ~temperature-invariant, <=2 pp):

  1. COMPONENT-level selective classification: accept/reject WHOLE components ranked by their mean confidence
     (you cannot cherry-pick confident pixels inside a bad region), integrated over pixel coverage -- directly
     comparable to the pixel-level curve. If component-AURC >> pixel-AURC, the pixel result was optimistic.
  2. COMPONENT-cluster BOOTSTRAP CI on the pixel-level L2A AURC (resample components, not pixels), so the
     informative gap (error - AURC) carries an honest, correlation-aware interval.

Verdict target: does the flagship confidence stay INFORMATIVE at the paper's UNIT under L1C->L2A?
"""
import glob
import json
import os

import numpy as np
from scipy.special import softmax
from scipy.integrate import trapezoid

_HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(_HERE, "..", "paper", "scenedump_rich")


def pix_aurc(conf, corr):
    o = np.argsort(-conf); c = corr[o].astype(float); k = np.arange(1, len(c) + 1)
    return float(trapezoid(1.0 - np.cumsum(c) / k, k / len(c))) * 100


def comp_aurc(conf, corr, comp, ucomp):
    """Accept WHOLE components in descending mean-confidence order; selective risk = pixel error among
    accepted components' pixels; x-axis = fraction of PIXELS accepted (comparable to the pixel curve)."""
    confc = np.array([conf[comp == u].mean() for u in ucomp])
    order = ucomp[np.argsort(-confc)]
    acc_c = 0.0; acc_n = 0; sr = []; cov = []
    for u in order:
        m = comp == u
        acc_c += corr[m].sum(); acc_n += int(m.sum())
        sr.append(1.0 - acc_c / acc_n); cov.append(acc_n / len(comp))
    return float(trapezoid(sr, cov)) * 100


def main():
    inp = np.load(os.path.join(DUMP, "inputs_test.npz"))
    y = inp["y"].astype(int); comp = inp["comp"].astype(int); ucomp = np.unique(comp)
    rows = []
    for f in sorted(glob.glob(os.path.join(DUMP, "logits_seed*.npz"))):
        z = np.load(f)
        pc = softmax(z["logits_clean_test"], axis=1); cc = pc.argmax(1) == y
        pl = softmax(z["logits_l2a_test"], axis=1); cl = pl.argmax(1) == y
        rows.append(dict(
            pix_clean=pix_aurc(pc.max(1), cc), pix_l2a=pix_aurc(pl.max(1), cl),
            comp_clean=comp_aurc(pc.max(1), cc, comp, ucomp), comp_l2a=comp_aurc(pl.max(1), cl, comp, ucomp),
            err_l2a=float((~cl).mean()) * 100))
    m = lambda k: float(np.mean([r[k] for r in rows]))

    # component-cluster bootstrap CI on the pixel-level L2A AURC (seed 0), resampling COMPONENTS
    z0 = np.load(sorted(glob.glob(os.path.join(DUMP, "logits_seed*.npz")))[0])
    pl0 = softmax(z0["logits_l2a_test"], axis=1); cl0 = pl0.argmax(1) == y; conf0 = pl0.max(1)
    by_c = {u: np.where(comp == u)[0] for u in ucomp}
    rng = np.random.default_rng(12345); boot = []
    for _ in range(500):
        pick = rng.choice(ucomp, size=len(ucomp), replace=True)
        idx = np.concatenate([by_c[u] for u in pick])
        boot.append(pix_aurc(conf0[idx], cl0[idx]))
    lo, hi = np.percentile(boot, [5, 95])

    el = m("err_l2a")
    print(f"CloudSEN12 flagship, {len(rows)} seeds, {len(ucomp)} scene-components. L2A error {el:.0f}.", flush=True)
    print(f"  PIXEL-level     AURC L2A {m('pix_l2a'):5.1f}  (clean {m('pix_clean'):.1f})", flush=True)
    print(f"  COMPONENT-level AURC L2A {m('comp_l2a'):5.1f}  (clean {m('comp_clean'):.1f})  "
          f"<- accept/reject WHOLE components", flush=True)
    print(f"  component-cluster bootstrap 90% CI on pixel L2A AURC: [{lo:.1f}, {hi:.1f}]", flush=True)
    infp = el - m("pix_l2a"); infc = el - m("comp_l2a")
    print(f"  informative gap (error - AURC): pixel +{infp:.0f}pp, component +{infc:.0f}pp  "
          f"=> {'INFORMATIVE at the paper unit' if infc > 5 else 'NOT informative at component level'}", flush=True)
    out = os.path.join(_HERE, "..", "paper", "results_aurc_scene_component.json")
    json.dump(dict(rows=rows, boot_ci_pixel_l2a=[float(lo), float(hi)], n_components=int(len(ucomp))), open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
