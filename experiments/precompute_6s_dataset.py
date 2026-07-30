#!/usr/bin/env python3
"""Generate a 6S gaseous-transmittance LUT for a specific HSI dataset's wavelength axis, so
phase8H_hsi6s_shift.py can run the 6S dry->humid reliability shift on datasets beyond Indian Pines:
  salinas  204-band AVIRIS  (400-2491 nm, HAS SWIR water-vapour bands -> a real shift)
  pavia    103-band ROSIS   (430-860 nm, VNIR only, NO SWIR water bands -> a NEAR-NULL negative control)

Same primitive + schema as precompute_6s_table.py (run_6s_transmittance, keyed by CWV), just a different
wavelength axis. Needs Py6S + the 6S binary (`sixs` on PATH). CPU-only; safe to run alongside GPU jobs.

  python experiments/precompute_6s_dataset.py --dataset salinas
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from bandsim.atmosphere import run_6s_transmittance, DEFAULT_GRID
from bandsim.io import ROSIS_WL_NM, AVIRIS_WL_NM, axis_sha256
import phase6_second_dataset as P6

AXES = {"indian_pines": np.asarray(AVIRIS_WL_NM, float),
        "salinas": np.asarray(P6.SALINAS_WL_NM, float),
        "pavia": np.asarray(ROSIS_WL_NM, float)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(AXES))
    ap.add_argument("--aod", type=float, default=0.1)   # inert for gaseous T; passed only to satisfy the 6S API
    args = ap.parse_args()
    wl = AXES[args.dataset]
    out = os.path.join(os.path.dirname(_HERE), "data", "srf_cache", f"T_6s_grid_{args.dataset}.npz")
    cwvs = DEFAULT_GRID["cwv_g_cm2"]
    meta = dict(generation_mode="direct_6s", schema_version=2, output_quantity="transmittance_global_gas",
                cwv_grid_g_cm2=list(cwvs), aod550_nominal=args.aod, axis_name=f"{args.dataset}_wl",
                n_bands=int(wl.size), wl_min_nm=float(wl[0]), wl_max_nm=float(wl[-1]),
                wl_sha256=axis_sha256(wl),
                generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    tables = {"wl_nm": wl, "wl_sha256": np.array(axis_sha256(wl)),
              "generation_mode": np.array("direct_6s"),
              "_metadata_json": np.array(json.dumps(meta, sort_keys=True))}
    import time
    print(f"6S LUT for {args.dataset}: {wl.size} bands {wl[0]:.0f}-{wl[-1]:.0f} nm", flush=True)
    for cwv in cwvs:
        t0 = time.time()
        T = run_6s_transmittance(wl, cwv, args.aod, n_threads=8)
        tables[f"cwv{float(cwv)}"] = T
        print(f"  cwv{cwv}: T [{T.min():.3f},{T.max():.3f}] mean {T.mean():.3f} ({time.time()-t0:.1f}s)", flush=True)
    np.savez_compressed(out, **tables)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
