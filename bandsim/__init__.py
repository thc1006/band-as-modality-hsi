"""bandsim — band-loss simulation testbed for band-as-modality HSI.

How strong a physical claim each stage supports differs, and the difference is load-bearing for the
paper: Design B is real 6S gaseous transmittance evaluated on the cube's own wavelength axis, and
Design A applies the exact NASA-HLS bandpass operator — but only to the SRF it is given. With
`srf.pyspectral_srf` that SRF is a measured ESA/USGS response; with `srf.gaussian_srf` (what the
YAML front-end and the demos use) it is a synthetic Gaussian of one fixed width, which fixes the
band SET correctly while approximating every band's SHAPE. Designs C and D are schematic. Each
config declares a `claim_scope` and each design a `validation_status`, and pipeline.simulate returns
both in `info` so the qualifier travels with the numbers.

Modules (WORKING = implemented + tested; SCHEMATIC = implemented but NOT sensor-calibrated,
excluded from physics claims — see each module header)
-------
model         GroupedCrossBandAttention (SGMAE) + MLP baselines                       WORKING
grouping      spectral group construction + group-centre wavelengths                  WORKING
srf           Design A: bandpass operator WORKING; measured SRF via pyspectral, or a
              synthetic fixed-FWHM Gaussian (approximate band shape) via gaussian_srf
atmosphere    Design B: 6S GASEOUS-absorption transmittance (needs Py6S to precompute) WORKING
cirrus        Design C: thin-cirrus per-band corruption                               SCHEMATIC
noise         Design D: per-band SNR + striping + dead detectors                      SCHEMATIC
metrics       mIoU / OA / AA / kappa / AUDC / retention                               WORKING
reliability   risk-coverage / AURC / AUGRC / selective-AUROC / conformal-risk-control  WORKING
io            HSI cube loading (.mat / ENVI)                                          WORKING
hw            device / threads / seeds / determinism                                  WORKING
parallel      task-level parallelism across GPUs / cores                              WORKING
config_runner / pipeline   YAML-driven simulation orchestration                       WORKING

See docs/guide/03_physical_simulation.md for the physics and docs/guide/00_ROADMAP.md for the plan.
"""

__version__ = "0.1.0"
