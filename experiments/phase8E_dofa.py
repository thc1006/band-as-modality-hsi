#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8E — Frozen-DOFA FEATURE baseline (foundation-model encoder, frozen) under our reliability
framework, SPATIAL.

Runs the real pretrained DOFA (channel-adaptive ViT-B foundation model; Xiong et al. 2024,
arXiv:2403.15356; SOURCE MIT, WEIGHTS distributed via HuggingFace earthflow/DOFA whose repo metadata
marks CC-BY-4.0 — two licences, see DOFA_LICENSES) as a FROZEN-FEATURE baseline on CloudSEN12 cloud
segmentation: the DOFA encoder is frozen and a light trainable decoder is added on top. This is NOT
the official DOFA segmentation recipe, so we call it a "frozen-DOFA feature baseline", not a
reproduced SOTA. Evaluated under the SAME reliability framework as phase8R. It shows two things a
top-journal reviewer will demand:

  (1) a REAL foundation model ALSO loses naive (clean-calibrated) risk control under the real
      L1C->L2A operational shift, and
  (2) our degradation-aware (Mondrian, per-state) recalibration is the MODEL-AGNOSTIC fix.

ESTIMAND — this script runs ONLY the plug-in conformal_at_risk operating point, so every risk it
reports is the CONDITIONAL selective risk P(wrong | accepted) and NOTHING here is certified. Do not
describe it as a guarantee, a certificate, or a bound; the word "guarantee" belongs to phase8R,
which runs Conformal Risk Control on the JOINT mass P(accepted AND wrong). Coverage is reported
alongside because a low risk bought by abstaining is not risk control.

BONUS: DOFA is SPATIAL (224x224 ViT) -> this doubles as the spatial-context variant that preempts
the "why per-pixel?" objection against phase8/phase8R.

WHICH ENCODER THIS ACTUALLY IS — measured, because an earlier version got it wrong. `vit_base_dofa`
defaults to global_pool=True, and OFAViT builds `fc_norm` OR `norm`, never both. This checkpoint
carries `norm.weight/bias` and NO `fc_norm`, so at the default, upstream's `strict=False` load
DISCARDED the pretrained normalisation as "unexpected" and left `fc_norm` at LayerNorm's
initialisation — measured mean 1.0000, std 0.0000, i.e. provably untrained — which this file then
applied PER SPATIAL TOKEN, a position official code never uses. The encoder was substituting 1.0 for
a learned scaling of ~2.78 and the sha256 check could not see it: a file digest says which bytes
arrived, not which parameters were populated. Now built at global_pool=False (norm.weight loads with
mean 2.7755, std 0.4248) with the load report asserted against a measured allowlist. Any number
produced before that fix was measured on a different encoder.

STILL NOT THE OFFICIAL ENDPOINT, and the name should say so. Official `forward_features` returns
x[:, 0] (CLS alone) at global_pool=False and a mean-pooled vector at global_pool=True; neither is a
14x14 map, so a spatial baseline must choose. This one applies the pretrained `norm` official code
applies and then keeps the grid. Call it a DOFA-token spatial baseline.

Design (see dofa-baseline-integration memory):
  - FROZEN DOFA encoder -> 14x14x768 patch-token features; the features are
    CACHED once per (patch-set, degradation-state) since DOFA is frozen (deterministic), so only a
    light segmentation head is trained -> fast, reproducible.
  - Light trainable head: Conv-BN-GELU-Conv on the 14x14 tokens -> bilinear upsample -> per-pixel
    4-class cloud logits.
  - Normalisation: per-band mean/std from CloudSEN12 TRAIN (DOFA's downstream convention is
    dataset-specific normalization_stats(), so this is the correct + fair choice).
  - DOFA S2 = 9 bands [B4,B3,B2,B5,B6,B7,B8,B11,B12] (no B1/B8A/B9/B10). Degradation states:
      clean       : 9 L1C bands
      dropSWIR    : drop B11,B12 (channel-adaptive, 7 bands) -> controlled band loss
      L2A_real    : 9 real Sen2Cor L2A bands (operational TOA->BOA shift) -> the FLAGSHIP failure
  - Guard 2: calibration/evaluation split at PATCH level within the roi-disjoint TEST split.

Conformal (naive vs Mondrian) mirrors the audited phase8R logic, minus the CRC certificate: here
conformal_at_risk is a demonstration operating point only. phase8R carries the rigorous version.

Outputs (../paper/):
  results_phase8E_dofa.csv                 per state: acc/AURC/sel-AUROC + naive vs Mondrian risk/cov
  figs/fig_phase8E_dofa_conformal.pdf      naive vs degradation-aware achieved risk (DOFA)
--smoke writes both under a `_smoke` suffix. Its 16-patch calibration set puts the risk columns on
the B/(n+1) floor rather than at the real operating point, so it must never touch the paths above.

Usage:
  python experiments/phase8E_dofa.py --smoke
  python experiments/phase8E_dofa.py --seeds 0 1 2 --patches-train 800 --epochs 20 --device cuda
"""
import os, sys, csv, argparse, contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
from bandsim.reliability import (confidence_msp, aurc, selective_auroc,
                                 fit_temperature, conformal_at_risk)
from bandsim.metrics import miou, per_class_iou
from bandsim import hw
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
def P(rel):
    return os.path.join(PAPER_DIR, rel)

# DOFA Sentinel-2 config: 9 bands (microns) -> indices into our 13-band L1C layout
DOFA_BANDS = ["B4", "B3", "B2", "B5", "B6", "B7", "B8", "B11", "B12"]
DOFA_WL = [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]
# Import-time CONTRACT, not a lookup: .index() raises if a DOFA band name is absent from our
# L1C layout, so a typo in DOFA_BANDS fails here instead of opening the wrong .dat. The loader
# reads by NAME, which is why the list itself is never indexed with.
DOFA_IDX = [P8.L1C_BANDS.index(b) for b in DOFA_BANDS]
SIDE = P8.VALID_SIDE                                        # 509 valid content (drop 512 padding)
IMG = 224                                                  # DOFA input size
NUM_CLASSES = P8.NUM_CLASSES                               # 4 (clear/thick/thin/shadow)
CLASS_NAMES = P8.CLASS_NAMES
TARGET_RISK = 0.10
# degradation states: (name, product, dropped-DOFA-band-indices)
STATES = [("clean", "L1C", ()), ("dropSWIR", "L1C", (7, 8)), ("L2A_real", "L2A", ())]


# ------------------------------------------------------------------------------------- DOFA
# REPRODUCIBILITY PIN. `torch.hub.load("zhu-xlab/DOFA", ...)` tracks the default branch, and upstream
# hubconf.py in turn fetches the weights from the MUTABLE HuggingFace ref
# earthflow/DOFA @ main -> DOFA_ViT_base_e100.pth. Either can change under us and silently move every
# number in this file. So we pin BOTH refs and verify the checkpoint BYTES:
#   - DOFA_HUB_REF   : zhu-xlab/DOFA at an explicit commit (torch.hub accepts owner/repo:<ref>)
#   - DOFA_CKPT_SHA256: sha256 of DOFA_ViT_base_e100.pth, checked after download
# (We authored upstream PRs zhu-xlab/DOFA#31/#32/#33 adding lazy hub deps and a commit-pinned,
# sha256-verified checkpoint, but this must not depend on them being merged.)
DOFA_HUB_REF = "zhu-xlab/DOFA:8346385695912606f74e00ef601b5c598a27df78"   # master @ 2026-07
DOFA_CKPT_NAME = "DOFA_ViT_base_e100.pth"
DOFA_CKPT_SHA256 = "4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1"
# provenance of the hash: HF earthflow/DOFA commit 7a5219e48d2f8848511b0fabea7920a8836bc480,
# whose x-linked-etag for this file equals the sha256 above.
DOFA_HF_REVISION = "7a5219e48d2f8848511b0fabea7920a8836bc480"
# The revision now CONTROLS the download instead of only being recorded next to it. Upstream
# hubconf.py hardcodes ".../resolve/main/...", a mutable ref: pinning the commit of the CODE while
# letting it fetch the WEIGHTS from a moving pointer pins half the input.
DOFA_CKPT_URL = f"https://huggingface.co/earthflow/DOFA/resolve/{DOFA_HF_REVISION}/{DOFA_CKPT_NAME}"
# What `load_state_dict(strict=False)` is ALLOWED to leave behind, measured on this checkpoint at
# global_pool=False rather than assumed: `head` is the ImageNet-style classifier this script never
# uses, and `mask_token`/`projector` are pretraining-only modules the encoder does not carry.
# Anything else means the encoder is partially initialised and the run must stop.
DOFA_ALLOWED_MISSING = {"head.weight", "head.bias"}
DOFA_ALLOWED_UNEXPECTED = {"mask_token", "projector.weight", "projector.bias"}
# LICENSE. Two different licences, and conflating them in one line was wrong: the DOFA SOURCE
# (zhu-xlab/DOFA) is MIT; the WEIGHTS are distributed through the HuggingFace repo earthflow/DOFA,
# whose repository metadata marks CC-BY-4.0. Cite both, separately, and re-check the model card
# before submission.
DOFA_LICENSES = {"source": "MIT (zhu-xlab/DOFA)",
                 "checkpoint": "distributed via HF earthflow/DOFA; repo metadata marks CC-BY-4.0"}


@contextlib.contextmanager
def cudnn_disabled(active=True):
    """Scope `torch.backends.cudnn.enabled = False` to this block and put it back afterwards.

    The flag is PROCESS-global and was previously set once in main() and never restored, so every
    convolution executed later in the same interpreter silently switched backend. That is not just a
    speed change: measured on this box, the same conv2d gives bitwise-different results with cuDNN
    off (max|delta| 1.5e-5 on a random 8x16x64x64 input). Anything that imported or ran after
    phase8E — another experiment in a driver script, the next test in a pytest session — inherited
    both effects with nothing in its own output to say so.

    `active=False` is the CPU path: with no CUDA device the flag changes nothing about this run, so
    the only thing setting it would achieve is the leak."""
    prev = torch.backends.cudnn.enabled
    if active:
        torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = prev


def _sha256(path, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def verify_dofa_checkpoint(path=None):
    """sha256 an EXISTING checkpoint file; raise if it is not the pinned one. Returns the digest.

    Kept as its own function because it is the tested one (tests/test_reliability_guards.py pins
    both the mismatch and the missing-file behaviour) and because the two jobs are separable: this
    one answers "are these the right bytes", fetch_verified_checkpoint answers "get me a file that
    passes this". An earlier revision deleted it in favour of the fetcher, which swapped a covered
    function for an uncovered one."""
    path = path or os.path.join(torch.hub.get_dir(), "checkpoints", DOFA_CKPT_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DOFA checkpoint not found at {path}")
    got = _sha256(path)
    if got != DOFA_CKPT_SHA256:
        raise RuntimeError(
            f"DOFA checkpoint sha256 mismatch at {path}\n  expected {DOFA_CKPT_SHA256}\n"
            f"  got      {got}\nupstream weights changed -> the reported numbers are NOT "
            f"comparable. Re-pin deliberately (HF earthflow/DOFA rev {DOFA_HF_REVISION}) rather "
            f"than silently.")
    return got


def fetch_verified_checkpoint():
    """Path to DOFA weights whose BYTES are the pinned sha256, hashed BEFORE any torch.load.

    The order was backwards and it mattered. `torch.hub.load(pretrained=True)` downloads AND
    DESERIALISES upstream's `.../resolve/main/...` file, and only then did this script hash what had
    already been unpickled. A digest checked afterwards can refuse to REPORT numbers computed from
    the wrong bytes; it cannot stop those bytes being executed as a pickle first, which is what
    torch's serialization notes warn about. Fetches the pinned REVISION, never `main`."""
    ckdir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    pinned = os.path.join(ckdir, f"dofa_{DOFA_HF_REVISION[:12]}_{DOFA_CKPT_NAME}")
    # Reuse any local copy that already hashes correctly -- including the one upstream's loader may
    # have left behind -- so this does not re-download on a box that already holds the right bytes.
    for cand in (pinned, os.path.join(ckdir, DOFA_CKPT_NAME)):
        try:
            verify_dofa_checkpoint(cand)
            return cand
        except (FileNotFoundError, RuntimeError):
            continue
    print(f"[dofa] fetching pinned checkpoint from revision {DOFA_HF_REVISION[:12]} ...")
    torch.hub.download_url_to_file(DOFA_CKPT_URL, pinned, progress=True)
    try:
        verify_dofa_checkpoint(pinned)
    except RuntimeError:
        os.remove(pinned)                    # never leave bytes that failed the pin on disk
        raise
    return pinned


def load_dofa(device):
    """Frozen DOFA encoder with the PRETRAINED final norm actually loaded, and a load PROVEN
    complete rather than assumed. Two defects this replaces, both measured on the pinned checkpoint:

      * `vit_base_dofa` defaults to global_pool=True, and OFAViT builds `fc_norm` OR `norm`, never
        both (dofa_v1.py:38-43). The checkpoint carries `norm.weight/bias` and NO `fc_norm`, so at
        the default the pretrained normalisation was reported UNEXPECTED and dropped, while
        `fc_norm` stayed at LayerNorm's initialisation: measured fc_norm.weight mean 1.0000, std
        0.0000 -- provably untrained. At global_pool=False the same checkpoint loads norm.weight
        with mean 2.7755, std 0.4248. The encoder was substituting 1.0 for a learned scaling of
        ~2.78 and calling the result a frozen foundation-model baseline.
      * upstream loads with strict=False and DISCARDS the report, and a comment here claimed the
        sha256 "is what actually rules that out". It does not, and this checkpoint is the proof: the
        hash PASSED while four keys were missing. A file digest says which bytes arrived, not which
        parameters were populated. Both are now checked, separately, and the allowlist below was
        measured rather than guessed."""
    m = torch.hub.load(DOFA_HUB_REF, "vit_base_dofa", pretrained=False, trust_repo=True,
                       global_pool=False)
    ckpt = fetch_verified_checkpoint()
    digest = _sha256(ckpt)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    rep = m.load_state_dict(sd, strict=False)
    miss = sorted(set(rep.missing_keys) - DOFA_ALLOWED_MISSING)
    extra = sorted(set(rep.unexpected_keys) - DOFA_ALLOWED_UNEXPECTED)
    if miss or extra:
        raise RuntimeError(
            f"DOFA state_dict load is not the expected one.\n  unexpectedly MISSING: {miss}\n"
            f"  unexpectedly PRESENT: {extra}\n"
            f"strict=False would leave those parameters at their initialisation and report nothing. "
            f"Allowed missing {sorted(DOFA_ALLOWED_MISSING)}, allowed unexpected "
            f"{sorted(DOFA_ALLOWED_UNEXPECTED)}.")
    if hasattr(m, "fc_norm"):
        raise RuntimeError("DOFA model exposes fc_norm: it was built global-pooled, and this "
                           "checkpoint has no pretrained fc_norm to fill it.")
    if not hasattr(m, "norm"):
        raise RuntimeError("DOFA model has no `norm`; the pretrained final normalisation cannot load.")
    w = m.norm.weight.detach()
    if float(w.std()) < 1e-6:                # ones-initialised LayerNorm -> nothing was loaded
        raise RuntimeError(f"DOFA `norm` looks untrained (weight std {float(w.std()):.2e}); the "
                           f"pretrained normalisation did not load despite a clean key report.")
    from importlib.metadata import version as _pkgver
    print(f"DOFA hub ref {DOFA_HUB_REF}\n     checkpoint sha256 {digest} (pinned rev "
          f"{DOFA_HF_REVISION[:12]}, hashed BEFORE load)\n"
          f"     load: missing {sorted(rep.missing_keys)} / unexpected {sorted(rep.unexpected_keys)} "
          f"(all allowlisted)\n     norm.weight mean {float(w.mean()):.4f} std {float(w.std()):.4f} "
          f"(pretrained; 1.0000/0.0000 would mean it did not load)\n"
          f"     runtime: timm {_pkgver('timm')} / torchvision {_pkgver('torchvision')} "
          f"(timm builds the ViT PatchEmbed/Block; pinned in requirements-lock.txt)")
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval().to(device)


@torch.no_grad()
def dofa_features(m, x, wave_list):
    """(B,C,224,224) -> (B,768,14,14) frozen DOFA patch tokens, through the PRETRAINED final norm.

    This is official `forward_features` (patch_embed -> +pos -> CLS -> blocks -> norm) with ONE
    deliberate, stated departure: it keeps the SPATIAL GRID instead of what forward_features
    returns. Neither official ending yields a 14x14 map -- at global_pool=False it returns x[:, 0],
    the CLS token alone; at global_pool=True it mean-pools the patch tokens and only then applies
    fc_norm -- so a spatial baseline has to choose. This one applies the same pretrained `norm`
    official code applies at this point, then drops CLS. Call it a DOFA-token spatial baseline, not
    "official DOFA feature extraction".

    An earlier version applied `fc_norm` PER TOKEN. That is a position official code never uses AND,
    on this checkpoint, an untrained module (see load_dofa) -- so it normalised away each token's
    scale using an operator the pretraining never produced.

    `waves` is a LOCAL. forward_features assigns self.waves and then passes it to patch_embed as an
    argument, so the attribute is bookkeeping; writing it made every call mutate shared model state
    for no benefit, which is a hazard the moment two extractions share one model."""
    waves = torch.as_tensor(wave_list, device=x.device, dtype=torch.float32)
    # Shape/finiteness contract. Without it a changed patch size or input size surfaces as a reshape
    # error far from its cause -- or, when N happens to stay a perfect square, as a silently
    # transposed feature map that trains and scores like a real one.
    if x.ndim != 4:
        raise ValueError(f"dofa_features expects (B,C,H,W), got {tuple(x.shape)}")
    if x.shape[1] != waves.numel():
        raise ValueError(f"{x.shape[1]} input channels but {waves.numel()} wavelengths")
    if tuple(x.shape[-2:]) != (IMG, IMG):
        raise ValueError(f"DOFA input must be {IMG}x{IMG}, got {tuple(x.shape[-2:])}")
    if not torch.isfinite(x).all():
        raise ValueError("non-finite values in the DOFA input batch")
    if not (torch.isfinite(waves).all() and (waves > 0).all()):
        raise ValueError(f"wavelengths must be finite and positive, got {wave_list}")
    x, _ = m.patch_embed(x, waves)
    x = x + m.pos_embed[:, 1:, :]
    cls = (m.cls_token + m.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    for blk in m.blocks:
        x = blk(x)
    x = m.norm(x)                                           # PRETRAINED final norm (see load_dofa)
    tok = x[:, 1:, :]                                       # drop CLS -> (B, N, 768)
    B, N, D = tok.shape
    s = int(round(N ** 0.5))
    if s * s != N:
        raise RuntimeError(f"{N} patch tokens is not a square grid; the spatial reshape would be "
                           f"meaningless. Check the input size and the patch size.")
    return tok.transpose(1, 2).reshape(B, D, s, s)          # (B,768,14,14)


class DofaSegHead(nn.Module):
    """Light trainable decoder on frozen DOFA tokens -> per-pixel logits (DOFA stays frozen).

    `norm` is a SCIENTIFIC CONTROL, not a hyperparameter.

      "batch" (default, historical) is BatchNorm2d. The head trains only on CLEAN features, so its
      running mean/var ARE the clean feature distribution. Under dropSWIR or L2A_real the features
      move but those statistics do not, so part of any degradation is the DECODER's normalisation
      failing to track the shift rather than the ENCODER losing information. That is the realistic
      deployment -- you do deploy a clean-trained decoder -- but it makes the attribution
      "DOFA loses risk control" unavailable, because the pipeline that lost it includes a
      clean-frozen BN.
      "group" is GroupNorm, which normalises per sample and carries NO running statistics, so there
      is nothing to go stale. Any degradation that survives it is not the BN artefact.

    Run BOTH. The encoder-level claim is only what the two agree on; the difference is the size of
    the decoder confound and belongs in the paper next to the claim, not omitted from it."""
    def __init__(self, in_dim=768, mid=256, num_classes=NUM_CLASSES, norm="batch"):
        super().__init__()
        if norm not in ("batch", "group"):
            raise ValueError(f"head norm must be 'batch' or 'group', got {norm!r}")
        nrm = nn.BatchNorm2d(mid) if norm == "batch" else nn.GroupNorm(32, mid)
        self.norm_kind = norm
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, mid, 3, padding=1), nrm, nn.GELU(),
            nn.Conv2d(mid, num_classes, 1))

    def forward(self, feat14):                              # (B,768,14,14)
        y = self.head(feat14)                              # (B,num_classes,14,14)
        return F.interpolate(y, size=(IMG, IMG), mode="bilinear", align_corners=False)


# ------------------------------------------------------------------------------------- data
def load_spatial(split, product, patch_ids, return_info=False):
    """Load spatial patches for DOFA's 9 bands -> (X [P,9,224,224] float reflectance, Y [P,224,224]),
    or (X, Y, info) with return_info=True.

    Images bilinear+ANTIALIAS resized 509->224 (a 2.27x downsample; plain bilinear samples rather
    than averages); labels NEAREST-EXACT (categorical, must not interpolate, and plain "nearest"
    carries a half-pixel offset kept for OpenCV compatibility). No normalization here (caller
    applies TRAIN mean/std).

    Shares phase8's loader contract rather than re-implementing a laxer one: this function had the
    same holes load_split did — an assert-based proj_shape guard (deleted by `python -O`), no .dat
    size check, a hardcoded 512 instead of P8.SIDE, and unvalidated patch indices, where a negative
    id silently returned the LAST patch via numpy wrap-around."""
    import pandas as pd
    if product not in P8.PRODUCTS:
        raise ValueError(f"unknown product {product!r}; expected one of {sorted(P8.PRODUCTS)}")
    root = os.path.join(P8.DATA, split)
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    n = len(meta)
    if "proj_shape" in meta.columns:
        shp = set(meta["proj_shape"].unique())
        if not shp <= {SIDE}:
            raise ValueError(f"{split}: proj_shape {sorted(shp)} != {SIDE} — re-check the padding crop")
    pid = np.asarray(patch_ids)
    if pid.ndim != 1 or pid.size == 0:
        raise ValueError(f"patch_ids must be a non-empty 1-D array, got shape {pid.shape}")
    if not np.issubdtype(pid.dtype, np.integer):
        raise ValueError(f"patch_ids must be integers, got dtype {pid.dtype}")
    if pid.min() < 0 or pid.max() >= n:
        raise IndexError(f"patch_ids out of range for split {split!r} ({n} patches): "
                         f"{pid[(pid < 0) | (pid >= n)][:8]}")
    if np.unique(pid).size != pid.size:
        raise ValueError("patch_ids contains duplicates — the same patch would be loaded twice")
    V = P8.VALID_SIDE; o = P8.VALID_OFF                            # padding-corrected real 509x509 region [o:o+509]
    bmm = [P8._memmap_checked(os.path.join(root, f"{product}_{b}.dat"), np.int16, n)
           for b in DOFA_BANDS]
    lab = P8._memmap_checked(os.path.join(root, "LABEL_manual_hq.dat"), np.uint8, n)
    X, Y = [], []
    native_counts = np.zeros(NUM_CLASSES, np.int64)
    for p in pid:
        img = np.stack([np.asarray(bm[p][o:o + V, o:o + V]) for bm in bmm], 0).astype(np.float32) * 1e-4  # (9,509,509)
        ll = np.asarray(lab[p][o:o + V, o:o + V]).astype(np.int64)                                          # (509,509)
        # Validate the NATIVE labels here, not only the resized ones: an earlier version skipped a
        # bad patch's contribution to prevalence_509 while the range check ran on `Ys`, so a patch
        # whose bad pixels happened to be dropped by the resize vanished from one side of the
        # prevalence pair that provenance then presents as a like-for-like comparison.
        if ll.min() < 0 or ll.max() >= NUM_CLASSES:
            raise ValueError(f"{root}/LABEL_manual_hq.dat patch {int(p)} has labels outside "
                             f"[0,{NUM_CLASSES}): {np.unique(ll[(ll < 0) | (ll >= NUM_CLASSES)])[:8]}")
        native_counts += np.bincount(ll.ravel(), minlength=NUM_CLASSES)
        # antialias=True: 509->224 is a 2.27x DOWNSAMPLE, and plain bilinear at that ratio samples
        # rather than averages -- a bilinear kernel only ever reads 4 source pixels, so ~80% of each
        # cell is skipped and the result is aliased. For a physical quantity like reflectance the
        # low-pass average is the meaningful one. (Applies only when downsampling; a no-op otherwise.)
        xt = F.interpolate(torch.from_numpy(img)[None], size=(IMG, IMG), mode="bilinear",
                           align_corners=False, antialias=True)[0]                                # (9,224,224)
        # nearest-exact, not nearest: torch's plain "nearest" reproduces a known off-by-half buggy
        # alignment kept for OpenCV INTER_NEAREST compatibility, which shifts a categorical mask by
        # up to half a source pixel. On a class boundary that is a relabelled pixel, and this label
        # grid is the ground truth every risk number is scored against.
        yt = F.interpolate(torch.from_numpy(ll)[None, None].float(), size=(IMG, IMG),
                           mode="nearest-exact")[0, 0].long()                                     # (224,224)
        X.append(xt.numpy()); Y.append(yt.numpy())
    Ys = np.stack(Y, 0)
    # NEAREST resize cannot invent a label, so an out-of-range value here came off the disk. It would
    # otherwise reach CrossEntropyLoss as an "index out of bounds" from inside training, or — worse —
    # simply be counted as permanently wrong by the `argmax == y` accuracy and coverage arithmetic.
    if Ys.min() < 0 or Ys.max() >= NUM_CLASSES:
        raise ValueError(f"{root}/LABEL_manual_hq.dat produced labels outside [0,{NUM_CLASSES}): "
                         f"{np.unique(Ys[(Ys < 0) | (Ys >= NUM_CLASSES)])[:8]}")
    Xs = np.stack(X, 0).astype(np.float32)
    # Reflectance contract. `astype(float32) * 1e-4` accepted anything on the disk: a nodata
    # sentinel, a NaN, a band that is constant. A constant band then reaches the normaliser, where
    # `sd + 1e-6` divides a zero range by 1e-6 and turns rounding noise into a feature with unit
    # variance -- silently, and only in whichever band the product happens to have dropped.
    if not np.isfinite(Xs).all():
        raise ValueError(f"{root}: {int((~np.isfinite(Xs)).sum())} non-finite reflectance values "
                         f"({product}); a nodata sentinel or a corrupt .dat reached the model")
    band_sd = Xs.std(axis=(0, 2, 3))
    dead = [DOFA_BANDS[i] for i, s in enumerate(band_sd) if s < 1e-8]
    if dead:
        raise ValueError(f"{root}: {product} bands {dead} are constant across all patches; "
                         f"normalisation would divide their rounding noise by 1e-6")
    # The estimand changed at the resize and the size of the change belongs beside the result, not
    # in a reader's head: 509->224 merges ~5 source pixels into one, which preferentially erases the
    # thin/scattered classes a cloud benchmark is judged on.
    res_counts = np.bincount(Ys.ravel(), minlength=NUM_CLASSES).astype(np.int64)
    info = {"prevalence_509": (native_counts / max(native_counts.sum(), 1)).round(5).tolist(),
            "prevalence_224": (res_counts / max(res_counts.sum(), 1)).round(5).tolist(),
            "class_names": list(CLASS_NAMES), "reflectance_band_sd": band_sd.round(5).tolist()}
    shift = [f"{CLASS_NAMES[c]} {info['prevalence_509'][c]:.4f}->{info['prevalence_224'][c]:.4f}"
             for c in range(NUM_CLASSES)]
    print(f"  [{split}/{product}] class prevalence 509->224: " + ", ".join(shift))
    return (Xs, Ys, info) if return_info else (Xs, Ys)


def assert_products_aligned(Xa, Xb, tag, band=0, sample=24, rng=None, roi=None):
    """Prove L1C and L2A patch p really is the SAME scene, with a check that can fail.

    Replaces `np.array_equal(Ycal, Yca)`, which could not: load_spatial reads labels from
    `<split>/LABEL_manual_hq.dat` for BOTH products -- same directory, same file -- so that
    comparison held an array against a re-read of its own bytes while claiming to protect the one
    thing the L2A_real state depends on.

    RANK-BASED, with no additive margin. A first version required
    `median_r_same > median_r_decoy + 0.2`, which is UNSATISFIABLE the moment the decoy baseline
    exceeds 0.8: it then demands a correlation above 1.0 and blames patch ordering for the
    impossibility. CloudSEN12 makes that likely rather than exotic -- five patches per ROI share a
    footprint, so an index neighbour is usually a same-location sibling. It passed on this data only
    because cloud fields decorrelate between dates, which is a property of the scenes and not of the
    check. A per-patch WIN RATE has no such ceiling, needs no scale, and the decoy is drawn from a
    DIFFERENT ROI when roi ids are supplied so the baseline is not a sibling by construction.

    WHAT IT DOES NOT CATCH, stated because an earlier bullet overclaimed. Correlation is
    affine-invariant, so a fabricated `Xb = a*Xa + b` -- an "L2A" carrying no atmospheric content --
    passes with r = 1. This detects DUPLICATION and MIS-ORDERING. It is not evidence that the second
    product is a real atmospheric correction; nothing in this repo establishes that."""
    blank = {"n_compared": 0, "win_rate": float("nan"), "median_r_same": float("nan"),
             "median_r_decoy": float("nan")}
    if Xa.shape != Xb.shape:
        raise RuntimeError(f"{tag}: product shapes differ, {Xa.shape} vs {Xb.shape}")
    if np.array_equal(Xa, Xb):
        raise RuntimeError(f"{tag}: L1C and L2A arrays are IDENTICAL. One .dat is standing in for "
                           f"the other, so the L2A_real state carries no shift at all.")
    P_ = Xa.shape[0]
    if P_ < 2:
        # Same KEYS either way: an earlier version returned a different dict here and the caller,
        # which formats median_r_same_patch unconditionally, died with a KeyError naming nothing
        # relevant -- reachable at --patches-test 2 or a lopsided ROI split.
        return dict(blank, note="fewer than 2 patches; ordering not checkable")
    rng = rng or np.random.default_rng(4242)
    idx = rng.choice(P_, size=min(sample, P_), replace=False)
    roi = None if roi is None else np.asarray(roi)

    def _r(u, v):
        u = u.ravel().astype(np.float64); v = v.ravel().astype(np.float64)
        u = u - u.mean(); v = v - v.mean()
        d = float(np.sqrt((u * u).sum() * (v * v).sum()))
        return float((u * v).sum() / d) if d > 0 else np.nan

    same, decoy = [], []
    for p in idx:
        cand = (np.where(roi != roi[p])[0] if roi is not None and (roi != roi[p]).any()
                else np.setdiff1d(np.arange(P_), [p]))
        if cand.size == 0:
            continue
        q = int(rng.choice(cand))
        same.append(_r(Xa[p, band], Xb[p, band]))
        decoy.append(_r(Xa[p, band], Xb[q, band]))
    if not same:
        return dict(blank, note="no decoy available; ordering not checkable")
    same = np.array(same, float); decoy = np.array(decoy, float)
    ok = np.isfinite(same) & np.isfinite(decoy)
    win = float((same[ok] > decoy[ok]).mean()) if ok.any() else float("nan")
    out = {"n_compared": int(ok.sum()), "win_rate": round(win, 4),
           "median_r_same": round(float(np.nanmedian(same)), 4),
           "median_r_decoy": round(float(np.nanmedian(decoy)), 4)}
    if not (win >= 0.9):
        raise RuntimeError(
            f"{tag}: L1C/L2A patch ordering not established. The same patch across products beats a "
            f"decoy patch in only {win:.0%} of {out['n_compared']} comparisons (medians "
            f"{out['median_r_same']} vs {out['median_r_decoy']}); patch p of one product is "
            f"evidently not patch p of the other, so L2A_real would be scored against a different "
            f"scene than the L1C states.")
    return out


def test_patch_split(n_test, calib_frac=0.5, max_patches=None, seed=0, roi_ids=None):
    """Disjoint calib/eval TEST-patch ids (Guard 2), split BY ROI when roi_ids is given.

    A patch-INDEX split is not a location split. CloudSEN12's 975 test patches come from only 195
    distinct roi_id values -- five patches per location that differ by acquisition DATE, over an
    identical footprint (proj_geometry is constant within an roi_id). Measured at this script's
    default (max_patches=300, calib_frac=0.5): ~51% of calibration patches had a same-ROI sibling in
    the evaluation set, so calibration and evaluation were not independent and the naive-vs-Mondrian
    comparison was made easier than the operational one it stands for. This mirrors the same fix in
    phase8R.test_patch_split, which is the arm that carries the certificate.

    roi_ids=None keeps the old index split (and says so), so callers without metadata still work.
    Both sides must be NON-EMPTY: n_test=1 used to return calib={0}, eval={} and every eval metric
    then averaged over nothing. Ordinary calls are unaffected by the clamp."""
    # calib_frac OUTSIDE [0,1] used to be absorbed by the min/max clamp below into some non-empty
    # split, so an upstream config error produced a plausible run instead of an error. The endpoints
    # 0.0 and 1.0 are NOT rejected: the clamp deliberately keeps both sides non-empty there, that
    # behaviour is pinned by tests/test_reliability_guards.py for BOTH copies of this helper, and a
    # stricter rule here would silently diverge phase8E from the phase8R it claims to mirror.
    # DIVERGENCE, stated: phase8R does not yet reject out-of-range values at all.
    cf = float(calib_frac)
    if not np.isfinite(cf) or not (0.0 <= cf <= 1.0):
        raise ValueError(f"calib_frac must be a finite fraction in [0,1], got {calib_frac!r}")
    rng = np.random.default_rng(70000 + seed)
    ids = np.arange(n_test)
    if max_patches is not None and max_patches < n_test:
        ids = rng.choice(n_test, size=max_patches, replace=False)
    if len(ids) < 2:
        raise ValueError(f"need >= 2 test patches for a disjoint calib/eval split, got {len(ids)}")
    if roi_ids is None:
        print("[warn] test_patch_split: no roi_ids given -- falling back to a patch-INDEX split. "
              "Calibration and evaluation may share locations; this is NOT an ROI-disjoint split.")
        rng.shuffle(ids)
        k = min(max(1, int(len(ids) * calib_frac)), len(ids) - 1)  # both sides non-empty
        return np.sort(ids[:k]), np.sort(ids[k:])
    r = np.asarray(roi_ids)
    # ndim: a shape-(n,1) column of ROI ids passes the length check, then `ids[np.isin(r, ...)]`
    # indexes with a 2-D boolean mask and dies with "too many indices" far from the cause.
    if r.ndim != 1:
        raise ValueError(f"roi_ids must be 1-D, got shape {r.shape}")
    if r.shape[0] != n_test:
        raise ValueError(f"roi_ids has {r.shape[0]} entries for {n_test} test patches")
    # NaN: np.unique KEEPS NaN, but np.isin(NaN, [NaN]) is False, so a patch with a missing ROI id
    # joins NEITHER side and vanishes from the experiment without changing any count that is printed.
    if r.dtype.kind == "f" and np.isnan(r).any():
        raise ValueError(f"roi_ids contains {int(np.isnan(r).sum())} NaN entries; those patches "
                         f"would silently fall out of both calibration and evaluation")
    r = r[ids]
    uniq = np.unique(r)
    if len(uniq) < 2:
        raise ValueError(f"need >= 2 distinct ROIs for an ROI-disjoint split, got {len(uniq)} "
                         f"from {len(ids)} patches")
    perm = rng.permutation(uniq)
    kr = min(max(1, int(len(perm) * calib_frac)), len(perm) - 1)
    cal = np.sort(ids[np.isin(r, perm[:kr])])
    ev = np.sort(ids[np.isin(r, perm[kr:])])
    # The split must PARTITION the sampled ids: no patch in both, none lost. Neither is guaranteed
    # by construction once roi_ids can hold anything np.isin treats as unequal to itself, and a lost
    # patch changes nothing visible -- the counts that get printed are len(cal) and len(ev), which
    # stay self-consistent while the union quietly shrinks.
    if np.intersect1d(cal, ev).size:
        raise RuntimeError(f"calib/eval overlap in {np.intersect1d(cal, ev).size} patches")
    if not np.array_equal(np.sort(np.concatenate([cal, ev])), np.sort(ids)):
        lost = np.setdiff1d(ids, np.concatenate([cal, ev]))
        raise RuntimeError(f"the ROI split lost {lost.size} of {ids.size} patches (e.g. {lost[:8]}); "
                           f"they would appear in neither calibration nor evaluation")
    return cal, ev


# ------------------------------------------------------------------------- frozen-feature cache
@torch.no_grad()
def extract_features(dofa, X, keep, device, bs=16):
    """X (P,9,224,224) normalized -> frozen DOFA features (P,768,14,14) using the `keep` band subset
    (channel-adaptive). Computed once per (patch-set, state) and reused across seeds."""
    wl = [DOFA_WL[i] for i in keep]
    Xt = torch.from_numpy(X[:, keep])                      # (P,|keep|,224,224)
    feats = []
    for s in range(0, Xt.shape[0], bs):
        feats.append(dofa_features(dofa, Xt[s:s + bs].to(device), wl).cpu())
    return torch.cat(feats, 0)                              # (P,768,14,14) on CPU


def train_head(feat_tr, Ytr, device, epochs, bs, lr, seed, head_norm="batch"):
    """Train the seg head on cached frozen features. feat_tr (P,768,14,14) CPU, Ytr (P,224,224)."""
    torch.manual_seed(seed)
    head = DofaSegHead(norm=head_norm).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Yt = torch.from_numpy(Ytr)
    n = feat_tr.shape[0]
    rng = np.random.default_rng(seed)
    head.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            fb = feat_tr[idx].to(device)
            yb = Yt[idx].to(device)
            logits = head(fb)                              # (b,4,224,224)
            loss = lossf(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def head_logits_perpix(head, feat, device, bs=32):
    """Cached features -> per-pixel logits (P*224*224, 4) in (patch,row,col) order (matches
    Y.reshape(-1))."""
    out = []
    for s in range(0, feat.shape[0], bs):
        lg = head(feat[s:s + bs].to(device))               # (b,4,224,224)
        out.append(lg.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES).cpu().numpy())
    return np.concatenate(out)


def _subsample(rng, n, k):
    return rng.choice(n, size=min(k, n), replace=False)


def _fmt(x, prec=2):
    """A statistic, or an EMPTY cell when it is undefined. Never the string "nan".

    Demonstrated, not stylistic: `conformal_at_risk` returns NaN when a threshold accepts nothing
    (correctly -- 0 would read as "no errors"), and phase8E wrote that straight into its CSV. Fed to
    integrity_check.csv_finite_and_sane -- the harness that runs THIS script at THIS command --
    the file is rejected outright: float("nan") parses, then trips the finiteness test. An empty
    cell hits `except ValueError: continue` and is skipped, and reads as "undefined" to pandas and
    to a human. Writing 0.00 instead would be worse than either.

    INF counts as non-finite here, and a first version of this helper missed that: when the target
    risk is unreachable `conformal_at_risk` returns threshold=+inf, which the per-seed CSV then
    carried as the literal "inf" -- and the same harness check rejects `inf` for the same reason it
    rejects `nan`. Nothing is lost by blanking it: `*_acc_n == 0` in the same row already says the
    threshold accepted nothing, unambiguously and as an integer."""
    x = float(x)
    return "" if not np.isfinite(x) else f"{x:.{prec}f}"


def design_effect(x, group):
    """How much the clustering inflates variance: n_eff = n / (1 + (m_bar - 1) * ICC).

    WHY THIS IS HERE. `conformal_at_risk`'s conservative margin is z*sqrt(p(1-p)/n) with n = the
    number of ROWS it was handed, and this script hands it ~400 pixels per patch. Neighbouring
    pixels of one cloud field are not independent draws, so the margin is computed on a sample size
    the data does not have and the plug-in point is anti-conservative by roughly sqrt(deff). The
    function cannot be told about groups (its signature has no `calib_group`; only
    `conformal_risk_control` does, and bandsim/reliability.py belongs to another branch), so this
    MEASURES the discrepancy instead of hiding it: report n, the number of exchangeable units, and
    the effective n the margin should have used.

    ICC from the one-way random-effects decomposition of the correctness indicator, with the
    unequal-cluster-size m0 rather than a plain mean, and clipped to [0,1] because the moment
    estimator can go slightly negative on noise."""
    x = np.asarray(x, float)
    g = np.asarray(group)
    uniq, inv = np.unique(g, return_inverse=True)
    k, nn_ = uniq.size, x.size
    blank = {"n_rows": int(nn_), "n_clusters": int(k), "icc": float("nan"),
             "design_effect": float("nan"), "n_effective": float("nan")}
    if k < 2 or nn_ <= k:
        return blank
    cnt = np.bincount(inv, minlength=k).astype(float)
    gm = np.bincount(inv, weights=x, minlength=k) / np.maximum(cnt, 1.0)
    msb = float((cnt * (gm - x.mean()) ** 2).sum()) / (k - 1)
    msw = float(((x - gm[inv]) ** 2).sum()) / (nn_ - k)
    m0 = (nn_ - (cnt ** 2).sum() / nn_) / (k - 1)
    den = msb + (m0 - 1.0) * msw
    icc = float(np.clip((msb - msw) / den, 0.0, 1.0)) if den > 0 else 0.0
    deff = 1.0 + (nn_ / k - 1.0) * icc
    return {"n_rows": int(nn_), "n_clusters": int(k), "icc": round(icc, 5),
            "design_effect": round(deff, 3), "n_effective": round(nn_ / max(deff, 1e-9), 1)}


def roi_bootstrap_risk(corr_ev, conf_ev, thr, group, n_boot=2000, seed=8125, alpha=0.05):
    """Cluster-bootstrap CI of the achieved conditional risk at a FIXED threshold.

    Resamples ROIs, not pixels: the pixels of one location are one draw, so a pixel bootstrap would
    reproduce the same understated spread the plug-in margin already has. The threshold is held
    fixed because this quantifies uncertainty in the ACHIEVED risk, not in the selection.

    Accumulates per-ROI (accepted, accepted-and-wrong) once and resamples those two counts, which is
    exact and avoids materialising ~60k indices per replicate."""
    sel = np.asarray(conf_ev, float) >= float(thr)
    wrong = (np.asarray(corr_ev) == 0) & sel
    uniq, inv = np.unique(np.asarray(group), return_inverse=True)
    k = uniq.size
    if k < 2:
        return {"lo": float("nan"), "hi": float("nan"), "n_clusters": int(k)}
    acc = np.bincount(inv, weights=sel.astype(float), minlength=k)
    err = np.bincount(inv, weights=wrong.astype(float), minlength=k)
    pick = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    A = acc[pick].sum(1); E = err[pick].sum(1)
    r = np.where(A > 0, E / np.maximum(A, 1.0), np.nan) * 100.0
    r = r[~np.isnan(r)]
    if r.size < n_boot // 10:                 # mostly zero-acceptance replicates -> not an interval
        return {"lo": float("nan"), "hi": float("nan"), "n_clusters": int(k)}
    return {"lo": round(float(np.quantile(r, alpha / 2)), 3),
            "hi": round(float(np.quantile(r, 1 - alpha / 2)), 3), "n_clusters": int(k)}


def _accept_counts(thr, conf_ev, corr_ev):
    """Exact (accepted, accepted-and-wrong, evaluated) for a threshold.

    Recomputed with conformal_at_risk's OWN acceptance rule (`sel = eval_conf >= thr`,
    reliability.py) rather than back-derived from the returned coverage and risk. Those are rounded
    floats, and at zero acceptance `risk` is NaN by design -- multiplying them back would give NaN
    where the truth is 0 errors out of 0 accepted, which is exactly the case that has to be
    countable for the pooled estimator below."""
    sel = np.asarray(conf_ev, float) >= float(thr)
    ce = np.asarray(corr_ev)
    return int(sel.sum()), int((ce[sel] == 0).sum()), int(sel.size)


def reliability_dofa(logits_by_state_cal, y_cal, logits_by_state_ev, y_ev, target=TARGET_RISK,
                     roi_cal=None, roi_ev=None):
    """Naive(clean-calibrated) vs Mondrian(state-calibrated) plug-in risk operating point per state.
    Mirrors the AUDITED phase8R.reliability_over_states logic (same math) for the plug-in arms only.
    Every `*_risk` returned is the CONDITIONAL P(wrong | accepted), NaN when nothing is accepted;
    none of it is CRC-certified. Read each risk with its `*_cov`.

    Returns the COUNTS behind every rate as well. Without them a multi-seed summary cannot be
    formed honestly: averaging conditional risks across seeds weights a seed that accepted 100
    pixels the same as one that accepted 100,000, which is a different estimand from the pooled
    error rate, and a seed that accepted nothing turns the whole state's mean into NaN.

    CALIBRATION REUSE, stated because it is a real limitation and not visible in the output. The
    temperature is fitted on `y_cal` and the threshold is then selected on the SAME labels, so the
    scores the threshold sees have already been tuned on them. Split conformal wants the score
    function fixed outside the calibration set. The effect is bounded here -- one scalar fitted on
    tens of thousands of pixels -- and this file certifies nothing, but it is one more reason the
    plug-in point is not a bound and phase8R is the arm that carries the guarantee."""
    lgc0 = logits_by_state_cal["clean"]
    T_clean = fit_temperature(lgc0, y_cal)
    conf_cal_clean = confidence_msp(lgc0 / T_clean)
    corr_cal_clean = (lgc0.argmax(1) == y_cal).astype(int)
    out = {}
    for name in logits_by_state_cal:
        lgc, lge = logits_by_state_cal[name], logits_by_state_ev[name]
        corr_cal = (lgc.argmax(1) == y_cal).astype(int)
        corr_ev = (lge.argmax(1) == y_ev).astype(int)
        T = fit_temperature(lgc, y_cal)
        conf_ev = confidence_msp(lge / T)
        conf_cal = confidence_msp(lgc / T)
        # THE MARGIN IS SIZED FROM THE ROI, NOT THE PIXEL (PR #6, merged into #20's structure).
        # Without calib_group the finite-sample margin counted every sampled pixel as independent
        # evidence, though pixels come in blocks from one 512x512 patch and patches in fives from
        # one location -- measured here: ICC 0.38-0.41, design effect 77-82, so ~1600 calibration
        # pixels carry ~19.5 independent units and the margin was ~9x understated. phase8E has no
        # CRC arm to fall back on; this plug-in point is the whole reliability claim. deff/bootstrap
        # below stay as DESCRIPTIVE companions; calib_group is what CORRECTS the operating margin.
        mond = conformal_at_risk(corr_cal, conf_cal, corr_ev, conf_ev, target_risk=target,
                                 calib_group=roi_cal)
        conf_ev_clean = confidence_msp(lge / T_clean)
        naive = conformal_at_risk(corr_cal_clean, conf_cal_clean, corr_ev, conf_ev_clean,
                                   target_risk=target, calib_group=roi_cal)
        m_n, m_e, n_ev = _accept_counts(mond["threshold"], conf_ev, corr_ev)
        v_n, v_e, _ = _accept_counts(naive["threshold"], conf_ev_clean, corr_ev)
        # The exchangeable unit is the LOCATION, not the pixel. deff says how far the plug-in
        # margin's n is from the sample size the data actually has; the bootstrap gives an interval
        # that respects it. Both are descriptive -- neither turns the operating point into a bound.
        deff = design_effect(corr_cal, roi_cal) if roi_cal is not None else {}
        bm = (roi_bootstrap_risk(corr_ev, conf_ev, mond["threshold"], roi_ev)
              if roi_ev is not None else {"lo": float("nan"), "hi": float("nan")})
        bn = (roi_bootstrap_risk(corr_ev, conf_ev_clean, naive["threshold"], roi_ev)
              if roi_ev is not None else {"lo": float("nan"), "hi": float("nan")})
        # Overall accuracy on 4 imbalanced cloud classes is dominated by clear and thick; the
        # classes a cloud benchmark is actually judged on (thin, shadow) can collapse without
        # moving it. mIoU and per-class IoU are computed on the SUBSAMPLED evaluation pixels, the
        # same population every risk column uses, so they are comparable to them and not to a
        # full-frame number.
        pred_ev = lge.argmax(1)
        pc = per_class_iou(y_ev, pred_ev, NUM_CLASSES)
        out[name] = {
            "acc": float(corr_ev.mean()) * 100,
            "miou": float(miou(y_ev, pred_ev, NUM_CLASSES)),
            **{f"iou_{CLASS_NAMES[c]}": float(pc[c]) for c in range(NUM_CLASSES)},
            "aurc": aurc(corr_ev, conf_ev) * 100,
            # selective_auroc already returns NaN when only one class is present (all-correct or
            # all-wrong evaluation subsets), so the undefined case arrives labelled rather than as a
            # fabricated 50 or 100; _fmt keeps it out of the CSV as an empty cell.
            "auroc": selective_auroc(corr_ev, conf_ev) * 100,
            "T_state": float(T), "T_clean": float(T_clean), "n_eval_px": n_ev,
            "mond_risk": mond["risk"] * 100, "mond_cov": mond["coverage"] * 100,
            "mond_thr": float(mond["threshold"]), "mond_acc_n": m_n, "mond_acc_err": m_e,
            "naive_risk": naive["risk"] * 100, "naive_cov": naive["coverage"] * 100,
            "naive_thr": float(naive["threshold"]), "naive_acc_n": v_n, "naive_acc_err": v_e,
            "mond_risk_roi_lo": bm["lo"], "mond_risk_roi_hi": bm["hi"],
            "naive_risk_roi_lo": bn["lo"], "naive_risk_roi_hi": bn["hi"],
            # From the CONFORMAL RESULT itself (PR #6, restored by the post-merge audit): the n and
            # ICC the returned margin was ACTUALLY sized from. deff's calib_* columns beside them
            # are the descriptive companion computed separately -- the pair lets a reader check the
            # two estimators against each other instead of trusting either.
            "plugin_margin_neff": float(mond.get("n_calib_units", float("nan"))),
            "plugin_margin_icc": float(mond.get("icc", float("nan"))),
            "calib_icc": deff.get("icc", float("nan")),
            "calib_design_effect": deff.get("design_effect", float("nan")),
            "calib_n_effective": deff.get("n_effective", float("nan")),
            "calib_n_rows": deff.get("n_rows", 0), "calib_n_rois": deff.get("n_clusters", 0),
        }
    return out


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--patches-train", type=int, default=800)
    ap.add_argument("--patches-test", type=int, default=300, help="test patches (calib+eval)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--px-per-patch", type=int, default=400, help="pixels/patch subsampled for reliability")
    ap.add_argument("--head-norm", default="batch", choices=["batch", "group"],
                    help="decoder normalisation. 'batch' (default, historical) carries CLEAN "
                         "running statistics into every shifted state, so degradation mixes a "
                         "decoder artefact with the encoder's; 'group' has no running stats. "
                         "The encoder-level claim is what the two runs AGREE on -- run both.")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed for the ROI calib/eval split. Separate from --seeds ON PURPOSE: "
                         "--seeds varies ONLY the head initialisation and minibatch order, so "
                         "the spread it produces is decoder-optimisation variance, not split or "
                         "sampling uncertainty. Vary this to see the latter.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    ap.add_argument("--nondeterministic", action="store_true")
    args = ap.parse_args()
    bad = []
    if args.patches_train < 1: bad.append(f"--patches-train must be >= 1 (got {args.patches_train})")
    if args.patches_test < 2: bad.append(f"--patches-test must be >= 2 for a calib/eval split "
                                        f"(got {args.patches_test})")
    if args.epochs < 1: bad.append(f"--epochs must be >= 1 (got {args.epochs}); 0 would write a "
                                   f"deliverable from an untrained head")
    if args.bs < 1: bad.append(f"--bs must be >= 1 (got {args.bs})")
    if args.lr <= 0: bad.append(f"--lr must be > 0 (got {args.lr})")
    if not 1 <= args.px_per_patch <= IMG * IMG:
        bad.append(f"--px-per-patch must be in [1,{IMG*IMG}] (got {args.px_per_patch}); larger "
                   f"values were silently clamped by min(k,n)")
    if len(set(args.seeds)) != len(args.seeds):
        bad.append(f"--seeds must be unique (got {args.seeds}); a repeat re-weights that run")
    if bad:
        sys.exit("ERROR:\n  " + "\n  ".join(bad))
    # Conformal calibration on 16 test patches is not a weaker version of the real result, it is a
    # different one: the risk/coverage columns are dominated by the B/(n+1) floor at that n. Sharing
    # the deliverable's filename would let that pass for the 300-patch run. Suffix smoke output.
    sfx = ""
    if args.smoke:
        args.seeds = [0]; args.patches_train = 24; args.patches_test = 16; args.epochs = 3
        args.px_per_patch = 200
        sfx = "_smoke"
        print("[smoke] 1 seed / 3 epochs / 16 test patches — writing *_smoke artefacts, NOT the deliverables")
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device)
    dev = hw.device(args.device)
    # This box's forward-compat driver (host libcuda 535, via scripts/gpu_env.sh) cannot
    # initialize the cu124 cuDNN 9.2.2 CONV kernels (CUDNN_STATUS_NOT_INITIALIZED). The native
    # CUDA conv works and is deterministic. DOFA's only conv is its patch_embed (the ViT body is
    # matmul-bound via cuBLAS, which is fine), so disabling cuDNN costs essentially nothing here.
    # SCOPED (see cudnn_disabled): the workaround is this run's, not the interpreter's. Same for the
    # serif/9pt rcParams, which used to stay applied to every later figure in the process.
    on_cuda = dev.type == "cuda"
    print("HW:", hw.info(), "| device", dev,
          "| cudnn disabled for this run (env conv workaround)" if on_cuda else "| cudnn untouched (cpu)")
    with cudnn_disabled(on_cuda), plt.rc_context({"font.size": 9, "font.family": "serif"}):
        _run(args, dev, sfx)


def _run(args, dev, sfx):
    """main()'s body, called inside the cuDNN/rcParams scope so both are restored on ANY exit."""
    import pandas as pd
    os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)   # in _run, not at import time
    n_train = len(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv")))
    n_test = len(pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv")))
    rng = np.random.default_rng(2024)
    train_ids = np.sort(rng.choice(n_train, size=min(args.patches_train, n_train), replace=False))
    # Split by SCENE-COMPONENT, not by patch index or by raw roi_id (P0-2): CloudSEN12 s2_id products
    # that span two roi_ids make a roi-disjoint split NOT scene-disjoint, and phase8E's plug-in point
    # IS the whole reliability claim (no CRC arm), so the leak matters. scene_component_ids unions
    # ROIs sharing any s2_id (195 -> 184). This is the exchangeable unit for the split AND roi_all.
    calib_ids, eval_ids = test_patch_split(n_test, calib_frac=0.5, max_patches=args.patches_test,
                                           seed=args.split_seed, roi_ids=P8.scene_component_ids("test"))
    roi_all = P8.scene_component_ids("test")
    print(f"loading spatial patches: train {len(train_ids)} | calib {len(calib_ids)} "
          f"({len(set(roi_all[calib_ids]))} ROIs) | eval {len(eval_ids)} "
          f"({len(set(roi_all[eval_ids]))} ROIs, {len(set(roi_all[calib_ids]) & set(roi_all[eval_ids]))} shared) ...")

    Xtr, Ytr, info_tr = load_spatial("train", "L1C", train_ids, return_info=True)
    # per-band TRAIN normalization (DOFA downstream convention: dataset-specific stats)
    mu = Xtr.mean(axis=(0, 2, 3)); sd = Xtr.std(axis=(0, 2, 3)) + 1e-6
    norm = lambda A: ((A - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float32)
    Xtr = norm(Xtr)
    Xcal_l1c, Ycal, info_cal = load_spatial("test", "L1C", calib_ids, return_info=True); Xcal_l1c = norm(Xcal_l1c)
    Xev_l1c, Yev, info_ev = load_spatial("test", "L1C", eval_ids, return_info=True); Xev_l1c = norm(Xev_l1c)
    Xcal_l2a, Yca, _ = load_spatial("test", "L2A", calib_ids, return_info=True); Xcal_l2a = norm(Xcal_l2a)
    Xev_l2a, Yea, _ = load_spatial("test", "L2A", eval_ids, return_info=True); Xev_l2a = norm(Xev_l2a)
    # The labels are the SAME FILE for both products by construction (load_spatial reads
    # <split>/LABEL_manual_hq.dat either way), so comparing Ycal against Yca could never fail and
    # said nothing about the thing at risk. Check what can actually be wrong instead: that the two
    # products differ at all, and that patch p of one IS patch p of the other.
    if not (np.array_equal(Ycal, Yca) and np.array_equal(Yev, Yea)):
        raise RuntimeError("the same label file read twice produced different arrays — the loader "
                           "or the memmap is not deterministic")
    align = {"calib": assert_products_aligned(Xcal_l1c, Xcal_l2a, "calib", roi=roi_all[calib_ids]),
             "eval": assert_products_aligned(Xev_l1c, Xev_l2a, "eval", roi=roi_all[eval_ids])}
    print(f"L1C/L2A alignment verified: same-patch beats an off-ROI decoy in "
          f"{align['calib']['win_rate']:.0%} of calib and {align['eval']['win_rate']:.0%} of eval "
          f"comparisons (median r {align['calib']['median_r_same']} vs "
          f"{align['calib']['median_r_decoy']})")
    print(f"train {Xtr.shape} | calib {Xcal_l1c.shape} | eval {Xev_l1c.shape} | target risk {TARGET_RISK:.0%}")

    dofa = load_dofa(dev)
    print(f"DOFA loaded (frozen, {sum(p.numel() for p in dofa.parameters())/1e6:.0f}M params). "
          f"Extracting frozen features (cached once per state) ...")
    keep_all = list(range(len(DOFA_BANDS)))
    feat_tr = extract_features(dofa, Xtr, keep_all, dev)                       # clean train features
    # per-state calib/eval features (frozen -> compute once)
    feat_cal, feat_ev = {}, {}
    src = {"L1C": (Xcal_l1c, Xev_l1c), "L2A": (Xcal_l2a, Xev_l2a)}
    for name, product, drop in STATES:
        keep = [i for i in range(len(DOFA_BANDS)) if i not in drop]
        Xc, Xe = src[product]
        feat_cal[name] = extract_features(dofa, Xc, keep, dev)
        feat_ev[name] = extract_features(dofa, Xe, keep, dev)
    print("features cached. training head per seed + reliability ...")

    # per-pixel label subsample indices (consistent across states; same pixels for a patch-set)
    rs = np.random.default_rng(999)
    def flat_labels_and_idx(Y, k):
        yf = Y.reshape(-1)
        idx = np.concatenate([_subsample(rs, IMG * IMG, k) + p * IMG * IMG for p in range(Y.shape[0])])
        return yf[idx], idx
    y_cal_s, idx_cal = flat_labels_and_idx(Ycal, args.px_per_patch)
    y_ev_s, idx_ev = flat_labels_and_idx(Yev, args.px_per_patch)
    # ROI of every SUBSAMPLED pixel: flat index -> patch -> ROI. This is the exchangeable unit the
    # conformal arms are actually calibrated on, and until now nothing in this file knew it.
    roi_cal_px = roi_all[calib_ids][idx_cal // (IMG * IMG)]
    roi_ev_px = roi_all[eval_ids][idx_ev // (IMG * IMG)]
    # PR #6's guard, restored by the post-merge audit: if the unit ids ever desynchronise from the
    # sampled pixels, every grouped margin downstream is sized from the wrong clustering -- and
    # nothing else would notice, because the arrays still broadcast.
    if roi_cal_px.size != y_cal_s.size or roi_ev_px.size != y_ev_s.size:
        raise ValueError(f"unit ids desynchronised from the sampled pixels "
                         f"({roi_cal_px.size} vs {y_cal_s.size}, {roi_ev_px.size} vs {y_ev_s.size})")
    print(f"exchangeable units: {len(set(roi_cal_px))} calib ROIs / {y_cal_s.size} calib pixels, "
          f"{len(set(roi_ev_px))} eval ROIs / {y_ev_s.size} eval pixels")

    metrics = (["acc", "miou"] + [f"iou_{c}" for c in CLASS_NAMES]
               + ["aurc", "auroc", "mond_risk", "mond_cov", "naive_risk", "naive_cov",
                  "mond_risk_roi_lo", "mond_risk_roi_hi", "naive_risk_roi_lo", "naive_risk_roi_hi",
                  "plugin_margin_neff", "plugin_margin_icc",
                  "calib_icc", "calib_design_effect", "calib_n_effective", "calib_n_rows",
                  "calib_n_rois"])
    rows = []                       # RAW per-seed rows; every aggregate below derives from these
    agg = {s: {m: [] for m in metrics} for s, _, _ in STATES}
    counts = {s: {k: 0 for k in ("mond_acc_n", "mond_acc_err", "naive_acc_n", "naive_acc_err")}
              for s, _, _ in STATES}
    for seed in args.seeds:
        head = train_head(feat_tr, Ytr, dev, args.epochs, args.bs, args.lr, seed,
                          head_norm=args.head_norm)
        lg_cal = {name: head_logits_perpix(head, feat_cal[name], dev)[idx_cal] for name, _, _ in STATES}
        lg_ev = {name: head_logits_perpix(head, feat_ev[name], dev)[idx_ev] for name, _, _ in STATES}
        res = reliability_dofa(lg_cal, y_cal_s, lg_ev, y_ev_s,
                               roi_cal=roi_cal_px, roi_ev=roi_ev_px)
        for name, _, _ in STATES:
            r = res[name]
            for m in metrics:
                agg[name][m].append(r[m])
            for k in counts[name]:
                counts[name][k] += int(r[k])
            # Counts stay OUT of the row for the >1e4 reason above; coverage x n_eval_px recovers
            # them exactly, and n_eval_px is in provenance.
            rows.append({"seed": seed, "state": name,
                         **{k: r[k] for k in sorted(r)
                            if not k.endswith(("_acc_n", "_acc_err"))
                            and k not in ("n_eval_px", "calib_n_rows", "calib_n_effective")}})
        print(f"  seed {seed}: L2A naive_risk={res['L2A_real']['naive_risk']:.1f} "
              f"mondrian_risk={res['L2A_real']['mond_risk']:.1f} (target {TARGET_RISK*100:.0f})")

    # NaN here means "this seed accepted nothing", which is a fact about that seed, not a value to
    # average in. np.mean over it made ONE such seed erase an entire state's column -- and it is
    # reachable: a smoke run at the harness's own command produced exactly that for dropSWIR.
    def _nm(v):
        a = np.asarray(v, float); f = a[~np.isnan(a)]
        return float(f.mean()) if f.size else float("nan")

    def _ns(v):        # ddof=1: these seeds are a sample of head initialisations, not the population
        a = np.asarray(v, float); f = a[~np.isnan(a)]
        return float(f.std(ddof=1)) if f.size > 1 else float("nan")

    mean = {s: {m: _nm(agg[s][m]) for m in metrics} for s, _, _ in STATES}
    std = {s: {m: _ns(agg[s][m]) for m in metrics} for s, _, _ in STATES}
    # TWO estimands, because they are not the same number. Averaging per-seed conditional risks
    # weights a seed that accepted 100 pixels like one that accepted 100,000; POOLING the accepted
    # errors over the accepted count is the error rate among all predictions actually made. Report
    # both, plus how many seeds were undefined, so neither can be mistaken for the other.
    pooled = {s: {a: (100.0 * counts[s][f"{a}_acc_err"] / counts[s][f"{a}_acc_n"]
                      if counts[s][f"{a}_acc_n"] else float("nan")) for a in ("mond", "naive")}
              for s, _, _ in STATES}
    n_undef = {s: {a: int(sum(1 for v in agg[s][f"{a}_risk"] if v != v)) for a in ("mond", "naive")}
               for s, _, _ in STATES}
    n = len(args.seeds)

    # per-seed RAW csv -- thresholds, temperatures and accepted counts included, so any interval or
    # alternative estimand can be recomputed without re-running the experiment.
    with open(P(f"results_phase8E_dofa_perseed{sfx}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (_fmt(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    with open(P(f"results_phase8E_dofa{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # *_plugin_cond_risk = CONDITIONAL P(wrong|accepted) from the plug-in point; no CRC here, so
        # no column may be read as certified. Each risk ships with its coverage.
        w.writerow(["model", "state", "acc", "mIoU"] + [f"IoU_{c}" for c in CLASS_NAMES]
                   + ["AURC", "sel_AUROC",
                    "mondrian_plugin_cond_risk", "mondrian_plugin_cov",
                    "naive_plugin_cond_risk", "naive_plugin_cov",
                    # pooled = accepted errors / accepted, over ALL seeds: the error rate among the
                    # predictions actually made. The *_cond_risk columns are the mean over seeds of
                    # each seed's own rate. They differ whenever acceptance differs by seed.
                    # "seed_weighted" and not "pooled_px": the denominator sums over seeds that
                    # SHARE the same evaluation pixels, so it is n_seeds x distinct pixels. The rate
                    # is a legitimate acceptance-weighted average; its n is NOT a sample size and a
                    # binomial interval built on it would be ~sqrt(n_seeds) too narrow.
                    "mondrian_seed_weighted_cond_risk", "naive_seed_weighted_cond_risk",
                    "seeds_with_zero_acceptance_mondrian", "seeds_with_zero_acceptance_naive",
                    # sel_AUROC is NaN whenever an evaluation subset is all-correct or all-wrong,
                    # and the nan-aware mean drops those seeds silently. Without this column the
                    # row would show a plausible AUROC over a denominator nobody can see.
                    "seeds_defined_auroc",
                    # The plug-in margin used n = CALIB PIXELS. These say what the exchangeable
                    # sample size really is, so nobody has to infer it: calib_n_rows / n_eff is the
                    # variance inflation the margin does not account for.
                    "calib_n_rois", "calib_icc", "calib_design_effect",
                    # ROI-cluster bootstrap of the ACHIEVED risk at the selected threshold.
                    "mondrian_risk_roi95_lo", "mondrian_risk_roi95_hi",
                    "naive_risk_roi95_lo", "naive_risk_roi95_hi", "n_seeds"])
        for s, _, _ in STATES:
            d = mean[s]
            w.writerow([f"DOFA(frozen)+head[{args.head_norm}norm]", s]
                       + [_fmt(d[k]) for k in (["acc", "miou"] + [f"iou_{c}" for c in CLASS_NAMES]
                                               + ["aurc", "auroc", "mond_risk", "mond_cov",
                                                  "naive_risk", "naive_cov"])]
                       + [_fmt(pooled[s]["mond"]), _fmt(pooled[s]["naive"]),
                          n_undef[s]["mond"], n_undef[s]["naive"],
                          int(np.isfinite(np.asarray(agg[s]["auroc"], float)).sum())]
                       + [_fmt(mean[s][k], 4) for k in ("calib_n_rois", "calib_icc",
                                                        "calib_design_effect",
                                                        "mond_risk_roi_lo", "mond_risk_roi_hi",
                                                        "naive_risk_roi_lo", "naive_risk_roi_hi")]
                       + [n])

    fig, ax = plt.subplots(figsize=(4.0, 2.9))          # style comes from main()'s plt.rc_context
    names = [s for s, _, _ in STATES]
    xs = np.arange(len(names))
    for key, lab, col in [("naive_risk", "Naive (clean-calibrated)", "#c0392b"),
                          ("mond_risk", "Degradation-aware (Mondrian)", "#1f6f3a")]:
        me = [mean[s][key] for s in names]; er = [std[s][key] for s in names]
        ax.errorbar(xs, me, yerr=er, marker="o", lw=1.8, ms=4, color=col, label=lab, capsize=2)
    ax.axhline(TARGET_RISK * 100, ls="--", color="k", lw=1, label=f"target {TARGET_RISK:.0%}")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Achieved CONDITIONAL selective risk  P(wrong | accepted)  (%)")
    # phase8E uses the plug-in conformal_at_risk OPERATING POINT (not CRC), so this is the achieved
    # CONDITIONAL selective risk, NOT a certified quantity and NOT the joint mass phase8R plots --
    # the title must not say "certified", and the label must not be shortened to "risk".
    ax.set_title("Frozen-DOFA feature baseline — achieved selective risk (plug-in) under degradation", fontsize=8)
    ax.grid(alpha=0.3); ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(P(f"figs/fig_phase8E_dofa_conformal{sfx}.pdf")); plt.close(fig)

    print(f"\n===== Phase 8E DOFA (frozen-feature baseline, mean over {n} seeds, target {TARGET_RISK:.0%}) =====")
    print("state         acc   AURC  selAUROC | naive_risk/cov   mondrian_risk/cov  (pooled n/m)")
    for s, _, _ in STATES:
        d = mean[s]
        # ANY risk above target is a violation. The old flag needed target+2 percentage points, so
        # an 11.9% risk against a 10% target printed clean -- a threshold nobody chose deciding what
        # counts as failure. Undefined (no seed accepted anything) is its own state, not a pass.
        r = d["naive_risk"]
        flag = ("  <-- naive UNDEFINED (no acceptance)" if r != r else
                "  <-- naive EXCEEDS target" if r > TARGET_RISK * 100 else "")
        z = n_undef[s]
        zmsg = f"  [{z['naive']}/{z['mond']} seeds accepted nothing]" if (z["naive"] or z["mond"]) else ""
        print(f"{s:<12} {_fmt(d['acc'],1):>5} {_fmt(d['aurc'],1):>5} {_fmt(d['auroc'],1):>6}  | "
              f"{_fmt(r,1):>5}/{_fmt(d['naive_cov'],1):>4}    "
              f"{_fmt(d['mond_risk'],1):>5}/{_fmt(d['mond_cov'],1):>4}  "
              f"({_fmt(pooled[s]['naive'],1)}/{_fmt(pooled[s]['mond'],1)}){flag}{zmsg}")
    # Every number here is conditional on WHICH DOFA weights were frozen, so the pinned+verified
    # checkpoint digest is the input identity that matters (load_dofa already refused to run on any
    # other bytes). The calibration-patch count travels with it: it sets the CRC B/(n+1) floor.
    # BOTH deliverables get a sidecar: scripts/doctor.py requires one for every paper/results_*.csv,
    # and the per-seed file is the one any re-analysis actually reads.
    stamp(P(f"results_phase8E_dofa_perseed{sfx}.csv"), args, extra={"see": "results_phase8E_dofa"
          f"{sfx}.csv.provenance.json", "content": "raw per-seed rows behind every aggregate"})
    stamp(P(f"results_phase8E_dofa{sfx}.csv"), args,
          extra={"dofa_hub_ref": DOFA_HUB_REF, "dofa_ckpt_sha256": DOFA_CKPT_SHA256,
                 "dofa_hf_revision": DOFA_HF_REVISION, "dofa_ckpt_verified": True,
                 "dofa_ckpt_url": DOFA_CKPT_URL, "dofa_licenses": DOFA_LICENSES,
                 # The endpoint is part of the model's identity: the same checkpoint read through
                 # global_pool=True yields a DIFFERENT encoder (untrained fc_norm, pretrained norm
                 # discarded), so a result is only comparable to another that names the same one.
                 "dofa_global_pool": False,
                 "dofa_feature_endpoint": "patch_embed -> +pos -> CLS -> blocks -> pretrained norm "
                                          "-> drop CLS -> 14x14 grid (spatial; NOT official "
                                          "forward_features, which returns CLS or a pooled vector)",
                 "dofa_allowed_missing": sorted(DOFA_ALLOWED_MISSING),
                 "dofa_allowed_unexpected": sorted(DOFA_ALLOWED_UNEXPECTED),
                 "l1c_l2a_alignment": align,
                 "resize": {"from": int(SIDE), "to": IMG, "reflectance": "bilinear+antialias",
                            "label": "nearest-exact",
                            "class_prevalence_train": info_tr, "class_prevalence_calib": info_cal,
                            "class_prevalence_eval": info_ev},
                 "target_risk": TARGET_RISK, "states": [s for s, _, _ in STATES],
                 "n_train_patches": int(len(train_ids)), "n_calib_patches": int(len(calib_ids)),
                 "n_eval_patches": int(len(eval_ids)),
                 "n_calib_px": int(y_cal_s.size), "n_eval_px": int(y_ev_s.size),
                 # Absolute counts live here, not in the CSVs (see the header comment): JSON has no
                 # magnitude gate, and these are what any recomputation of the seed-weighted rate
                 # needs. `*_acc_n` sums over seeds sharing the same pixels -- divide by n_seeds for
                 # distinct pixels.
                 "accepted_counts_summed_over_seeds": counts,
                 # n_rows / n_effective are COUNTS and stay out of the CSV for the >1e4 reason
                 # above; the CSV carries their ratio (calib_design_effect), which is O(100).
                 "calibration_units": {s: {k: mean[s][k] for k in
                                           ("calib_n_rows", "calib_n_rois", "calib_icc",
                                            "calib_design_effect", "calib_n_effective")}
                                       for s, _, _ in STATES},
                 "seeds_with_zero_acceptance": n_undef,
                 "n_distinct_eval_px": int(y_ev_s.size), "n_seeds": n})
    print(f"wrote: {P(f'results_phase8E_dofa{sfx}.csv')}")
    print(f"       {P(f'figs/fig_phase8E_dofa_conformal{sfx}.pdf')}")


if __name__ == "__main__":
    main()
