"""Lightweight torch models for band-as-modality missing-band robustness (Phase 2).

Contents
--------
- sinusoidal_wavelength_pe : wavelength (nm) -> sinusoidal positional encoding
- GroupedCrossBandAttention : PROPOSED. Each spectral group = a token; pairwise cross-band
  (cross-group) self-attention with wavelength PE; masked pooling supports arbitrary
  present-group subsets (i.e. missing groups = removed tokens). Includes an SGMAE decoder
  head for spectral-group masked reconstruction pretraining.
- MLPBaseline : channel-stack MLP for B1/B2/B3 baselines.

Design notes
------------
- Kept SMALL on purpose — the contribution is the missing-band mechanism, not model size. Measured
  on the 200-band Indian Pines setup (10 groups, 16 classes): GroupedCrossBandAttention 70,692
  params, MLPBaseline 121,360. A blanket "<100k params" claim used to appear here and in the phase 2
  docstring; it is false for the MLP baseline, whose first layer scales with the full band count
  (200*C) while the grouped model's does not. That the ATTENTION model is the SMALLER of the two is
  worth stating plainly — but state it with the numbers, which the experiment scripts also report.
- All groups are zero-padded to the max group size so one Linear handles unequal groups.
- CPU best practice: caller sets torch.set_num_threads(physical_cores).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def sinusoidal_wavelength_pe(center_wl_nm, d_model, wl_scale=2500.0):
    """(G,) group centre wavelengths -> (G, d_model) sinusoidal PE.

    Uses normalized wavelength as the 'position' so the encoding is sensitive to spectral
    order and spacing (cf. Panopticon / HyperspectralMAE wavelength encodings).
    """
    wl = torch.as_tensor(np.asarray(center_wl_nm, float) / wl_scale, dtype=torch.float32)
    G = wl.shape[0]
    pe = torch.zeros(G, d_model)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
    ang = wl[:, None] * div[None, :] * 100.0  # *100: spread normalized wl across freqs
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang[:, : pe[:, 1::2].shape[1]])  # handle odd d_model safely
    return pe


def pad_groups(group_index_list):
    """Return (padded_index (G,S), valid_mask (G,S)) padding each group to max size S."""
    G = len(group_index_list)
    S = max(len(g) for g in group_index_list)
    idx = np.zeros((G, S), int)
    valid = np.zeros((G, S), bool)
    for g, ind in enumerate(group_index_list):
        idx[g, :len(ind)] = ind
        valid[g, :len(ind)] = True
    return idx, valid


class GroupedCrossBandAttention(nn.Module):
    """Proposed model: spectral-group tokens + cross-band attention + wavelength PE.

    Forward accepts x (N, C) reflectance and a present-group mask (N, G) (True=present).
    Missing groups are replaced by a learnable mask token and excluded from the pooled
    representation used for classification.
    """

    def __init__(self, groups, center_wl_nm, num_classes, d_model=64, nhead=4, layers=2,
                 dim_ff=128, dropout=0.0, pe_type="sinusoidal"):
        super().__init__()
        # pe_type: "sinusoidal" = wavelength-conditioned PE (our method's ingredient);
        #          "learned"    = a free per-group embedding (ChannelViT/SatMAE-style, no
        #                         physical wavelength) -> used to build the B4/B6 baselines.
        self.pe_type = pe_type
        idx, valid = pad_groups(groups)
        self.register_buffer("group_idx", torch.as_tensor(idx, dtype=torch.long))
        self.register_buffer("group_valid", torch.as_tensor(valid, dtype=torch.float32))
        self.G, self.S = idx.shape
        self.d_model = d_model
        self.embed = nn.Linear(self.S, d_model)
        if pe_type == "learned":
            self.pe = nn.Parameter(torch.zeros(self.G, d_model))
            nn.init.normal_(self.pe, std=0.02)               # learned per-group embedding
        else:
            self.register_buffer("pe", sinusoidal_wavelength_pe(center_wl_nm, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                         batch_first=True, activation="gelu")
        # enable_nested_tensor=False: dense, deterministic path (avoids the prototype
        # nested-tensor fast-path warning triggered by src_key_padding_mask on small seqs).
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, num_classes)
        self.decoder = nn.Linear(d_model, self.S)  # SGMAE reconstruction head

    def _tokens(self, x):
        """x (N,C) -> group value tensor (N, G, S) with padded bands zeroed."""
        # gather bands per group; padded positions gather index 0 then zeroed by valid mask
        xg = x[:, self.group_idx]                 # (N, G, S)
        xg = xg * self.group_valid[None]          # zero padded slots
        return xg

    def encode(self, x, present_mask, use_padding_mask, attend_mask=None):
        """Return encoder outputs (N, G, d). present_mask (N,G) True=present (real band content),
        False=hidden (replaced by the learnable mask token).

        Two correctness invariants (both were bugs before):
        (1) the wavelength/group PE is added to ALL tokens INCLUDING masked ones, so multiple
            simultaneously-masked groups keep DISTINCT positional identity -- otherwise every
            masked token is the identical `mask_token` and the decoder reconstructs them identically.
        (2) a masked token carries NO band value (its content embedding is zeroed), so the value the
            model must reconstruct/ignore never leaks in.
        attend_mask (N,G) True=token may be attended to (default: present_mask); lets a caller allow
        attention over mask tokens (e.g. a fully-missing row) WITHOUT feeding the raw input back in."""
        xg = self._tokens(x)                                     # (N, G, S)
        content = self.embed(xg)                                 # (N, G, d) band-value embedding
        present = present_mask.float()[:, :, None]
        tok = content * present + self.mask_token[None, None] * (1 - present)   # hide value of masked groups
        tok = tok + self.pe[None]                                # keep group/wavelength position for ALL tokens
        key_padding = None
        if use_padding_mask:
            am = present_mask if attend_mask is None else attend_mask
            key_padding = ~am.bool()
        return self.encoder(tok, src_key_padding_mask=key_padding)

    def forward(self, x, present_mask):
        """Classification logits (N, num_classes). Missing groups are hidden (mask token, no value)
        and excluded from attention + pooling.

        A fully-missing row (no present group) would make src_key_padding_mask all-True -> softmax
        over an all-masked row -> NaN. Such rows never occur in the experiments (we never drop all
        groups); we guard defensively by letting that row ATTEND to its own mask tokens (finite,
        uninformed output) -- WITHOUT feeding the raw input back in (the old guard set the row
        all-present, which read the full raw spectrum it was supposed to be missing)."""
        present = present_mask.bool()
        fully_missing = ~present.any(dim=1)                       # (N,)
        attend = present.clone()
        attend[fully_missing] = True                              # all-mask rows may attend to their mask tokens
        h = self.encode(x, present, use_padding_mask=True, attend_mask=attend)  # masked=mask token (no raw x)
        w = attend.float()[:, :, None]
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1e-6)        # masked mean pool
        return self.classifier(pooled)

    def reconstruct(self, x, masked_mask):
        """SGMAE: given masked groups (masked_mask True=masked/hidden), reconstruct their
        band values from context. All tokens attend (no padding mask). Returns (N,G,S)."""
        present = ~masked_mask
        h = self.encode(x, present, use_padding_mask=False)      # masked=mask_token input
        return self.decoder(h)                                    # (N,G,S) predicted values


class MLPBaseline(nn.Module):
    """Channel-stack MLP (B1/B2/B3). Input is the full C-band vector (missing bands zeroed
    or imputed by the caller)."""

    def __init__(self, n_bands, num_classes, hidden=256, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bands, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
