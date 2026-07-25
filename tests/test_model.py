"""Unit tests for the model + guards added after adversarial review. Run: pytest tests/ -v"""
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention, sinusoidal_wavelength_pe
from bandsim.metrics import average_accuracy


def _model(G=10, C=200, K=16):
    groups = contiguous_groups(C, G)
    cwl = group_center_wavelengths(np.linspace(400, 2500, C), groups)
    torch.manual_seed(0)
    return GroupedCrossBandAttention(groups, cwl, K), groups


def test_fully_missing_row_is_finite_not_nan():
    # A row with NO present group must not produce NaN (defensive guard).
    m, groups = _model()
    m.eval()
    x = torch.randn(4, 200)
    pm = np.ones((4, len(groups)), bool)
    pm[1] = False   # sample 1: all groups missing
    with torch.no_grad():
        out = m(x, torch.from_numpy(pm))
    assert torch.isfinite(out).all(), "fully-missing row produced NaN/inf"


def test_missing_group_values_do_not_change_output():
    # True masking: garbage in a missing group must not change logits.
    m, groups = _model()
    m.eval()
    x = torch.randn(8, 200)
    pm = np.ones((8, len(groups)), bool); pm[:, [2, 5]] = False
    with torch.no_grad():
        o1 = m(x, torch.from_numpy(pm))
        x2 = x.clone()
        for g in (2, 5):
            x2[:, groups[g]] = torch.randn_like(x2[:, groups[g]]) * 10 + 50
        o2 = m(x2, torch.from_numpy(pm))
    assert torch.allclose(o1, o2, atol=1e-5), "missing-group values leaked into output"


def test_present_mask_changes_output():
    m, groups = _model()
    m.eval()
    x = torch.randn(8, 200)
    full = np.ones((8, len(groups)), bool)
    part = full.copy(); part[:, [1, 4]] = False
    with torch.no_grad():
        assert not torch.allclose(m(x, torch.from_numpy(full)), m(x, torch.from_numpy(part)), atol=1e-4)


def test_pe_type_learned_vs_sinusoidal():
    # sinusoidal PE is a fixed buffer (not a parameter); learned PE is a trainable parameter.
    m_sin, groups = _model()
    assert not any(n == "pe" for n, _ in m_sin.named_parameters())
    cwl = group_center_wavelengths(np.linspace(400, 2500, 200), groups)
    m_learn = GroupedCrossBandAttention(groups, cwl, 16, pe_type="learned")
    assert any(n == "pe" for n, _ in m_learn.named_parameters())
    x = torch.randn(4, 200); pm = torch.ones(4, len(groups), dtype=torch.bool)
    assert torch.isfinite(m_learn(x, pm)).all()


def test_sinusoidal_pe_handles_odd_dmodel():
    pe = sinusoidal_wavelength_pe([500.0, 1500.0, 2400.0], d_model=7)  # odd
    assert pe.shape == (3, 7)
    assert torch.isfinite(pe).all()


def test_group_dropout_mask_equals_inplace_zeroing():
    # The device-agnostic refactor zeroes whole groups via a 0/1 band-mask multiply; verify it
    # is byte-identical to in-place zeroing (x*1.0=x, x*0.0=0 exactly in float32).
    x = torch.randn(32, 200)
    groups = [list(range(i * 20, (i + 1) * 20)) for i in range(10)]
    drop = [1, 4, 8]
    a = x.clone().numpy()
    for g in drop:
        a[:, groups[g]] = 0.0
    bmask = np.ones((32, 200), np.float32)
    for g in drop:
        bmask[:, groups[g]] = 0.0
    b = (x * torch.from_numpy(bmask)).numpy()
    assert np.array_equal(a, b)


def test_average_accuracy_present_classes_only():
    # Perfect predictions with a class absent from y_true -> AA should be 100 (not deflated).
    yt = np.array([0, 0, 1, 1]); yp = yt.copy()
    assert abs(average_accuracy(yt, yp, num_classes=3) - 100.0) < 1e-6
    # all classes present + one class fully wrong -> AA reflects the miss
    yt = np.array([0, 0, 1, 1, 2, 2]); yp = np.array([0, 0, 1, 1, 0, 0])
    assert abs(average_accuracy(yt, yp, num_classes=3) - (100 + 100 + 0) / 3) < 1e-6
