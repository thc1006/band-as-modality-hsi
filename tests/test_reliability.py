"""Unit tests for reliability / selective-prediction metrics (Phase 4R).

Guard the MATH so the honest-ace figures can be trusted. Run: pytest tests/ -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandsim.reliability import (
    confidence_msp, risk_coverage_curve, aurc, augrc, selective_auroc,
    fit_temperature, conformal_at_risk, softmax,
)


def test_msp_in_unit_range():
    logits = np.random.default_rng(0).normal(size=(50, 16))
    c = confidence_msp(logits)
    assert c.shape == (50,)
    assert np.all(c >= 1.0 / 16 - 1e-6) and np.all(c <= 1.0)


def test_full_coverage_risk_equals_error_rate():
    correct = np.array([1, 1, 0, 1, 0, 1], float)
    conf = np.array([0.9, 0.8, 0.4, 0.7, 0.3, 0.6])
    cov, risk = risk_coverage_curve(correct, conf)
    assert np.isclose(cov[-1], 1.0)
    assert np.isclose(risk[-1], 1 - correct.mean())  # at full coverage risk = overall error


def test_perfect_confidence_gives_low_aurc_high_auroc():
    # confidence perfectly ranks correct above incorrect -> AURC ~ 0, AUROC = 1
    rng = np.random.default_rng(1)
    correct = np.array([1] * 40 + [0] * 10)
    conf = np.where(correct == 1, rng.uniform(0.7, 1.0, 50), rng.uniform(0.0, 0.3, 50))
    assert selective_auroc(correct, conf) > 0.99
    # abstaining on the low-confidence (all wrong) drives risk to 0 at low coverage
    cov, risk = risk_coverage_curve(correct, conf)
    assert risk[:40].max() < 1e-9          # top-40 most-confident are all correct
    assert aurc(correct, conf) < augrc(correct, conf) + 0.05  # both small


def test_random_confidence_auroc_near_half():
    rng = np.random.default_rng(2)
    correct = rng.integers(0, 2, size=4000)
    conf = rng.uniform(0, 1, size=4000)     # independent of correctness
    assert abs(selective_auroc(correct, conf) - 0.5) < 0.05


def test_augrc_le_aurc():
    # generalized risk g(c)=risk*c <= risk, so AUGRC <= AURC always
    rng = np.random.default_rng(3)
    correct = rng.integers(0, 2, size=500)
    conf = rng.uniform(0, 1, size=500)
    assert augrc(correct, conf) <= aurc(correct, conf) + 1e-9


def test_temperature_reduces_nll_when_overconfident():
    # overconfident logits (scaled up) -> optimal T > 1 lowers NLL
    rng = np.random.default_rng(4)
    K = 5; n = 400
    labels = rng.integers(0, K, size=n)
    base = rng.normal(size=(n, K))
    base[np.arange(n), labels] += 1.0        # mildly informative
    logits = base * 5.0                       # over-confident
    def nll(T):
        p = softmax(logits / T)
        return -np.log(p[np.arange(n), labels] + 1e-12).mean()
    T = fit_temperature(logits, labels)
    assert T > 1.0
    assert nll(T) < nll(1.0)


def test_conformal_meets_target_risk_in_distribution():
    rng = np.random.default_rng(5)
    n = 3000
    conf = rng.uniform(0, 1, n)
    # higher confidence -> more likely correct
    correct = (rng.uniform(0, 1, n) < conf).astype(int)
    calib = slice(0, n // 2); ev = slice(n // 2, n)
    out = conformal_at_risk(correct[calib], conf[calib], correct[ev], conf[ev], target_risk=0.10)
    # in-distribution: eval risk should be close to target (allow sampling slack)
    assert out["risk"] <= 0.15
    assert 0.0 < out["coverage"] <= 1.0
