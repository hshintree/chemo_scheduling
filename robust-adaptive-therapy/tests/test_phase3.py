"""
Phase 3 unit tests (feasibility, CVaR, policies, CVXPY relaxation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cvar_optimizer import CVaROptimizer
from src.cvxpy_relaxation import CVXPYRobustScheduler, _model_from_nominal
from src.feasibility import FeasibilityClassifier
from src.lotka_volterra import LotkaVolterraModel
from src.policy import MTDPolicy, RangeBoundedPolicy, ThresholdPolicy
from src.psa_model import PSAModel
from src.uncertainty import PARAM_NAMES as UQ_PARAM_NAMES
from src.uncertainty import EllipsoidalUncertaintySet


def _nominal_dict() -> dict:
    m = LotkaVolterraModel()
    return m.to_dict()


def test_feasibility_drug_eligible_delta_plus() -> None:
    """AT-eligible iff delta_plus > drug-effect threshold."""
    clf = FeasibilityClassifier(LotkaVolterraModel, PSAModel, t_max=400.0)
    p = _nominal_dict()
    assert clf.is_feasible(p) is True
    p_low = dict(p)
    p_low["delta_plus"] = 0.01
    assert clf.is_feasible(p_low) is False
    out = clf.classify_cohort([p])
    assert "ttp_zhang" in out["boundary_features"].columns
    assert "ttp_mtd" in out["boundary_features"].columns
    assert "zhang_beats_mtd" in out["boundary_features"].columns
    assert out["feasible_fraction"] == 1.0


def test_cvar_between_min_and_mean() -> None:
    clf = FeasibilityClassifier(LotkaVolterraModel, PSAModel)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 7))
    ell = EllipsoidalUncertaintySet(X.mean(0), np.cov(X.T) + 1e-3 * np.eye(7), 0.9)
    scenarios = []
    names = ell.param_names
    for i in range(8):
        row = np.exp(X[i])
        d = dict(zip(names, row.tolist()))
        d["K"] = 10_000.0
        d["delta_prod"] = 0.5 * float(d["delta_plus"])
        d["alpha"] = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]], dtype=float)
        scenarios.append(d)
    opt = CVaROptimizer(
        ell, clf, alpha=0.2, n_scenarios=len(scenarios), fixed_scenarios=scenarios, t_max_sim=200.0,
    )
    phi = (0.45, 1.05)
    cv = opt.compute_cvar(phi, scenarios)
    raw = []
    psa_m = PSAModel()
    for p in scenarios:
        m = LotkaVolterraModel(p)
        out = ThresholdPolicy(*phi).simulate_patient(m, psa_m, t_max=200, check_interval=1)
        raw.append(float(out["ttp"]) if np.isfinite(out["ttp"]) else 1e6)
    mn, mean = float(np.min(raw)), float(np.mean(raw))
    assert mn <= cv + 1e-6
    assert cv <= mean + 1e-2


def test_cvar_alpha_one_equals_mean() -> None:
    clf = FeasibilityClassifier(LotkaVolterraModel, PSAModel)
    scenarios = []
    for i in range(8):
        p = _nominal_dict()
        p["T0_minus"] = 20.0 + i * 3.0
        scenarios.append(p)
    ell = EllipsoidalUncertaintySet(np.ones(7), np.eye(7) * 0.01, 0.9)
    opt = CVaROptimizer(
        ell, clf, alpha=1.0, n_scenarios=len(scenarios), fixed_scenarios=scenarios, t_max_sim=200.0,
    )
    phi = (0.5, 1.0)
    cv = opt.compute_cvar(phi, scenarios)
    psa_m = PSAModel()
    ttps = []
    for p in scenarios:
        m = LotkaVolterraModel(p)
        out = ThresholdPolicy(*phi).simulate_patient(m, psa_m, t_max=200, check_interval=1)
        ttps.append(float(out["ttp"]) if np.isfinite(out["ttp"]) else 1e6)
    assert abs(cv - float(np.mean(ttps))) < 5.0


def test_cvar_small_alpha_near_min() -> None:
    clf = FeasibilityClassifier(LotkaVolterraModel, PSAModel)
    scenarios = []
    rng = np.random.default_rng(1)
    for _ in range(10):
        p = _nominal_dict()
        p["T0_minus"] = float(rng.uniform(10, 80))
        scenarios.append(p)
    ell = EllipsoidalUncertaintySet(np.ones(7), np.eye(7) * 0.01, 0.9)
    opt_lo = CVaROptimizer(
        ell, clf, alpha=0.05, n_scenarios=len(scenarios), fixed_scenarios=scenarios, t_max_sim=220.0,
    )
    phi = (0.5, 1.0)
    cv_lo = opt_lo.compute_cvar(phi, scenarios)
    psa_m = PSAModel()
    ttps = []
    for p in scenarios:
        m = LotkaVolterraModel(p)
        out = ThresholdPolicy(*phi).simulate_patient(m, psa_m, t_max=220, check_interval=1)
        ttps.append(float(out["ttp"]) if np.isfinite(out["ttp"]) else 1e6)
    assert cv_lo <= float(np.min(ttps)) + 50.0


def test_threshold_policy_zhang_hand_sequence() -> None:
    pol = ThresholdPolicy(0.5, 1.0)
    b = 100.0
    assert pol.get_drug_decision(40.0, b, True) == 0
    assert pol.get_drug_decision(60.0, b, True) == 1
    assert pol.get_drug_decision(90.0, b, False) == 0
    assert pol.get_drug_decision(100.0, b, False) == 1


def test_range_bounded_dose_midpoints() -> None:
    pol = RangeBoundedPolicy(0.3, 0.8)
    b = 100.0
    assert pol.get_dose(0.2 * b, b) == pytest.approx(0.0)
    assert pol.get_dose(1.0 * b, b) == pytest.approx(1.0)
    assert pol.get_dose(0.55 * b, b) == pytest.approx(0.5)


def test_cvxpy_socp_status_optimal() -> None:
    m = LotkaVolterraModel()
    d = m.to_dict()
    X_log = np.log([[d[k] for k in UQ_PARAM_NAMES]])
    ell = EllipsoidalUncertaintySet(X_log.ravel(), np.eye(7) * 0.05, 0.9)
    sched = CVXPYRobustScheduler(d, ell, T_horizon=29, dt=28)
    out = sched.solve(verbose=False)
    assert out["status"] in ("optimal", "optimal_inaccurate")


def test_socp_dose_sum_constraint() -> None:
    m = LotkaVolterraModel()
    d = m.to_dict()
    X_log = np.log([[d[k] for k in UQ_PARAM_NAMES]])
    ell = EllipsoidalUncertaintySet(X_log.ravel(), np.eye(7) * 0.05, 0.9)
    sched = CVXPYRobustScheduler(d, ell, T_horizon=29, dt=28)
    out = sched.solve(verbose=False)
    u = out["optimal_schedule"]
    assert np.isfinite(u).all()
    assert float(np.sum(u >= 0.99)) <= 0.71 * len(u) + 1


def test_subgradient_cvar_trend() -> None:
    clf = FeasibilityClassifier(LotkaVolterraModel, PSAModel)
    scenarios = []
    rng = np.random.default_rng(42)
    for i in range(6):
        p = _nominal_dict()
        p["T0_minus"] = float(rng.uniform(15, 60))
        scenarios.append(p)
    ell = EllipsoidalUncertaintySet(np.ones(7), np.eye(7) * 0.02, 0.9)
    opt = CVaROptimizer(
        ell, clf, alpha=0.25, n_scenarios=len(scenarios), fixed_scenarios=scenarios, t_max_sim=300.0,
    )
    res = opt.optimize_gradient(
        n_iterations=25,
        step_size=0.02,
        init_phi=(0.5, 1.0),
        seed=42,
        tqdm_disable=True,
    )
    hist = res["convergence_history"]
    cvars = [h[1] for h in hist]
    early = float(np.mean(cvars[:5]))
    late = float(np.mean(cvars[-5:]))
    assert late >= early - 40.0
