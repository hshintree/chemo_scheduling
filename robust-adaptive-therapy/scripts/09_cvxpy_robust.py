#!/usr/bin/env python
"""
SCRIPT 09: CVXPY Linearized SOCP
"""

from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.cvxpy_relaxation import CVXPYRobustScheduler, _rk_step, _model_from_nominal
from src.feasibility import FeasibilityClassifier
from src.lotka_volterra import LotkaVolterraModel
from src.policy import ThresholdPolicy
from src.population_model import PopulationModel
from src.psa_model import PSAModel
from src.uncertainty import EllipsoidalUncertaintySet

ROOT = Path(__file__).resolve().parent.parent
DATA_SYN = ROOT / "data" / "synthetic"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "cvxpy"


def psa_track_from_u(
    model: LotkaVolterraModel,
    psa_m: PSAModel,
    u_series: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.array([model.T0_plus, model.T0_prod, model.T0_minus], dtype=float)
    bl = float(
        psa_m.compute_psa({
            "T_plus": np.array([y[0]]),
            "T_prod": np.array([y[1]]),
            "T_minus": np.array([y[2]]),
        })[0]
    )
    ts, psa_n, tm = [0.0], [], []
    for u in u_series:
        psa = float(
            psa_m.compute_psa({
                "T_plus": np.array([y[0]]),
                "T_prod": np.array([y[1]]),
                "T_minus": np.array([y[2]]),
            })[0]
        )
        psa_n.append(psa / bl if bl > 0 else 0.0)
        total = float(y.sum())
        tm.append(float(y[2] / total) if total > 0 else 0.0)
        y = _rk_step(model, y, float(u), dt)
        ts.append(ts[-1] + dt)
    return np.array(ts[:-1]), np.array(psa_n), np.array(tm)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_SYN / "population_params.csv")
    records = df[PopulationModel.param_names].to_dict(orient="records")
    pop = PopulationModel()
    fit = pop.fit(records)
    ell = EllipsoidalUncertaintySet(
        fit["mean_log"], fit["cov_log"], confidence=0.90,
        param_names=PopulationModel.param_names,
    )
    ell_w = ell.weighted_ellipsoid({3: 1.5, 6: 2.0})

    X = np.log(df[PopulationModel.param_names].astype(float).values)
    nominal_log = X.mean(axis=0)
    nominal_params = {k: float(np.exp(v)) for k, v in zip(PopulationModel.param_names, nominal_log)}
    nominal_params["K"] = 10_000.0
    nominal_params["delta_prod"] = 0.5 * nominal_params["delta_plus"]
    nominal_params["alpha"] = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]], dtype=float)

    clf = FeasibilityClassifier(LotkaVolterraModel, PSAModel)
    if not clf.is_feasible(nominal_params):
        print("Nominal mean patient flagged infeasible; using first feasible cohort row.")
        for _, row in df.iterrows():
            d = row.to_dict()
            d.pop("patient_id", None)
            cand = LotkaVolterraModel.from_dict(d).to_dict()
            if clf.is_feasible(cand):
                nominal_params = cand
                break

    sched = CVXPYRobustScheduler(nominal_params, ell_w, T_horizon=364, dt=28)
    A_list, B_list, C_list, x_nom, u_nom = sched.linearize_dynamics()
    for k, A in enumerate(A_list[:3]):
        print(f"Step {k}: cond(A) = {np.linalg.cond(A):.3e}")

    sol = sched.solve(verbose=False)
    print(f"SOCP status: {sol['status']}, solve_time={sol['solve_time']:.3f}s, objective={sol['objective_value']:.4f}")

    mpc = sched.mpc_solve(nominal_params, n_steps=13)
    print(f"MPC: mean solve time per step = {float(mpc['solve_times'].mean()):.4f}s")

    m_nom = _model_from_nominal(nominal_params)
    psa_m = PSAModel()
    zhang = ThresholdPolicy(0.5, 1.0)
    zout = zhang.simulate_patient(m_nom, psa_m, t_max=364, check_interval=1)

    u_socp = sol["optimal_schedule"]
    u_mpc = mpc["u_applied"]
    t1, p1, tm1 = psa_track_from_u(m_nom, psa_m, u_socp, 28.0)
    t2, p2, tm2 = psa_track_from_u(m_nom, psa_m, u_mpc, 28.0)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    days_socp = np.arange(len(u_socp), dtype=float) * 28.0
    axes[0, 0].step(days_socp, u_socp, where="post")
    axes[0, 0].set_title("SOCP open-loop dose")
    axes[1, 0].plot(t1, p1, "o-")
    axes[1, 0].set_ylabel("PSA norm")
    axes[2, 0].plot(t1, tm1, "o-", color="red")
    axes[2, 0].set_ylabel("T− fraction")

    axes[0, 1].step(np.arange(len(u_mpc), dtype=float) * 28.0, u_mpc, where="post")
    axes[0, 1].set_title("MPC dose")
    axes[1, 1].plot(t2, p2, "o-")
    axes[2, 1].plot(t2, tm2, "o-", color="red")

    axes[0, 2].plot(zout["t"], zout["on_treatment"], drawstyle="steps-post")
    axes[0, 2].set_title("Zhang binary")
    axes[1, 2].plot(zout["t"], zout["psa_normalized"], lw=1)
    axes[2, 2].axis("off")

    for ax in axes[2, :]:
        ax.set_xlabel("Day")
    fig.tight_layout()
    fig.savefig(FIG / "schedule_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    dev_norms = []
    for k in range(min(3, len(A_list))):
        y0 = x_nom[k]
        u0 = float(sol["optimal_schedule"][k]) if k < len(sol["optimal_schedule"]) else 0.5
        y_nl = _rk_step(m_nom, y0, u0, 28.0)
        lin_delta = A_list[k] @ np.zeros(3) + B_list[k].ravel() * (u0 - u_nom[k])
        y_lin = y0 + lin_delta
        dev_norms.append(float(np.linalg.norm(y_nl - y_lin)))
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(dev_norms) + 1), dev_norms, "o-")
    ax.set_xlabel("MPC step")
    ax.set_ylabel(r"$\|x_{\mathrm{nl}} - x_{\mathrm{lin}}\|$")
    ax.set_title("Linearization error (first steps)")
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG / "linearization_validity.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)

    thr = 0.1 * (np.linalg.norm(x_nom[0]) + 1.0)
    valid_days = 28 * (int(np.searchsorted(dev_norms, thr)) if dev_norms else 0)
    print(f"Linearization valid for ~{valid_days} days before 10% scale deviation threshold (heuristic).")

    sim_ttp = float(zout["ttp"]) if np.isfinite(zout["ttp"]) else 364.0
    gap = abs(sol["objective_value"] - sim_ttp) / max(sim_ttp, 1.0)
    if gap > 0.3:
        print("WARNING: Linearization error is large for this patient. CVaR grid search result is more reliable.")

    with open(PROC / "cvxpy_mpc_schedule.pkl", "wb") as f:
        pickle.dump({"open_loop_u": u_socp, "mpc_u": u_mpc, "sol": sol, "mpc": mpc}, f)
    print(f"Saved {PROC / 'cvxpy_mpc_schedule.pkl'} and figures to {FIG}/")


if __name__ == "__main__":
    main()
