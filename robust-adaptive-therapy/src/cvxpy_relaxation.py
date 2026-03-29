"""
LINEARIZED ROBUST SCHEDULING VIA CVXPY

This module approximates the LV ODE as a linear time-varying (LTV) system
around a nominal trajectory, then solves the robust scheduling problem as an SOCP.

IMPORTANT DISCLAIMER
--------------------
This is an approximation valid only near the nominal trajectory.
The linearization error grows with deviation from nominal.
Use as: (1) a lower bound on achievable robust TTP, (2) a warm-start
for the gradient-based optimizer, (3) a fast feasibility check.
Re-linearize every 56 days in MPC mode to maintain validity.

MATHEMATICAL SETUP
------------------
Let x(t) = [T+, Tp, T-] be the state, u(t) in [0,1] the control.
LV dynamics: dx/dt = f(x, u, theta).

Linearize around nominal trajectory (x_nom(t), u_nom(t)):
  delta_x(t+dt) ≈ A(t) * delta_x(t) + B(t) * delta_u(t) + C(t) * delta_theta

where A(t) = df/dx, B(t) = df/du, C(t) = df/d(log theta) (implemented as
sensitivity of the one-step flow to log-parameter perturbations).

The robust objective (minimize worst-case T_minus growth) over an ellipsoidal
uncertainty set in theta becomes, for the T_minus component:
  min_u  sum_t ( T_minus_nom(t) + delta_x_2(t) + kappa * ||C(t)[2,:]||_2 )

The SOCP uses the closed-form robust margin kappa * ||C[2,:]||_2 per step.

Note on identifiability
-----------------------
Competition coefficients alpha_ij are fixed at 0.5 (not in the 7-D ellipsoid).
The ellipsoid covers r_plus, r_prod, r_minus, delta_plus, T0_plus, T0_prod,
T0_minus. Alpha sensitivity can be analysed separately (Phase 4).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import cvxpy as cp
import numpy as np

from .lotka_volterra import LotkaVolterraModel
from .uncertainty import EllipsoidalUncertaintySet

log = logging.getLogger(__name__)

PARAM_NAMES_7 = [
    "r_plus", "r_prod", "r_minus", "delta_plus",
    "T0_plus", "T0_prod", "T0_minus",
]


def _rk_step(model: LotkaVolterraModel, y: np.ndarray, u: float, dt: float) -> np.ndarray:
    uu = float(u)
    drug_fn = lambda t, uu=uu: uu
    sub = model.simulate(
        (0.0, dt),
        np.array([dt]),
        drug_fn,
        y0=y,
        rtol=1e-4,
        atol=1e-1,
    )
    return np.array(
        [sub["T_plus"][-1], sub["T_prod"][-1], sub["T_minus"][-1]],
        dtype=float,
    )


def _model_from_nominal(nominal_params: dict) -> LotkaVolterraModel:
    p = dict(nominal_params)
    keys = [f"alpha_{i}{j}" for i in range(3) for j in range(3)]
    if "alpha" not in p and all(k in p for k in keys):
        return LotkaVolterraModel.from_dict(p)
    if "alpha" not in p:
        p["alpha"] = np.array([
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ])
    if "delta_prod" not in p and "delta_plus" in p:
        p["delta_prod"] = 0.5 * float(p["delta_plus"])
    if "K" not in p:
        p["K"] = 10_000.0
    return LotkaVolterraModel(p)


def _perturb_log_param(base: dict, j: int, h_log: float, sign: float) -> dict:
    p = dict(base)
    name = PARAM_NAMES_7[j]
    v = float(p[name])
    p[name] = v * np.exp(sign * h_log)
    if name == "delta_plus":
        p["delta_prod"] = 0.5 * float(p["delta_plus"])
    return p


class CVXPYRobustScheduler:
    def __init__(
        self,
        nominal_params: dict,
        uncertainty_set: EllipsoidalUncertaintySet,
        T_horizon: int = 365,
        dt: int = 28,
    ) -> None:
        self.nominal_params = dict(nominal_params)
        self.uncertainty_set = uncertainty_set
        self.T_horizon = int(T_horizon)
        self.dt = int(dt)
        self.N = max(1, self.T_horizon // self.dt)
        self.kappa = float(uncertainty_set.kappa)

    def linearize_dynamics(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        model = _model_from_nominal(self.nominal_params)
        dt = float(self.dt)
        N = self.N
        u_nom = np.full(N, 0.5, dtype=float)
        x_nom = np.zeros((N + 1, 3), dtype=float)
        x_nom[0] = np.array([model.T0_plus, model.T0_prod, model.T0_minus], dtype=float)
        for k in range(N):
            x_nom[k + 1] = _rk_step(model, x_nom[k], u_nom[k], dt)

        h_y = 1e-4 * (np.linalg.norm(x_nom[0]) + 1.0)
        h_u = 1e-4
        h_log = 1e-5

        A_list: list[np.ndarray] = []
        B_list: list[np.ndarray] = []
        C_list: list[np.ndarray] = []

        for k in range(N):
            y0 = x_nom[k]
            u0 = u_nom[k]
            x_base = _rk_step(model, y0, u0, dt)

            A = np.zeros((3, 3))
            for i in range(3):
                ey = np.zeros(3)
                ey[i] = h_y
                xp = _rk_step(model, y0 + ey, u0, dt)
                xm = _rk_step(model, y0 - ey, u0, dt)
                A[:, i] = (xp - xm) / (2 * h_y)

            B = np.zeros(3)
            xp = _rk_step(model, y0, u0 + h_u, dt)
            xm = _rk_step(model, y0, u0 - h_u, dt)
            B[:] = (xp - xm) / (2 * h_u)

            C = np.zeros((3, 7))
            base_dict = model.to_dict()
            for j in range(7):
                if k > 0 and j >= 4:
                    C[:, j] = 0.0
                    continue
                if k == 0 and j >= 4:
                    yp = y0.copy()
                    ym = y0.copy()
                    idx = j - 4
                    yp[idx] = y0[idx] * np.exp(h_log)
                    ym[idx] = y0[idx] * np.exp(-h_log)
                    xp2 = _rk_step(model, yp, u0, dt)
                    xm2 = _rk_step(model, ym, u0, dt)
                    C[:, j] = (xp2 - xm2) / (2 * h_log)
                    continue
                pp = _perturb_log_param(base_dict, j, h_log, 1.0)
                pm = _perturb_log_param(base_dict, j, h_log, -1.0)
                mp = _model_from_nominal(pp)
                mm = _model_from_nominal(pm)
                xp2 = _rk_step(mp, y0, u0, dt)
                xm2 = _rk_step(mm, y0, u0, dt)
                C[:, j] = (xp2 - xm2) / (2 * h_log)

            A_list.append(A)
            B_list.append(B.reshape(3, 1))
            C_list.append(C)

        return A_list, B_list, C_list, x_nom, u_nom

    def build_socp(
        self,
        A_list: list[np.ndarray],
        B_list: list[np.ndarray],
        C_list: list[np.ndarray],
        x_nom: np.ndarray,
        u_nom: np.ndarray,
    ) -> tuple[cp.Problem, cp.Variable, np.ndarray]:
        N = self.N
        K = float(self.nominal_params.get("K", 10_000.0))
        T_minus_max = 0.3 * K
        delta_x = cp.Variable((N + 1, 3))
        u = cp.Variable(N)
        worst = np.array([self.kappa * np.linalg.norm(C_list[k][2, :]) for k in range(N)])
        Tm_nom = x_nom[:N, 2]

        cons = [delta_x[0, :] == 0, u >= 0, u <= 1, cp.sum(u) <= 0.7 * N]
        obj_terms = []
        for k in range(N):
            Ak = A_list[k]
            Bk = B_list[k].ravel()
            cons.append(
                delta_x[k + 1, :]
                == Ak @ delta_x[k, :] + Bk * (u[k] - u_nom[k])
            )
            T_rob = Tm_nom[k] + delta_x[k, 2] + worst[k]
            cons.append(T_rob <= T_minus_max)
            obj_terms.append(T_rob)
        obj = cp.Minimize(cp.sum(obj_terms))
        prob = cp.Problem(obj, cons)
        return prob, u, worst

    def solve(self, verbose: bool = False) -> dict:
        A_list, B_list, C_list, x_nom, u_nom = self.linearize_dynamics()
        prob, u_var, worst = self.build_socp(A_list, B_list, C_list, x_nom, u_nom)
        t0 = time.time()
        status = "not_solved"
        try:
            prob.solve(solver=cp.MOSEK, verbose=verbose)
            status = prob.status
        except Exception as e:
            log.info("MOSEK unavailable (%s); falling back to SCS.", e)
            prob.solve(solver=cp.SCS, verbose=verbose, max_iters=25_000)
            status = prob.status
        elapsed = time.time() - t0
        u_opt = np.array(u_var.value).ravel() if u_var.value is not None else np.full(self.N, np.nan)
        Tm_traj = x_nom[: self.N, 2] + worst
        return {
            "status": str(status),
            "solve_time": float(elapsed),
            "optimal_schedule": u_opt,
            "objective_value": float(prob.value) if prob.value is not None else float("nan"),
            "T_minus_trajectory": Tm_traj,
            "x_nom": x_nom,
            "u_nom": u_nom,
            "A_list": A_list,
            "B_list": B_list,
            "C_list": C_list,
        }

    def mpc_solve(
        self,
        patient_params: dict,
        n_steps: int = 13,
        seed: int = 0,
    ) -> dict:
        dt = float(self.dt)
        m0 = _model_from_nominal(patient_params)
        y = np.array([m0.T0_plus, m0.T0_prod, m0.T0_minus], dtype=float)
        u_applied: list[float] = []
        x_hist: list[np.ndarray] = [y.copy()]
        t_hist: list[float] = [0.0]
        solve_times: list[float] = []
        _ = np.random.default_rng(seed)

        for step in range(n_steps):
            dyn = dict(patient_params)
            dyn["T0_plus"] = float(y[0])
            dyn["T0_prod"] = float(y[1])
            dyn["T0_minus"] = float(y[2])
            self.nominal_params = dyn
            self.N = max(1, self.T_horizon // int(self.dt))
            out = self.solve(verbose=False)
            solve_times.append(float(out["solve_time"]))
            u0 = float(out["optimal_schedule"][0]) if np.isfinite(out["optimal_schedule"][0]) else 0.5
            u0 = float(np.clip(u0, 0.0, 1.0))
            u_applied.append(u0)
            model = _model_from_nominal(dyn)
            y = _rk_step(model, y, u0, dt)
            x_hist.append(y.copy())
            t_hist.append(t_hist[-1] + dt)

        return {
            "times": np.array(t_hist),
            "states": np.array(x_hist),
            "u_applied": np.array(u_applied),
            "solve_times": np.array(solve_times),
        }
