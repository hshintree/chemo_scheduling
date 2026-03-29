#!/usr/bin/env python
"""
Script 02 — Fit Lotka-Volterra model to patient PSA data.

Given a patient's PSA time series (real or synthetic), fits the LV model
parameters via nonlinear least squares with L-BFGS-B and 20 random restarts.

Usage
-----
# Validate on the first 5 synthetic patients (default):
python scripts/02_fit_patient_parameters.py

# Fit a specific real patient CSV:
python scripts/02_fit_patient_parameters.py --patient data/raw/patient_01.csv

Outputs
-------
For each patient: a fitted parameter CSV + residual plot saved to
data/processed/patient_{id}_fit.{csv,png}
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
from tqdm import tqdm

from src.lotka_volterra import LotkaVolterraModel
from src.psa_model import PSAModel
from src.utils import compute_rmse, compute_r2, plot_psa

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"

N_RESTARTS    = 20
K_FIXED       = 10_000.0
ALPHA_FIXED   = np.array([[1.0, 0.5, 0.5],
                           [0.5, 1.0, 0.5],
                           [0.5, 0.5, 1.0]])

# Parameter names in the optimization vector
PARAM_NAMES   = ["r_plus", "r_prod", "r_minus", "delta_plus", "T0_plus", "T0_prod", "T0_minus"]

# Bounds (lower, upper) for each parameter
PARAM_BOUNDS  = [
    (1e-5, 0.05),      # r_plus
    (1e-5, 0.05),      # r_prod
    (1e-5, 0.05),      # r_minus
    (1e-5, 1.0),       # delta_plus
    (10.0, 50_000.0),  # T0_plus
    (1.0,  20_000.0),  # T0_prod
    (0.1,  5_000.0),   # T0_minus
]

# Published means used as reference for random restarts
PARAM_NOMINAL = np.array([
    np.log(2) / 250,   # r_plus
    np.log(2) / 200,   # r_prod
    np.log(2) / 104,   # r_minus
    0.15,              # delta_plus
    5_000.0,           # T0_plus
    500.0,             # T0_prod
    50.0,              # T0_minus
])


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def _unpack(x: np.ndarray) -> dict:
    """Map optimization vector to parameter dict."""
    return {
        "r_plus":    x[0],
        "r_prod":    x[1],
        "r_minus":   x[2],
        "delta_plus": x[3],
        "delta_prod": 0.5 * x[3],   # fixed ratio
        "T0_plus":   x[4],
        "T0_prod":   x[5],
        "T0_minus":  x[6],
        "K":         K_FIXED,
        "alpha":     ALPHA_FIXED.copy(),
    }


def _build_drug_schedule(days: np.ndarray, on_treatment: np.ndarray):
    """
    Build a piecewise-constant drug schedule function from discrete observations.

    Between observed time points, the drug value is held constant at the last
    known value (zero-order hold).
    """
    days_arr  = np.asarray(days, dtype=float)
    on_arr    = np.asarray(on_treatment, dtype=int)
    max_day   = days_arr[-1]

    def schedule(t: float) -> int:
        if t <= days_arr[0]:
            return int(on_arr[0])
        if t > max_day:
            return int(on_arr[-1])
        idx = np.searchsorted(days_arr, t, side="right") - 1
        idx = int(np.clip(idx, 0, len(on_arr) - 1))
        return int(on_arr[idx])

    return schedule


def _objective(
    x: np.ndarray,
    obs_days: np.ndarray,
    obs_psa_norm: np.ndarray,
    drug_fn,
    psa_model: PSAModel,
) -> float:
    """Sum of squared residuals between observed and modeled normalized PSA."""
    try:
        params = _unpack(x)
        model  = LotkaVolterraModel(params)
        sim    = model.simulate(
            t_span=(obs_days[0], obs_days[-1]),
            t_eval=obs_days,
            drug_schedule=drug_fn,
        )
        psa_vals  = psa_model.compute_psa(sim)
        psa_norm  = psa_vals / psa_vals[0] if psa_vals[0] > 0 else psa_vals
        residuals = psa_norm - obs_psa_norm
        return float(np.sum(residuals ** 2))
    except Exception:
        return 1e12


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_patient(
    days: np.ndarray,
    psa_norm: np.ndarray,
    on_treatment: np.ndarray,
    n_restarts: int = N_RESTARTS,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[dict, float]:
    """
    Fit LV model to a single patient.

    Parameters
    ----------
    days : 1-D ndarray of observation times (days)
    psa_norm : 1-D ndarray of normalized PSA values (PSA / PSA[0])
    on_treatment : 1-D ndarray of {0, 1} treatment indicators
    n_restarts : int
    seed : int
    verbose : bool

    Returns
    -------
    best_params : dict
    best_rmse   : float
    """
    rng      = np.random.default_rng(seed)
    drug_fn  = _build_drug_schedule(days, on_treatment)
    psa_mod  = PSAModel()

    best_result = None
    best_cost   = np.inf

    for trial in range(n_restarts):
        # Random initial condition around nominal parameters
        if trial == 0:
            x0 = PARAM_NOMINAL.copy()
        else:
            # Log-normal perturbation: multiply each parameter by exp(N(0, 0.5))
            x0 = PARAM_NOMINAL * np.exp(rng.normal(0, 0.5, size=len(PARAM_NOMINAL)))

        # Clip to bounds
        x0 = np.array([np.clip(x0[i], PARAM_BOUNDS[i][0], PARAM_BOUNDS[i][1])
                        for i in range(len(x0))])

        result = minimize(
            fun=_objective,
            x0=x0,
            args=(days, psa_norm, drug_fn, psa_mod),
            method="L-BFGS-B",
            bounds=PARAM_BOUNDS,
            options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-8},
        )

        if verbose:
            print(f"  Restart {trial:2d}: cost={result.fun:.4f}")

        if result.fun < best_cost:
            best_cost   = result.fun
            best_result = result

    best_params_flat  = best_result.x
    best_params       = _unpack(best_params_flat)

    # Compute predicted PSA for diagnostics
    model = LotkaVolterraModel(best_params)
    sim   = model.simulate(
        t_span=(days[0], days[-1]),
        t_eval=days,
        drug_schedule=drug_fn,
    )
    psa_pred     = psa_mod.compute_psa(sim)
    psa_pred_norm = psa_pred / psa_pred[0] if psa_pred[0] > 0 else psa_pred

    rmse = compute_rmse(psa_norm, psa_pred_norm)
    r2   = compute_r2(psa_norm, psa_pred_norm)

    if verbose:
        print(f"  Best RMSE: {rmse:.4f}  R²: {r2:.4f}")

    return best_params, rmse, r2, psa_pred_norm


# ---------------------------------------------------------------------------
# Validation against synthetic ground truth
# ---------------------------------------------------------------------------

def validate_synthetic_patients(n: int = 5) -> None:
    """
    Fit the model to the first n synthetic patients and compare with
    the known true parameters.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pop_params = pd.read_csv(SYNTHETIC_DIR / "population_params.csv")

    print(f"\nValidating parameter recovery on {n} synthetic patients")
    print("=" * 70)

    records = []
    for i in range(n):
        csv_path = SYNTHETIC_DIR / f"patient_{i:03d}_adaptive.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path} not found — run script 01 first.")
            continue

        df = pd.read_csv(csv_path)
        # Use total cell count as "PSA" proxy (all c_i = 1)
        total_cells = df["T_plus"] + df["T_prod"] + df["T_minus"]
        psa_norm    = (total_cells / total_cells.iloc[0]).values
        days        = df["day"].values.astype(float)
        on_tx       = df["on_treatment"].values.astype(int)

        print(f"\nPatient {i:03d} — fitting ({N_RESTARTS} restarts)...")
        best_params, rmse, r2, psa_pred = fit_patient(
            days, psa_norm, on_tx, verbose=False
        )

        # True parameter from ground truth
        true_row = pop_params[pop_params["patient_id"] == i].iloc[0]
        rec = {
            "patient_id": i,
            "rmse":  rmse,
            "r2":    r2,
        }
        for pname in ["r_plus", "r_prod", "r_minus", "delta_plus", "T0_plus", "T0_prod", "T0_minus"]:
            rec[f"true_{pname}"]   = float(true_row[pname]) if pname in true_row else np.nan
            rec[f"fitted_{pname}"] = best_params[pname]

        records.append(rec)
        print(f"  RMSE={rmse:.4f}  R²={r2:.4f}")
        print(f"  r_plus:  true={rec['true_r_plus']:.5f}  fitted={rec['fitted_r_plus']:.5f}")
        print(f"  delta+:  true={rec['true_delta_plus']:.4f}  fitted={rec['fitted_delta_plus']:.4f}")

        # --- Save fitted params ---
        flat = {k: v for k, v in best_params.items() if k != "alpha"}
        pd.DataFrame([flat]).to_csv(
            PROCESSED_DIR / f"patient_{i:03d}_fit_params.csv", index=False
        )

        # --- Plot ---
        drug_fn    = _build_drug_schedule(days, on_tx)
        model_best = LotkaVolterraModel(best_params)
        sim_best   = model_best.simulate((days[0], days[-1]), days, drug_fn)

        fig = plot_psa(
            t=days,
            psa_model=psa_pred,
            psa_observed=psa_norm,
            t_observed=days,
            drug=np.array([float(drug_fn(t)) for t in days]),
            title=f"Patient {i:03d} — fitted vs observed (RMSE={rmse:.4f}, R²={r2:.3f})",
        )
        fig.savefig(PROCESSED_DIR / f"patient_{i:03d}_fit.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    if records:
        pd.DataFrame(records).to_csv(PROCESSED_DIR / "fit_validation.csv", index=False)
        print(f"\n  Validation results saved to {PROCESSED_DIR}/fit_validation.csv")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit LV model to patient PSA data."
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Path to a patient CSV file (real data). "
             "If omitted, validates on first 5 synthetic patients.",
    )
    parser.add_argument(
        "--n-validate",
        type=int,
        default=5,
        help="Number of synthetic patients to validate against (default 5).",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.patient is not None:
        # --- Fit a single real patient CSV ---
        df   = pd.read_csv(args.patient)
        days = df["day"].values.astype(float)
        psa  = df["psa"].values.astype(float)
        on_tx = df["on_treatment"].values.astype(int)
        psa_norm = psa / psa[0] if psa[0] > 0 else psa

        print(f"Fitting patient: {args.patient}")
        best_params, rmse, r2, psa_pred = fit_patient(
            days, psa_norm, on_tx, verbose=True
        )
        print(f"\nFitted parameters:")
        for k, v in best_params.items():
            if k != "alpha":
                print(f"  {k}: {v:.6g}")
        print(f"RMSE: {rmse:.4f}  R²: {r2:.4f}")

        # Save
        stem = Path(args.patient).stem
        flat = {k: v for k, v in best_params.items() if k != "alpha"}
        pd.DataFrame([flat]).to_csv(
            PROCESSED_DIR / f"{stem}_fit_params.csv", index=False
        )

        drug_fn = _build_drug_schedule(days, on_tx)
        fig = plot_psa(
            t=days,
            psa_model=psa_pred,
            psa_observed=psa_norm,
            t_observed=days,
            drug=np.array([float(drug_fn(t)) for t in days]),
            title=f"{stem} — fitted (RMSE={rmse:.4f}, R²={r2:.3f})",
        )
        out_png = PROCESSED_DIR / f"{stem}_fit.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {out_png}")

    else:
        validate_synthetic_patients(n=args.n_validate)


if __name__ == "__main__":
    main()
