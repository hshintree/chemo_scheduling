#!/usr/bin/env python
"""
Script 01 — Generate synthetic patient cohort.

Draws N=50 patients from published population parameter distributions
(log-normal with coefficients of variation from Zhang et al. 2017) and
simulates each patient under:
  1. Zhang adaptive therapy (stop at PSA ≤ 50% baseline, restart at PSA ≥ 100%)
  2. Maximum tolerated dose (MTD, continuous treatment)

Outputs
-------
data/synthetic/patient_{i:03d}_adaptive.csv
data/synthetic/patient_{i:03d}_mtd.csv
data/synthetic/population_params.csv
"""

from __future__ import annotations

import sys
import os

# Allow running from project root: python scripts/01_generate_synthetic_data.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.lotka_volterra import LotkaVolterraModel
from src.psa_model import PSAModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_PATIENTS = 50
T_DAYS     = 1825          # 5 years
DT         = 1.0           # 1-day time resolution
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "synthetic"

# Published population mean parameter values
POP_MEANS = {
    "r_plus":    np.log(2) / 250,
    "r_prod":    np.log(2) / 200,
    "r_minus":   np.log(2) / 104,
    "delta_plus": 0.15,
    "T0_plus":   5_000.0,
    "T0_prod":   500.0,
    "T0_minus":  50.0,
}

# Coefficients of variation (σ / μ) for log-normal sampling
POP_CV = {
    "r_plus":    0.30,
    "r_prod":    0.30,
    "r_minus":   0.25,
    "delta_plus": 0.40,
    "T0_plus":   0.50,
    "T0_prod":   0.80,
    "T0_minus":  1.00,
}

# Zhang adaptive therapy thresholds (fractions of baseline PSA)
AT_STOP_THRESHOLD    = 0.50   # stop treatment when PSA ≤ 50% of baseline
AT_RESTART_THRESHOLD = 1.00   # restart when PSA ≥ 100% of baseline

# Progression criteria
PSA_PROGRESSION_FACTOR    = 2.0   # PSA > 2× baseline on treatment
RESISTANT_FRACTION_CUTOFF = 0.80  # T- > 80% of total cells


# ---------------------------------------------------------------------------
# Log-normal sampling helpers
# ---------------------------------------------------------------------------

def lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """
    Convert (mean, CV) to (mu, sigma) for np.random.lognormal.
    The parameterisation is: X = exp(mu + sigma * Z), Z ~ N(0,1)
    E[X] = exp(mu + sigma^2/2) = mean
    CV[X] = sqrt(exp(sigma^2) - 1) ≈ cv  (for small cv)
    """
    sigma2 = np.log(1.0 + cv ** 2)
    mu     = np.log(mean) - 0.5 * sigma2
    return mu, np.sqrt(sigma2)


def sample_patient_params(rng: np.random.Generator) -> dict:
    """Sample one patient's parameters from population distributions."""
    params = {}
    for key in ["r_plus", "r_prod", "r_minus", "delta_plus",
                "T0_plus", "T0_prod", "T0_minus"]:
        mu, sigma = lognormal_params(POP_MEANS[key], POP_CV[key])
        params[key] = float(rng.lognormal(mu, sigma))

    # delta_prod is a fixed ratio of delta_plus
    params["delta_prod"] = 0.5 * params["delta_plus"]

    # Competition coefficients: uniform on [0.3, 0.8] for off-diagonal entries
    alpha = np.eye(3)
    for i in range(3):
        for j in range(3):
            if i != j:
                alpha[i, j] = rng.uniform(0.3, 0.8)
    params["alpha"] = alpha
    params["K"] = 10_000.0

    return params


# ---------------------------------------------------------------------------
# Drug schedule builders
# ---------------------------------------------------------------------------

def build_mtd_schedule():
    """Always on."""
    return lambda t: 1


class AdaptiveSchedule:
    """
    Zhang adaptive therapy: event-driven feedback controller.

    The schedule is integrated *alongside* the ODE — the drug state is updated
    at integer day boundaries based on the PSA/cell trajectory so far.
    Because solve_ivp does not expose state between steps, we simulate day-by-day
    with short sub-intervals and update the drug flag each day.
    """

    def __init__(
        self,
        baseline_psa: float,
        stop_fraction: float = AT_STOP_THRESHOLD,
        restart_fraction: float = AT_RESTART_THRESHOLD,
    ) -> None:
        self.baseline_psa    = baseline_psa
        self.stop_threshold  = stop_fraction  * baseline_psa
        self.restart_threshold = restart_fraction * baseline_psa
        self._drug_on        = True    # start on treatment

    def update(self, current_psa: float) -> None:
        """Call once per day with the current PSA value to update drug flag."""
        if self._drug_on and current_psa <= self.stop_threshold:
            self._drug_on = False
        elif not self._drug_on and current_psa >= self.restart_threshold:
            self._drug_on = True

    def __call__(self, t: float) -> int:
        return int(self._drug_on)


# ---------------------------------------------------------------------------
# Day-by-day simulation with adaptive controller
# ---------------------------------------------------------------------------

def simulate_adaptive(
    model: LotkaVolterraModel,
    psa_model: PSAModel,
    t_max: float = T_DAYS,
    stop_fraction: float = AT_STOP_THRESHOLD,
    restart_fraction: float = AT_RESTART_THRESHOLD,
) -> pd.DataFrame:
    """
    Simulate the adaptive therapy protocol day-by-day.

    Returns a DataFrame with columns:
        day, T_plus, T_prod, T_minus, psa_normalized, on_treatment, protocol
    """
    baseline_psa = psa_model.compute_psa({
        "T_plus":  np.array([model.T0_plus]),
        "T_prod":  np.array([model.T0_prod]),
        "T_minus": np.array([model.T0_minus]),
    })[0]

    schedule = AdaptiveSchedule(baseline_psa, stop_fraction, restart_fraction)

    rows = []
    y = np.array([model.T0_plus, model.T0_prod, model.T0_minus], dtype=float)

    for day in range(int(t_max) + 1):
        T_plus, T_prod, T_minus = y
        current_psa = psa_model.compute_psa({
            "T_plus":  np.array([T_plus]),
            "T_prod":  np.array([T_prod]),
            "T_minus": np.array([T_minus]),
        })[0]
        psa_norm = current_psa / baseline_psa if baseline_psa > 0 else 0.0

        on_tx = schedule(float(day))
        rows.append({
            "day":           day,
            "T_plus":        T_plus,
            "T_prod":        T_prod,
            "T_minus":       T_minus,
            "psa_normalized": psa_norm,
            "on_treatment":  on_tx,
            "protocol":      "adaptive",
        })

        # Check for progression
        total = T_plus + T_prod + T_minus
        initial_total = model.T0_plus + model.T0_prod + model.T0_minus
        resistant_frac = T_minus / total if total > 0 else 0.0
        if (psa_norm >= PSA_PROGRESSION_FACTOR and on_tx == 1) or \
           (resistant_frac >= RESISTANT_FRACTION_CUTOFF and day > 30):
            break

        # Advance one day
        if day < t_max:
            sub_sim = model.simulate(
                t_span=(float(day), float(day + 1)),
                t_eval=np.array([float(day + 1)]),
                drug_schedule=schedule,
                y0=y,
            )
            y = np.array([sub_sim["T_plus"][-1],
                          sub_sim["T_prod"][-1],
                          sub_sim["T_minus"][-1]])
            # Update schedule for next step based on current PSA
            schedule.update(current_psa)

    return pd.DataFrame(rows)


def simulate_mtd(
    model: LotkaVolterraModel,
    psa_model: PSAModel,
    t_max: float = T_DAYS,
) -> pd.DataFrame:
    """
    Simulate MTD (always-on) protocol.

    Returns a DataFrame with columns:
        day, T_plus, T_prod, T_minus, psa_normalized, on_treatment, protocol
    """
    t_eval = np.arange(0, int(t_max) + 1, 1.0)
    drug   = build_mtd_schedule()
    sim    = model.simulate((0.0, t_max), t_eval, drug)

    baseline_psa = psa_model.compute_psa({
        "T_plus":  np.array([model.T0_plus]),
        "T_prod":  np.array([model.T0_prod]),
        "T_minus": np.array([model.T0_minus]),
    })[0]

    psa_vals = psa_model.compute_psa(sim)
    psa_norm = psa_vals / baseline_psa if baseline_psa > 0 else psa_vals

    df = pd.DataFrame({
        "day":           sim["t"].astype(int),
        "T_plus":        sim["T_plus"],
        "T_prod":        sim["T_prod"],
        "T_minus":       sim["T_minus"],
        "psa_normalized": psa_norm,
        "on_treatment":  sim["drug"].astype(int),
        "protocol":      "mtd",
    })

    # Truncate at progression (PSA > 2× baseline on continuous treatment)
    prog_mask = df["psa_normalized"] >= PSA_PROGRESSION_FACTOR
    if prog_mask.any():
        cutoff = df.index[prog_mask][0]
        df = df.iloc[: cutoff + 1].copy()

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    psa_model = PSAModel()

    all_params: list[dict] = []
    summary_rows: list[dict] = []

    print(f"Generating {N_PATIENTS} synthetic patients...")

    for i in tqdm(range(N_PATIENTS), ncols=70):
        rng    = np.random.default_rng(seed=i)
        params = sample_patient_params(rng)
        model  = LotkaVolterraModel(params)

        # --- Adaptive ---
        df_adaptive = simulate_adaptive(model, psa_model)
        df_adaptive.to_csv(OUTPUT_DIR / f"patient_{i:03d}_adaptive.csv", index=False)

        # --- MTD ---
        df_mtd = simulate_mtd(model, psa_model)
        df_mtd.to_csv(OUTPUT_DIR / f"patient_{i:03d}_mtd.csv", index=False)

        # --- Summary metrics ---
        ttp_adaptive = float(df_adaptive["day"].iloc[-1])
        ttp_mtd      = float(df_mtd["day"].iloc[-1])
        frac_off     = float((df_adaptive["on_treatment"] == 0).mean())

        summary_rows.append({
            "patient_id":     i,
            "ttp_adaptive":   ttp_adaptive,
            "ttp_mtd":        ttp_mtd,
            "frac_off_tx":    frac_off,
            "ratio_ttp":      ttp_adaptive / ttp_mtd if ttp_mtd > 0 else np.nan,
        })

        # --- Store params (flatten alpha) ---
        flat = {
            "patient_id":    i,
            "r_plus":        params["r_plus"],
            "r_prod":        params["r_prod"],
            "r_minus":       params["r_minus"],
            "delta_plus":    params["delta_plus"],
            "delta_prod":    params["delta_prod"],
            "K":             params["K"],
            "T0_plus":       params["T0_plus"],
            "T0_prod":       params["T0_prod"],
            "T0_minus":      params["T0_minus"],
        }
        alpha = params["alpha"]
        for ii in range(3):
            for jj in range(3):
                flat[f"alpha_{ii}{jj}"] = alpha[ii, jj]
        all_params.append(flat)

    # --- Save population parameters ---
    pd.DataFrame(all_params).to_csv(
        OUTPUT_DIR / "population_params.csv", index=False
    )

    # --- Print summary table ---
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 65)
    print("SYNTHETIC COHORT SUMMARY")
    print("=" * 65)
    print(f"  N patients         : {N_PATIENTS}")
    print(f"  Simulation horizon : {T_DAYS} days ({T_DAYS / 365:.1f} years)")
    print()
    print(f"  TTP adaptive  — mean: {df_summary['ttp_adaptive'].mean():.0f} d  "
          f"median: {df_summary['ttp_adaptive'].median():.0f} d")
    print(f"  TTP MTD       — mean: {df_summary['ttp_mtd'].mean():.0f} d  "
          f"median: {df_summary['ttp_mtd'].median():.0f} d")
    print(f"  TTP ratio (AT/MTD) — mean: {df_summary['ratio_ttp'].mean():.2f}")
    print(f"  % time off tx (AT) — mean: {df_summary['frac_off_tx'].mean() * 100:.1f}%")
    print()
    print(f"  Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
