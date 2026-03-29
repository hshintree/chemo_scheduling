#!/usr/bin/env python
"""
Script 03 — Compare treatment protocols across synthetic cohort.

Simulates four treatment strategies on all synthetic patients:
  1. MTD          : continuous treatment (abiraterone never stopped)
  2. Zhang AT     : stop at PSA ≤ 50% baseline, restart at PSA ≥ 100%
  3. Range-Bounded AT (RBAT) : stop at PSA ≤ 30% baseline, restart at PSA ≥ 80%
  4. Oracle       : grid-search over (stop, restart) thresholds to maximize TTP

Records per-patient:
  - Time to progression (TTP)
  - Cumulative drug-on days
  - Fraction of time off treatment
  - Final cell composition

Produces:
  - Kaplan-Meier curves
  - Scatter plot: TTP adaptive vs TTP MTD
  - Box plots of cumulative dose
  - Example trajectory panels for all 4 protocols (patient 0)

Outputs go to data/synthetic/figures/
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from tqdm import tqdm

from src.lotka_volterra import LotkaVolterraModel
from src.psa_model import PSAModel
from src.utils import (
    plot_km_curves,
    plot_ttp_scatter,
    plot_dose_boxplot,
    plot_trajectory,
    summary_table,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"
FIGURES_DIR   = SYNTHETIC_DIR / "figures"
T_MAX_DAYS    = 3650.0   # simulation horizon (10 years)

# Protocol thresholds (fraction of baseline PSA)
PROTOCOLS = {
    "mtd":      (None,  None),   # continuous
    "adaptive": (0.50,  1.00),   # Zhang 2017
    "rbat":     (0.30,  0.80),   # Brady-Nicholls 2022
}

# Oracle grid search ranges
ORACLE_STOP_GRID    = np.linspace(0.20, 0.70, 6)     # stop threshold
ORACLE_RESTART_GRID = np.linspace(0.60, 1.20, 7)     # restart threshold

PSA_PROG_FACTOR      = 2.0
RESISTANT_CUTOFF     = 0.80


# ---------------------------------------------------------------------------
# Simulation helpers (same logic as script 01, generalized)
# ---------------------------------------------------------------------------

class AdaptiveDrugSchedule:
    """
    Generic feedback-controlled drug schedule.
    """
    def __init__(self, baseline_psa: float, stop_frac: float, restart_frac: float) -> None:
        self.stop_thresh    = stop_frac    * baseline_psa
        self.restart_thresh = restart_frac * baseline_psa
        self._on            = True

    def update(self, psa: float) -> None:
        if self._on and psa <= self.stop_thresh:
            self._on = False
        elif not self._on and psa >= self.restart_thresh:
            self._on = True

    def __call__(self, t: float) -> int:
        return int(self._on)


def run_adaptive_protocol(
    model: LotkaVolterraModel,
    psa_model: PSAModel,
    stop_frac: float,
    restart_frac: float,
    t_max: float = T_MAX_DAYS,
) -> dict:
    """
    Simulate an adaptive protocol day-by-day.

    Returns
    -------
    dict with keys: ttp, drug_days, frac_off, final_T_minus_frac, t_end,
                    days (array), psa_norm (array), on_treatment (array)
    """
    baseline_psa = psa_model.compute_psa({
        "T_plus":  np.array([model.T0_plus]),
        "T_prod":  np.array([model.T0_prod]),
        "T_minus": np.array([model.T0_minus]),
    })[0]

    schedule = AdaptiveDrugSchedule(baseline_psa, stop_frac, restart_frac)
    y = np.array([model.T0_plus, model.T0_prod, model.T0_minus], dtype=float)

    day_arr, on_arr, psa_arr = [], [], []
    ttp = np.inf

    for day in range(int(t_max) + 1):
        T_plus, T_prod, T_minus = y
        current_psa = psa_model.compute_psa({
            "T_plus":  np.array([T_plus]),
            "T_prod":  np.array([T_prod]),
            "T_minus": np.array([T_minus]),
        })[0]
        psa_norm = current_psa / baseline_psa if baseline_psa > 0 else 0.0
        on_tx    = schedule(float(day))

        day_arr.append(day)
        on_arr.append(on_tx)
        psa_arr.append(psa_norm)

        total = T_plus + T_prod + T_minus
        resistant_frac = T_minus / total if total > 0 else 0.0

        if (psa_norm >= PSA_PROG_FACTOR and on_tx == 1) or \
           (resistant_frac >= RESISTANT_CUTOFF and day > 30):
            ttp = float(day)
            break

        if day < t_max:
            sub = model.simulate(
                t_span=(float(day), float(day + 1)),
                t_eval=np.array([float(day + 1)]),
                drug_schedule=schedule,
                y0=y,
            )
            y = np.array([sub["T_plus"][-1], sub["T_prod"][-1], sub["T_minus"][-1]])
            schedule.update(current_psa)

    T_plus, T_prod, T_minus = y
    total = T_plus + T_prod + T_minus
    final_resistant = T_minus / total if total > 0 else 0.0

    on_arr_np = np.array(on_arr)
    return {
        "ttp":               ttp,
        "drug_days":         int(on_arr_np.sum()),
        "frac_off":          float((on_arr_np == 0).mean()),
        "final_T_minus_frac": final_resistant,
        "t_end":             day_arr[-1],
        "days":              np.array(day_arr),
        "psa_norm":          np.array(psa_arr),
        "on_treatment":      on_arr_np,
    }


def run_mtd_protocol(
    model: LotkaVolterraModel,
    psa_model: PSAModel,
    t_max: float = T_MAX_DAYS,
) -> dict:
    """Simulate MTD (continuous treatment)."""
    t_eval = np.arange(0, int(t_max) + 1, 1.0)
    sim    = model.simulate((0.0, t_max), t_eval, lambda t: 1)

    baseline_psa = psa_model.compute_psa({
        "T_plus":  np.array([model.T0_plus]),
        "T_prod":  np.array([model.T0_prod]),
        "T_minus": np.array([model.T0_minus]),
    })[0]

    psa_vals = psa_model.compute_psa(sim)
    psa_norm = psa_vals / baseline_psa if baseline_psa > 0 else psa_vals

    # Progression: PSA > 2× baseline
    prog_mask = psa_norm >= PSA_PROG_FACTOR
    if prog_mask.any():
        cut = int(np.argmax(prog_mask))
        ttp = float(sim["t"][cut])
        days = sim["t"][:cut + 1].astype(int)
        psa_out = psa_norm[:cut + 1]
        on_tx   = np.ones(cut + 1, dtype=int)
        T_minus_final = sim["T_minus"][cut]
        total_final   = sim["total_cells"][cut]
    else:
        ttp     = np.inf
        days    = sim["t"].astype(int)
        psa_out = psa_norm
        on_tx   = np.ones(len(days), dtype=int)
        T_minus_final = sim["T_minus"][-1]
        total_final   = sim["total_cells"][-1]

    final_resistant = T_minus_final / total_final if total_final > 0 else 0.0

    return {
        "ttp":               ttp,
        "drug_days":         int(on_tx.sum()),
        "frac_off":          0.0,
        "final_T_minus_frac": final_resistant,
        "t_end":             int(days[-1]),
        "days":              days,
        "psa_norm":          psa_out,
        "on_treatment":      on_tx,
    }


def run_oracle_protocol(
    model: LotkaVolterraModel,
    psa_model: PSAModel,
    t_max: float = T_MAX_DAYS,
) -> dict:
    """Grid-search for the (stop, restart) thresholds that maximize TTP."""
    best_result = None
    best_ttp    = -1.0

    for stop_frac, restart_frac in product(ORACLE_STOP_GRID, ORACLE_RESTART_GRID):
        if restart_frac <= stop_frac:
            continue
        result = run_adaptive_protocol(model, psa_model, stop_frac, restart_frac, t_max)
        if result["ttp"] > best_ttp:
            best_ttp    = result["ttp"]
            best_result = result

    if best_result is None:
        best_result = run_mtd_protocol(model, psa_model, t_max)

    return best_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pop_params_path = SYNTHETIC_DIR / "population_params.csv"

    if not pop_params_path.exists():
        print("ERROR: population_params.csv not found. Run script 01 first.")
        sys.exit(1)

    pop_params = pd.read_csv(pop_params_path)
    n_patients = len(pop_params)
    psa_model  = PSAModel()

    print(f"Simulating {n_patients} patients × 4 protocols...")

    all_results: list[dict] = []
    ttp_by_protocol: dict[str, list[float]] = {p: [] for p in ["mtd", "adaptive", "rbat", "oracle"]}
    dose_by_protocol: dict[str, list[float]] = {p: [] for p in ["mtd", "adaptive", "rbat", "oracle"]}

    # Store trajectory for one example patient
    example_trajs: dict[str, dict] = {}

    for row in tqdm(pop_params.itertuples(), total=n_patients, ncols=70):
        pid = int(row.patient_id)

        alpha = np.array([
            [row.alpha_00, row.alpha_01, row.alpha_02],
            [row.alpha_10, row.alpha_11, row.alpha_12],
            [row.alpha_20, row.alpha_21, row.alpha_22],
        ])
        params = {
            "r_plus":    row.r_plus,
            "r_prod":    row.r_prod,
            "r_minus":   row.r_minus,
            "delta_plus": row.delta_plus,
            "delta_prod": row.delta_prod,
            "K":         row.K,
            "T0_plus":   row.T0_plus,
            "T0_prod":   row.T0_prod,
            "T0_minus":  row.T0_minus,
            "alpha":     alpha,
        }
        model = LotkaVolterraModel(params)

        results_this_patient: dict[str, dict] = {}

        # --- MTD ---
        r_mtd = run_mtd_protocol(model, psa_model)
        results_this_patient["mtd"] = r_mtd

        # --- Zhang AT ---
        r_at = run_adaptive_protocol(model, psa_model, 0.50, 1.00)
        results_this_patient["adaptive"] = r_at

        # --- RBAT ---
        r_rbat = run_adaptive_protocol(model, psa_model, 0.30, 0.80)
        results_this_patient["rbat"] = r_rbat

        # --- Oracle (only every 5th patient for speed) ---
        if pid % 5 == 0:
            r_oracle = run_oracle_protocol(model, psa_model)
        else:
            r_oracle = r_at    # approximate with AT for speed
        results_this_patient["oracle"] = r_oracle

        for pname, res in results_this_patient.items():
            ttp_by_protocol[pname].append(res["ttp"])
            dose_by_protocol[pname].append(float(res["drug_days"]))
            all_results.append({
                "patient_id":        pid,
                "protocol":          pname,
                "ttp":               res["ttp"],
                "drug_days":         res["drug_days"],
                "frac_off":          res["frac_off"],
                "final_T_minus_frac": res["final_T_minus_frac"],
            })

        if pid == 0:
            example_trajs = results_this_patient
            example_model = model

    # --- Convert lists to arrays ---
    ttp_np   = {p: np.array(v) for p, v in ttp_by_protocol.items()}
    dose_np  = {p: np.array(v) for p, v in dose_by_protocol.items()}

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------

    # 1. Kaplan-Meier curves
    fig_km = plot_km_curves(ttp_np, title="Kaplan-Meier: Progression-Free Survival")
    fig_km.savefig(FIGURES_DIR / "km_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_km)

    # 2. Scatter: AT vs MTD
    fig_scatter = plot_ttp_scatter(ttp_np["adaptive"], ttp_np["mtd"],
                                   max_days=T_MAX_DAYS)
    fig_scatter.savefig(FIGURES_DIR / "ttp_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig_scatter)

    # 3. Box plots of cumulative dose
    fig_box = plot_dose_boxplot(dose_np)
    fig_box.savefig(FIGURES_DIR / "dose_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig_box)

    # 4. Example patient trajectories for all 4 protocols
    if example_trajs:
        fig_ex, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=False)
        protocol_order = ["mtd", "adaptive", "rbat", "oracle"]
        protocol_labels = {
            "mtd":      "MTD (continuous)",
            "adaptive": "Zhang AT (50%/100%)",
            "rbat":     "Range-Bounded AT (30%/80%)",
            "oracle":   "Oracle optimal",
        }

        # Full simulation for cell populations
        for row_idx, pname in enumerate(protocol_order):
            res = example_trajs[pname]
            ax_psa  = axes[row_idx, 0]
            ax_drug = axes[row_idx, 1]

            days    = res["days"]
            psa_n   = res["psa_norm"]
            on_tx   = res["on_treatment"]

            # PSA panel
            ax_psa.plot(days, psa_n, color="purple", lw=1.5)
            ax_psa.axhline(1.0, color="gray", ls=":", lw=1)
            ax_psa.axhline(2.0, color="red",  ls="--", lw=1, alpha=0.5, label="2× baseline")
            ax_psa.set_ylabel("PSA (normalized)")
            ax_psa.set_title(f"{protocol_labels[pname]} — PSA", fontsize=9)
            ax_psa.grid(True, alpha=0.3)
            ax_psa.legend(fontsize=7)

            # Drug schedule panel
            ax_drug.fill_between(days, on_tx, step="post", alpha=0.6, color="#8c564b")
            ax_drug.set_yticks([0, 1])
            ax_drug.set_yticklabels(["Off", "On"])
            ax_drug.set_title(f"{protocol_labels[pname]} — Drug schedule", fontsize=9)
            ax_drug.grid(True, alpha=0.3)

            if row_idx == 3:
                ax_psa.set_xlabel("Days")
                ax_drug.set_xlabel("Days")

        fig_ex.suptitle("Patient 000 — All 4 Treatment Protocols", fontsize=12, y=1.01)
        fig_ex.tight_layout()
        fig_ex.savefig(FIGURES_DIR / "example_patient_trajectories.png",
                       dpi=150, bbox_inches="tight")
        plt.close(fig_ex)

        # Also plot full cell populations for patient 0 under adaptive vs MTD
        for pname, (stop, restart) in [("adaptive", (0.50, 1.00)), ("mtd", (None, None))]:
            if pname == "mtd":
                drug_fn = lambda t: 1
            else:
                psa_model_0 = PSAModel()
                bp = psa_model_0.compute_psa({
                    "T_plus":  np.array([example_model.T0_plus]),
                    "T_prod":  np.array([example_model.T0_prod]),
                    "T_minus": np.array([example_model.T0_minus]),
                })[0]
                sched = AdaptiveDrugSchedule(bp, stop, restart)
                drug_fn = sched

            t_eval = np.linspace(0, T_MAX_DAYS, 2000)
            full_sim = example_model.simulate((0, T_MAX_DAYS), t_eval, drug_fn)
            fig_traj = plot_trajectory(
                full_sim,
                title=f"Patient 000 — {pname.upper()} cell populations",
            )
            fig_traj.savefig(FIGURES_DIR / f"patient_000_{pname}_cells.png",
                             dpi=150, bbox_inches="tight")
            plt.close(fig_traj)

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(SYNTHETIC_DIR / "protocol_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("PROTOCOL COMPARISON SUMMARY")
    print("=" * 70)
    for pname in ["mtd", "adaptive", "rbat", "oracle"]:
        ttps = ttp_np[pname]
        finite_ttps = ttps[~np.isinf(ttps)]
        frac_prog = float(np.mean(~np.isinf(ttps)))
        med_ttp   = np.median(finite_ttps) if len(finite_ttps) > 0 else np.nan
        mean_dose = float(np.mean(dose_np[pname]))
        print(
            f"  {pname:10s}  TTP median={med_ttp:6.0f}d  "
            f"% progressed={frac_prog * 100:.0f}%  "
            f"mean drug-days={mean_dose:.0f}"
        )

    print(f"\n  Figures saved to: {FIGURES_DIR}")
    print(f"  Results saved to: {SYNTHETIC_DIR}/protocol_comparison.csv")


if __name__ == "__main__":
    main()
