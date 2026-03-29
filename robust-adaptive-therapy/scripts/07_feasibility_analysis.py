#!/usr/bin/env python
"""
SCRIPT 07: Feasibility Analysis
Validates that the simulation reproduces the Cunningham trial
(adaptive median TTP ~998 days) and tags **drug-eligible** patients
(δ+ > threshold) in the Wasserstein set for CVaR optimization.

Zhang vs MTD TTP is reported for science (subgroup where robust scheduling
matters most) but does **not** gate the optimization cohort.

Must pass validation before proceeding to script 08.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.feasibility import DRUG_EFFECT_THRESHOLD, FeasibilityClassifier, T_MAX_FEASIBILITY
from src.lotka_volterra import LotkaVolterraModel
from src.population_model import PopulationModel
from src.psa_model import PSAModel
from src.policy import ThresholdPolicy
from src.uncertainty import EllipsoidalUncertaintySet, WassersteinUncertaintySet

ROOT = Path(__file__).resolve().parent.parent
DATA_SYN = ROOT / "data" / "synthetic"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "feasibility"

# Trial ground truth from Cunningham et al. — used for validation only
TRIAL_MEDIAN_TTP_DAYS = 998.0
TRIAL_TTP_TOLERANCE = 200.0  # warn if simulated median is outside this band

T_MAX_DAYS = int(T_MAX_FEASIBILITY)


def rows_to_params(mat: np.ndarray, param_names: list[str]) -> list[dict]:
    out = []
    for i in range(len(mat)):
        d = dict(zip(param_names, mat[i].tolist()))
        d["K"] = 10_000.0
        d["delta_prod"] = 0.5 * float(d["delta_plus"])
        d["alpha"] = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]], dtype=float)
        out.append(d)
    return out


def validate_simulation(param_list: list[dict], t_max: int, n_validate: int = 30) -> np.ndarray:
    """
    Run Zhang AT on a sample of the cohort.
    Warn if median TTP is far from the trial benchmark of 998 days.
    """
    rng = np.random.default_rng(0)
    sample = rng.choice(len(param_list), size=min(n_validate, len(param_list)), replace=False)
    psa_m = PSAModel()
    ttps = []
    for i in sample:
        p = param_list[i]
        m = LotkaVolterraModel(p)
        out = ThresholdPolicy(0.5, 1.0).simulate_patient(m, psa_m, t_max=float(t_max), check_interval=1)
        ttp = float(out["ttp"]) if np.isfinite(out["ttp"]) else float(t_max)
        ttps.append(ttp)
    return np.array(ttps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--fast", action="store_true", help="100 samples")
    args = parser.parse_args()
    n = 100 if args.fast else args.n_samples

    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_SYN / "population_params.csv")
    records = df[PopulationModel.param_names].to_dict(orient="records")
    pop = PopulationModel()
    fit = pop.fit(records)

    ell = EllipsoidalUncertaintySet(
        fit["mean_log"], fit["cov_log"], confidence=0.90,
        param_names=PopulationModel.param_names)
    X_log = np.log(df[PopulationModel.param_names].astype(float).values)
    w = WassersteinUncertaintySet(X_log, epsilon=0.1,
                                  param_names=PopulationModel.param_names)
    w.epsilon = w.compute_epsilon_from_confidence(0.90)
    print(f"Wasserstein epsilon: {w.epsilon:.4f}")

    w_mat = w.sample_perturbed_params(n, seed=42)
    e_mat = ell.sample_interior(n, seed=43)
    params_w = rows_to_params(w_mat, PopulationModel.param_names)
    params_e = rows_to_params(e_mat, PopulationModel.param_names)

    # === STEP 1: VALIDATION ===
    print("\n[Step 1] Validating simulation against Cunningham trial...")
    val_ttps = validate_simulation(params_w, T_MAX_DAYS, n_validate=30)
    sim_median = float(np.median(val_ttps))
    print(f"  Simulated median TTP (Zhang AT, n=30):  {sim_median:.0f} days")
    print(f"  Trial benchmark (Cunningham adaptive):  {TRIAL_MEDIAN_TTP_DAYS:.0f} days")
    if abs(sim_median - TRIAL_MEDIAN_TTP_DAYS) > TRIAL_TTP_TOLERANCE:
        print(
            f"  WARNING: Simulated median is {abs(sim_median - TRIAL_MEDIAN_TTP_DAYS):.0f} days "
            f"from trial benchmark — check population_params calibration."
        )
    else:
        print(f"  PASS: Within {TRIAL_TTP_TOLERANCE:.0f}-day tolerance of trial benchmark.")

    # === STEP 2: ELIGIBILITY (drug effect) + Zhang/MTD reporting ===
    print(
        f"\n[Step 2] Drug eligibility (δ+ > {DRUG_EFFECT_THRESHOLD}) + Zhang vs MTD (informational)…"
    )
    clf = FeasibilityClassifier(LotkaVolterraModel, PSAModel, t_max=float(T_MAX_DAYS))
    out_w = clf.classify_cohort(params_w)
    out_e = clf.classify_cohort(params_e)
    bf = out_w["boundary_features"]

    # === STEP 3: PLOTS ===
    drug_ok = bf[bf["feasible"]]
    drug_bad = bf[~bf["feasible"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(drug_ok["T0_minus"], drug_ok["ratio_rm_dp"],
               c="green", alpha=0.6, s=20, label=f"Drug-eligible (n={len(drug_ok)})")
    ax.scatter(drug_bad["T0_minus"], drug_bad["ratio_rm_dp"],
               c="red", alpha=0.4, s=20, label=f"Not drug-eligible (n={len(drug_bad)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T0_minus (initial resistant cells)")
    ax.set_ylabel("r_minus / delta_plus")
    ax.set_title(
        f"AT cohort: δ+ > {DRUG_EFFECT_THRESHOLD} (day⁻¹)\n(Wasserstein-perturbed samples)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "landscape.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    wins = bf[bf["zhang_beats_mtd"]]
    loses = bf[~bf["zhang_beats_mtd"]]
    if len(bf) > 2:
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(wins) > 0:
            ax.hist(wins["ttp_gap"], bins=min(20, max(5, len(wins))), color="steelblue",
                    alpha=0.65, label=f"Zhang > MTD (n={len(wins)})")
        if len(loses) > 0:
            ax.hist(loses["ttp_gap"], bins=min(20, max(5, len(loses))), color="coral",
                    alpha=0.55, label=f"Zhang ≤ MTD (n={len(loses)})")
        ax.set_xlabel("TTP gap (Zhang − MTD, days)")
        ax.set_ylabel("Count")
        ax.set_title("Informational: Zhang vs MTD (all Wasserstein samples)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "ttp_gap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["r_minus", "delta_plus", "T0_minus"]):
        if len(drug_ok) > 1:
            drug_ok[col].plot.density(ax=ax, label="drug-eligible", color="green", alpha=0.7)
        if len(drug_bad) > 1:
            drug_bad[col].plot.density(ax=ax, label="not eligible", color="red", alpha=0.7)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Drug-eligible vs not (Wasserstein samples)")
    fig.tight_layout()
    fig.savefig(FIG / "infeasible_params.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        PROC / "feasible_scenarios.npz",
        idx_w_feasible=np.array(out_w["feasible_indices"], dtype=int),
        idx_e_feasible=np.array(out_e["feasible_indices"], dtype=int),
        samples_w=w_mat,
        samples_e=e_mat,
        param_names=np.array(PopulationModel.param_names, dtype=object),
        t_max_days=np.array([T_MAX_DAYS], dtype=np.int64),
        wasserstein_epsilon=np.array([w.epsilon]),
        drug_effect_threshold=np.array([DRUG_EFFECT_THRESHOLD], dtype=float),
    )

    n_w = int(out_w["n_zhang_beats_mtd"])
    n_l = int(out_w["n_zhang_loses_mtd"])
    pct_elig = 100 * out_w["feasible_fraction"]
    sim_ok = 798 <= sim_median <= 1198
    feas_ok = pct_elig >= 80.0

    print("\n=== FEASIBILITY ANALYSIS REPORT ===")
    print(f"\nSimulation horizon:    {T_MAX_DAYS} days (5-year, covers observed 342–1701 d range)")
    print(
        f"Optimization cohort:   δ+ > {DRUG_EFFECT_THRESHOLD} day⁻¹ (drug-responsive / AT-eligible)"
    )
    print("Zhang vs MTD:          reported only — does NOT remove patients from CVaR cohort")
    print(f"Trial benchmark TTP:   {TRIAL_MEDIAN_TTP_DAYS:.0f} days (Cunningham adaptive arm)")
    print(f"Simulated median TTP:  {sim_median:.0f} days")
    print(f"\nWasserstein epsilon:   {w.epsilon:.4f}")
    print(
        f"Drug-eligible (W):     {len(out_w['feasible_indices'])} / {n} ({pct_elig:.1f}%)"
    )
    print(
        f"Drug-eligible (ell.):  {len(out_e['feasible_indices'])} / {n} "
        f"({100 * out_e['feasible_fraction']:.1f}%)"
    )
    print(f"\n--- Zhang vs MTD (informational, same Wasserstein draw) ---")
    print(f"  Zhang TTP > MTD TTP:  {n_w} / {n} ({100 * n_w / n:.1f}%)")
    print(f"  Zhang TTP ≤ MTD TTP: {n_l} / {n} ({100 * n_l / n:.1f}%)  ← key subgroup for robust schedules")
    print(f"  Mean TTP gap (Zhang > MTD):   {out_w['mean_ttp_gap_zhang_gt_mtd']:.1f} days")
    print(f"  Mean TTP gap (Zhang ≤ MTD): {out_w['mean_ttp_gap_zhang_le_mtd']:.1f} days")
    if out_w["feasible_indices"]:
        print(f"  Median Zhang TTP (drug-eligible): {out_w['median_zhang_ttp_feasible']:.0f} days")

    print("\n[Gate check] Script 08 should only run if:")
    print(f"  (1) Simulated median TTP in [798, 1198] d: {sim_median:.0f} — {'PASS' if sim_ok else 'FAIL'}")
    print(f"  (2) Drug-eligible fraction ≥ 80%: {pct_elig:.1f}% — {'PASS' if feas_ok else 'FAIL'}")
    if sim_ok and feas_ok:
        print("\n  READY FOR SCRIPT 08.")
    else:
        print("\n  DO NOT PROCEED. Fix calibration or eligibility threshold first.")


if __name__ == "__main__":
    main()
