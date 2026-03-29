#!/usr/bin/env python
"""
SCRIPT 08: CVaR-Robust Threshold Grid Search

Optimizes threshold-policy (stop/restart) parameters via CVaR over a **fixed subset
of drug-eligible scenarios** from script 07: patients with ``delta_plus`` above the
eligibility floor (abiraterone has meaningful effect). Patients where Zhang TTP ≤ MTD
remain in this set — robust thresholds can target schedules that beat both.

The ``WassersteinUncertaintySet`` matches cohort ε from script 07; **CVaR and policy
comparisons use ``fixed_scenarios``** from ``feasible_scenarios.npz``. The uncertainty
object is only required for the optimizer API when fresh sampling is enabled.
"""

from __future__ import annotations

import argparse
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

from src.cvar_optimizer import CVaROptimizer
from src.feasibility import FeasibilityClassifier, _params_to_model
from src.lotka_volterra import LotkaVolterraModel
from src.policy import MTDPolicy, ThresholdPolicy
from src.population_model import PopulationModel
from src.psa_model import PSAModel
from src.robust_optimizer import RobustThresholdOptimizer
from src.uncertainty import WassersteinUncertaintySet

ROOT = Path(__file__).resolve().parent.parent
DATA_SYN = ROOT / "data" / "synthetic"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "robust_opt"
PKL_PATH = PROC / "grid_search_results.pkl"
T_MAX_DAYS = 1825.0
CVAR_ALPHA = 0.20


def _feasibility_classifier_from_npz(z: np.lib.npyio.NpzFile) -> FeasibilityClassifier:
    t_max = float(z["t_max_days"][0]) if "t_max_days" in z.files else 1825.0
    return FeasibilityClassifier(LotkaVolterraModel, PSAModel, t_max=t_max)


def _plots_and_tables(
    grid_out: dict,
    comp: pd.DataFrame,
    scenarios: list,
) -> None:
    df_grid = grid_out["grid_results"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df_grid["restart"], df_grid["stop"],
        c=df_grid["cvar_ttp"], cmap="viridis", s=80, marker="s",
    )
    plt.colorbar(sc, ax=ax, label="CVaR TTP (days)")
    ax.scatter([1.0], [0.5], marker="x", s=120, c="red", label="Zhang (0.5, 1.0)", zorder=6)
    ax.scatter(
        [grid_out["optimal_restart"]], [grid_out["optimal_stop"]],
        marker="*", s=200, c="gold", edgecolors="k", label="CVaR-robust", zorder=7,
    )
    ax.scatter([0.8], [0.3], marker="^", s=100, c="cyan", edgecolors="k", label="RBAT (0.3, 0.8)", zorder=6)
    ax.set_xlabel("Restart fraction")
    ax.set_ylabel("Stop fraction")
    ax.set_title("CVaR TTP heatmap (stop vs restart)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cvar_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc2 = ax.scatter(
        df_grid["median_ttp"], df_grid["cvar_ttp"],
        c=df_grid["stop"], cmap="coolwarm", s=70, alpha=0.85,
    )
    plt.colorbar(sc2, ax=ax, label="stop fraction")
    zrow = df_grid[
        ((df_grid["stop"] - 0.5).abs() < 0.02)
        & ((df_grid["restart"] - 1.0).abs() < 0.02)
    ]
    if len(zrow):
        ax.scatter(
            [zrow["median_ttp"].values[0]], [zrow["cvar_ttp"].values[0]],
            s=140, facecolors="none", edgecolors="red", linewidths=2, label="Zhang",
        )
    ax.set_xlabel("Median TTP (days)")
    ax.set_ylabel("CVaR TTP (days)")
    ax.set_title("Robustness–performance tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "median_vs_cvar_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nProtocol               | Median TTP | CVaR TTP | P5 TTP | % Off Treatment")
    print("---------------------- | ---------- | -------- | ------ | ---------------")
    for _, row in comp.iterrows():
        print(
            f"{row['protocol']:<22} | {row['median_ttp']:.1f}      | {row['cvar_ttp']:.1f}    | "
            f"{row['p5_ttp']:.1f}  | {100*row['frac_off_treatment']:.1f}%"
        )

    oracle_med = []
    if scenarios:
        for p in scenarios[: min(20, len(scenarios))]:
            m = _params_to_model(LotkaVolterraModel, p)
            best_ttp = 0.0
            for s in np.linspace(0.2, 0.7, 6):
                for r in np.linspace(0.6, 1.2, 7):
                    if r <= s:
                        continue
                    out = ThresholdPolicy(s, r).simulate_patient(m, PSAModel(), t_max=T_MAX_DAYS, check_interval=1)
                    ttp = float(out["ttp"]) if np.isfinite(out["ttp"]) else 1e6
                    if ttp > best_ttp:
                        best_ttp = ttp
            oracle_med.append(best_ttp)
    if oracle_med:
        om = float(np.median(oracle_med))
        ocv = float(np.mean(np.sort(oracle_med)[: max(1, int(np.ceil(0.2 * len(oracle_med))))]))
        print(f"{'Oracle (subset)':<22} | {om:.1f}      | {ocv:.1f}    | —      | —")

    print(f"\nSaved figures to {FIG}/\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="7x7 grid, 50 scenarios (~5 min)")
    parser.add_argument("--full", action="store_true", help="13x13 grid, 200 scenarios")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Skip grid search, reload saved results and regenerate plots only",
    )
    args = parser.parse_args()
    n_grid = 7 if args.fast else 13
    n_scenarios = 50 if args.fast else 200

    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    if args.rerun and PKL_PATH.exists():
        with open(PKL_PATH, "rb") as f:
            saved = pickle.load(f)
        grid_out = {
            "grid_results": saved["grid_results"],
            "optimal_stop": saved["optimal_stop"],
            "optimal_restart": saved["optimal_restart"],
            "optimal_cvar": saved["optimal_cvar"],
            "zhang_cvar": saved["zhang_cvar"],
        }
        comp = saved["comparison"]
        scenarios = list(saved.get("scenarios_used", []))
        print("\n=== RERUN: loaded grid_search_results.pkl ===")
        print(f"  scenario_source: {saved.get('scenario_source', 'unknown')}")
        print(f"  t_max_sim (saved): {saved.get('t_max_sim', 'unknown')}")
        print(f"  scenarios in file: {len(scenarios)}")
        _plots_and_tables(grid_out, comp, scenarios)
        return

    npz_path = PROC / "feasible_scenarios.npz"
    if not npz_path.exists():
        print("Run scripts/07_feasibility_analysis.py first.")
        sys.exit(1)
    z = np.load(npz_path, allow_pickle=True)
    names = list(z["param_names"].tolist())
    w_mat = z["samples_w"]
    idx_feas = z["idx_w_feasible"]
    feas_rows = w_mat[idx_feas]
    n_avail = len(feas_rows)
    param_list = []
    for i in range(n_avail):
        d = dict(zip(names, feas_rows[i].tolist()))
        d["K"] = 10_000.0
        d["delta_prod"] = 0.5 * float(d["delta_plus"])
        d["alpha"] = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]], dtype=float)
        param_list.append(d)
    if len(param_list) > n_scenarios:
        rng = np.random.default_rng(123)
        pick = rng.choice(len(param_list), size=n_scenarios, replace=False)
        scenarios = [param_list[i] for i in pick]
    else:
        scenarios = param_list
        print(f"Warning: only {len(scenarios)} feasible scenarios available.")

    df = pd.read_csv(DATA_SYN / "population_params.csv")
    records = df[PopulationModel.param_names].to_dict(orient="records")
    pop = PopulationModel()
    pop.fit(records)
    X_log = np.log(df[PopulationModel.param_names].astype(float).values)
    # Structural match to script 07; grid objective uses fixed_scenarios only.
    w_uq = WassersteinUncertaintySet(X_log, epsilon=0.1, param_names=PopulationModel.param_names)
    w_uq.epsilon = w_uq.compute_epsilon_from_confidence(0.90)

    clf = _feasibility_classifier_from_npz(z)
    opt = CVaROptimizer(
        w_uq,
        clf,
        alpha=CVAR_ALPHA,
        n_scenarios=len(scenarios),
        fixed_scenarios=scenarios,
        t_max_sim=T_MAX_DAYS,
    )

    print("\n=== ROBUST THRESHOLD SEARCH (drug-eligible Wasserstein scenarios) ===\n")
    print(f"  scenario_source: drug-eligible subset from script 07 npz (δ+ > threshold)")
    print(f"  {opt.scenario_source_description}")
    print(f"  drug-eligible scenarios available: {n_avail}")
    print(f"  drug-eligible scenarios used in this run: {len(scenarios)}")
    print(f"  simulation horizon t_max_sim: {T_MAX_DAYS} days")
    print(f"  CVaR alpha: {CVAR_ALPHA}")
    print(f"  Grid uses same scenario draw for all cells: yes (fresh_scenarios_per_cell=False)")
    t_meta = float(z["t_max_days"][0]) if "t_max_days" in z.files else T_MAX_DAYS
    print(f"  t_max_days (from feasible_scenarios.npz): {t_meta:.0f}")
    print(
        "\n  Note: WassersteinUncertaintySet matches cohort ε for API consistency; "
        "CVaR and policy metrics are evaluated on the fixed scenario list above.\n"
    )

    stop_grid = np.linspace(0.20, 0.80, n_grid)
    restart_grid = np.linspace(0.60, 1.50, n_grid)
    grid_out = opt.optimize_grid(stop_grid=stop_grid, restart_grid=restart_grid, tqdm_disable=False)

    zhang = ThresholdPolicy(0.5, 1.0)
    rbat = ThresholdPolicy(0.3, 0.8)
    mtd = MTDPolicy()
    rob = RobustThresholdOptimizer(opt)
    comp = rob.full_comparison({
        "MTD": mtd,
        "Zhang": zhang,
        "RBAT_thr": rbat,
        "CVaR_opt": ThresholdPolicy(grid_out["optimal_stop"], grid_out["optimal_restart"]),
    })

    scenario_source = "drug-eligible Wasserstein subset (script 07 idx_w_feasible)"
    with open(PKL_PATH, "wb") as f:
        pickle.dump({
            **{k: v for k, v in grid_out.items() if k != "scenarios_used"},
            "comparison": comp,
            "scenarios_used": grid_out["scenarios_used"],
            "scenario_source": scenario_source,
            "t_max_sim": float(T_MAX_DAYS),
            "cvar_alpha": CVAR_ALPHA,
            "wasserstein_epsilon": float(w_uq.epsilon),
        }, f)
    print(f"Saved {PKL_PATH}")

    _plots_and_tables(grid_out, comp, list(grid_out["scenarios_used"]))


if __name__ == "__main__":
    main()
