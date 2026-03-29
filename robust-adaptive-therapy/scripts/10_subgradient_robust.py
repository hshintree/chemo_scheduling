#!/usr/bin/env python
"""
SCRIPT 10: Subgradient Ascent on CVaR Objective
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
from pathlib import Path

from src.cvar_optimizer import CVaROptimizer
from src.feasibility import FeasibilityClassifier
from src.lotka_volterra import LotkaVolterraModel
from src.psa_model import PSAModel
from src.robust_optimizer import SubgradientRobustOptimizer
from src.uncertainty import WassersteinUncertaintySet

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "subgradient"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pkl = PROC / "grid_search_results.pkl"
    if not pkl.exists():
        print("Run scripts/08_robust_threshold_search.py first.")
        sys.exit(1)
    with open(pkl, "rb") as f:
        grid = pickle.load(f)

    scenarios = grid.get("scenarios_used")
    if scenarios is None:
        print("grid_search_results.pkl missing scenarios_used; re-run script 08.")
        sys.exit(1)

    ell_w = grid.get("weighted_ellipsoid")
    if ell_w is None:
        import pandas as pd
        from src.population_model import PopulationModel

        df = pd.read_csv(ROOT / "data" / "synthetic" / "population_params.csv")
        X_log = np.log(df[PopulationModel.param_names].astype(float).values)
        ell_w = WassersteinUncertaintySet(
            X_log, epsilon=0.1, param_names=PopulationModel.param_names,
        )
        ell_w.epsilon = ell_w.compute_epsilon_from_confidence(0.90)

    clf = FeasibilityClassifier(LotkaVolterraModel, PSAModel)
    t_max = float(grid.get("t_max_sim", 1825.0))
    opt = CVaROptimizer(
        ell_w,
        clf,
        alpha=0.20,
        n_scenarios=len(scenarios),
        fixed_scenarios=scenarios,
        t_max_sim=t_max,
    )
    sg = SubgradientRobustOptimizer(opt)

    init = (float(grid["optimal_stop"]), float(grid["optimal_restart"]))
    grid_cvar = float(grid["optimal_cvar"])

    runs = []
    for r, phi0 in enumerate([init, (0.35, 1.1), (0.55, 0.95)]):
        res = opt.optimize_gradient(
            n_iterations=150,
            step_size=0.015,
            init_phi=phi0,
            seed=4242 + r,
            tqdm_disable=True,
        )
        runs.append(res)

    fig = sg.plot_convergence(runs, save_path=FIG / "convergence.png", grid_cvar_line=grid_cvar)
    plt.close(fig)

    df_grid = grid["grid_results"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df_grid["restart"], df_grid["stop"],
        c=df_grid["cvar_ttp"], cmap="viridis", s=50, alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="CVaR TTP")
    ax.scatter([init[1]], [init[0]], c="red", s=80, label="start", zorder=5)
    best = max(runs, key=lambda x: x["final_cvar"])
    ax.scatter(
        [best["optimal_phi"][1]], [best["optimal_phi"][0]],
        marker="*", s=220, c="lime", edgecolors="k", label="subgradient end", zorder=6,
    )
    ax.scatter([1.0], [0.5], marker="x", s=100, c="black", label="Zhang", zorder=6)
    ax.set_xlabel("Restart")
    ax.set_ylabel("Stop")
    ax.legend(fontsize=8)
    ax.set_title("Policy trajectories over CVaR landscape")
    fig.tight_layout()
    fig.savefig(FIG / "policy_evolution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    s_grid, r_grid = init
    cvar_grid = grid_cvar
    s_sg, r_sg = best["optimal_phi"]
    cvar_sg = best["final_cvar"]
    delta = cvar_sg - cvar_grid
    pct = 100 * delta / max(abs(cvar_grid), 1e-6)
    print(f"Grid search optimum:     stop={s_grid:.3f}, restart={r_grid:.3f}, CVaR={cvar_grid:.1f}d")
    print(f"Subgradient refinement:  stop={s_sg:.3f},   restart={r_sg:.3f},   CVaR={cvar_sg:.1f}d")
    print(f"Improvement from gradient refinement: {delta:+.1f} days CVaR TTP ({pct:+.1f}%)")

    with open(PROC / "subgradient_results.pkl", "wb") as f:
        pickle.dump({"runs": runs, "best": best, "grid_cvar": grid_cvar}, f)


if __name__ == "__main__":
    main()
