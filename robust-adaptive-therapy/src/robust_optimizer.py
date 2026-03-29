"""
Phase 3 — Wrappers for CVaR grid / subgradient robust optimisation.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cvar_optimizer import CVaROptimizer
from .feasibility import _params_to_model
from .policy import MTDPolicy, RangeBoundedPolicy, ThresholdPolicy


class RobustThresholdOptimizer:
    """Thin wrapper around CVaROptimizer.optimize_grid() with protocol comparison."""

    def __init__(self, cvar_optimizer: CVaROptimizer) -> None:
        self.opt = cvar_optimizer

    def full_comparison(self, protocols: dict[str, Any]) -> pd.DataFrame:
        scen = self.opt.fixed_scenarios
        if scen is None:
            scen = self.opt.sample_feasible_scenarios(seed=123)
        rows: list[dict] = []
        for name, pol in protocols.items():
            ttps: list[float] = []
            offs: list[float] = []
            for p in scen:
                m = _params_to_model(self.opt.lv_model_class, p)
                psa_m = self.opt.psa_model_class()
                tm = float(self.opt.t_max_sim)
                if isinstance(pol, ThresholdPolicy):
                    out = pol.simulate_patient(m, psa_m, t_max=tm, check_interval=1)
                elif isinstance(pol, RangeBoundedPolicy):
                    out = pol.simulate_patient(m, psa_m, t_max=tm, check_interval=1)
                elif isinstance(pol, MTDPolicy):
                    out = pol.simulate_patient(m, psa_m, t_max=tm)
                else:
                    raise TypeError(type(pol))
                ttp = float(out["ttp"]) if np.isfinite(out["ttp"]) else 1e6
                ttps.append(ttp)
                offs.append(float(out["fraction_off_treatment"]))
            arr = np.sort(np.asarray(ttps, dtype=float))
            n = len(arr)
            al = self.opt.alpha
            k = max(1, int(np.ceil(al * n)))
            rows.append({
                "protocol": name,
                "cvar_ttp": float(np.mean(arr[:k])),
                "median_ttp": float(np.median(arr)),
                "p5_ttp": float(np.percentile(arr, 5)),
                "frac_off_treatment": float(np.mean(offs)),
            })
        return pd.DataFrame(rows)


class SubgradientRobustOptimizer:
    """Runs CVaROptimizer.optimize_gradient with diagnostics and plots."""

    def __init__(self, cvar_optimizer: CVaROptimizer) -> None:
        self.opt = cvar_optimizer

    def run_with_diagnostics(
        self,
        n_runs: int = 3,
        n_iterations: int = 150,
        step_size: float = 0.015,
        inits: list[tuple[float, float]] | None = None,
        seed0: int = 0,
    ) -> dict:
        runs = []
        best = None
        if inits is None:
            rng = np.random.default_rng(seed0)
            inits = [(0.4 + 0.1 * rng.random(), 0.9 + 0.2 * rng.random()) for _ in range(n_runs)]
        for r in range(n_runs):
            res = self.opt.optimize_gradient(
                n_iterations=n_iterations,
                step_size=step_size,
                init_phi=inits[r] if r < len(inits) else None,
                seed=seed0 + r * 9973,
                tqdm_disable=True,
            )
            runs.append(res)
            if best is None or res["final_cvar"] > best["final_cvar"]:
                best = res
        return {"runs": runs, "best": best}

    def plot_convergence(
        self,
        runs: list[dict],
        save_path: str | None = None,
        grid_cvar_line: float | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, run in enumerate(runs):
            h = run["convergence_history"]
            xs = [x[0] for x in h]
            ys = [x[1] for x in h]
            ax.plot(xs, ys, alpha=0.8, label=f"run {i+1}")
        if grid_cvar_line is not None:
            ax.axhline(grid_cvar_line, color="k", ls="--", lw=1, label="grid start CVaR")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("CVaR TTP (days)")
        ax.set_title("Subgradient ascent on CVaR")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        return fig
