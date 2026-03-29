"""
Phase 3 — CVaR-based robust optimisation over threshold policies.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .feasibility import _params_to_model
from .lotka_volterra import LotkaVolterraModel
from .policy import ThresholdPolicy
from .psa_model import PSAModel
from .uncertainty import EllipsoidalUncertaintySet, WassersteinUncertaintySet

log = logging.getLogger(__name__)


def _row_to_param_dict(row: np.ndarray, names: list[str]) -> dict:
    d = dict(zip(names, row.tolist()))
    if "alpha" not in d:
        d["alpha"] = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ])
        np.fill_diagonal(d["alpha"], 1.0)
    if "delta_prod" not in d and "delta_plus" in d:
        d["delta_prod"] = 0.5 * float(d["delta_plus"])
    if "K" not in d:
        d["K"] = 10_000.0
    return d


def _sample_raw_uncertainty(
    uncertainty_set: Any,
    n: int,
    seed: int,
) -> list[dict]:
    names = list(getattr(uncertainty_set, "param_names", []))
    if not names:
        names = [
            "r_plus", "r_prod", "r_minus", "delta_plus",
            "T0_plus", "T0_prod", "T0_minus",
        ]
    rng = np.random.default_rng(seed)
    if isinstance(uncertainty_set, EllipsoidalUncertaintySet):
        mat = uncertainty_set.sample_interior(n, seed=int(rng.integers(1 << 30)))
    elif isinstance(uncertainty_set, WassersteinUncertaintySet):
        mat = uncertainty_set.sample_perturbed_params(n, seed=int(rng.integers(1 << 30)))
    else:
        raise TypeError(f"Unsupported uncertainty set type: {type(uncertainty_set)}")
    return [_row_to_param_dict(mat[i], names) for i in range(len(mat))]


class CVaROptimizer:
    """
    Maximise CVaR_alpha of TTP over threshold policies using scenario sampling.
    """

    def __init__(
        self,
        uncertainty_set: EllipsoidalUncertaintySet | WassersteinUncertaintySet,
        feasibility_classifier: Any,
        alpha: float = 0.20,
        n_scenarios: int = 200,
        oversample_factor: int = 5,
        fixed_scenarios: Optional[list[dict]] = None,
        lv_model_class: type = LotkaVolterraModel,
        psa_model_class: type = PSAModel,
        t_max_sim: float = 1825.0,
    ) -> None:
        self.uncertainty_set = uncertainty_set
        self.feas = feasibility_classifier
        self.alpha = float(alpha)
        self.n_scenarios = int(n_scenarios)
        self.oversample_factor = int(oversample_factor)
        self.fixed_scenarios = fixed_scenarios
        self.lv_model_class = lv_model_class
        self.psa_model_class = psa_model_class
        self.t_max_sim = float(t_max_sim)

    @property
    def scenario_source_description(self) -> str:
        """Human-readable provenance for logging (fixed list vs fresh sampling)."""
        if self.fixed_scenarios is not None:
            return "fixed feasible scenarios (caller-supplied)"
        return "freshly sampled feasible scenarios from uncertainty_set"

    def sample_feasible_scenarios(self, seed: int = 0) -> list[dict]:
        if self.fixed_scenarios is not None:
            fs = self.fixed_scenarios
            if len(fs) >= self.n_scenarios:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(fs), size=self.n_scenarios, replace=False)
                return [dict(fs[i]) for i in idx]
            log.warning(
                "fixed_scenarios has %d < n_scenarios=%d; using all available.",
                len(fs), self.n_scenarios,
            )
            return [dict(p) for p in fs]

        want = self.n_scenarios * self.oversample_factor
        collected: list[dict] = []
        trial = 0
        max_trials = 5
        while len(collected) < self.n_scenarios and trial < max_trials:
            raw = _sample_raw_uncertainty(
                self.uncertainty_set,
                n=want,
                seed=seed + 10_000 * trial,
            )
            for p in raw:
                if self.feas.is_feasible(p):
                    collected.append(p)
                    if len(collected) >= self.n_scenarios:
                        break
            trial += 1
            want = min(want * 2, 50_000)

        if len(collected) == 0:
            raise RuntimeError(
                "CVaROptimizer.sample_feasible_scenarios: zero feasible scenarios after sampling. "
                "Increase oversample_factor, relax feasibility settings, or pass fixed_scenarios."
            )
        if len(collected) < self.n_scenarios:
            log.warning(
                "Only %d feasible scenarios after sampling (requested %d); returning all collected "
                "without changing n_scenarios.",
                len(collected),
                self.n_scenarios,
            )
        return collected[: self.n_scenarios]

    def compute_cvar(self, phi: tuple[float, float], scenarios: list[dict]) -> float:
        stop, restart = float(phi[0]), float(phi[1])
        pol = ThresholdPolicy(stop, restart)
        psa_m = self.psa_model_class()
        ttps: list[float] = []
        for p in scenarios:
            m = _params_to_model(self.lv_model_class, p)
            out = pol.simulate_patient(m, psa_m, t_max=self.t_max_sim, check_interval=1)
            ttps.append(float(out["ttp"]) if np.isfinite(out["ttp"]) else 1e6)
        arr = np.sort(np.asarray(ttps, dtype=float))
        n = len(arr)
        if n == 0:
            return float("nan")
        k = max(1, int(np.ceil(self.alpha * n)))
        return float(np.mean(arr[:k]))

    def _ttp_bundle(
        self,
        phi: tuple[float, float],
        scenarios: list[dict],
    ) -> tuple[np.ndarray, float, float, float, float]:
        stop, restart = float(phi[0]), float(phi[1])
        pol = ThresholdPolicy(stop, restart)
        psa_m = self.psa_model_class()
        ttps: list[float] = []
        off_fracs: list[float] = []
        for p in scenarios:
            m = _params_to_model(self.lv_model_class, p)
            out = pol.simulate_patient(m, psa_m, t_max=self.t_max_sim, check_interval=1)
            ttp = float(out["ttp"]) if np.isfinite(out["ttp"]) else 1e6
            ttps.append(ttp)
            off_fracs.append(float(out["fraction_off_treatment"]))
        arr = np.sort(np.asarray(ttps, dtype=float))
        n = len(arr)
        k = max(1, int(np.ceil(self.alpha * n)))
        cvar = float(np.mean(arr[:k]))
        med = float(np.median(arr))
        p5 = float(np.percentile(arr, 5))
        frac_off = float(np.mean(off_fracs)) if off_fracs else 0.0
        return arr, cvar, med, p5, frac_off

    def compute_cvar_gradient(
        self,
        phi: tuple[float, float],
        scenarios: list[dict],
        h: float = 0.01,
    ) -> np.ndarray:
        g = np.zeros(2, dtype=float)
        for _ in range(3):
            rng = np.random.default_rng()
            if self.fixed_scenarios is None:
                scen = self.sample_feasible_scenarios(seed=int(rng.integers(1 << 30)))
            else:
                idx = rng.choice(len(self.fixed_scenarios), size=min(len(self.fixed_scenarios), self.n_scenarios), replace=False)
                scen = [dict(self.fixed_scenarios[i]) for i in idx]
            e0 = np.array([h, 0.0])
            e1 = np.array([0.0, h])
            g[0] += (
                self.compute_cvar((phi[0] + e0[0], phi[1]), scen)
                - self.compute_cvar((phi[0] - e0[0], phi[1]), scen)
            ) / (2 * h)
            g[1] += (
                self.compute_cvar((phi[0], phi[1] + e1[1]), scen)
                - self.compute_cvar((phi[0], phi[1] - e1[1]), scen)
            ) / (2 * h)
        return g / 3.0

    def optimize_grid(
        self,
        stop_grid: Optional[np.ndarray] = None,
        restart_grid: Optional[np.ndarray] = None,
        tqdm_disable: bool = False,
        fresh_scenarios_per_cell: bool = False,
    ) -> dict:
        if stop_grid is None:
            stop_grid = np.linspace(0.20, 0.80, 13)
        if restart_grid is None:
            restart_grid = np.linspace(0.60, 1.50, 13)
        rows: list[dict] = []
        best_cvar = -np.inf
        best_phi = (0.5, 1.0)

        scenarios0 = self.sample_feasible_scenarios(seed=42)

        total = len(stop_grid) * len(restart_grid)
        it = tqdm(
            ((s, r) for s in stop_grid for r in restart_grid),
            total=total,
            disable=tqdm_disable,
            desc="CVaR grid",
        )
        for stop, restart in it:
            if restart <= stop:
                continue
            scen = self.sample_feasible_scenarios(seed=hash((stop, restart)) % (2**31)) if fresh_scenarios_per_cell else scenarios0
            _, cvar, med, p5, fo = self._ttp_bundle((float(stop), float(restart)), scen)
            rows.append({
                "stop": float(stop),
                "restart": float(restart),
                "cvar_ttp": cvar,
                "median_ttp": med,
                "p5_ttp": p5,
                "frac_off_treatment": fo,
            })
            if cvar > best_cvar:
                best_cvar = cvar
                best_phi = (float(stop), float(restart))

        zhang_cvar = self.compute_cvar((0.5, 1.0), scenarios0)
        return {
            "grid_results": pd.DataFrame(rows),
            "optimal_stop": best_phi[0],
            "optimal_restart": best_phi[1],
            "optimal_cvar": float(best_cvar),
            "zhang_cvar": float(zhang_cvar),
            "scenarios_used": scenarios0,
        }

    def optimize_gradient(
        self,
        n_iterations: int = 150,
        step_size: float = 0.015,
        init_phi: Optional[tuple[float, float]] = None,
        seed: int = 0,
        tqdm_disable: bool = False,
    ) -> dict:
        rng = np.random.default_rng(seed)
        phi = np.array(
            init_phi if init_phi is not None else (0.45, 1.05),
            dtype=float,
        )
        best_phi = phi.copy()
        best_cvar = -np.inf
        hist: list[tuple[int, float, float, float]] = []
        avg_phi = np.zeros(2, dtype=float)
        scen = self.sample_feasible_scenarios(seed=seed)

        it = range(n_iterations)
        if not tqdm_disable:
            it = tqdm(it, desc="CVaR grad ascent", ncols=70)

        for itn in it:
            if itn % 10 == 0:
                scen = self.sample_feasible_scenarios(seed=seed + itn)
            g = self.compute_cvar_gradient((float(phi[0]), float(phi[1])), scen)
            phi = phi + step_size * g
            phi[0] = np.clip(phi[0], 0.15, 0.85)
            phi[1] = np.clip(phi[1], 0.50, 1.60)
            cv = self.compute_cvar((float(phi[0]), float(phi[1])), scen)
            if cv > best_cvar:
                best_cvar = cv
                best_phi = phi.copy()
            avg_phi += phi
            hist.append((itn, cv, float(phi[0]), float(phi[1])))

        avg_phi /= max(n_iterations, 1)
        return {
            "optimal_phi": (float(best_phi[0]), float(best_phi[1])),
            "averaged_phi": (float(avg_phi[0]), float(avg_phi[1])),
            "convergence_history": hist,
            "final_cvar": float(best_cvar),
        }
