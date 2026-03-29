from __future__ import annotations

from typing import Type

import numpy as np
import pandas as pd

from .lotka_volterra import LotkaVolterraModel
from .policy import MTDPolicy, ThresholdPolicy
from .psa_model import PSAModel

T_MAX_FEASIBILITY = 1825.0  # 5-year horizon matches observed trial range

# day^-1; ~7× below published cohort mean delta_plus ≈ 0.15
DRUG_EFFECT_THRESHOLD = 0.02


def _params_to_model(
    lv_model_class: Type[LotkaVolterraModel],
    params: dict,
) -> LotkaVolterraModel:
    p = dict(params)
    if "alpha" not in p:
        keys = [f"alpha_{i}{j}" for i in range(3) for j in range(3)]
        if all(k in p for k in keys):
            return lv_model_class.from_dict(p)
    return lv_model_class(p)


class FeasibilityClassifier:
    """
    AT **eligibility** (for CVaR / robust cohort): abiraterone must have meaningful
    effect on sensitive cells (``delta_plus`` above a floor).

    Zhang vs MTD TTP is computed for **reporting only** — patients where Zhang loses
    to MTD remain eligible; optimized schedules may still outperform both.
    """

    def __init__(
        self,
        lv_model_class: type = LotkaVolterraModel,
        psa_model_class: type = PSAModel,
        *,
        t_max: float = T_MAX_FEASIBILITY,
        drug_effect_threshold: float = DRUG_EFFECT_THRESHOLD,
    ) -> None:
        self.lv_model_class = lv_model_class
        self.psa_model_class = psa_model_class
        self.t_max = float(t_max)
        self.drug_effect_threshold = float(drug_effect_threshold)

    def _simulate_both(self, params: dict) -> tuple[float, float]:
        """Returns (ttp_zhang, ttp_mtd) for reporting; not used for eligibility."""
        model = _params_to_model(self.lv_model_class, params)
        psa_m = self.psa_model_class()
        zhang_out = ThresholdPolicy(0.5, 1.0).simulate_patient(
            model, psa_m, t_max=self.t_max, check_interval=1)
        mtd_out = MTDPolicy().simulate_patient(
            model, psa_m, t_max=self.t_max)
        ttp_z = float(zhang_out["ttp"]) if np.isfinite(zhang_out["ttp"]) else self.t_max
        ttp_m = float(mtd_out["ttp"]) if np.isfinite(mtd_out["ttp"]) else self.t_max
        return ttp_z, ttp_m

    def is_feasible(self, params: dict) -> bool:
        """
        True if this patient is drug-responsive: ``delta_plus`` exceeds the floor.

        Patients where Zhang TTP ≤ MTD TTP **remain** eligible — a better schedule
        may still help them.
        """
        return float(params["delta_plus"]) > self.drug_effect_threshold

    def ttp_zhang(self, params: dict) -> float:
        ttp_z, _ = self._simulate_both(params)
        return ttp_z

    def classify_cohort(self, param_list: list[dict]) -> dict:
        feasible_idx: list[int] = []
        infeasible_idx: list[int] = []
        rows: list[dict] = []
        gaps_zhang_wins: list[float] = []
        gaps_zhang_loses: list[float] = []

        for i, p in enumerate(param_list):
            ttp_z, ttp_m = self._simulate_both(p)
            gap = ttp_z - ttp_m
            zhang_beats = ttp_z > ttp_m
            if zhang_beats:
                gaps_zhang_wins.append(gap)
            else:
                gaps_zhang_loses.append(gap)

            drug_ok = self.is_feasible(p)
            if drug_ok:
                feasible_idx.append(i)
            else:
                infeasible_idx.append(i)

            rows.append({
                "r_minus": float(p["r_minus"]),
                "delta_plus": float(p["delta_plus"]),
                "T0_minus": float(p["T0_minus"]),
                "ratio_rm_dp": float(p["r_minus"]) / max(float(p["delta_plus"]), 1e-12),
                "ttp_zhang": ttp_z,
                "ttp_mtd": ttp_m,
                "ttp_gap": gap,
                "zhang_beats_mtd": zhang_beats,
                "feasible": drug_ok,
            })

        n = len(param_list)
        elig_ttps = [rows[i]["ttp_zhang"] for i in feasible_idx]

        return {
            "feasible_indices": feasible_idx,
            "infeasible_indices": infeasible_idx,
            "feasible_fraction": len(feasible_idx) / n if n else 0.0,
            "mean_ttp_gap_zhang_gt_mtd": float(np.mean(gaps_zhang_wins)) if gaps_zhang_wins else 0.0,
            "mean_ttp_gap_zhang_le_mtd": float(np.mean(gaps_zhang_loses)) if gaps_zhang_loses else 0.0,
            "median_zhang_ttp_feasible": float(np.median(elig_ttps)) if elig_ttps else 0.0,
            "n_zhang_beats_mtd": sum(1 for r in rows if r["zhang_beats_mtd"]),
            "n_zhang_loses_mtd": sum(1 for r in rows if not r["zhang_beats_mtd"]),
            "boundary_features": pd.DataFrame(rows),
        }
