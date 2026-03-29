"""
Phase 3 — Treatment policy parameterisations (threshold, range-bounded, MTD).
"""

from __future__ import annotations

import numpy as np

from .lotka_volterra import LotkaVolterraModel
from .psa_model import PSAModel

# Match scripts/01 and 03 progression rules
PSA_PROG_FACTOR = 2.0
RESISTANT_CUTOFF = 0.80
PROGRESSION_MIN_DAY = 30


def _baseline_psa(model: LotkaVolterraModel, psa_model: PSAModel) -> float:
    return float(
        psa_model.compute_psa({
            "T_plus":  np.array([model.T0_plus]),
            "T_prod":  np.array([model.T0_prod]),
            "T_minus": np.array([model.T0_minus]),
        })[0]
    )


def _ttp_and_end_state(
    day_arr: list[int],
    psa_arr: list[float],
    on_arr: list[float],
    y_final: np.ndarray,
    ttp: float,
) -> dict:
    on_np = np.asarray(on_arr, dtype=float)
    drug_days = float(np.sum(on_np))
    n = len(day_arr)
    frac_off = float(np.mean(on_np < 0.5)) if n else 0.0
    Tp, Tpr, Tm = y_final
    total = Tp + Tpr + Tm
    final_tm = float(Tm / total) if total > 0 else 0.0
    return {
        "t": np.asarray(day_arr, dtype=float),
        "T_plus": None,
        "T_prod": None,
        "T_minus": None,
        "psa_normalized": np.asarray(psa_arr, dtype=float),
        "on_treatment": on_np,
        "ttp": float(ttp),
        "cumulative_dose_days": drug_days,
        "fraction_off_treatment": frac_off,
        "final_T_minus_frac": final_tm,
        "y_final": y_final.copy(),
    }


class ThresholdPolicy:
    """Binary on/off policy parameterized by (stop_fraction, restart_fraction)."""

    def __init__(self, stop_fraction: float = 0.5, restart_fraction: float = 1.0) -> None:
        self.stop_fraction = float(stop_fraction)
        self.restart_fraction = float(restart_fraction)

    def get_drug_decision(
        self,
        psa_current: float,
        psa_baseline: float,
        currently_on_treatment: bool,
    ) -> int:
        if currently_on_treatment:
            return 0 if psa_current < self.stop_fraction * psa_baseline else 1
        return 1 if psa_current >= self.restart_fraction * psa_baseline else 0

    def simulate_patient(
        self,
        lv_model: LotkaVolterraModel,
        psa_model: PSAModel,
        t_max: float = 1825.0,
        check_interval: int = 28,
    ) -> dict:
        baseline = _baseline_psa(lv_model, psa_model)
        on_tx = True
        y = np.array([lv_model.T0_plus, lv_model.T0_prod, lv_model.T0_minus], dtype=float)
        day_arr: list[int] = []
        psa_arr: list[float] = []
        on_arr: list[float] = []
        ttp = float(np.inf)
        max_day = int(t_max)

        for day in range(max_day + 1):
            Tp, Tpr, Tm = y
            psa = float(
                psa_model.compute_psa({
                    "T_plus":  np.array([Tp]),
                    "T_prod":  np.array([Tpr]),
                    "T_minus": np.array([Tm]),
                })[0]
            )
            psa_norm = psa / baseline if baseline > 0 else 0.0

            if day % check_interval == 0:
                on_tx = self.get_drug_decision(psa, baseline, bool(on_tx)) == 1

            day_arr.append(day)
            psa_arr.append(psa_norm)
            on_arr.append(1.0 if on_tx else 0.0)

            total = Tp + Tpr + Tm
            rf = Tm / total if total > 0 else 0.0
            if (psa_norm >= PSA_PROG_FACTOR and on_tx) or (
                rf >= RESISTANT_CUTOFF and day > PROGRESSION_MIN_DAY
            ):
                ttp = float(day)
                out = _ttp_and_end_state(day_arr, psa_arr, on_arr, y, ttp)
                out["T_plus"] = np.array([Tp])
                out["T_prod"] = np.array([Tpr])
                out["T_minus"] = np.array([Tm])
                return out

            if day < max_day:
                drug_fn = (lambda u: (lambda t: u))(1.0 if on_tx else 0.0)
                sub = lv_model.simulate(
                    t_span=(float(day), float(day + 1)),
                    t_eval=np.array([float(day + 1)]),
                    drug_schedule=drug_fn,
                    y0=y,
                    rtol=1e-4,
                    atol=1e-1,
                )
                y = np.array(
                    [sub["T_plus"][-1], sub["T_prod"][-1], sub["T_minus"][-1]],
                    dtype=float,
                )

        out = _ttp_and_end_state(day_arr, psa_arr, on_arr, y, ttp)
        Tp, Tpr, Tm = y
        out["T_plus"] = np.array([Tp])
        out["T_prod"] = np.array([Tpr])
        out["T_minus"] = np.array([Tm])
        return out


class RangeBoundedPolicy:
    """
    Continuous dose modulation to keep PSA in [lo, hi] × baseline (RBAT-style).
    """

    def __init__(self, lo_fraction: float = 0.3, hi_fraction: float = 0.8) -> None:
        self.lo = float(lo_fraction)
        self.hi = float(hi_fraction)

    def get_dose(self, psa_current: float, psa_baseline: float) -> float:
        if psa_baseline <= 0:
            return 0.0
        psa_norm = psa_current / psa_baseline
        if psa_norm <= self.lo:
            return 0.0
        if psa_norm >= self.hi:
            return 1.0
        return (psa_norm - self.lo) / (self.hi - self.lo)

    def simulate_patient(
        self,
        lv_model: LotkaVolterraModel,
        psa_model: PSAModel,
        t_max: float = 1825.0,
        check_interval: int = 28,
    ) -> dict:
        baseline = _baseline_psa(lv_model, psa_model)
        y = np.array([lv_model.T0_plus, lv_model.T0_prod, lv_model.T0_minus], dtype=float)
        day_arr: list[int] = []
        psa_arr: list[float] = []
        dose_arr: list[float] = []
        ttp = float(np.inf)
        max_day = int(t_max)
        u_hold = 0.0

        for day in range(max_day + 1):
            Tp, Tpr, Tm = y
            psa = float(
                psa_model.compute_psa({
                    "T_plus":  np.array([Tp]),
                    "T_prod":  np.array([Tpr]),
                    "T_minus": np.array([Tm]),
                })[0]
            )
            psa_norm = psa / baseline if baseline > 0 else 0.0

            if day % check_interval == 0:
                u_hold = self.get_dose(psa, baseline)

            day_arr.append(day)
            psa_arr.append(psa_norm)
            dose_arr.append(u_hold)

            total = Tp + Tpr + Tm
            rf = Tm / total if total > 0 else 0.0
            if (psa_norm >= PSA_PROG_FACTOR and u_hold >= 0.5) or (
                rf >= RESISTANT_CUTOFF and day > PROGRESSION_MIN_DAY
            ):
                ttp = float(day)
                on_np = np.asarray(dose_arr, dtype=float)
                drug_days = float(np.sum(on_np))
                frac_off = float(np.mean(on_np < 0.5))
                final_tm = float(Tm / total) if total > 0 else 0.0
                return {
                    "t": np.asarray(day_arr, dtype=float),
                    "psa_normalized": np.asarray(psa_arr, dtype=float),
                    "on_treatment": on_np,
                    "ttp": ttp,
                    "cumulative_dose_days": drug_days,
                    "fraction_off_treatment": frac_off,
                    "final_T_minus_frac": final_tm,
                    "y_final": y.copy(),
                    "T_plus": np.array([Tp]),
                    "T_prod": np.array([Tpr]),
                    "T_minus": np.array([Tm]),
                }

            if day < max_day:
                u = u_hold
                drug_fn = (lambda uu: (lambda t: uu))(u)
                sub = lv_model.simulate(
                    t_span=(float(day), float(day + 1)),
                    t_eval=np.array([float(day + 1)]),
                    drug_schedule=drug_fn,
                    y0=y,
                    rtol=1e-4,
                    atol=1e-1,
                )
                y = np.array(
                    [sub["T_plus"][-1], sub["T_prod"][-1], sub["T_minus"][-1]],
                    dtype=float,
                )

        Tp, Tpr, Tm = y
        total = Tp + Tpr + Tm
        on_np = np.asarray(dose_arr, dtype=float)
        return {
            "t": np.asarray(day_arr, dtype=float),
            "psa_normalized": np.asarray(psa_arr, dtype=float),
            "on_treatment": on_np,
            "ttp": ttp,
            "cumulative_dose_days": float(np.sum(on_np)),
            "fraction_off_treatment": float(np.mean(on_np < 0.5)),
            "final_T_minus_frac": float(Tm / total) if total > 0 else 0.0,
            "y_final": y.copy(),
            "T_plus": np.array([Tp]),
            "T_prod": np.array([Tpr]),
            "T_minus": np.array([Tm]),
        }


class MTDPolicy:
    """Continuous maximum tolerated dose — drug = 1 every day."""

    def simulate_patient(
        self,
        lv_model: LotkaVolterraModel,
        psa_model: PSAModel,
        t_max: float = 1825.0,
    ) -> dict:
        t_eval = np.arange(0.0, float(t_max) + 1.0, 1.0)
        sim = lv_model.simulate((0.0, float(t_max)), t_eval, lambda t: 1, rtol=1e-4, atol=1e-1)
        baseline = _baseline_psa(lv_model, psa_model)
        psa_vals = psa_model.compute_psa(sim)
        psa_norm = psa_vals / baseline if baseline > 0 else psa_vals
        prog = psa_norm >= PSA_PROG_FACTOR
        if prog.any():
            cut = int(np.argmax(prog))
            ttp = float(sim["t"][cut])
            days = sim["t"][: cut + 1]
            pn = psa_norm[: cut + 1]
            on = np.ones(len(days))
            yf = np.array(
                [sim["T_plus"][cut], sim["T_prod"][cut], sim["T_minus"][cut]],
                dtype=float,
            )
        else:
            ttp = float(np.inf)
            days = sim["t"]
            pn = psa_norm
            on = np.ones(len(days))
            yf = np.array(
                [sim["T_plus"][-1], sim["T_prod"][-1], sim["T_minus"][-1]],
                dtype=float,
            )
        total = float(np.sum(yf))
        return {
            "t": days,
            "psa_normalized": pn,
            "on_treatment": on.astype(float),
            "ttp": ttp,
            "cumulative_dose_days": float(np.sum(on)),
            "fraction_off_treatment": 0.0,
            "final_T_minus_frac": float(yf[2] / total) if total > 0 else 0.0,
            "y_final": yf,
            "T_plus": np.array([yf[0]]),
            "T_prod": np.array([yf[1]]),
            "T_minus": np.array([yf[2]]),
        }
