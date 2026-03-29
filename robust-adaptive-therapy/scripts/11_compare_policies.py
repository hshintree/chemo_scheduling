#!/usr/bin/env python
"""
SCRIPT 11: Full Policy Comparison — Main Results
"""

from __future__ import annotations

import os
import pickle
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.cvxpy_relaxation import _rk_step
from src.feasibility import FeasibilityClassifier, _params_to_model
from src.lotka_volterra import LotkaVolterraModel
from src.policy import MTDPolicy, RangeBoundedPolicy, ThresholdPolicy
from src.population_model import PopulationModel
from src.psa_model import PSAModel
from src.uncertainty import WassersteinUncertaintySet
from src.utils import plot_km_curves

ROOT = Path(__file__).resolve().parent.parent
DATA_SYN = ROOT / "data" / "synthetic"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "comparison"

T_MAX_DAYS = 1825.0


def load_synthetic_params() -> list[dict]:
    df = pd.read_csv(DATA_SYN / "population_params.csv")
    out = []
    for row in df.itertuples(index=False):
        d = dict(row._asdict())
        d.pop("patient_id", None)
        out.append(LotkaVolterraModel.from_dict(d).to_dict())
    return out


def sample_full_uncertainty_set(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> list[dict]:
    """Draw n parameter vectors from the full Wasserstein-perturbed uncertainty set."""
    X_log = np.log(df[PopulationModel.param_names].astype(float).values)
    w = WassersteinUncertaintySet(X_log, epsilon=0.1, param_names=PopulationModel.param_names)
    w.epsilon = w.compute_epsilon_from_confidence(0.90)
    mat = w.sample_perturbed_params(n, seed=seed)
    names = PopulationModel.param_names
    out: list[dict] = []
    for i in range(len(mat)):
        d = dict(zip(names, mat[i].tolist()))
        d["K"] = 10_000.0
        d["delta_prod"] = 0.5 * float(d["delta_plus"])
        d["alpha"] = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]], dtype=float)
        out.append(d)
    return out


def oracle_ttp(m: LotkaVolterraModel, psa_m: PSAModel) -> float:
    best = -1.0
    for s, r in product(np.linspace(0.2, 0.7, 5), np.linspace(0.65, 1.2, 5)):
        if r <= s:
            continue
        out = ThresholdPolicy(s, r).simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)
        ttp = float(out["ttp"]) if np.isfinite(out["ttp"]) else 1e6
        best = max(best, ttp)
    return float(best)


def run_socp_style(
    m: LotkaVolterraModel,
    psa_m: PSAModel,
    mpc_u: np.ndarray,
    dt: float = 28.0,
    t_max: float = T_MAX_DAYS,
) -> dict:
    y = np.array([m.T0_plus, m.T0_prod, m.T0_minus], dtype=float)
    bl = float(
        psa_m.compute_psa({
            "T_plus": np.array([y[0]]),
            "T_prod": np.array([y[1]]),
            "T_minus": np.array([y[2]]),
        })[0]
    )
    drug_days = 0.0
    days = 0
    ttp = np.inf
    max_steps = int(t_max / dt) + 2
    for _k in range(max_steps):
        u = float(mpc_u[_k % len(mpc_u)])
        drug_days += u * dt
        psa = float(
            psa_m.compute_psa({
                "T_plus": np.array([y[0]]),
                "T_prod": np.array([y[1]]),
                "T_minus": np.array([y[2]]),
            })[0]
        )
        psa_n = psa / bl if bl > 0 else 0.0
        tot = float(y.sum())
        rf = float(y[2] / tot) if tot > 0 else 0.0
        if (psa_n >= 2.0 and u >= 0.5) or (rf >= 0.8 and days > 30):
            ttp = float(days)
            break
        if days >= t_max:
            break
        y = _rk_step(m, y, u, dt)
        days += int(dt)
    frac_off = 1.0 - drug_days / max(days, 1)
    return {
        "ttp": ttp,
        "cumulative_dose_days": drug_days,
        "fraction_off_treatment": frac_off,
    }


def _ttp_finite(x: float) -> float:
    if not np.isfinite(x):
        return float(T_MAX_DAYS)
    return float(x)


def run_stratified_comparison(
    all_scenarios: list[dict],
    classifier: FeasibilityClassifier,
    psa_m: PSAModel,
    sg_phi: tuple[float, float],
    mpc_u: np.ndarray | None,
) -> dict[str, list[float]]:
    """
    For each patient:
      1. Classify feasible or infeasible.
      2. If feasible: simulate ALL policies and record TTP.
      3. If infeasible: TTP = 0 for adaptive threshold policies and SOCP-style;
         MTD gets true MTD TTP; Oracle still evaluated; Stratified = 0.

    Stratified: CVaR-robust when feasible; when AT is inappropriate, clinical
    fallback is MTD (same TTP as MTD for that patient — do not waste time on AT).
    """
    res: dict[str, list[float]] = {k: [] for k in [
        "MTD", "Zhang", "RBAT", "CVaR_robust", "SOCP_MPC", "Oracle", "Stratified",
    ]}
    zhang = ThresholdPolicy(0.5, 1.0)
    rbat = RangeBoundedPolicy(0.3, 0.8)
    cvar_pol = ThresholdPolicy(sg_phi[0], sg_phi[1])
    mtd_pol = MTDPolicy()

    for p in all_scenarios:
        m = _params_to_model(LotkaVolterraModel, p)
        feas = classifier.is_feasible(p)
        if feas:
            mtd_t = _ttp_finite(mtd_pol.simulate_patient(m, psa_m, t_max=T_MAX_DAYS)["ttp"])
            res["MTD"].append(mtd_t)
            res["Zhang"].append(_ttp_finite(zhang.simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)["ttp"]))
            res["RBAT"].append(_ttp_finite(rbat.simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)["ttp"]))
            res["CVaR_robust"].append(_ttp_finite(cvar_pol.simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)["ttp"]))
            res["Stratified"].append(_ttp_finite(cvar_pol.simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)["ttp"]))
            if mpc_u is not None:
                res["SOCP_MPC"].append(_ttp_finite(run_socp_style(m, psa_m, mpc_u, t_max=T_MAX_DAYS)["ttp"]))
            else:
                res["SOCP_MPC"].append(_ttp_finite(zhang.simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)["ttp"]))
            res["Oracle"].append(_ttp_finite(oracle_ttp(m, psa_m)))
        else:
            mtd_t = _ttp_finite(mtd_pol.simulate_patient(m, psa_m, t_max=T_MAX_DAYS)["ttp"])
            res["MTD"].append(mtd_t)
            res["Zhang"].append(0.0)
            res["RBAT"].append(0.0)
            res["CVaR_robust"].append(0.0)
            res["Stratified"].append(mtd_t)
            res["SOCP_MPC"].append(0.0)
            res["Oracle"].append(_ttp_finite(oracle_ttp(m, psa_m)))

    return res


def cvar_ttp(ttps: np.ndarray, alpha: float = 0.2) -> float:
    a = np.sort(ttps)
    k = max(1, int(np.ceil(alpha * len(a))))
    return float(np.mean(a[:k]))


def simulate_dose_unstratified(
    name: str,
    pol: object | None,
    p: dict,
    psa_m: PSAModel,
    mpc_u: np.ndarray | None,
) -> tuple[float, float]:
    """One patient: (fraction_off, dose_days) with full simulation (not stratified)."""
    m = _params_to_model(LotkaVolterraModel, p)
    if name == "SOCP_MPC" and mpc_u is not None:
        out = run_socp_style(m, psa_m, mpc_u, t_max=T_MAX_DAYS)
    elif name == "SOCP_MPC":
        out = ThresholdPolicy(0.5, 1.0).simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)
    elif isinstance(pol, MTDPolicy):
        out = pol.simulate_patient(m, psa_m, t_max=T_MAX_DAYS)
    elif isinstance(pol, RangeBoundedPolicy):
        out = pol.simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)
    elif isinstance(pol, ThresholdPolicy):
        out = pol.simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)
    else:
        return float("nan"), float("nan")
    return float(out["fraction_off_treatment"]), float(out["cumulative_dose_days"])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    psa_m = PSAModel()
    clf = FeasibilityClassifier(LotkaVolterraModel, PSAModel, t_max=T_MAX_DAYS)
    df = pd.read_csv(DATA_SYN / "population_params.csv")

    all_scenarios = sample_full_uncertainty_set(df, n=200, seed=11)
    syn = load_synthetic_params()

    sg_phi = (0.5, 1.0)
    p_sub = PROC / "subgradient_results.pkl"
    if p_sub.exists():
        with open(p_sub, "rb") as f:
            sub = pickle.load(f)
            sg_phi = tuple(sub["best"]["optimal_phi"])

    mpc_u = None
    p_cv = PROC / "cvxpy_mpc_schedule.pkl"
    if p_cv.exists():
        with open(p_cv, "rb") as f:
            mpc_u = pickle.load(f).get("mpc_u")

    strat = run_stratified_comparison(all_scenarios, clf, psa_m, sg_phi, mpc_u)

    cohort_100 = all_scenarios[: min(100, len(all_scenarios))]
    mtd_doses = [
        float(MTDPolicy().simulate_patient(_params_to_model(LotkaVolterraModel, p), psa_m, t_max=T_MAX_DAYS)["cumulative_dose_days"])
        for p in cohort_100
    ]
    mtd_med_dose = float(np.median(mtd_doses))

    policy_for_dose: list[tuple[str, object | None]] = [
        ("MTD", MTDPolicy()),
        ("Zhang", ThresholdPolicy(0.5, 1.0)),
        ("RBAT", RangeBoundedPolicy(0.3, 0.8)),
        ("CVaR_robust", ThresholdPolicy(sg_phi[0], sg_phi[1])),
        ("SOCP_MPC", None),
    ]

    rows = []
    km_in: dict[str, np.ndarray] = {}
    order = ["MTD", "Zhang", "RBAT", "CVaR_robust", "SOCP_MPC", "Oracle", "Stratified"]
    for key in order:
        ttps = np.asarray(strat[key], dtype=float)
        km_in[key] = np.where(ttps >= T_MAX_DAYS - 1, np.inf, ttps)
        med = float(np.median(ttps))
        cv = cvar_ttp(ttps)
        if key == "Stratified":
            fo = float("nan")
            dr = float("nan")
        elif key == "Oracle":
            fo = float("nan")
            dr = float("nan")
        else:
            pol = dict(policy_for_dose)[key] if key != "SOCP_MPC" else None
            offs = []
            doses = []
            for p in cohort_100:
                o, d = simulate_dose_unstratified(key, pol, p, psa_m, mpc_u)
                offs.append(o)
                doses.append(d)
            fo = float(np.mean(offs) * 100)
            dr = float(100 * (1 - float(np.median(doses)) / mtd_med_dose)) if mtd_med_dose > 0 else 0.0
        rows.append({
            "policy": key,
            "n": len(all_scenarios),
            "median_ttp": med,
            "cvar_ttp": cv,
            "pct_off": fo,
            "dose_reduction_pct": dr,
        })

    fig_km = plot_km_curves(
        km_in,
        title="Kaplan–Meier: full uncertainty cohort (stratified TTP for adaptive policies)",
        figsize=(10, 6),
    )
    fig_km.savefig(FIG / "kaplan_meier_main.png", dpi=300, bbox_inches="tight")
    fig_km.savefig(FIG / "kaplan_meier_main.pdf", bbox_inches="tight")
    plt.close(fig_km)

    zhang_t, robust_t, t0s = [], [], []
    for p in all_scenarios[: min(120, len(all_scenarios))]:
        m = _params_to_model(LotkaVolterraModel, p)
        zhang_t.append(
            float(ThresholdPolicy(0.5, 1.0).simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)["ttp"])
        )
        robust_t.append(
            float(ThresholdPolicy(sg_phi[0], sg_phi[1]).simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)["ttp"])
        )
        t0s.append(float(p["T0_minus"]))
    zhang_t = np.array([t if np.isfinite(t) else T_MAX_DAYS for t in zhang_t])
    robust_t = np.array([t if np.isfinite(t) else T_MAX_DAYS for t in robust_t])
    t0s = np.asarray(t0s)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(zhang_t, robust_t, c=t0s, cmap="viridis", alpha=0.8, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label=r"$T_{0}^-$")
    mx = max(zhang_t.max(), robust_t.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=1)
    imp = float(np.mean(robust_t > zhang_t))
    med_gain = float(np.median(robust_t - zhang_t))
    ax.set_xlabel("TTP Zhang (days)")
    ax.set_ylabel("TTP CVaR-robust (days)")
    ax.text(0.04, 0.96, f"{100*imp:.0f}% improved\nmedian gain {med_gain:+.0f} d", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "scatter_robust_vs_zhang.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    dose_series: dict[str, list[float]] = {k: [] for k, _ in policy_for_dose}
    for p in cohort_100:
        for key, pol in policy_for_dose:
            _, d = simulate_dose_unstratified(key, pol, p, psa_m, mpc_u)
            dose_series[key].append(d)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(dose_series.values(), labels=list(dose_series.keys()), patch_artist=True)
    ax.set_ylabel("Cumulative dose-days (proxy)")
    ax.set_title("Dose intensity by policy (unstratified, n=100)")
    fig.tight_layout()
    fig.savefig(FIG / "dose_reduction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n=== MAIN RESULTS (Zhang/RBAT/CVaR/SOCP → 0 when infeasible; Stratified → MTD TTP) ===\n")
    print("Policy               | N cohort | Median TTP | CVaR TTP | % Off Tx | Dose red. vs MTD")
    print("-------------------- | -------- | ---------- | -------- | -------- | -----------------")
    for r in rows:
        po = r["pct_off"]
        po_s = f"{po:7.1f}%" if np.isfinite(po) else "   n/a  "
        dr = r["dose_reduction_pct"]
        dr_s = f"{dr:17.1f}%" if np.isfinite(dr) else "              n/a"
        print(
            f"{r['policy']:<20} | {r['n']:8d} | {r['median_ttp']:10.1f} | {r['cvar_ttp']:8.1f} | "
            f"{po_s} | {dr_s}"
        )

    elig = clf.classify_cohort(all_scenarios)
    feas_frac = float(elig["feasible_fraction"])
    zhang_wins = float(elig["boundary_features"]["zhang_beats_mtd"].mean())
    print("\nFeasibility / reporting (full uncertainty sample):")
    print(f"  ~{100*feas_frac:.0f}% drug-eligible (δ+ > threshold).")
    print(f"  ~{100*zhang_wins:.0f}% have Zhang TTP > MTD (informational).\n")

    print("(Synthetic cohort medians — unstratified)")
    mtd_p = MTDPolicy()
    zh = ThresholdPolicy(0.5, 1.0)
    rb = RangeBoundedPolicy(0.3, 0.8)
    cv = ThresholdPolicy(sg_phi[0], sg_phi[1])
    for label, pol in [
        ("MTD", mtd_p),
        ("Zhang", zh),
        ("RBAT", rb),
        ("CVaR_robust", cv),
    ]:
        ttps = []
        for p in syn:
            m = _params_to_model(LotkaVolterraModel, p)
            if label == "MTD":
                ttps.append(_ttp_finite(mtd_p.simulate_patient(m, psa_m, t_max=T_MAX_DAYS)["ttp"]))
            elif label == "RBAT":
                ttps.append(_ttp_finite(rb.simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)["ttp"]))
            else:
                ttps.append(_ttp_finite(pol.simulate_patient(m, psa_m, t_max=T_MAX_DAYS, check_interval=1)["ttp"]))
        print(f"  {label}: median TTP = {np.median(ttps):.0f} d")


if __name__ == "__main__":
    main()
