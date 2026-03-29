#!/usr/bin/env python
"""
Script 04 — Bootstrap parameter uncertainty quantification.

Runs residual bootstrap on the first N synthetic patients (default 3) and
reports 90% confidence intervals for all 7 LV model parameters.  Saves
marginal distribution plots to figures/bootstrap_uncertainty/.

Key diagnostic question: how uncertain are the parameters given only
~20–30 PSA observations per patient?

Runtime note
------------
Each bootstrap iteration requires fitting a 7-parameter nonlinear ODE model
(L-BFGS-B with numerical gradients).  Expect ~8–15 seconds per bootstrap
iteration per patient.  Defaults are set for a ~5–10 minute run:
  - n=3 patients × n_bootstrap=30 × ~10s ≈ 15 min
For a quick smoke test, use n=1 --n-bootstrap 5 (~1 min).
For production phase-2 results use n=10 --n-bootstrap 200 (~300 min total,
run overnight or in parallel on a cluster).

Usage
-----
python scripts/04_bootstrap_uncertainty.py                   # 3 patients, 30 bootstraps
python scripts/04_bootstrap_uncertainty.py --n 1 --n-bootstrap 5   # quick smoke test
python scripts/04_bootstrap_uncertainty.py --real                   # Cunningham trial

Outputs
-------
figures/bootstrap_uncertainty/patient_{id}_marginals.png
figures/bootstrap_uncertainty/bootstrap_summary_table.csv
"""

from __future__ import annotations

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.patient import Patient
from src.parameter_fitting import BootstrapFitter, PARAM_NAMES, PARAM_BOUNDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FIGURES_DIR  = Path(__file__).parent.parent / "figures" / "bootstrap_uncertainty"
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PARAM_UNITS = {
    "r_plus":    "day⁻¹",
    "r_prod":    "day⁻¹",
    "r_minus":   "day⁻¹",
    "delta_plus": "day⁻¹",
    "T0_plus":   "cells",
    "T0_prod":   "cells",
    "T0_minus":  "cells",
}


def load_synthetic_patients(n: int, every_n_days: int = 28) -> list[Patient]:
    """
    Load the first n synthetic patients from data/synthetic/.

    Subsamples to every `every_n_days` days (default 28 = monthly) to match
    the real clinical trial measurement frequency and keep fitting fast.
    The full daily-resolution data has ~1000 time points which is impractical;
    real patients have ~20–35 measurements.
    """
    patients = []
    for i in range(n):
        path = SYNTHETIC_DIR / f"patient_{i:03d}_adaptive.csv"
        if not path.exists():
            log.warning("Synthetic patient file not found: %s", path)
            continue
        df = pd.read_csv(path)
        req = {"day", "psa_normalized", "on_treatment"}
        if not req.issubset(df.columns):
            log.warning("File %s missing required columns %s", path, req)
            continue
        psa_df = df[["day", "psa_normalized", "on_treatment"]].copy()
        psa_df = psa_df.rename(columns={"psa_normalized": "psa"})
        # Subsample to clinical measurement frequency
        psa_df = psa_df[psa_df["day"] % every_n_days == 0].reset_index(drop=True)
        if len(psa_df) < 3:
            psa_df = df[["day", "psa_normalized", "on_treatment"]].rename(
                columns={"psa_normalized": "psa"}
            ).reset_index(drop=True)
        patients.append(Patient(
            patient_id=f"synthetic_{i:03d}",
            psa_data=psa_df,
        ))
        log.debug("  synthetic_%03d: %d observations after subsampling", i, len(psa_df))
    return patients


def load_real_patients(n: int) -> list[Patient]:
    """Load Cunningham trial adaptive-arm patients."""
    from src.data_loaders import CunninghamTrialLoader
    loader   = CunninghamTrialLoader()
    patients = loader.load_patients(groups=["Adaptive"])
    return patients[:n]


def plot_marginals(result: dict, patient_id: str) -> None:
    """Plot marginal bootstrap distributions for all 7 parameters."""
    samples = result["bootstrap_samples"]
    nominal = result["nominal"]
    ci_lo   = result["ci_lower"]
    ci_hi   = result["ci_upper"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, param in enumerate(PARAM_NAMES):
        ax  = axes[idx]
        col = samples[:, idx]

        ax.hist(col, bins=40, density=True, alpha=0.65,
                color="#4c72b0", edgecolor="white", linewidth=0.4)

        nom_val = nominal.get(param, np.nan)
        ax.axvline(nom_val,  color="black",   lw=2,   label=f"Nominal={nom_val:.4g}")
        ax.axvline(ci_lo[param], color="crimson", lw=1.5, ls="--",
                   label=f"5th={ci_lo[param]:.4g}")
        ax.axvline(ci_hi[param], color="crimson", lw=1.5, ls="--",
                   label=f"95th={ci_hi[param]:.4g}")

        cv = float(col.std() / col.mean() * 100) if col.mean() > 0 else 0.0
        ax.set_title(f"{param}  (CV={cv:.0f}%)", fontsize=10)
        ax.set_xlabel(PARAM_UNITS.get(param, ""), fontsize=8)
        ax.legend(fontsize=6.5)

    axes[-1].set_visible(False)
    fig.suptitle(
        f"Bootstrap parameter distributions — Patient {patient_id}\n"
        f"(n_boot={len(samples)}, RMSE_nominal={result['nominal_rmse']:.4f})",
        fontsize=12,
    )
    plt.tight_layout()

    out = FIGURES_DIR / f"patient_{patient_id}_marginals.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", out)


def summary_table(results: list[tuple[str, dict]]) -> pd.DataFrame:
    """Build a summary table: param | nominal | 5th | 95th | CV% for each patient."""
    rows = []
    for patient_id, res in results:
        nominal  = res["nominal"]
        ci_lo    = res["ci_lower"]
        ci_hi    = res["ci_upper"]
        samples  = res["bootstrap_samples"]
        for idx, param in enumerate(PARAM_NAMES):
            nom = nominal.get(param, np.nan)
            lo  = ci_lo.get(param, np.nan)
            hi  = ci_hi.get(param, np.nan)
            col = samples[:, idx]
            cv  = float(col.std() / col.mean() * 100) if col.mean() > 0 else 0.0
            rows.append({
                "patient_id":    patient_id,
                "parameter":     param,
                "unit":          PARAM_UNITS.get(param, ""),
                "nominal":       round(nom, 6),
                "ci_5th":        round(lo,  6),
                "ci_95th":       round(hi,  6),
                "cv_%":          round(cv,  1),
                "n_successful":  res["n_successful"],
                "nominal_rmse":  round(res["nominal_rmse"], 5),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",            type=int,   default=3,
                        help="Number of patients to analyse (default 3).")
    parser.add_argument("--n-bootstrap",  type=int,   default=30,
                        help="Bootstrap replicates per patient (default 30; use 200 for final results).")
    parser.add_argument("--n-restarts",   type=int,   default=5,
                        help="L-BFGS-B restarts for nominal fit (default 5).")
    parser.add_argument("--real",         action="store_true",
                        help="Use Cunningham trial data instead of synthetic.")
    args = parser.parse_args()

    # ── Load patients ──────────────────────────────────────────────────────
    if args.real:
        log.info("Loading Cunningham trial patients ...")
        patients = load_real_patients(args.n)
    else:
        log.info("Loading synthetic patients ...")
        patients = load_synthetic_patients(args.n)

    if not patients:
        log.error("No patients loaded. Run scripts/01_generate_synthetic_data.py first.")
        sys.exit(1)

    log.info("Analysing %d patients with n_bootstrap=%d", len(patients), args.n_bootstrap)

    # ── Bootstrap fitting ──────────────────────────────────────────────────
    fitter  = BootstrapFitter(
        n_bootstrap=args.n_bootstrap,
        n_restarts_nominal=args.n_restarts,
        n_restarts_bootstrap=5,
    )
    results: list[tuple[str, dict]] = []

    for patient in tqdm(patients, desc="Bootstrap UQ"):
        log.info("  Patient %s  (n_obs=%d)", patient.patient_id,
                 len(patient.psa_data) if patient.psa_data is not None else 0)
        try:
            res = fitter.fit(patient)
            results.append((patient.patient_id, res))
            plot_marginals(res, patient.patient_id)
            log.info(
                "    RMSE=%.5f  n_ok=%d/%d  r_plus 90%%CI=[%.5g, %.5g]",
                res["nominal_rmse"],
                res["n_successful"], args.n_bootstrap,
                res["ci_lower"]["r_plus"], res["ci_upper"]["r_plus"],
            )
        except Exception as exc:
            log.warning("  Patient %s failed: %s", patient.patient_id, exc)

    if not results:
        log.error("No bootstrap results obtained.")
        sys.exit(1)

    # ── Summary table ──────────────────────────────────────────────────────
    df_summary = summary_table(results)
    out_csv    = FIGURES_DIR / "bootstrap_summary_table.csv"
    df_summary.to_csv(out_csv, index=False)
    log.info("\nBootstrap summary saved → %s", out_csv)

    print("\n" + "=" * 80)
    print("BOOTSTRAP UNCERTAINTY SUMMARY  (90% CI across all patients per parameter)")
    print("=" * 80)

    grouped = df_summary.groupby("parameter").agg(
        nominal_mean   = ("nominal",  "mean"),
        ci5_mean       = ("ci_5th",   "mean"),
        ci95_mean      = ("ci_95th",  "mean"),
        cv_mean        = ("cv_%",     "mean"),
    ).round(6)
    grouped["fold_CI"] = (grouped["ci95_mean"] / grouped["ci5_mean"]).round(1)
    print(grouped[["nominal_mean", "ci5_mean", "ci95_mean", "cv_mean", "fold_CI"]]
          .to_string())
    print("=" * 80)
    print("\nKey question: how much do the parameters vary across bootstraps?")
    most_uncertain = grouped["cv_mean"].idxmax()
    print(f"  Most uncertain parameter: {most_uncertain}  "
          f"(mean CV={grouped.loc[most_uncertain, 'cv_mean']:.1f}%)")
    print(f"  Least uncertain: {grouped['cv_mean'].idxmin()}  "
          f"(mean CV={grouped['cv_mean'].min():.1f}%)")
    print(f"\nPlots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
