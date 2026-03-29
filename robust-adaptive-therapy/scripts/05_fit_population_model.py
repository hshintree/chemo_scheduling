#!/usr/bin/env python
"""
Script 05 — Fit the population parameter distribution.

1. Loads population_params.csv (true parameters for all 50 synthetic patients).
2. Fits a multivariate log-normal PopulationModel.
3. Runs KS tests on each marginal (they should pass — generated log-normally).
4. Plots marginal histograms with fitted log-normal overlaid.
5. Plots correlation matrix of log-parameters.
6. Samples 100 virtual patients from the population model and simulates
   them under Zhang adaptive therapy.
7. Compares simulated TTP to the original 50 patients (validates the model
   as a generative model).

Usage
-----
python scripts/05_fit_population_model.py
python scripts/05_fit_population_model.py --n-virtual 200

Outputs
-------
data/processed/population_model.npz   (saved model for downstream scripts)
figures/population_model/marginals.png
figures/population_model/correlation.png
figures/population_model/ttp_comparison.png
figures/population_model/summary_table.csv
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
import matplotlib.cm as cm
from pathlib import Path
from scipy.stats import norm as sp_norm

from src.population_model import PopulationModel, PARAM_LABELS
from src.lotka_volterra import LotkaVolterraModel
from src.psa_model import PSAModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FIGURES_DIR   = Path(__file__).parent.parent / "figures" / "population_model"
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

K_FIXED     = 10_000.0
ALPHA_FIXED = np.array([[1.0, 0.5, 0.5],
                         [0.5, 1.0, 0.5],
                         [0.5, 0.5, 1.0]])


def load_population_params() -> list[dict]:
    """Load population_params.csv from data/synthetic/."""
    path = SYNTHETIC_DIR / "population_params.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. "
            "Run scripts/01_generate_synthetic_data.py first."
        )
    df = pd.read_csv(path)
    required = PopulationModel.param_names
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"population_params.csv missing columns: {missing}")
    return df[required].to_dict(orient="records")


def simulate_zhang_at(params: dict, max_days: int = 1825) -> float:
    """
    Simulate Zhang adaptive therapy and return TTP (days).

    Stop treatment when PSA < 50% of baseline; restart when PSA = baseline.
    Progression = PSA > 2× baseline on treatment, or T− > 80% of total.
    """
    full_params = {**params,
                   "delta_prod": 0.5 * params["delta_plus"],
                   "K":     K_FIXED,
                   "alpha": ALPHA_FIXED.copy()}

    model   = LotkaVolterraModel(full_params)
    psa_mod = PSAModel()

    t_span = (0.0, float(max_days))
    t_eval = np.arange(0, max_days + 1, dtype=float)

    baseline_total = params["T0_plus"] + params["T0_prod"] + params["T0_minus"]
    psa_baseline   = baseline_total   # c_i = 1

    on_tx    = 1
    y        = np.array([params["T0_plus"], params["T0_prod"], params["T0_minus"]])
    day      = 0
    psa_stop = 0.50 * psa_baseline

    while day < max_days:
        end_day = min(day + 28, max_days)
        t_seg   = np.arange(day, end_day + 1, dtype=float)

        def schedule(t, _on=on_tx):
            return _on

        seg = model.simulate(
            t_span=(float(day), float(end_day)),
            t_eval=t_seg,
            drug_schedule=schedule,
            y0=y,
        )
        y   = np.array([seg["T_plus"][-1], seg["T_prod"][-1], seg["T_minus"][-1]])
        psa = psa_mod.compute_psa(seg)[-1]

        total = float(y.sum())
        t_minus_frac = float(y[2]) / total if total > 0 else 0.0

        # Progression check
        if on_tx and (psa > 2.0 * psa_baseline or t_minus_frac > 0.80):
            return float(day)

        # Treatment switching
        if on_tx and psa < psa_stop:
            on_tx = 0
        elif not on_tx and psa >= psa_baseline:
            on_tx = 1

        day = end_day

    return float(max_days)


def plot_marginals(model: PopulationModel, param_records: list[dict]) -> None:
    """Plot marginal histograms with fitted log-normal overlaid."""
    X = np.exp(model.log_samples())   # original parameter space

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, param in enumerate(PopulationModel.param_names):
        ax  = axes[idx]
        col = X[:, idx]

        ax.hist(col, bins=20, density=True, alpha=0.6,
                color="#4c72b0", edgecolor="white", label="Data")

        # Fitted log-normal overlay
        mu_log  = float(model.mean_log[idx])
        std_log = float(np.sqrt(model.cov_log[idx, idx]))
        x_min, x_max = col.min() * 0.5, col.max() * 1.5
        x_range = np.linspace(x_min, x_max, 300)
        # log-normal PDF
        x_pos = x_range[x_range > 0]
        log_pdf = sp_norm.pdf(np.log(x_pos), mu_log, std_log) / x_pos
        ax.plot(x_pos, log_pdf, "r-", lw=2, label="Log-normal fit")

        ax.set_title(PARAM_LABELS.get(param, param), fontsize=10)
        ax.set_xlabel("Value", fontsize=8)
        ax.legend(fontsize=7)

    axes[-1].set_visible(False)
    fig.suptitle("Population parameter distributions (synthetic cohort, N=50)",
                 fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "marginals.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


def plot_correlation(model: PopulationModel) -> None:
    """Plot correlation matrix of log-parameters."""
    corr_df = model.correlation_matrix()
    labels  = [PARAM_LABELS.get(p, p) for p in model.param_names]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1,
                   cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson correlation (log-space)")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr_df.values[i, j]) > 0.5 else "black")

    ax.set_title("Log-parameter correlation matrix (population model)", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "correlation.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


def plot_ttp_comparison(
    ttp_original:  np.ndarray,
    ttp_virtual:   np.ndarray,
) -> None:
    """KM-style TTP comparison: original 50 patients vs 100 virtual patients."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for ttp, label, colour in [
        (ttp_original, f"Original cohort (N={len(ttp_original)})", "#1f77b4"),
        (ttp_virtual,  f"Virtual patients from pop. model (N={len(ttp_virtual)})", "#d62728"),
    ]:
        ttp_sorted = np.sort(ttp)
        survival   = 1.0 - np.arange(1, len(ttp) + 1) / len(ttp)
        ax.step(ttp_sorted, survival, where="post",
                label=label, color=colour, lw=2)

    ax.set_xlabel("Time to Progression (days)", fontsize=12)
    ax.set_ylabel("Proportion progression-free", fontsize=12)
    ax.set_title("TTP distribution: original vs virtual patients (Zhang AT)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    for ttp, colour in [(ttp_original, "#1f77b4"), (ttp_virtual, "#d62728")]:
        med = np.median(ttp)
        ax.axvline(med, color=colour, ls="--", lw=1.2, alpha=0.7)
        ax.text(med + 10, 0.5, f"Med={med:.0f}d",
                color=colour, fontsize=8, va="center")

    plt.tight_layout()
    out = FIGURES_DIR / "ttp_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-virtual", type=int, default=100,
                        help="Virtual patients to simulate (default 100).")
    args = parser.parse_args()

    # ── Load true parameters ───────────────────────────────────────────────
    log.info("Loading population_params.csv ...")
    param_records = load_population_params()
    log.info("  Loaded %d patient parameter sets.", len(param_records))

    # ── Fit population model ───────────────────────────────────────────────
    log.info("Fitting PopulationModel ...")
    pop_model = PopulationModel()
    result    = pop_model.fit(param_records)

    print("\n" + "=" * 70)
    print("POPULATION MODEL FIT RESULTS")
    print("=" * 70)
    print(f"N patients: {result['n_patients']}")
    print("\nSummary statistics (parameter space):")
    print(pop_model.summary_dataframe().round(6).to_string())
    print("\nKS-test p-values (log-normality of each marginal):")
    for param, pval in result["marginal_ks_pvalues"].items():
        flag = "✓" if pval > 0.05 else "✗ (reject log-normal at 5%)"
        print(f"  {param:<15} p={pval:.3f}  {flag}")

    # ── Plots: marginals and correlation ───────────────────────────────────
    log.info("Plotting marginals ...")
    plot_marginals(pop_model, param_records)
    log.info("Plotting correlation matrix ...")
    plot_correlation(pop_model)

    # ── Simulate original cohort TTP ───────────────────────────────────────
    log.info("Simulating original cohort TTP (Zhang AT) ...")
    ttp_original = np.array([
        simulate_zhang_at(p) for p in param_records
    ])
    log.info("  Original: median TTP = %.0f days", np.median(ttp_original))

    # ── Sample virtual patients and simulate ───────────────────────────────
    log.info("Sampling %d virtual patients from population model ...", args.n_virtual)
    virtual_params = pop_model.sample_as_dicts(args.n_virtual, seed=0)

    log.info("Simulating virtual cohort TTP (Zhang AT) ...")
    ttp_virtual = np.array([
        simulate_zhang_at(p) for p in virtual_params
    ])
    log.info("  Virtual:  median TTP = %.0f days", np.median(ttp_virtual))

    # ── TTP comparison plot ────────────────────────────────────────────────
    plot_ttp_comparison(ttp_original, ttp_virtual)

    # ── Save model for downstream use ─────────────────────────────────────
    out_npz = PROCESSED_DIR / "population_model.npz"
    np.savez(
        out_npz,
        mean_log   = pop_model.mean_log,
        cov_log    = pop_model.cov_log,
        X_log      = pop_model.log_samples(),
        param_names= np.array(pop_model.param_names),
    )
    log.info("Population model saved → %s", out_npz)

    # ── Summary table ──────────────────────────────────────────────────────
    summary_df = pop_model.summary_dataframe().round(6)
    summary_df["ks_pvalue"] = pd.Series(result["marginal_ks_pvalues"])
    out_csv = FIGURES_DIR / "summary_table.csv"
    summary_df.to_csv(out_csv)
    log.info("Summary table saved → %s", out_csv)

    print("\n" + "=" * 70)
    print("TTP COMPARISON: ORIGINAL vs. VIRTUAL PATIENTS")
    print("=" * 70)
    for label, ttp in [("Original (N=50)", ttp_original), (f"Virtual (N={args.n_virtual})", ttp_virtual)]:
        print(f"  {label}")
        print(f"    Median TTP: {np.median(ttp):.0f} days")
        print(f"    Mean TTP:   {np.mean(ttp):.0f} days")
        print(f"    5th–95th:   {np.percentile(ttp, 5):.0f} – {np.percentile(ttp, 95):.0f} days")
    print(f"\nConclusion: population model is {'✓ valid' if abs(np.median(ttp_virtual) - np.median(ttp_original)) < 200 else '⚠ check'} as a generative model.")
    print(f"(Median TTP difference: {abs(np.median(ttp_virtual) - np.median(ttp_original)):.0f} days)")
    print(f"\nAll outputs saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
