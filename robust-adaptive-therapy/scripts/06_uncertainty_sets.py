#!/usr/bin/env python
"""
Script 06 — Construct and evaluate uncertainty sets.

Builds the two uncertainty sets that Phase 3 will robustify over, evaluates
them by sampling parameter vectors and simulating Zhang adaptive therapy,
and reports the worst-case TTP distribution.

Steps
-----
1. Load the population model fitted by script 05.
2. Build EllipsoidalUncertaintySet (confidence = 0.90).
3. Build WassersteinUncertaintySet (ε auto-set via finite-sample bound).
4. Sample 200 parameter vectors from each set.
5. Simulate Zhang AT for each.
6. Plot TTP distributions:
   (a) Nominal parameters (population mean)
   (b) Ellipsoidal boundary samples
   (c) Wasserstein ball samples
7. Print: nominal TTP | mean TTP | 5th pct TTP | worst-case TTP.

Usage
-----
python scripts/06_uncertainty_sets.py
python scripts/06_uncertainty_sets.py --n-samples 400 --confidence 0.95

Outputs
-------
figures/uncertainty_sets/ttp_distributions.png
figures/uncertainty_sets/ellipsoid_2d_projection.png
figures/uncertainty_sets/uncertainty_summary.csv
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

from src.population_model import PopulationModel, PARAM_NAMES, PARAM_LABELS
from src.uncertainty import EllipsoidalUncertaintySet, WassersteinUncertaintySet
from src.lotka_volterra import LotkaVolterraModel
from src.psa_model import PSAModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FIGURES_DIR   = Path(__file__).parent.parent / "figures" / "uncertainty_sets"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

K_FIXED     = 10_000.0
ALPHA_FIXED = np.array([[1.0, 0.5, 0.5],
                         [0.5, 1.0, 0.5],
                         [0.5, 0.5, 1.0]])


def load_population_model() -> PopulationModel:
    """Load population model from data/processed/population_model.npz."""
    npz_path = PROCESSED_DIR / "population_model.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. "
            "Run scripts/05_fit_population_model.py first."
        )
    data = np.load(npz_path, allow_pickle=True)
    model = PopulationModel()
    # Reconstruct fitted state
    model.mean_log    = data["mean_log"]
    model.cov_log     = data["cov_log"]
    model._X_log      = data["X_log"]
    model.n_patients  = len(data["X_log"])
    model.diag_cov_log = np.diag(np.diag(model.cov_log))
    n   = len(model.mean_log)
    reg = 1e-8 * np.eye(n)
    try:
        model._cov_inv = np.linalg.inv(model.cov_log + reg)
    except np.linalg.LinAlgError:
        model._cov_inv = np.diag(1.0 / (np.diag(model.cov_log) + 1e-8))
    model._fitted = True
    log.info("Population model loaded: %d patients, %d params.",
             model.n_patients, len(model.mean_log))
    return model


def simulate_zhang_at(params: dict, max_days: int = 1825) -> float:
    """Simulate Zhang AT and return TTP (days)."""
    full_params = {
        **params,
        "delta_prod": 0.5 * params["delta_plus"],
        "K":     K_FIXED,
        "alpha": ALPHA_FIXED.copy(),
    }
    model   = LotkaVolterraModel(full_params)
    psa_mod = PSAModel()

    psa_baseline = params["T0_plus"] + params["T0_prod"] + params["T0_minus"]
    psa_stop     = 0.50 * psa_baseline
    on_tx = 1
    y     = np.array([params["T0_plus"], params["T0_prod"], params["T0_minus"]])
    day   = 0

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

        total        = float(y.sum())
        t_minus_frac = float(y[2]) / total if total > 0 else 0.0

        if on_tx and (psa > 2.0 * psa_baseline or t_minus_frac > 0.80):
            return float(day)

        if on_tx and psa < psa_stop:
            on_tx = 0
        elif not on_tx and psa >= psa_baseline:
            on_tx = 1

        day = end_day

    return float(max_days)


def params_from_row(row: np.ndarray, names: list[str] = PARAM_NAMES) -> dict:
    return dict(zip(names, row.tolist()))


def simulate_batch(
    param_rows: np.ndarray,
    label:      str,
    max_days:   int = 1825,
) -> np.ndarray:
    """Simulate TTP for each row in param_rows; show tqdm progress."""
    ttp_list = []
    for row in tqdm(param_rows, desc=f"Sim {label}", leave=False):
        params = params_from_row(row)
        ttp_list.append(simulate_zhang_at(params, max_days))
    return np.array(ttp_list)


def plot_ttp_distributions(
    ttp_nominal:     float,
    ttp_ellipsoidal: np.ndarray,
    ttp_wasserstein: np.ndarray,
    confidence:      float,
    epsilon:         float,
) -> None:
    """Plot TTP distributions for all three sources side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Histogram ──────────────────────────────────────────────────────────
    bins = np.linspace(0, max(ttp_ellipsoidal.max(), ttp_wasserstein.max()) * 1.05,
                       35)
    ax1.axvline(ttp_nominal, color="black", lw=2.5, label=f"Nominal (μ) = {ttp_nominal:.0f}d", zorder=5)
    ax1.hist(ttp_ellipsoidal, bins=bins, density=True, alpha=0.60,
             color="#2ca02c", label=f"Ellipsoid boundary (conf={confidence:.0%})")
    ax1.hist(ttp_wasserstein, bins=bins, density=True, alpha=0.50,
             color="#ff7f0e", label=f"Wasserstein ball (ε={epsilon:.3f})")
    ax1.set_xlabel("Time to Progression (days)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("TTP distribution under uncertainty", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── KM-style survival ──────────────────────────────────────────────────
    for ttp, label, colour in [
        (ttp_ellipsoidal, f"Ellipsoid (N={len(ttp_ellipsoidal)})", "#2ca02c"),
        (ttp_wasserstein, f"Wasserstein (N={len(ttp_wasserstein)})", "#ff7f0e"),
    ]:
        ttp_s  = np.sort(ttp)
        surv   = 1.0 - np.arange(1, len(ttp) + 1) / len(ttp)
        ax2.step(ttp_s, surv, where="post", color=colour, lw=2, label=label)
        med = np.median(ttp)
        ax2.axvline(med, color=colour, ls="--", lw=1.2, alpha=0.7)

    ax2.axvline(ttp_nominal, color="black", lw=2.5, ls="-",
                label=f"Nominal = {ttp_nominal:.0f}d")
    ax2.set_xlabel("Time to Progression (days)", fontsize=11)
    ax2.set_ylabel("Proportion progression-free", fontsize=11)
    ax2.set_title("Kaplan-Meier under uncertainty", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Phase 2: Uncertainty Set TTP Evaluation", fontsize=13, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "ttp_distributions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


def plot_ellipsoid_2d(
    ell_set:     EllipsoidalUncertaintySet,
    was_set:     WassersteinUncertaintySet,
    boundary_pts: np.ndarray,
    was_pts:      np.ndarray,
    params_i:    int = 0,
    params_j:    int = 3,
) -> None:
    """2-D projection of the ellipsoid and Wasserstein samples."""
    pname_i = PARAM_NAMES[params_i]
    pname_j = PARAM_NAMES[params_j]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Wasserstein perturbed samples (in parameter space)
    ax.scatter(was_pts[:, params_i], was_pts[:, params_j],
               s=20, alpha=0.4, color="#ff7f0e", label="Wasserstein samples", zorder=2)

    # Ellipsoid boundary points
    ax.scatter(boundary_pts[:, params_i], boundary_pts[:, params_j],
               s=25, alpha=0.7, color="#2ca02c", marker="^",
               label="Ellipsoid boundary", zorder=3)

    # Original cohort in log-space
    X_orig = np.exp(ell_set.mean_log[None, :] + was_set.samples)
    ax.scatter(X_orig[:, params_i], X_orig[:, params_j],
               s=40, alpha=0.8, color="#1f77b4", marker="o",
               label="Original patients", zorder=4)

    # Population mean
    mu = np.exp(ell_set.mean_log)
    ax.scatter(mu[params_i], mu[params_j],
               s=200, color="black", marker="*", zorder=5, label="Population mean μ")

    ax.set_xlabel(PARAM_LABELS.get(pname_i, pname_i), fontsize=11)
    ax.set_ylabel(PARAM_LABELS.get(pname_j, pname_j), fontsize=11)
    ax.set_title(f"Uncertainty set projections: {pname_i} vs {pname_j}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "ellipsoid_2d_projection.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples",  type=int,   default=200,
                        help="Parameter vectors to sample from each set (default 200).")
    parser.add_argument("--confidence", type=float, default=0.90,
                        help="Ellipsoid confidence level (default 0.90).")
    parser.add_argument("--max-days",   type=int,   default=1825,
                        help="Max simulation horizon in days (default 1825 = 5 years).")
    args = parser.parse_args()

    # ── Load population model ──────────────────────────────────────────────
    pop_model = load_population_model()

    # ── Build uncertainty sets ─────────────────────────────────────────────
    log.info("Building EllipsoidalUncertaintySet (confidence=%.2f) ...", args.confidence)
    ell_set = EllipsoidalUncertaintySet(
        mean_log   = pop_model.mean_log,
        cov_log    = pop_model.cov_log,
        confidence = args.confidence,
    )
    log.info("  κ (radius) = %.4f", ell_set.kappa)

    # Wasserstein: use the empirical log-parameter samples from the cohort
    X_log = pop_model.log_samples()
    was_tmp = WassersteinUncertaintySet(X_log, epsilon=1.0)
    eps     = was_tmp.compute_epsilon_from_confidence(confidence=args.confidence)
    log.info("Building WassersteinUncertaintySet (ε=%.4f) ...", eps)
    was_set = WassersteinUncertaintySet(X_log, epsilon=eps)

    # ── Nominal TTP ────────────────────────────────────────────────────────
    mu_params = params_from_row(np.exp(pop_model.mean_log))
    log.info("Simulating nominal TTP (population mean parameters) ...")
    ttp_nominal = simulate_zhang_at(mu_params, max_days=args.max_days)
    log.info("  Nominal TTP = %.0f days", ttp_nominal)

    # ── Sample and simulate: ellipsoidal ───────────────────────────────────
    log.info("Sampling %d parameter vectors from ellipsoid boundary ...", args.n_samples)
    boundary_pts = ell_set.sample_boundary(n_points=args.n_samples, seed=0)

    log.info("Simulating Zhang AT for ellipsoidal samples ...")
    ttp_ell = simulate_batch(boundary_pts, "ellipsoid", args.max_days)

    # ── Sample and simulate: Wasserstein ───────────────────────────────────
    log.info("Sampling %d parameter vectors from Wasserstein ball ...", args.n_samples)
    was_params_log = was_set.sample_perturbed(n_samples=args.n_samples, seed=0)
    was_pts        = np.exp(was_params_log)

    log.info("Simulating Zhang AT for Wasserstein samples ...")
    ttp_was = simulate_batch(was_pts, "wasserstein", args.max_days)

    # ── Plots ──────────────────────────────────────────────────────────────
    log.info("Generating plots ...")
    plot_ttp_distributions(ttp_nominal, ttp_ell, ttp_was, args.confidence, eps)
    plot_ellipsoid_2d(ell_set, was_set, boundary_pts, was_pts)

    # ── Summary table ──────────────────────────────────────────────────────
    rows = []
    for label, ttp in [
        ("ellipsoidal", ttp_ell),
        ("wasserstein", ttp_was),
    ]:
        rows.append({
            "set":              label,
            "n_samples":        len(ttp),
            "nominal_ttp":      round(ttp_nominal, 0),
            "mean_ttp":         round(float(np.mean(ttp)), 0),
            "median_ttp":       round(float(np.median(ttp)), 0),
            "p5_ttp":           round(float(np.percentile(ttp, 5)), 0),
            "worst_case_ttp":   round(float(np.min(ttp)), 0),
            "pct_below_nominal": round(100 * np.mean(ttp < ttp_nominal), 1),
        })
    summary = pd.DataFrame(rows)
    out_csv = FIGURES_DIR / "uncertainty_summary.csv"
    summary.to_csv(out_csv, index=False)
    log.info("Summary saved → %s", out_csv)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("UNCERTAINTY SET TTP EVALUATION  (Zhang Adaptive Therapy)")
    print("=" * 75)
    print(f"  Nominal TTP (population mean μ):  {ttp_nominal:.0f} days\n")
    print(f"  {'Set':<14} {'Mean':>8} {'Median':>8} {'5th%ile':>10} {'Worst':>10} {'% < nominal':>13}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*13}")
    for _, row in summary.iterrows():
        print(f"  {row['set']:<14} {row['mean_ttp']:>8.0f} {row['median_ttp']:>8.0f} "
              f"{row['p5_ttp']:>10.0f} {row['worst_case_ttp']:>10.0f} "
              f"{row['pct_below_nominal']:>12.1f}%")
    print("=" * 75)
    print("\nInterpretation:")
    for _, row in summary.iterrows():
        print(f"  {row['set'].capitalize()} set: {row['pct_below_nominal']:.1f}% of sampled patients "
              f"have TTP < nominal ({ttp_nominal:.0f}d).")
        print(f"    Worst-case TTP = {row['worst_case_ttp']:.0f}d  "
              f"(Phase 3 will try to maximise this quantity.)")
    print(f"\n  This is the key quantity the robust optimiser will maximise in Phase 3.")
    print(f"  All plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
