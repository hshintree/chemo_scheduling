"""
Plotting, metrics, and survival analysis utilities.

All plot functions return the Figure object so callers can save or display it.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Color palette (color-blind-friendly)
# ---------------------------------------------------------------------------
COLORS = {
    "T_plus":  "#1f77b4",   # blue
    "T_prod":  "#ff7f0e",   # orange
    "T_minus": "#d62728",   # red
    "total":   "#2ca02c",   # green
    "psa":     "#9467bd",   # purple
    "drug":    "#8c564b",   # brown
    "mtd":     "#e377c2",
    "adaptive": "#17becf",
    "rbat":    "#bcbd22",
    "oracle":  "#7f7f7f",
}

PROTOCOL_COLORS = {
    "mtd":      COLORS["mtd"],
    "adaptive": COLORS["adaptive"],
    "rbat":     COLORS["rbat"],
    "oracle":   COLORS["oracle"],
}

PROTOCOL_LABELS = {
    "mtd":      "MTD (continuous)",
    "adaptive": "Zhang AT (50%/100%)",
    "rbat":     "Range-bounded AT (30%/80%)",
    "oracle":   "Oracle optimal",
}


# ---------------------------------------------------------------------------
# Single-patient trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory(
    sim: dict,
    title: str = "Patient trajectory",
    normalize: bool = True,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot cell populations, total, and drug schedule for one simulation.

    Parameters
    ----------
    sim : dict from LotkaVolterraModel.simulate()
    title : str
    normalize : bool
        If True, normalize all populations to initial total cell count.
    figsize : (width, height)

    Returns
    -------
    matplotlib.figure.Figure
    """
    t           = sim["t"]
    T_plus      = sim["T_plus"]
    T_prod      = sim["T_prod"]
    T_minus     = sim["T_minus"]
    total       = sim["total_cells"]
    drug        = sim["drug"]

    if normalize:
        scale = total[0] if total[0] > 0 else 1.0
        T_plus  = T_plus  / scale
        T_prod  = T_prod  / scale
        T_minus = T_minus / scale
        total   = total   / scale
        ylabel  = "Cells (normalized to baseline)"
    else:
        ylabel = "Cell count"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    gridspec_kw={"height_ratios": [4, 1]},
                                    sharex=True)

    ax1.plot(t, T_plus,  color=COLORS["T_plus"],  lw=1.5, label="T⁺ (sensitive)")
    ax1.plot(t, T_prod,  color=COLORS["T_prod"],  lw=1.5, label="Tᵖ (producing)")
    ax1.plot(t, T_minus, color=COLORS["T_minus"], lw=1.5, label="T⁻ (resistant)")
    ax1.plot(t, total,   color=COLORS["total"],   lw=2.0, ls="--", label="Total")
    ax1.set_ylabel(ylabel)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Drug schedule as filled step
    ax2.fill_between(t, drug, step="post",
                     color=COLORS["drug"], alpha=0.6, label="Abiraterone")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Off", "On"])
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Drug")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PSA overlay plot
# ---------------------------------------------------------------------------

def plot_psa(
    t: np.ndarray,
    psa_model: np.ndarray,
    psa_observed: Optional[np.ndarray] = None,
    t_observed: Optional[np.ndarray] = None,
    drug: Optional[np.ndarray] = None,
    title: str = "PSA trajectory",
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Plot modeled (and optionally observed) PSA with optional drug schedule.
    """
    nrows = 2 if drug is not None else 1
    height_ratios = [4, 1] if drug is not None else [1]
    fig, axes = plt.subplots(
        nrows, 1, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
        squeeze=False,
    )
    ax1 = axes[0, 0]

    ax1.plot(t, psa_model, color=COLORS["psa"], lw=2, label="Model PSA")
    if psa_observed is not None and t_observed is not None:
        ax1.scatter(t_observed, psa_observed, color="black", zorder=5,
                    s=20, label="Observed PSA", marker="o")
    ax1.axhline(1.0, color="gray", ls=":", lw=1, label="Baseline")
    ax1.set_ylabel("PSA (normalized)")
    ax1.legend(fontsize=9)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    if drug is not None:
        ax2 = axes[1, 0]
        ax2.fill_between(t, drug, step="post",
                         color=COLORS["drug"], alpha=0.6)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Off", "On"])
        ax2.set_ylabel("Drug")
        ax2.set_xlabel("Time (days)")
        ax2.grid(True, alpha=0.3)
    else:
        axes[0, 0].set_xlabel("Time (days)")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Kaplan-Meier estimator
# ---------------------------------------------------------------------------

def kaplan_meier(
    times: np.ndarray,
    events: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple Kaplan-Meier survival estimator.

    Parameters
    ----------
    times : 1-D ndarray
        Time-to-event (or censoring) for each subject.
    events : 1-D ndarray of bool/int, optional
        1 = event occurred, 0 = censored.  Default: all events observed.

    Returns
    -------
    t_km : ndarray of event times
    S_km : ndarray of survival probabilities
    """
    times = np.asarray(times, dtype=float)
    if events is None:
        events = np.ones(len(times), dtype=int)
    else:
        events = np.asarray(events, dtype=int)

    n = len(times)
    order = np.argsort(times)
    sorted_t = times[order]
    sorted_e = events[order]

    t_km = [0.0]
    S_km = [1.0]
    S = 1.0
    at_risk = n

    i = 0
    while i < n:
        t_i = sorted_t[i]
        # Collect all tied events at this time
        j = i
        d = 0
        while j < n and sorted_t[j] == t_i:
            d += sorted_e[j]
            j += 1
        if d > 0:
            S *= (1.0 - d / at_risk)
            t_km.append(t_i)
            S_km.append(S)
        at_risk -= (j - i)
        i = j

    return np.array(t_km), np.array(S_km)


def plot_km_curves(
    ttp_by_protocol: dict[str, np.ndarray],
    title: str = "Kaplan-Meier: Time to Progression",
    figsize: tuple[float, float] = (9, 6),
) -> plt.Figure:
    """
    Plot KM survival curves for multiple protocols.

    Parameters
    ----------
    ttp_by_protocol : dict mapping protocol_name -> array of TTP values
        Use np.inf for patients who did not progress.
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for protocol, ttps in ttp_by_protocol.items():
        finite_ttps = np.where(np.isinf(ttps), np.nanmax(ttps[~np.isinf(ttps)]) * 1.05
                               if np.any(~np.isinf(ttps)) else 2000.0, ttps)
        events = (~np.isinf(ttps)).astype(int)
        t_km, S_km = kaplan_meier(finite_ttps, events)

        color = PROTOCOL_COLORS.get(protocol, None)
        label = PROTOCOL_LABELS.get(protocol, protocol)
        ax.step(t_km, S_km, where="post", lw=2, color=color, label=label)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Progression-free survival")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scatter: adaptive TTP vs MTD TTP
# ---------------------------------------------------------------------------

def plot_ttp_scatter(
    ttp_adaptive: np.ndarray,
    ttp_mtd: np.ndarray,
    max_days: float = 1825,
    title: str = "TTP: Adaptive vs MTD",
    figsize: tuple[float, float] = (6, 6),
) -> plt.Figure:
    """
    Scatter plot comparing time-to-progression under adaptive therapy vs MTD.
    Points above the diagonal indicate adaptive therapy benefit.
    """
    cap = max_days
    ta = np.minimum(ttp_adaptive, cap)
    tm = np.minimum(ttp_mtd, cap)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(tm, ta, alpha=0.7, edgecolors="k", linewidths=0.5, color=COLORS["adaptive"])
    diag = np.linspace(0, cap, 200)
    ax.plot(diag, diag, "k--", lw=1, label="y = x (no benefit)")
    ax.set_xlabel("TTP under MTD (days)")
    ax.set_ylabel("TTP under Adaptive Therapy (days)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, cap * 1.05)
    ax.set_ylim(0, cap * 1.05)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Box plots of cumulative dose
# ---------------------------------------------------------------------------

def plot_dose_boxplot(
    dose_by_protocol: dict[str, np.ndarray],
    title: str = "Cumulative Drug-On Days by Protocol",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Side-by-side box plots of cumulative treatment days per protocol.
    """
    protocols = list(dose_by_protocol.keys())
    data      = [dose_by_protocol[p] for p in protocols]
    labels    = [PROTOCOL_LABELS.get(p, p) for p in protocols]
    colors    = [PROTOCOL_COLORS.get(p, "gray") for p in protocols]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(protocols) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Cumulative days on abiraterone")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def compute_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination R²."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def summary_table(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of per-patient result dicts to a formatted DataFrame.

    Each dict should have keys like: patient_id, protocol, ttp, drug_days, ...
    """
    return pd.DataFrame(results).sort_values(["patient_id", "protocol"]).reset_index(drop=True)
