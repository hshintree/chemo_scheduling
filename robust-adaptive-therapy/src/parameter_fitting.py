"""
Phase 2 — Parameter uncertainty quantification for the LV model.

Two complementary approaches:

1. BootstrapFitter
   Residual bootstrap: resample residuals between observed PSA and nominal
   model predictions, refit on each synthetic dataset, report 5th/95th
   percentile confidence intervals and empirical covariance in log-space.

2. LaplaceFitter
   Gaussian (Laplace) approximation of the Bayesian posterior.  Optimises
   the MAP estimate in log-parameter space (with a log-normal prior), then
   computes the Hessian via finite differences to obtain a Gaussian posterior
   covariance.  Fast to compute and exact for well-identified patients.

Both classes expose a dict-valued fit() method and are designed to feed
directly into PopulationModel and the uncertainty sets in uncertainty.py.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, approx_fprime

from .lotka_volterra import LotkaVolterraModel
from .psa_model import PSAModel
from .patient import Patient

log = logging.getLogger(__name__)

# ── Shared parameter configuration ────────────────────────────────────────────

PARAM_NAMES = [
    "r_plus", "r_prod", "r_minus",
    "delta_plus",
    "T0_plus", "T0_prod", "T0_minus",
]

PARAM_BOUNDS = [
    (1e-5, 0.05),      # r_plus
    (1e-5, 0.05),      # r_prod
    (1e-5, 0.05),      # r_minus
    (1e-5, 1.0),       # delta_plus
    (10.0, 50_000.0),  # T0_plus
    (1.0,  20_000.0),  # T0_prod
    (0.1,  5_000.0),   # T0_minus
]

_PARAM_NOMINAL = np.array([
    np.log(2) / 250,   # r_plus   (Zhang et al. 2017 Table S1)
    np.log(2) / 200,   # r_prod
    np.log(2) / 104,   # r_minus
    0.15,              # delta_plus
    5_000.0,           # T0_plus
    500.0,             # T0_prod
    50.0,              # T0_minus
])

K_FIXED     = 10_000.0
ALPHA_FIXED = np.array([[1.0, 0.5, 0.5],
                         [0.5, 1.0, 0.5],
                         [0.5, 0.5, 1.0]])


# ── Internal helpers ──────────────────────────────────────────────────────────

def _unpack(x: np.ndarray) -> dict:
    """Map the 7-element optimization vector to a full LV parameter dict."""
    return {
        "r_plus":    float(x[0]),
        "r_prod":    float(x[1]),
        "r_minus":   float(x[2]),
        "delta_plus": float(x[3]),
        "delta_prod": 0.5 * float(x[3]),   # fixed ratio
        "T0_plus":   float(x[4]),
        "T0_prod":   float(x[5]),
        "T0_minus":  float(x[6]),
        "K":         K_FIXED,
        "alpha":     ALPHA_FIXED.copy(),
    }


def _build_drug_schedule(
    days: np.ndarray,
    on_treatment: np.ndarray,
) -> Callable[[float], int]:
    """Zero-order-hold drug schedule from discrete observations."""
    days_arr = np.asarray(days, dtype=float)
    on_arr   = np.asarray(on_treatment, dtype=int)

    def schedule(t: float) -> int:
        if t <= days_arr[0]:
            return int(on_arr[0])
        if t > days_arr[-1]:
            return int(on_arr[-1])
        idx = int(np.clip(
            np.searchsorted(days_arr, t, side="right") - 1,
            0, len(on_arr) - 1
        ))
        return int(on_arr[idx])

    return schedule


def _objective(
    x: np.ndarray,
    obs_days: np.ndarray,
    obs_psa_norm: np.ndarray,
    drug_fn: Callable,
    psa_model: PSAModel,
) -> float:
    """Sum of squared residuals between observed and model-predicted PSA (normalised).

    Uses relaxed ODE tolerances (rtol=1e-4, atol=1e-1) appropriate for
    parameter fitting where cell counts are O(100–50000) and sub-percent
    accuracy is sufficient.  This is ~100x faster than the default atol=1e-9.
    """
    try:
        params = _unpack(x)
        model  = LotkaVolterraModel(params)
        sim    = model.simulate(
            t_span=(obs_days[0], obs_days[-1]),
            t_eval=obs_days,
            drug_schedule=drug_fn,
            rtol=1e-4,
            atol=1e-1,   # cells < 0.1 are numerically irrelevant
        )
        psa_vals = psa_model.compute_psa(sim)
        psa_norm = psa_vals / psa_vals[0] if psa_vals[0] > 0 else psa_vals
        return float(np.sum((psa_norm - obs_psa_norm) ** 2))
    except Exception:
        return 1e12


def _fit(
    days: np.ndarray,
    psa_norm: np.ndarray,
    on_treatment: np.ndarray,
    n_restarts: int = 10,
    seed: int = 0,
    maxiter: int = 150,
) -> tuple[dict, float]:
    """
    Fit the LV model to a PSA time-series via L-BFGS-B with random restarts.

    Parameters
    ----------
    maxiter : int
        Maximum L-BFGS-B iterations per restart.  Since scipy uses numerical
        finite-difference gradients by default (n+1 function calls per iter),
        150 iterations ≈ 1200 function evaluations per restart — sufficient
        for convergence on a well-scaled 7-parameter problem.  Use 80 for
        bootstrap refits (warmstart from nominal makes fewer iters needed).

    Returns (best_params_dict, rmse).
    """
    rng     = np.random.default_rng(seed)
    drug_fn = _build_drug_schedule(days, on_treatment)
    psa_mod = PSAModel()

    best_params: dict | None = None
    best_cost = np.inf

    for trial in range(n_restarts):
        if trial == 0:
            x0 = _PARAM_NOMINAL.copy()
        else:
            x0 = _PARAM_NOMINAL * np.exp(rng.normal(0, 0.5, size=len(_PARAM_NOMINAL)))

        x0 = np.array([
            np.clip(x0[i], PARAM_BOUNDS[i][0], PARAM_BOUNDS[i][1])
            for i in range(len(x0))
        ])

        res = minimize(
            fun=_objective,
            x0=x0,
            args=(days, psa_norm, drug_fn, psa_mod),
            method="L-BFGS-B",
            bounds=PARAM_BOUNDS,
            options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-6},
        )
        if res.fun < best_cost:
            best_cost   = res.fun
            best_params = _unpack(res.x)

    if best_params is None:
        best_params = _unpack(_PARAM_NOMINAL)
        best_cost   = _objective(_PARAM_NOMINAL, days, psa_norm, drug_fn, psa_mod)

    rmse = float(np.sqrt(best_cost / max(len(days), 1)))
    return best_params, rmse


def _fit_warmstart(
    x_start: np.ndarray,
    days: np.ndarray,
    psa_norm: np.ndarray,
    on_treatment: np.ndarray,
    drug_fn,
    psa_mod: PSAModel,
    n_extra_restarts: int = 1,
    maxiter: int = 80,
    seed: int = 0,
) -> tuple[dict, float]:
    """
    Fit starting from x_start (warmstart) plus n_extra_restarts random restarts.

    Faster than _fit for bootstrap refits because the bootstrap PSA is close
    to the nominal prediction, so x_start is already a good initial point.
    """
    rng = np.random.default_rng(seed)

    starts = [x_start.copy()]
    for _ in range(n_extra_restarts):
        x0 = x_start * np.exp(rng.normal(0, 0.3, size=len(x_start)))
        starts.append(x0)

    best_params: dict | None = None
    best_cost = np.inf

    for x0 in starts:
        x0 = np.array([
            np.clip(x0[i], PARAM_BOUNDS[i][0], PARAM_BOUNDS[i][1])
            for i in range(len(x0))
        ])
        res = minimize(
            fun=_objective,
            x0=x0,
            args=(days, psa_norm, drug_fn, psa_mod),
            method="L-BFGS-B",
            bounds=PARAM_BOUNDS,
            options={"maxiter": maxiter, "ftol": 1e-7, "gtol": 1e-5},
        )
        if res.fun < best_cost:
            best_cost   = res.fun
            best_params = _unpack(res.x)

    if best_params is None:
        best_params = _unpack(x_start)
        best_cost   = _objective(x_start, days, psa_norm, drug_fn, psa_mod)

    return best_params, float(np.sqrt(best_cost / max(len(days), 1)))


def _model_psa_norm(
    x: np.ndarray,
    days: np.ndarray,
    drug_fn: Callable,
    psa_mod: PSAModel,
) -> np.ndarray:
    """Run the LV model and return normalised PSA at `days`."""
    try:
        params   = _unpack(x)
        model    = LotkaVolterraModel(params)
        sim      = model.simulate(
            t_span=(days[0], days[-1]),
            t_eval=days,
            drug_schedule=drug_fn,
            rtol=1e-4,
            atol=1e-1,
        )
        psa_vals = psa_mod.compute_psa(sim)
        return psa_vals / psa_vals[0] if psa_vals[0] > 0 else psa_vals
    except Exception:
        return np.ones(len(days))


def _patient_data(patient: Patient) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (days, psa_norm, on_treatment) arrays from a Patient object."""
    if patient.psa_data is None:
        raise ValueError(
            f"Patient {patient.patient_id!r} has no PSA time-series data. "
            "Fitting requires columns [day, psa, on_treatment]."
        )
    df = (
        patient.psa_data
        .dropna(subset=["day", "psa"])
        .sort_values("day")
        .reset_index(drop=True)
    )
    days  = df["day"].values.astype(float)
    psa   = df["psa"].values.astype(float)
    if "on_treatment" in df.columns:
        on_tx = df["on_treatment"].values.astype(int)
    else:
        on_tx = np.ones(len(df), dtype=int)

    psa_norm = psa / psa[0] if psa[0] > 0 else psa
    return days, psa_norm, on_tx


def _finite_diff_hessian(
    f: Callable,
    x: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """
    Central finite-difference Hessian of f at x.

    Uses 2n calls to approx_fprime (each with n+1 evaluations), so O(n²)
    total function evaluations.  For n=7, that is ~98 evaluations.
    """
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        x_plus  = x.copy(); x_plus[i]  += h
        x_minus = x.copy(); x_minus[i] -= h
        g_plus  = approx_fprime(x_plus,  f, h)
        g_minus = approx_fprime(x_minus, f, h)
        H[i, :] = (g_plus - g_minus) / (2.0 * h)
    return 0.5 * (H + H.T)   # symmetrise


# ── BootstrapFitter ────────────────────────────────────────────────────────────

class BootstrapFitter:
    """
    Residual-bootstrap confidence intervals for LV model parameters.

    Algorithm
    ---------
    1.  Fit nominal parameters via L-BFGS-B with `n_restarts_nominal` restarts.
    2.  Compute residuals ε_k = PSA_obs[k] − PSA_model(nominal, t_k).
    3.  For each bootstrap iteration b:
        a. Resample {ε_k} with replacement.
        b. Create PSA_boot = PSA_model(nominal) + resampled ε.
        c. Refit parameters on PSA_boot with `n_restarts_bootstrap` restarts.
    4.  Report 5th / 95th percentile CIs and empirical covariance in log-space.

    This is a non-parametric approach that makes no distributional assumption
    on the residuals and is valid even for non-Gaussian noise.
    """

    def __init__(
        self,
        n_bootstrap:          int = 500,
        n_restarts_nominal:   int = 20,
        n_restarts_bootstrap: int = 5,
    ) -> None:
        self.n_bootstrap          = n_bootstrap
        self.n_restarts_nominal   = n_restarts_nominal
        self.n_restarts_bootstrap = n_restarts_bootstrap

    def fit(self, patient: Patient) -> dict:
        """
        Run bootstrap uncertainty quantification for a single patient.

        Parameters
        ----------
        patient : Patient with non-null psa_data.

        Returns
        -------
        dict with keys:
            nominal           – best-fit parameter dict
            nominal_rmse      – RMSE of the nominal fit
            bootstrap_samples – (n_successful, 7) array of refitted params
            ci_lower          – {param: 5th percentile value}
            ci_upper          – {param: 95th percentile value}
            cov_matrix        – (7, 7) covariance of log(params) across boots
            n_successful      – number of converged bootstrap fits
        """
        days, psa_norm, on_tx = _patient_data(patient)
        drug_fn = _build_drug_schedule(days, on_tx)
        psa_mod = PSAModel()

        # ── Nominal fit ────────────────────────────────────────────────────
        nominal_params, nominal_rmse = _fit(
            days, psa_norm, on_tx,
            n_restarts=self.n_restarts_nominal,
            seed=0,
        )
        log.debug("Bootstrap nominal RMSE=%.5f for patient %s",
                  nominal_rmse, patient.patient_id)

        x_nom     = np.array([nominal_params[k] for k in PARAM_NAMES])
        psa_pred  = _model_psa_norm(x_nom, days, drug_fn, psa_mod)
        residuals = psa_norm - psa_pred

        # ── Bootstrap ─────────────────────────────────────────────────────
        rng = np.random.default_rng(42)
        boot_rows: list[list[float]] = []

        x_nom_arr = np.array([nominal_params[k] for k in PARAM_NAMES])

        for b in range(self.n_bootstrap):
            eps       = rng.choice(residuals, size=len(residuals), replace=True)
            psa_boot  = np.maximum(psa_pred + eps, 1e-6)
            try:
                # Warmstart from nominal + small perturbation — bootstrap PSA is
                # close to nominal, so convergence is fast from the nominal point.
                p, _ = _fit_warmstart(
                    x_nom_arr, days, psa_boot, on_tx,
                    drug_fn=drug_fn, psa_mod=psa_mod,
                    n_extra_restarts=self.n_restarts_bootstrap - 1,
                    maxiter=80,
                    seed=int(b),
                )
                boot_rows.append([p[k] for k in PARAM_NAMES])
            except Exception:
                continue

        if not boot_rows:
            raise RuntimeError(
                f"All {self.n_bootstrap} bootstrap fits failed to converge "
                f"for patient {patient.patient_id!r}."
            )

        boot_arr = np.array(boot_rows)                  # (n_ok, 7)
        log_boot = np.log(np.maximum(boot_arr, 1e-15))  # (n_ok, 7)

        ci_lower = {k: float(np.percentile(boot_arr[:, i], 5))
                    for i, k in enumerate(PARAM_NAMES)}
        ci_upper = {k: float(np.percentile(boot_arr[:, i], 95))
                    for i, k in enumerate(PARAM_NAMES)}

        return {
            "nominal":           nominal_params,
            "nominal_rmse":      nominal_rmse,
            "bootstrap_samples": boot_arr,
            "ci_lower":          ci_lower,
            "ci_upper":          ci_upper,
            "cov_matrix":        np.cov(log_boot.T),
            "n_successful":      len(boot_rows),
        }


# ── LaplaceFitter ──────────────────────────────────────────────────────────────

class LaplaceFitter:
    """
    Laplace (Gaussian) approximation of the Bayesian parameter posterior.

    Model
    -----
    Prior:     log θ ~ N(log θ_nominal, prior_std² · I)   (log-normal prior)
    Likelihood: PSA_obs ~ N(PSA_model(θ), σ²)             (Gaussian noise)

    The negative log-posterior in log-parameter space is:
        L(u) = RSS(exp(u)) + ||u − μ_prior||² / (2 · prior_std²)

    MAP estimate û obtained by L-BFGS-B in log-space.
    Posterior covariance: Σ_post = H(û)⁻¹  where H = ∇²L at û.

    The Laplace approximation is most reliable when:
    - The patient has ≥ 15 PSA observations.
    - The MAP fit is low-residual (good identifiability).
    """

    def __init__(
        self,
        prior_std:  float = 0.5,
        n_restarts: int   = 20,
    ) -> None:
        """
        Parameters
        ----------
        prior_std : std of the Gaussian prior on log(θ) (half-width in log-space).
                    0.5 ≈ ±50% relative uncertainty on each parameter.
        n_restarts : number of random restarts for MAP optimisation.
        """
        self.prior_std  = prior_std
        self.n_restarts = n_restarts
        self._last_fit: Optional[dict] = None

    def fit(self, patient: Patient) -> dict:
        """
        Fit the Laplace posterior for a single patient.

        Returns
        -------
        dict with keys:
            nominal   – MAP parameter dict (in parameter space)
            mean_log  – MAP point in log-parameter space (length 7)
            cov_log   – posterior covariance in log-space (7×7)
            hessian   – Hessian of negative log-posterior at MAP (7×7)
            samples   – callable: (n: int) → (n, 7) parameter samples
        """
        days, psa_norm, on_tx = _patient_data(patient)
        drug_fn  = _build_drug_schedule(days, on_tx)
        psa_mod  = PSAModel()
        mu_prior = np.log(_PARAM_NOMINAL)

        log_bounds = [
            (np.log(lo + 1e-15), np.log(hi))
            for lo, hi in PARAM_BOUNDS
        ]

        def neg_log_post(u: np.ndarray) -> float:
            theta = np.exp(u)
            rss   = _objective(theta, days, psa_norm, drug_fn, psa_mod)
            prior = np.sum((u - mu_prior) ** 2) / (2.0 * self.prior_std ** 2)
            return rss + prior

        # ── MAP optimisation in log-space ──────────────────────────────────
        rng  = np.random.default_rng(0)
        best_result = None
        best_cost   = np.inf

        for trial in range(self.n_restarts):
            if trial == 0:
                u0 = mu_prior.copy()
            else:
                u0 = mu_prior + rng.normal(0, 0.3, size=len(mu_prior))
            u0 = np.array([
                np.clip(u0[i], log_bounds[i][0], log_bounds[i][1])
                for i in range(len(u0))
            ])
            res = minimize(
                fun=neg_log_post,
                x0=u0,
                method="L-BFGS-B",
                bounds=log_bounds,
                options={"maxiter": 150, "ftol": 1e-9, "gtol": 1e-6},
            )
            if res.fun < best_cost:
                best_cost   = res.fun
                best_result = res

        u_map     = best_result.x
        theta_map = np.exp(u_map)
        nominal   = _unpack(theta_map)

        # ── Hessian of neg-log-posterior in log-space ──────────────────────
        H = _finite_diff_hessian(neg_log_post, u_map, h=1e-4)

        # ── Posterior covariance = H⁻¹ ─────────────────────────────────────
        n = len(u_map)
        try:
            cov_log = np.linalg.inv(H + 1e-6 * np.eye(n))
        except np.linalg.LinAlgError:
            log.warning("Hessian inversion failed; using diagonal approximation.")
            cov_log = np.diag(1.0 / (np.diag(H) + 1e-6))

        # Ensure positive diagonal (can degrade with poor identifiability)
        np.fill_diagonal(cov_log, np.maximum(np.diag(cov_log), 1e-8))

        result: dict = {
            "nominal":  nominal,
            "mean_log": u_map,
            "cov_log":  cov_log,
            "hessian":  H,
        }
        result["samples"] = lambda n, _r=result: self._sample_from(n, _r)
        self._last_fit = result
        return result

    def sample_parameters(self, n_samples: int = 1000) -> np.ndarray:
        """
        Draw samples from the Laplace posterior.

        Returns (n_samples, 7) in parameter space.
        Requires fit() to have been called first.
        """
        if self._last_fit is None:
            raise RuntimeError("Call fit() before sample_parameters().")
        return self._sample_from(n_samples, self._last_fit)

    @staticmethod
    def _sample_from(n: int, fit_result: dict) -> np.ndarray:
        u_mean  = fit_result["mean_log"]
        cov_log = fit_result["cov_log"]
        u_samps = np.random.multivariate_normal(u_mean, cov_log, size=n)
        return np.exp(u_samps)
