"""
Phase 2 — Population-level parameter distribution.

PopulationModel fits a multivariate Gaussian to the log-transformed
per-patient parameters (i.e. a log-normal population model).

This is the empirical distribution P̂_n that Phase 3 will robustify against:
the Wasserstein ball is centred here.

Usage example
-------------
    from src.population_model import PopulationModel
    import pandas as pd

    df = pd.read_csv("data/synthetic/population_params.csv")
    records = df[PopulationModel.param_names].to_dict(orient="records")

    model = PopulationModel()
    result = model.fit(records)

    print(result["marginal_ks_pvalues"])  # check log-normality
    samples = model.sample(100)           # 100 virtual patients
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm as sp_norm

log = logging.getLogger(__name__)

PARAM_NAMES = [
    "r_plus", "r_prod", "r_minus",
    "delta_plus",
    "T0_plus", "T0_prod", "T0_minus",
]

PARAM_LABELS = {
    "r_plus":    "r₊  (growth, T+)",
    "r_prod":    "rₚ  (growth, Tp)",
    "r_minus":   "r₋  (growth, T−)",
    "delta_plus": "δ₊  (drug kill)",
    "T0_plus":   "T₀₊ (init T+)",
    "T0_prod":   "T₀ₚ (init Tp)",
    "T0_minus":  "T₀₋ (init T−)",
}


class PopulationModel:
    """
    Multivariate log-normal population model for LV parameters.

    The model assumes:
        log(θ) ~ N(μ, Σ)

    where μ and Σ are estimated from the observed cohort by maximum
    likelihood (sample mean and covariance of log-parameters).

    Provides:
    - KS test for log-normality of each marginal.
    - Sampling of virtual patients.
    - Mahalanobis distance for outlier identification and Wasserstein ball sizing.
    - Summary statistics table.
    """

    param_names = PARAM_NAMES

    def __init__(self) -> None:
        self.mean_log:     Optional[np.ndarray] = None
        self.cov_log:      Optional[np.ndarray] = None
        self.diag_cov_log: Optional[np.ndarray] = None
        self.n_patients:   int = 0
        self._cov_inv:     Optional[np.ndarray] = None
        self._X_log:       Optional[np.ndarray] = None
        self._fitted:      bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, patient_params: list[dict]) -> dict:
        """
        Fit a multivariate Gaussian to the log-parameter matrix.

        Parameters
        ----------
        patient_params : list[dict]
            Each dict must contain keys matching `param_names`.
            All values must be strictly positive.

        Returns
        -------
        dict with keys:
            mean_log            – population mean in log-space  (length 7)
            cov_log             – population covariance in log-space  (7×7)
            diag_cov_log        – diagonal-only covariance (7×7)
            marginal_ks_pvalues – KS test p-values per parameter (dict)
            n_patients          – number of patients used
        """
        if not patient_params:
            raise ValueError("patient_params must be a non-empty list.")

        n = len(self.param_names)

        # Build log-parameter matrix (N × 7)
        try:
            X_log = np.array([
                [np.log(float(p[k])) for k in self.param_names]
                for p in patient_params
            ])
        except KeyError as e:
            raise ValueError(
                f"Patient parameter dict is missing required key {e}. "
                f"Required keys: {self.param_names}"
            ) from e

        self._X_log      = X_log
        self.n_patients  = X_log.shape[0]
        self.mean_log    = X_log.mean(axis=0)

        if self.n_patients > 1:
            self.cov_log = np.cov(X_log.T)
        else:
            # Degenerate case: single patient → use prior variance
            self.cov_log = np.diag(np.full(n, 0.25))

        self.diag_cov_log = np.diag(np.diag(self.cov_log))

        # Regularised inverse for Mahalanobis distance
        reg = 1e-8 * np.eye(n)
        try:
            self._cov_inv = np.linalg.inv(self.cov_log + reg)
        except np.linalg.LinAlgError:
            self._cov_inv = np.diag(1.0 / (np.diag(self.cov_log) + 1e-8))

        # KS test: does each marginal follow a Gaussian in log-space?
        ks_pvalues: dict[str, float] = {}
        for i, name in enumerate(self.param_names):
            col = X_log[:, i]
            mu  = float(col.mean())
            std = float(col.std(ddof=1)) if self.n_patients > 1 else 1.0
            std = max(std, 1e-12)
            _, pval = kstest(col, lambda x, m=mu, s=std: sp_norm.cdf(x, m, s))
            ks_pvalues[name] = float(pval)

        self._fitted = True

        return {
            "mean_log":            self.mean_log,
            "cov_log":             self.cov_log,
            "diag_cov_log":        self.diag_cov_log,
            "marginal_ks_pvalues": ks_pvalues,
            "n_patients":          self.n_patients,
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw n_samples virtual patients from the fitted population model.

        Parameters
        ----------
        n_samples : int
        seed      : random seed for reproducibility.

        Returns
        -------
        (n_samples, 7) array in parameter space (exp of log-samples).
        """
        self._check_fitted()
        rng         = np.random.default_rng(seed)
        log_samples = rng.multivariate_normal(self.mean_log, self.cov_log,
                                              size=n_samples)
        return np.exp(log_samples)

    def sample_as_dicts(
        self,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> list[dict]:
        """Return samples as a list of parameter dicts."""
        samples = self.sample(n_samples, seed=seed)
        return [
            dict(zip(self.param_names, row.tolist()))
            for row in samples
        ]

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------

    def mahalanobis_distance(self, params: dict) -> float:
        """
        Mahalanobis distance of a parameter vector from the population mean.

        D(θ) = sqrt( (log θ − μ)ᵀ Σ⁻¹ (log θ − μ) )

        Used to:
        - Identify outlier patients (large D → atypical).
        - Size the Wasserstein ball radius ε in Phase 3.
        """
        self._check_fitted()
        theta_log = np.array([np.log(float(params[k])) for k in self.param_names])
        diff = theta_log - self.mean_log
        return float(np.sqrt(diff @ self._cov_inv @ diff))

    def all_mahalanobis(self) -> np.ndarray:
        """Return Mahalanobis distances for all fitted patients."""
        self._check_fitted()
        diffs = self._X_log - self.mean_log
        return np.sqrt(np.einsum("ni,ij,nj->n", diffs, self._cov_inv, diffs))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def log_samples(self) -> np.ndarray:
        """Return the (N, 7) log-parameter matrix used for fitting."""
        self._check_fitted()
        return self._X_log.copy()

    def summary_dataframe(self) -> pd.DataFrame:
        """
        Summary table of population statistics in parameter space.

        Returns a DataFrame indexed by parameter name with columns:
        mean, std, p5, p25, p50, p75, p95, cv_pct.
        """
        self._check_fitted()
        X = np.exp(self._X_log)
        rows = []
        for i, name in enumerate(self.param_names):
            col = X[:, i]
            rows.append({
                "parameter": name,
                "label":     PARAM_LABELS.get(name, name),
                "mean":      float(col.mean()),
                "std":       float(col.std(ddof=1)) if self.n_patients > 1 else 0.0,
                "p5":        float(np.percentile(col, 5)),
                "p25":       float(np.percentile(col, 25)),
                "p50":       float(np.percentile(col, 50)),
                "p75":       float(np.percentile(col, 75)),
                "p95":       float(np.percentile(col, 95)),
                "cv_%":      float(col.std(ddof=1) / col.mean() * 100)
                             if col.mean() > 0 and self.n_patients > 1 else 0.0,
            })
        return pd.DataFrame(rows).set_index("parameter")

    def correlation_matrix(self) -> pd.DataFrame:
        """Return correlation matrix of log-parameters as a labelled DataFrame."""
        self._check_fitted()
        std = np.sqrt(np.diag(self.cov_log))
        std[std < 1e-12] = 1.0
        corr = self.cov_log / np.outer(std, std)
        return pd.DataFrame(corr, index=self.param_names, columns=self.param_names)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "PopulationModel has not been fitted. Call fit() first."
            )
