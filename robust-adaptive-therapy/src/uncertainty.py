"""
Phase 2 — Uncertainty set construction for robust optimisation.

Two uncertainty sets are implemented:

1. EllipsoidalUncertaintySet
   An ellipsoid in log-parameter space centred at the population mean μ with
   shape defined by the population covariance Σ:

       U_ell = { θ : (log θ − μ)ᵀ Σ⁻¹ (log θ − μ) ≤ κ² }

   Radius κ² = χ²_α(d) ensures the set covers confidence α under the
   Gaussian assumption.  Suitable for SOCP-based robust optimisation.

2. WassersteinUncertaintySet
   A Wasserstein-1 ball of radius ε around the empirical distribution P̂_n of
   patient log-parameters:

       W_ε(P̂_n) = { Q : W₁(Q, P̂_n) ≤ ε }

   This is the core object for Phase 3 distributionally robust optimisation
   (DRO) following Mohajerin Esfahani & Kuhn (2018), Math. Programming.

Reference
---------
Mohajerin Esfahani, P. & Kuhn, D. (2018).
Data-driven distributionally robust optimization using the Wasserstein metric.
Mathematical Programming, 171(1), 115–166.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from scipy.stats import chi2

log = logging.getLogger(__name__)

PARAM_NAMES = [
    "r_plus", "r_prod", "r_minus",
    "delta_plus",
    "T0_plus", "T0_prod", "T0_minus",
]

# Natural-space bounds applied after exp(log θ) in WassersteinUncertaintySet.sample_perturbed_params
WASSERSTEIN_NATURAL_BOUNDS: dict[str, tuple[float, float]] = {
    "r_plus":    (1e-4, 0.05),
    "r_prod":    (1e-4, 0.05),
    "r_minus":   (1e-4, 0.10),
    "delta_plus": (0.01, 1.0),
    "T0_plus":   (100, 20000),
    "T0_prod":   (10, 5000),
    "T0_minus":  (1, 1000),
}


# ── EllipsoidalUncertaintySet ─────────────────────────────────────────────────

class EllipsoidalUncertaintySet:
    """
    Ellipsoidal uncertainty set in log-parameter space.

        U_ell = { θ : (log θ − μ)ᵀ Σ⁻¹ (log θ − μ) ≤ κ² }

    Parameters
    ----------
    mean_log   : population mean in log-space  (length d)
    cov_log    : population covariance in log-space  (d × d)
    confidence : probability mass the ellipsoid covers under the Gaussian
                 model.  κ² = chi2.ppf(confidence, df=d).
    param_names: names of the d parameters (default: 7-param LV names).
    """

    def __init__(
        self,
        mean_log:    np.ndarray,
        cov_log:     np.ndarray,
        confidence:  float = 0.90,
        param_names: list[str] | None = None,
    ) -> None:
        self.mean_log    = np.asarray(mean_log, dtype=float)
        self.cov_log     = np.asarray(cov_log,  dtype=float)
        self.confidence  = confidence
        self.param_names = param_names or PARAM_NAMES[:len(mean_log)]

        d = len(self.mean_log)
        self.kappa = float(np.sqrt(chi2.ppf(confidence, df=d)))

        reg = 1e-8 * np.eye(d)
        self.L       = np.linalg.cholesky(self.cov_log + reg)
        self.cov_inv = np.linalg.inv(self.cov_log + reg)

    # ------------------------------------------------------------------

    def contains(self, params: dict) -> bool:
        """Return True if params lies inside (or on the boundary of) the ellipsoid."""
        theta_log = np.array([np.log(float(params[k])) for k in self.param_names])
        diff = theta_log - self.mean_log
        return float(diff @ self.cov_inv @ diff) <= self.kappa ** 2

    def mahalanobis(self, params: dict) -> float:
        """Normalised Mahalanobis distance (= 1 on the ellipsoid boundary)."""
        theta_log = np.array([np.log(float(params[k])) for k in self.param_names])
        diff = theta_log - self.mean_log
        return float(np.sqrt(diff @ self.cov_inv @ diff)) / self.kappa

    # ------------------------------------------------------------------

    def sample_boundary(self, n_points: int = 100, seed: int = 0) -> np.ndarray:
        """
        Sample points uniformly on the boundary of the ellipsoid.

        Algorithm:
          log θ = μ + κ · L · u    where u ~ Uniform(unit sphere)

        Returns (n_points, d) array in parameter space (positive values).
        """
        rng = np.random.default_rng(seed)
        d   = len(self.mean_log)
        u   = rng.standard_normal((n_points, d))
        u  /= np.linalg.norm(u, axis=1, keepdims=True)
        log_pts = self.mean_log + self.kappa * (u @ self.L.T)
        return np.exp(log_pts)

    def sample_interior(self, n_points: int = 100, seed: int = 0) -> np.ndarray:
        """
        Sample points uniformly inside the ellipsoid.

        Uses scaled-sphere method: radius ~ Uniform([0,1])^(1/d) for
        approximate volume-uniform density.

        Returns (n_points, d) in parameter space.
        """
        rng = np.random.default_rng(seed)
        d   = len(self.mean_log)
        u   = rng.standard_normal((n_points, d))
        u  /= np.linalg.norm(u, axis=1, keepdims=True)
        r   = rng.uniform(0.0, 1.0, n_points) ** (1.0 / d)
        log_pts = self.mean_log + (r[:, None] * self.kappa) * (u @ self.L.T)
        return np.exp(log_pts)

    # ------------------------------------------------------------------

    def worst_case_direction(self, gradient: np.ndarray) -> np.ndarray:
        """
        Closed-form worst-case perturbation for a linear objective.

        Given ∇f(log θ), the point in U_ell maximising gradient · δ is:

            δ* = κ · Σ · g / ‖g‖_Σ    where ‖g‖_Σ = √(gᵀ Σ g)

        This is the steepest-ascent direction inside the Σ-shaped ellipsoid.
        """
        g   = np.asarray(gradient, dtype=float)
        Sg  = self.cov_log @ g
        nrm = np.sqrt(float(g @ Sg))
        if nrm < 1e-12:
            return np.zeros_like(g)
        return self.kappa * Sg / nrm

    def worst_case_params(self, gradient: np.ndarray) -> np.ndarray:
        """
        Return the worst-case parameter vector (in parameter space) for a
        linear objective with gradient ∇f in log-space.
        """
        delta = self.worst_case_direction(gradient)
        return np.exp(self.mean_log + delta)

    # ------------------------------------------------------------------

    def weighted_ellipsoid(
        self,
        weight_overrides: dict[int, float],
    ) -> "EllipsoidalUncertaintySet":
        """
        Return a new EllipsoidalUncertaintySet with a modified covariance matrix
        that stretches specified parameter axes to reflect greater uncertainty.

        weight_overrides : dict mapping parameter index to scale factor.
        Recommended defaults from Phase 2 (higher CV on δ₊ and T₀₋):
          ``{3: 1.5, 6: 2.0}``  (delta_plus, T0_minus).
        """
        scaled_cov = self.cov_log.copy()
        for idx, scale in weight_overrides.items():
            scaled_cov[idx, idx] *= scale
            scaled_cov[:, idx] *= np.sqrt(scale)
            scaled_cov[idx, :] *= np.sqrt(scale)
        return EllipsoidalUncertaintySet(
            self.mean_log,
            scaled_cov,
            self.confidence,
            param_names=self.param_names,
        )

    # ------------------------------------------------------------------

    def volume_fraction(self, other_set_samples: np.ndarray) -> float:
        """
        Fraction of the given samples that lie inside this ellipsoid.

        Useful for comparing how well the ellipsoidal set covers the
        Wasserstein perturbed samples.
        """
        n = 0
        for row in other_set_samples:
            params = dict(zip(self.param_names, row.tolist()))
            if self.contains(params):
                n += 1
        return n / max(len(other_set_samples), 1)


# ── WassersteinUncertaintySet ─────────────────────────────────────────────────

class WassersteinUncertaintySet:
    """
    Wasserstein-1 ball of radius ε around the empirical distribution P̂_n.

    Stores n observed patient log-parameter vectors and provides:
    - Data-driven radius selection (Mohajerin Esfahani & Kuhn 2018).
    - Worst-case expectation (mean–std approximation).
    - CVaR approximation for heavy-tailed objectives.
    - Perturbed sampling for Monte Carlo evaluation.

    Parameters
    ----------
    empirical_samples : (N, d) array of patient log-parameters.
    epsilon           : Wasserstein ball radius.  Use
                        compute_epsilon_from_confidence() if unknown.
    param_names       : names corresponding to the d columns.
    """

    def __init__(
        self,
        empirical_samples: np.ndarray,
        epsilon:           float,
        param_names:       list[str] | None = None,
    ) -> None:
        self.samples     = np.asarray(empirical_samples, dtype=float)
        self.epsilon     = float(epsilon)
        self.n, self.d   = self.samples.shape
        self.param_names = param_names or PARAM_NAMES[:self.d]

    # ------------------------------------------------------------------

    def compute_epsilon_from_confidence(
        self,
        confidence: float = 0.90,
    ) -> float:
        """
        Data-driven ε using the finite-sample bound of Theorem 3.4 in
        Mohajerin Esfahani & Kuhn (2018).

        For sub-Gaussian distributions the bound is approximately:

            ε_n ≈ C · √(log(1 / (1 − confidence)) / n)

        where C = empirical scale of the distribution (estimated as the
        mean standard deviation across dimensions × √d to account for
        the multivariate geometry).

        NOTE: This is an asymptotic approximation.  The exact bound
        depends on the support diameter and is conservative in practice.
        Calibration via cross-validation is recommended for Phase 3.
        """
        scale = float(np.std(self.samples) * np.sqrt(self.d))
        raw_epsilon = scale * np.sqrt(np.log(1.0 / (1.0 - confidence)) / max(self.n, 1))
        # Cap epsilon at 0.5 to prevent biologically impossible samples.
        # The finite-sample bound is conservative; this keeps perturbations
        # within ~0.5 log-units = roughly 1.65x multiplicative factor.
        epsilon = min(float(raw_epsilon), 0.5)
        return float(epsilon)

    # ------------------------------------------------------------------

    def sample_perturbed(
        self,
        n_samples: int = 500,
        seed:      int = 0,
    ) -> np.ndarray:
        """
        Draw n_samples parameter vectors from a Wasserstein-perturbed distribution.

        Implementation: bootstrap resample from P̂_n and add isotropic Gaussian
        noise with std = ε (Gaussian perturbation kernel).  This is a
        valid construction for Wasserstein-1 balls when the kernel is Lipschitz.

        Returns (n_samples, d) in log-space.
        """
        rng   = np.random.default_rng(seed)
        idx   = rng.integers(0, self.n, size=n_samples)
        noise = rng.standard_normal((n_samples, self.d)) * self.epsilon
        return self.samples[idx] + noise

    def sample_perturbed_params(
        self,
        n_samples: int = 500,
        seed:      int = 0,
    ) -> np.ndarray:
        """Sample perturbed parameters in parameter space (exp of log-samples)."""
        log_samp = self.sample_perturbed(n_samples, seed=seed)
        nat = np.exp(log_samp)
        n_clip = 0
        for i in range(n_samples):
            row_clipped = False
            for j, name in enumerate(self.param_names):
                lo, hi = WASSERSTEIN_NATURAL_BOUNDS[name]
                before = float(nat[i, j])
                after = float(np.clip(before, lo, hi))
                if not np.isclose(before, after):
                    nat[i, j] = after
                    row_clipped = True
            if row_clipped:
                n_clip += 1
        frac = n_clip / max(n_samples, 1)
        if frac > 0.05:
            log.warning(
                "Wasserstein sample_perturbed_params: %.1f%% of samples required bound clipping (>5%%).",
                100.0 * frac,
            )
        else:
            log.info(
                "Wasserstein sample_perturbed_params: %d/%d samples required clipping (%.1f%%).",
                n_clip,
                n_samples,
                100.0 * frac,
            )
        return nat

    # ------------------------------------------------------------------

    def worst_case_expectation(
        self,
        objective_fn: Callable[[np.ndarray], float],
        n_samples:    int = 500,
        seed:         int = 0,
    ) -> float:
        """
        Monte Carlo approximation of the worst-case expected objective.

        Approximation (mean–std, Lagrangian dual):

            sup_{Q ∈ W_ε(P̂)} E_Q[f(θ)] ≈ E_{P̂}[f(θ)] + ε · std(f(θ))

        This follows from the dual of the Wasserstein problem assuming f is
        Lipschitz (Blanchet & Murthy 2019, JRSS-B).  It is tight when ε → 0
        and n → ∞, and overestimates when f has large local curvature.

        NOTE: This is an approximation, not a rigorous upper bound in general.
        Use cvar_approximation() for a complementary robustness measure.

        Parameters
        ----------
        objective_fn : callable taking a 1-D array of log-parameters and
                       returning a scalar.
        n_samples    : how many empirical samples to evaluate on.
        seed         : not used here (evaluates on empirical samples, not perturbed).
        """
        eval_n     = min(n_samples, self.n)
        objectives = np.array([
            objective_fn(self.samples[i]) for i in range(eval_n)
        ])
        mu  = float(np.mean(objectives))
        std = float(np.std(objectives))
        return mu + self.epsilon * std

    def cvar_approximation(
        self,
        objective_fn: Callable[[np.ndarray], float],
        alpha:        float = 0.10,
        n_samples:    int   = 500,
        seed:         int   = 0,
    ) -> float:
        """
        CVaR(1−α) of the objective over Wasserstein-perturbed samples.

        A tighter worst-case measure than the mean–std approximation when
        the objective has heavy tails.

        Steps:
        1. Draw n_samples from the Wasserstein-perturbed distribution.
        2. Compute the objective for each.
        3. Return the mean of the top α-fraction of objective values.
        """
        perturbed  = self.sample_perturbed(n_samples, seed=seed)
        objectives = np.array([objective_fn(s) for s in perturbed])
        k          = max(1, int(np.ceil((1.0 - alpha) * len(objectives))))
        return float(np.mean(np.sort(objectives)[k:]))

    # ------------------------------------------------------------------

    def empirical_cdf_values(self, objective_fn: Callable) -> np.ndarray:
        """Evaluate objective_fn on all empirical samples. Returns 1-D array."""
        return np.array([objective_fn(s) for s in self.samples])

    # ------------------------------------------------------------------

    def to_ellipsoidal(self, confidence: float = 0.90) -> "EllipsoidalUncertaintySet":
        """
        Fit an EllipsoidalUncertaintySet to the empirical samples.

        This converts the Wasserstein set to an ellipsoid with the same
        centre and covariance, useful for comparing the two representations.
        """
        mean = self.samples.mean(axis=0)
        cov  = np.cov(self.samples.T) if self.n > 1 else np.eye(self.d) * 0.25
        return EllipsoidalUncertaintySet(
            mean_log=mean,
            cov_log=cov,
            confidence=confidence,
            param_names=self.param_names,
        )
