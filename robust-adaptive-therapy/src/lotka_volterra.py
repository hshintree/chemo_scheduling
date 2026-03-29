"""
Three-species competitive Lotka-Volterra ODE model for adaptive therapy.

References
----------
Zhang J, et al. (2017). Integrating evolutionary dynamics into treatment of
metastatic castration-resistant prostate cancer. Nature Communications, 8, 1816.
https://doi.org/10.1038/s41467-017-01968-5
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable


# ---------------------------------------------------------------------------
# Default published parameters (Zhang et al. 2017, Table S1)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    # Growth rates (day^-1) from published doubling times
    "r_plus": np.log(2) / 250,    # T+:  doubling time 250 days  → 0.002773
    "r_prod": np.log(2) / 200,    # Tp:  doubling time 200 days  → 0.003466
    "r_minus": np.log(2) / 104,   # T-:  doubling time 104 days  → 0.006668

    # Shared carrying capacity (arbitrary units)
    "K": 10_000.0,

    # Competition matrix alpha[i, j]: effect of species j on species i
    # Rows/cols: 0=T+, 1=Tp, 2=T-
    "alpha": np.array([
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
    ]),

    # Drug kill rates (day^-1) under full abiraterone treatment
    "delta_plus": 0.15,    # T+ cells: highly sensitive
    "delta_prod": 0.075,   # Tp cells: half-sensitive (make own androgen)
    # T- cells: completely insensitive (delta_minus = 0, hardcoded in rhs)

    # Initial conditions (cell counts, arbitrary units)
    "T0_plus": 5_000.0,
    "T0_prod": 500.0,
    "T0_minus": 50.0,
}


def _make_drug_schedule(on_treatment: bool) -> Callable[[float], int]:
    """Convenience: constant drug schedule for testing."""
    value = 1 if on_treatment else 0
    return lambda t: value


class LotkaVolterraModel:
    """
    Three-species competitive Lotka-Volterra model with abiraterone treatment.

    Species
    -------
    T+  (index 0) : androgen-dependent cells — killed by abiraterone
    Tp  (index 1) : androgen-producing cells — partially killed by abiraterone
    T-  (index 2) : androgen-independent cells — unaffected by abiraterone

    ODE system
    ----------
    dT+/dt  = r+  * T+ * (1 - (T+ + a12*Tp + a13*T-)/K) - drug*delta+  * T+
    dTp/dt  = rp  * Tp * (1 - (a21*T+ + Tp + a23*T-)/K) - drug*deltap * Tp
    dT-/dt  = r-  * T- * (1 - (a31*T+ + a32*Tp + T-)/K)

    where drug(t) ∈ {0, 1}.
    """

    def __init__(self, params: dict | None = None) -> None:
        """
        Parameters
        ----------
        params : dict, optional
            Override any subset of DEFAULT_PARAMS. Keys:
            r_plus, r_prod, r_minus, K, alpha (3×3 ndarray),
            delta_plus, delta_prod, T0_plus, T0_prod, T0_minus.
        """
        p = {**DEFAULT_PARAMS}
        if params is not None:
            # Deep-copy alpha if provided so callers can't mutate our state
            if "alpha" in params:
                params = {**params, "alpha": np.asarray(params["alpha"], dtype=float)}
            p.update(params)

        self.r_plus: float = float(p["r_plus"])
        self.r_prod: float = float(p["r_prod"])
        self.r_minus: float = float(p["r_minus"])
        self.K: float = float(p["K"])
        self.alpha: np.ndarray = np.asarray(p["alpha"], dtype=float)
        self.delta_plus: float = float(p["delta_plus"])
        self.delta_prod: float = float(p["delta_prod"])
        self.T0_plus: float = float(p["T0_plus"])
        self.T0_prod: float = float(p["T0_prod"])
        self.T0_minus: float = float(p["T0_minus"])

        # Validate shapes
        if self.alpha.shape != (3, 3):
            raise ValueError(f"alpha must be a 3×3 matrix, got {self.alpha.shape}")
        for name, val in [("r_plus", self.r_plus), ("r_prod", self.r_prod),
                          ("r_minus", self.r_minus), ("K", self.K),
                          ("delta_plus", self.delta_plus), ("delta_prod", self.delta_prod)]:
            if val < 0:
                raise ValueError(f"Parameter '{name}' must be non-negative, got {val}")

    # ------------------------------------------------------------------
    # Core ODE
    # ------------------------------------------------------------------

    def rhs(
        self,
        t: float,
        y: np.ndarray,
        drug_schedule: Callable[[float], int],
    ) -> np.ndarray:
        """
        Right-hand side of the Lotka-Volterra ODE.

        Parameters
        ----------
        t : float
            Current time (days).
        y : ndarray of shape (3,)
            Cell counts [T+, Tp, T-].
        drug_schedule : callable(t) -> {0, 1}
            Returns 1 when abiraterone is active, 0 otherwise.

        Returns
        -------
        dydt : ndarray of shape (3,)
        """
        T_plus, T_prod, T_minus = y
        # Clamp negatives from numerical noise
        T_plus = max(T_plus, 0.0)
        T_prod = max(T_prod, 0.0)
        T_minus = max(T_minus, 0.0)

        drug = float(drug_schedule(t))
        K = self.K
        a = self.alpha

        # Competitive load for each species (numerator of logistic suppression)
        load_plus  = (T_plus + a[0, 1] * T_prod + a[0, 2] * T_minus) / K
        load_prod  = (a[1, 0] * T_plus + T_prod  + a[1, 2] * T_minus) / K
        load_minus = (a[2, 0] * T_plus + a[2, 1] * T_prod + T_minus ) / K

        dT_plus  = self.r_plus  * T_plus  * (1.0 - load_plus)  - drug * self.delta_plus  * T_plus
        dT_prod  = self.r_prod  * T_prod  * (1.0 - load_prod)  - drug * self.delta_prod  * T_prod
        dT_minus = self.r_minus * T_minus * (1.0 - load_minus)   # drug has no effect on T-

        return np.array([dT_plus, dT_prod, dT_minus])

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_span: tuple[float, float],
        t_eval: np.ndarray,
        drug_schedule: Callable[[float], int],
        y0: np.ndarray | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ) -> dict:
        """
        Integrate the ODE system over ``t_span``.

        Parameters
        ----------
        t_span : (t_start, t_end)
        t_eval : 1-D ndarray
            Specific times at which to store the solution.
        drug_schedule : callable(t) -> {0, 1}
        y0 : ndarray of shape (3,), optional
            Initial conditions [T+, Tp, T-].  Defaults to the model's
            ``T0_plus``, ``T0_prod``, ``T0_minus`` attributes.  Pass an
            explicit array when continuing a simulation from a mid-point.
        rtol, atol : solver tolerances

        Returns
        -------
        dict with keys:
            t           ndarray  time points
            T_plus      ndarray  T+ cell counts
            T_prod      ndarray  Tp cell counts
            T_minus     ndarray  T- cell counts
            total_cells ndarray  sum of all three populations
            drug        ndarray  drug indicator at each evaluation point
        """
        if y0 is None:
            y0 = np.array([self.T0_plus, self.T0_prod, self.T0_minus], dtype=float)
        else:
            y0 = np.asarray(y0, dtype=float)

        sol = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, drug_schedule),
            t_span=t_span,
            y0=y0,
            method="RK45",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        T_plus  = np.maximum(sol.y[0], 0.0)
        T_prod  = np.maximum(sol.y[1], 0.0)
        T_minus = np.maximum(sol.y[2], 0.0)

        drug_vals = np.array([float(drug_schedule(t)) for t in sol.t])

        return {
            "t":           sol.t,
            "T_plus":      T_plus,
            "T_prod":      T_prod,
            "T_minus":     T_minus,
            "total_cells": T_plus + T_prod + T_minus,
            "drug":        drug_vals,
        }

    # ------------------------------------------------------------------
    # Clinical endpoints
    # ------------------------------------------------------------------

    def time_to_progression(
        self,
        simulation_result: dict,
        progression_threshold: float = 2.0,
    ) -> float:
        """
        Earliest time at which **either** progression criterion holds:

        **A.** ``total_cells > progression_threshold * initial_total`` (default
        threshold 2.0).

        **B.** Resistant fraction ``T_minus / total_cells > 0.6`` and
        ``total_cells > 0.05 * initial_total`` (avoids spurious ratios when
        total is tiny).

        Returns ``np.inf`` if neither is met in the simulated window.

        Parameters
        ----------
        simulation_result : dict returned by :meth:`simulate`
        progression_threshold : float
            Multiplier on initial total for criterion A.

        Returns
        -------
        float
            Day of progression, or ``np.inf`` if progression never occurs.
        """
        total = simulation_result["total_cells"]
        t     = simulation_result["t"]

        if len(total) == 0:
            return np.inf

        initial_total = total[0]
        if initial_total <= 0:
            return np.inf

        threshold = progression_threshold * initial_total
        T_minus = simulation_result["T_minus"]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_minus_fraction = T_minus / np.maximum(total, 1e-300)
        mask_a = total > threshold
        mask_b = (t_minus_fraction > 0.6) & (total > 0.05 * initial_total)
        mask = mask_a | mask_b
        if not np.any(mask):
            return np.inf

        return float(t[np.argmax(mask)])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def initial_total_cells(self) -> float:
        return self.T0_plus + self.T0_prod + self.T0_minus

    def to_dict(self) -> dict:
        """Serialize parameters to a flat dictionary (for CSV export)."""
        return {
            "r_plus":    self.r_plus,
            "r_prod":    self.r_prod,
            "r_minus":   self.r_minus,
            "K":         self.K,
            "delta_plus": self.delta_plus,
            "delta_prod": self.delta_prod,
            "T0_plus":   self.T0_plus,
            "T0_prod":   self.T0_prod,
            "T0_minus":  self.T0_minus,
            **{f"alpha_{i}{j}": self.alpha[i, j]
               for i in range(3) for j in range(3)},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LotkaVolterraModel":
        """Reconstruct a model from a flat parameter dictionary."""
        alpha = np.array([
            [d["alpha_00"], d["alpha_01"], d["alpha_02"]],
            [d["alpha_10"], d["alpha_11"], d["alpha_12"]],
            [d["alpha_20"], d["alpha_21"], d["alpha_22"]],
        ])
        params = {k: v for k, v in d.items() if not k.startswith("alpha_")}
        params["alpha"] = alpha
        return cls(params)

    def __repr__(self) -> str:
        return (
            f"LotkaVolterraModel("
            f"r=[{self.r_plus:.5f}, {self.r_prod:.5f}, {self.r_minus:.5f}], "
            f"delta=[{self.delta_plus:.4f}, {self.delta_prod:.4f}, 0], "
            f"K={self.K:.0f})"
        )
