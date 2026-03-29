"""
PSA observation model.

PSA (prostate-specific antigen) is a secreted protein, not the cells themselves.
We model the observed PSA as a linear combination of the three cell populations:

    PSA(t) = c_plus * T+(t) + c_prod * Tp(t) + c_minus * T-(t)

In practice all three cell types produce similar amounts of PSA, so the default
coefficients are all 1.0. This module can be extended to fit the c_i per patient
if better identifiability is needed.

Reference
---------
Zhang J, et al. (2017). Nature Communications, 8, 1816.
"""

from __future__ import annotations

import numpy as np


class PSAModel:
    """
    Linear PSA secretion model.

    Parameters
    ----------
    c_plus : float
        PSA production coefficient for T+ cells (arbitrary units per cell).
    c_prod : float
        PSA production coefficient for Tp cells.
    c_minus : float
        PSA production coefficient for T- cells.
    """

    def __init__(
        self,
        c_plus: float = 1.0,
        c_prod: float = 1.0,
        c_minus: float = 1.0,
    ) -> None:
        if any(c < 0 for c in (c_plus, c_prod, c_minus)):
            raise ValueError("PSA production coefficients must be non-negative.")
        self.c_plus = float(c_plus)
        self.c_prod = float(c_prod)
        self.c_minus = float(c_minus)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def compute_psa(self, simulation_result: dict) -> np.ndarray:
        """
        Compute PSA time series from ODE simulation output.

        Parameters
        ----------
        simulation_result : dict
            Output of ``LotkaVolterraModel.simulate()``.
            Must contain keys: T_plus, T_prod, T_minus.

        Returns
        -------
        ndarray of shape (n_timepoints,)
            PSA values (same units as cell counts × c_i coefficients).
        """
        psa = (
            self.c_plus  * simulation_result["T_plus"]
            + self.c_prod  * simulation_result["T_prod"]
            + self.c_minus * simulation_result["T_minus"]
        )
        return np.maximum(psa, 0.0)

    def normalize_psa(self, psa_series: np.ndarray) -> np.ndarray:
        """
        Normalize PSA to its pre-treatment baseline (first value = 1.0).

        Parameters
        ----------
        psa_series : 1-D ndarray

        Returns
        -------
        ndarray
            PSA / PSA[0].  If PSA[0] == 0, returns the series unchanged
            (avoids divide-by-zero for edge cases in parameter fitting).
        """
        psa_series = np.asarray(psa_series, dtype=float)
        baseline = psa_series[0]
        if baseline == 0.0:
            return psa_series.copy()
        return psa_series / baseline

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def psa_at_times(
        self,
        simulation_result: dict,
        query_times: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate PSA to arbitrary query times using the simulation grid.

        Parameters
        ----------
        simulation_result : dict
        query_times : 1-D ndarray

        Returns
        -------
        ndarray of shape (len(query_times),)
        """
        psa = self.compute_psa(simulation_result)
        t   = simulation_result["t"]
        return np.interp(query_times, t, psa)

    def psa_nadir(self, psa_series: np.ndarray) -> tuple[float, int]:
        """
        Return (min_value, argmin_index) of the PSA series.
        """
        idx = int(np.argmin(psa_series))
        return float(psa_series[idx]), idx

    def __repr__(self) -> str:
        return (
            f"PSAModel(c_plus={self.c_plus}, c_prod={self.c_prod}, "
            f"c_minus={self.c_minus})"
        )
