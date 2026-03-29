"""
Patient data class.

Stores observed PSA time-series (from either real digitized data or synthetic
generation) and fitted LV model parameters. The class is intentionally agnostic
to data source so that real CSV data can be dropped in without changing anything
downstream.

Expected CSV format (columns):
    day           int    Days from treatment start (day 0 = first dose)
    psa           float  PSA value (ng/mL or normalized)
    on_treatment  int    1 if abiraterone active that day, 0 otherwise
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class Patient:
    """
    Container for a single patient's PSA trajectory and fitted model.

    Parameters
    ----------
    patient_id : str
        Unique identifier (e.g. "patient_001" or "P03").
    psa_data : pd.DataFrame, optional
        DataFrame with columns [day, psa, on_treatment].
    params : dict, optional
        Fitted LotkaVolterraModel parameters for this patient.
        Set after calling ``scripts/02_fit_patient_parameters.py``.
    """

    REQUIRED_COLUMNS = {"day", "psa", "on_treatment"}

    def __init__(
        self,
        patient_id: str,
        psa_data: Optional[pd.DataFrame] = None,
        params: Optional[dict] = None,
    ) -> None:
        self.patient_id: str = patient_id
        self.params: Optional[dict] = params

        if psa_data is not None:
            self._validate_dataframe(psa_data)
            self.psa_data: pd.DataFrame = psa_data.copy().sort_values("day").reset_index(drop=True)
        else:
            self.psa_data = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, csv_path: str | Path, patient_id: Optional[str] = None) -> "Patient":
        """
        Load a patient from a CSV file.

        Parameters
        ----------
        csv_path : str or Path
            Path to CSV with columns [day, psa, on_treatment].
        patient_id : str, optional
            Defaults to the stem of the filename.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Patient CSV not found: {path}")
        df = pd.read_csv(path)
        pid = patient_id or path.stem
        return cls(patient_id=pid, psa_data=df)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Patient {self.patient_id}: missing columns {missing}. "
                f"Required: {self.REQUIRED_COLUMNS}"
            )
        if df["on_treatment"].isin([0, 1]).sum() != len(df):
            raise ValueError(
                f"Patient {self.patient_id}: 'on_treatment' column must contain only 0 or 1."
            )

    # ------------------------------------------------------------------
    # Clinical properties
    # ------------------------------------------------------------------

    def baseline_psa(self) -> float:
        """
        PSA value at day 0 (first observation).

        Returns
        -------
        float
            PSA at the earliest recorded day.
        """
        if self.psa_data is None:
            raise RuntimeError(f"Patient {self.patient_id} has no PSA data.")
        return float(self.psa_data.iloc[0]["psa"])

    def normalized_psa(self) -> pd.DataFrame:
        """
        Copy of psa_data with the 'psa' column divided by the baseline PSA.

        Returns
        -------
        pd.DataFrame
        """
        if self.psa_data is None:
            raise RuntimeError(f"Patient {self.patient_id} has no PSA data.")
        baseline = self.baseline_psa()
        if baseline == 0:
            raise ValueError(
                f"Patient {self.patient_id}: baseline PSA is 0, cannot normalize."
            )
        df = self.psa_data.copy()
        df["psa"] = df["psa"] / baseline
        return df

    def time_to_progression(self, baseline_multiplier: float = 1.0) -> float:
        """
        Days from treatment start until PSA returns to baseline (or above) for
        the last time — used as a simple proxy for clinical progression.

        The progression event is defined as PSA ≥ baseline_multiplier × PSA[0]
        after it has first fallen below that level.

        Parameters
        ----------
        baseline_multiplier : float
            Default 1.0 → PSA returns to pre-treatment level.

        Returns
        -------
        float
            Day of last baseline crossing.  Returns ``np.inf`` if PSA never
            rises back to baseline (e.g., patient stays in remission).
        """
        if self.psa_data is None:
            raise RuntimeError(f"Patient {self.patient_id} has no PSA data.")

        baseline = self.baseline_psa()
        threshold = baseline_multiplier * baseline
        psa_vals = self.psa_data["psa"].values
        days     = self.psa_data["day"].values

        # Find the last time PSA crosses back above threshold after going below
        below_threshold = psa_vals < threshold
        if not np.any(below_threshold):
            # Never went below baseline — progression on day 0
            return float(days[0])

        last_crossing = np.inf
        went_below = False
        for i, (val, day) in enumerate(zip(psa_vals, days)):
            if val < threshold:
                went_below = True
            elif went_below and val >= threshold:
                last_crossing = float(day)

        return last_crossing

    def treatment_schedule(self) -> np.ndarray:
        """
        Returns the on_treatment array aligned with the day column.

        Returns
        -------
        ndarray of int, shape (n_obs,)
        """
        if self.psa_data is None:
            raise RuntimeError(f"Patient {self.patient_id} has no PSA data.")
        return self.psa_data["on_treatment"].values.astype(int)

    def days(self) -> np.ndarray:
        """Day indices of all observations."""
        if self.psa_data is None:
            raise RuntimeError(f"Patient {self.patient_id} has no PSA data.")
        return self.psa_data["day"].values.astype(float)

    def psa_values(self) -> np.ndarray:
        """Raw PSA values at each observation day."""
        if self.psa_data is None:
            raise RuntimeError(f"Patient {self.patient_id} has no PSA data.")
        return self.psa_data["psa"].values.astype(float)

    def fraction_time_off_treatment(self) -> float:
        """Fraction of observation days with on_treatment == 0."""
        schedule = self.treatment_schedule()
        if len(schedule) == 0:
            return np.nan
        return float(np.mean(schedule == 0))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Save psa_data to CSV."""
        if self.psa_data is None:
            raise RuntimeError(f"Patient {self.patient_id} has no PSA data to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.psa_data.to_csv(path, index=False)

    def __repr__(self) -> str:
        n = len(self.psa_data) if self.psa_data is not None else 0
        has_params = self.params is not None
        return (
            f"Patient(id={self.patient_id!r}, "
            f"n_obs={n}, "
            f"fitted={has_params})"
        )
