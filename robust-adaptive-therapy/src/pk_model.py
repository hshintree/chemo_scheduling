"""
Pharmacokinetic models for mCRPC drugs.

Primary model: Abiraterone acetate (Zytiga) two-compartment oral PK.
Also includes a generic NONMEM .ext-file parser so FDA public-submission
models can be loaded directly.

References
----------
Xu XS, et al. (2015). Population pharmacokinetics of abiraterone in patients
    with metastatic castration-resistant prostate cancer.
    J Clin Pharmacol, 55(12), 1356–1364.
FDA Clinical Pharmacology Review, NDA 202379 (2011).
    https://www.accessdata.fda.gov/drugsatfda_docs/nda/2011/202379Orig1s000ClinPharmR.pdf
FDA Pharmacometrics public submissions:
    https://www.fda.gov/drugs/regulatory-science-and-research-priorities/population-pharmacokinetics
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp
from typing import Optional


# ---------------------------------------------------------------------------
# Two-compartment oral PK model
# ---------------------------------------------------------------------------
#
# Unit convention used throughout this module
# ───────────────────────────────────────────
# ODE state variables  : amounts in mg
# Volumes              : L
# CL / Q               : L/h
# Ka                   : h⁻¹
# Concentration (internal): mg/L  (= µg/mL)
# Concentration (output)  : ng/mL   (internal × 1000)
# AUC (output)            : ng·h/mL (internal × 1000)
#
# Derivation of parameters from FDA Prescribing Information (PI):
#   CL/F [L/h] = Dose[mg] / AUC[mg·h/L]  where AUC[mg·h/L] = AUC[ng·h/mL] / 1000
#   V1/F [L]   ≈ Dose[mg] / Cmax[mg/L]   where Cmax[mg/L]  = Cmax[ng/mL]  / 1000
#
# ─────────────────────────────────────────────────────────────────────────────

# Abiraterone acetate (Zytiga) 1000 mg QD with food, steady state.
# Source: FDA Prescribing Information NDA 202379, Table 2 (2022 label).
#   Cmax,ss  = 17.1 ng/mL  → V1/F ≈ 1000 mg / (0.0171 mg/L) = 58 480 L
#   AUC₀₋₂₄ = 272  ng·h/mL → CL/F = 1000 mg / (0.272 mg·h/L) = 3 676 L/h
#   Tmax ≈ 2–4 h with food; terminal t½ ≈ 12 h.
#   Large V1/F reflects low oral bioavailability (~7% fasted, ~18% fed)
#   and extensive tissue distribution (abiraterone is highly lipophilic).
# IIV estimates: Xu XS et al. (2015). J Clin Pharmacol 55:1356–1364.
# IC50 (CYP17A1 inhibition in LNCaP cells): Jarman et al. (2010). BJC 103:148.
ABIRATERONE_PK_PARAMS = {
    "drug":    "abiraterone",
    "dose_mg": 1000.0,
    "Ka":      0.50,             # h⁻¹  (Tmax ≈ 2 h with food)
    "CL_F":    3_676.0,          # L/h  (= 1000 mg / 0.272 mg·h/L)
    "V1_F":    58_480.0,         # L    (= 1000 mg / 0.01711 mg/L)
    "V2_F":    75_000.0,         # L    (estimated; gives terminal t½ ≈ 12 h)
    "Q_F":     500.0,            # L/h  (estimated inter-compartmental CL)
    "IIV_Ka":  36.0,             # % CV (Xu 2015)
    "IIV_CL":  32.0,
    "IIV_V1":  55.0,
    "food_effect_cmax": 10.0,    # fasted → fed multiplier
    "food_effect_auc":  5.0,
    "IC50_ng_mL": 4.0,
    "MW_g_mol":   391.56,
}

# Enzalutamide (Xtandi) 160 mg QD with food, steady state.
# Source: FDA Prescribing Information NDA 203415; Gibbons et al. (2015) JCCP.
#   Cmax,ss  = 7 100 ng/mL = 7.1 mg/L → V1/F ≈ 160/7.1 = 22.5 L
#   AUC₀₋₂₄ = 123 µg·h/mL = 123 mg/L × h → CL/F = 160/123 = 1.30 L/h
#   Oral bioavailability ≈ 84%; terminal t½ ≈ 5.8 days = 139 h.
# IC50 in LNCaP cells: Tran et al. (2009) Science 324:787; ≈ 0.5–2 µM ≈ 230–930 ng/mL.
ENZALUTAMIDE_PK_PARAMS = {
    "drug":    "enzalutamide",
    "dose_mg": 160.0,
    "Ka":      0.88,             # h⁻¹
    "CL_F":    1.30,             # L/h
    "V1_F":    22.5,             # L
    "V2_F":    238.0,            # L   (single-cpt V/F ≈ CL/F / k_el = 1.30/0.00499)
    "Q_F":     2.0,              # L/h
    "IIV_Ka":  54.0,
    "IIV_CL":  27.0,
    "IIV_V1":  40.0,
    "IC50_ng_mL": 600.0,
    "MW_g_mol":   464.44,
}

# Docetaxel 75 mg/m² IV q3w (IV bolus, no absorption phase).
# Source: Bruno et al. (1998) J Pharmacokinet Biopharm 26:521.
# IC50 in PC3/LNCaP: Fabbri et al. (2008) Prostate 68:1_270; ≈ 1–10 nM ≈ 0.8–8 ng/mL.
DOCETAXEL_PK_PARAMS = {
    "drug":    "docetaxel",
    "dose_mg_m2": 75.0,
    "Ka":      None,             # IV dosing: no depot compartment
    "CL_F":    21.3,             # L/h  (IV clearance = CL, not CL/F)
    "V1_F":    8.5,              # L
    "V2_F":    156.0,            # L
    "Q_F":     42.6,             # L/h
    "IIV_CL":  29.0,
    "IIV_V1":  36.0,
    "IC50_ng_mL": 4.0,
    "MW_g_mol":   807.88,
}


class TwoCompartmentPK:
    """
    Oral two-compartment pharmacokinetic model (NONMEM ADVAN4 equivalent).

    ODEs
    ----
    dA_depot/dt        = -Ka * A_depot
    dA_central/dt      = Ka * A_depot - (CL_F + Q_F) * C_central + Q_F * C_periph
    dA_peripheral/dt   = Q_F * C_central - Q_F * C_periph

    where C_central = A_central / V1_F, C_periph = A_peripheral / V2_F.

    For IV drugs (Ka=None), depot is replaced by an instantaneous bolus into
    the central compartment at t=0.

    Parameters
    ----------
    params : dict
        Must contain Ka (None for IV), CL_F, V1_F, V2_F, Q_F.
        Use one of the predefined DRUG_PK_PARAMS dicts or pass custom values.
    """

    def __init__(self, params: dict) -> None:
        self.params = params
        self.drug   = params.get("drug", "unknown")
        self.Ka     = params.get("Ka", None)           # h⁻¹; None → IV
        self.CL_F   = float(params["CL_F"])            # L/h
        self.V1_F   = float(params["V1_F"])            # L
        self.V2_F   = float(params["V2_F"])            # L
        self.Q_F    = float(params["Q_F"])             # L/h
        self.IC50   = params.get("IC50_ng_mL", None)   # ng/mL

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------

    def _rhs_oral(self, t: float, y: np.ndarray, dose_mg: float) -> np.ndarray:
        """RHS for oral two-compartment model. y = [A_depot, A_central, A_periph]."""
        A_d, A_c, A_p = y
        C_c = A_c / self.V1_F
        C_p = A_p / self.V2_F
        Ka  = self.Ka

        dA_d = -Ka * A_d
        dA_c =  Ka * A_d - (self.CL_F + self.Q_F) * C_c + self.Q_F * C_p
        dA_p =  self.Q_F * C_c - self.Q_F * C_p
        return np.array([dA_d, dA_c, dA_p])

    def _rhs_iv(self, t: float, y: np.ndarray) -> np.ndarray:
        """RHS for IV two-compartment model. y = [A_central, A_periph]."""
        A_c, A_p = y
        C_c = A_c / self.V1_F
        C_p = A_p / self.V2_F
        dA_c = -(self.CL_F + self.Q_F) * C_c + self.Q_F * C_p
        dA_p =   self.Q_F * C_c - self.Q_F * C_p
        return np.array([dA_c, dA_p])

    # ------------------------------------------------------------------
    # Single-dose simulation
    # ------------------------------------------------------------------

    def simulate_single_dose(
        self,
        dose_mg: float,
        t_max_h: float = 72.0,
        n_points: int = 500,
    ) -> dict:
        """
        Simulate a single oral or IV dose.

        Parameters
        ----------
        dose_mg : float
            Administered dose in mg.
        t_max_h : float
            Simulation horizon in hours.
        n_points : int

        Returns
        -------
        dict with keys: t_h, C_central_ng_mL, C_periph_ng_mL, AUC_ng_h_mL
        """
        t_eval = np.linspace(0.0, t_max_h, n_points)

        if self.Ka is not None:
            # Oral: amounts in mg; C_internal = mg/L (= µg/mL)
            y0  = np.array([dose_mg, 0.0, 0.0])
            sol = solve_ivp(
                lambda t, y: self._rhs_oral(t, y, dose_mg),
                t_span=(0.0, t_max_h), y0=y0, t_eval=t_eval,
                method="RK45", rtol=1e-8, atol=1e-10,
            )
            C_central_mgL = sol.y[1] / self.V1_F   # mg/L
            C_periph_mgL  = sol.y[2] / self.V2_F
        else:
            # IV bolus: entire dose enters central compartment at t=0
            y0  = np.array([dose_mg, 0.0])
            sol = solve_ivp(
                lambda t, y: self._rhs_iv(t, y),
                t_span=(0.0, t_max_h), y0=y0, t_eval=t_eval,
                method="RK45", rtol=1e-8, atol=1e-10,
            )
            C_central_mgL = sol.y[0] / self.V1_F
            C_periph_mgL  = sol.y[1] / self.V2_F

        # Convert mg/L → ng/mL (×1000)
        C_central = np.maximum(C_central_mgL * 1000.0, 0.0)
        C_periph  = np.maximum(C_periph_mgL  * 1000.0, 0.0)
        auc = float(np.trapezoid(C_central, sol.t))
        return {
            "t_h":             sol.t,
            "C_central_ng_mL": C_central,
            "C_periph_ng_mL":  C_periph,
            "AUC_ng_h_mL":     auc,
            "Cmax_ng_mL":      float(np.max(C_central)),
        }

    # ------------------------------------------------------------------
    # Multiple-dose simulation (steady state)
    # ------------------------------------------------------------------

    def simulate_multiple_doses(
        self,
        dose_mg: float,
        interval_h: float,
        n_doses: int,
        t_max_h: Optional[float] = None,
        n_points_per_interval: int = 100,
    ) -> dict:
        """
        Simulate repeated oral dosing until steady state.

        Parameters
        ----------
        dose_mg : float
        interval_h : float
            Dosing interval in hours (e.g. 24 for QD).
        n_doses : int
        t_max_h : float, optional
            Total simulation time; defaults to n_doses * interval_h.
        n_points_per_interval : int

        Returns
        -------
        dict with keys: t_h, C_central_ng_mL, dose_times_h
        """
        if t_max_h is None:
            t_max_h = n_doses * interval_h

        t_all = []
        C_all = []
        dose_times = [i * interval_h for i in range(n_doses)]

        # Initial state (amounts in mg)
        if self.Ka is not None:
            state = np.zeros(3)
        else:
            state = np.zeros(2)

        for i_dose, t_dose in enumerate(dose_times):
            # Add dose (mg) to depot (oral) or central (IV)
            state[0] += dose_mg

            t_end      = t_dose + interval_h if i_dose < n_doses - 1 else t_max_h
            t_eval_seg = np.linspace(t_dose, t_end, n_points_per_interval)

            if self.Ka is not None:
                sol = solve_ivp(
                    lambda t, y: self._rhs_oral(t, y, dose_mg),
                    t_span=(t_dose, t_end), y0=state, t_eval=t_eval_seg,
                    method="RK45", rtol=1e-8, atol=1e-10,
                )
                state     = sol.y[:, -1].copy()
                C_segment = sol.y[1] / self.V1_F * 1000.0   # mg/L → ng/mL
            else:
                sol = solve_ivp(
                    lambda t, y: self._rhs_iv(t, y),
                    t_span=(t_dose, t_end), y0=state, t_eval=t_eval_seg,
                    method="RK45", rtol=1e-8, atol=1e-10,
                )
                state     = sol.y[:, -1].copy()
                C_segment = sol.y[0] / self.V1_F * 1000.0

            t_all.append(sol.t)
            C_all.append(C_segment)

        return {
            "t_h":              np.concatenate(t_all),
            "C_central_ng_mL":  np.maximum(np.concatenate(C_all), 0.0),
            "dose_times_h":     np.array(dose_times),
        }

    # ------------------------------------------------------------------
    # LV model coupling: effective drug kill rate
    # ------------------------------------------------------------------

    def effective_kill_rate(
        self,
        concentration_ng_mL: float,
        E_max: float = 1.0,
        hill_coeff: float = 1.0,
    ) -> float:
        """
        Hill-equation drug effect (Emax model) linking plasma concentration
        to the fractional kill rate delta used in the LV model.

        E(C) = E_max * C^n / (IC50^n + C^n)

        Parameters
        ----------
        concentration_ng_mL : float
        E_max : float
            Maximum kill rate (= delta_plus at saturation).
        hill_coeff : float
            Hill exponent n (usually 1 for simple competition binding).

        Returns
        -------
        float
            Effective kill rate (day⁻¹).  Divide by 24 if concentration
            is sampled at hourly intervals.
        """
        if self.IC50 is None:
            raise ValueError(f"IC50 not set for {self.drug}.")
        IC50 = self.IC50
        C    = max(concentration_ng_mL, 0.0)
        return float(E_max * (C ** hill_coeff) / (IC50 ** hill_coeff + C ** hill_coeff))

    def daily_average_concentration(
        self,
        dose_mg: float,
        interval_h: float = 24.0,
        n_ss_doses: int = 14,
    ) -> float:
        """
        Average steady-state plasma concentration over one dosing interval.
        Uses the last interval of a multi-dose simulation.

        Returns
        -------
        float : C_avg (ng/mL)
        """
        sim = self.simulate_multiple_doses(dose_mg, interval_h, n_ss_doses)
        t   = sim["t_h"]
        C   = sim["C_central_ng_mL"]
        # Last interval
        t_last_dose = (n_ss_doses - 1) * interval_h
        mask = t >= t_last_dose
        if not np.any(mask):
            return float(np.mean(C))
        return float(np.trapezoid(C[mask], t[mask]) / interval_h)

    def __repr__(self) -> str:
        return (
            f"TwoCompartmentPK({self.drug}, "
            f"Ka={self.Ka}, CL/F={self.CL_F}, V1/F={self.V1_F})"
        )


# ---------------------------------------------------------------------------
# Convenience constructors for common drugs
# ---------------------------------------------------------------------------

def abiraterone_pk() -> TwoCompartmentPK:
    """Two-compartment PK model for abiraterone 1000 mg QD (fasted)."""
    return TwoCompartmentPK(ABIRATERONE_PK_PARAMS)


def enzalutamide_pk() -> TwoCompartmentPK:
    """Two-compartment PK model for enzalutamide 160 mg QD."""
    return TwoCompartmentPK(ENZALUTAMIDE_PK_PARAMS)


def docetaxel_pk() -> TwoCompartmentPK:
    """Two-compartment PK model for docetaxel 75 mg/m² IV q3w."""
    pk = TwoCompartmentPK({**DOCETAXEL_PK_PARAMS, "CL_F": DOCETAXEL_PK_PARAMS["CL"],
                           "V1_F": DOCETAXEL_PK_PARAMS["V1"],
                           "V2_F": DOCETAXEL_PK_PARAMS["V2"],
                           "Q_F":  DOCETAXEL_PK_PARAMS["Q"]})
    return pk


# ---------------------------------------------------------------------------
# NONMEM .ext file parser
# ---------------------------------------------------------------------------

class NONMEMParser:
    """
    Parser for NONMEM output files (.ext tables).

    NONMEM .ext files are plain-text tables generated by the
    `$ESTIMATION` step.  Each row corresponds to one estimation iteration.
    The final row (ITERATION=-1000000006) contains the final parameter
    estimates; the preceding row (ITERATION=-1000000007) has the standard
    errors.

    The columns follow the NONMEM THETA/OMEGA/SIGMA naming convention:

        ITERATION THETA1 THETA2 ... OMEGA(1,1) OMEGA(2,1) OMEGA(2,2) ... SIGMA(1,1)

    How to obtain NONMEM files
    --------------------------
    FDA public submissions are hosted at the Pharmacometrics library:
        https://www.fda.gov/drugs/regulatory-science-and-research-priorities/population-pharmacokinetics

    Direct NONMEM control stream and data files are also included in FDA
    New Drug Application (NDA) packages, accessible via:
        https://www.accessdata.fda.gov/scripts/cder/daf/

    For abiraterone (Zytiga): NDA 202379
    For enzalutamide (Xtandi): NDA 203415

    Usage
    -----
    >>> parser = NONMEMParser("path/to/run1.ext")
    >>> params = parser.get_final_estimates()
    >>> theta  = params["THETA"]    # dict {1: value, 2: value, ...}
    >>> omega  = params["OMEGA"]    # lower-triangular dict {(i,j): value}
    """

    def __init__(self, ext_path: str | Path) -> None:
        self.path = Path(ext_path)
        if not self.path.exists():
            raise FileNotFoundError(f"NONMEM .ext file not found: {self.path}")
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        # Skip comment lines starting with "TABLE"
        lines = []
        with open(self.path) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("TABLE") or stripped == "":
                    continue
                lines.append(stripped)
        if not lines:
            raise ValueError(f"Empty or unreadable .ext file: {self.path}")
        from io import StringIO
        self._df = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+")
        return self._df

    def get_final_estimates(self) -> dict:
        """
        Return the final parameter estimates (ITERATION = -1000000006).

        Returns
        -------
        dict with keys:
            THETA  : dict {1: val, ...}
            OMEGA  : dict {(i, j): val, ...}  lower-triangular
            SIGMA  : dict {(i, j): val, ...}
            SE     : dict with same structure, standard errors
        """
        df = self._load()
        final_row = df[df["ITERATION"] == -1000000006]
        se_row    = df[df["ITERATION"] == -1000000007]

        if final_row.empty:
            raise ValueError(
                "Final estimates row (ITERATION=-1000000006) not found. "
                "Check that the run completed successfully."
            )

        cols    = [c for c in df.columns if c != "ITERATION" and c != "OBJ"]
        values  = final_row[cols].iloc[0].to_dict()
        se_vals = se_row[cols].iloc[0].to_dict() if not se_row.empty else {}

        theta  = {}
        omega  = {}
        sigma  = {}
        se_theta  = {}
        se_omega  = {}
        se_sigma  = {}

        theta_re = re.compile(r"^THETA(\d+)$")
        omega_re = re.compile(r"^OMEGA\((\d+),(\d+)\)$")
        sigma_re = re.compile(r"^SIGMA\((\d+),(\d+)\)$")

        for col, val in values.items():
            if (m := theta_re.match(col)):
                idx = int(m.group(1))
                theta[idx] = float(val)
                se_theta[idx] = float(se_vals.get(col, np.nan))
            elif (m := omega_re.match(col)):
                i, j = int(m.group(1)), int(m.group(2))
                omega[(i, j)] = float(val)
                se_omega[(i, j)] = float(se_vals.get(col, np.nan))
            elif (m := sigma_re.match(col)):
                i, j = int(m.group(1)), int(m.group(2))
                sigma[(i, j)] = float(val)
                se_sigma[(i, j)] = float(se_vals.get(col, np.nan))

        return {
            "THETA": theta,
            "OMEGA": omega,
            "SIGMA": sigma,
            "SE":    {"THETA": se_theta, "OMEGA": se_omega, "SIGMA": se_sigma},
        }

    def to_two_compartment_params(
        self,
        theta_map: dict[int, str],
        drug: str = "unknown",
        ic50_ng_mL: Optional[float] = None,
    ) -> dict:
        """
        Convert parsed NONMEM THETAs to TwoCompartmentPK parameter dict.

        Parameters
        ----------
        theta_map : dict mapping THETA index → param name
            e.g. {1: "Ka", 2: "CL_F", 3: "V1_F", 4: "Q_F", 5: "V2_F"}
        drug : str
        ic50_ng_mL : float, optional

        Returns
        -------
        dict compatible with TwoCompartmentPK.__init__
        """
        estimates = self.get_final_estimates()
        theta     = estimates["THETA"]
        params    = {"drug": drug}
        if ic50_ng_mL is not None:
            params["IC50_ng_mL"] = ic50_ng_mL
        for idx, name in theta_map.items():
            if idx not in theta:
                raise KeyError(f"THETA({idx}) not found in .ext file.")
            params[name] = theta[idx]
        return params

    def omega_matrix(self) -> np.ndarray:
        """Return the OMEGA matrix as a square numpy array."""
        estimates = self.get_final_estimates()
        omega_d   = estimates["OMEGA"]
        if not omega_d:
            return np.array([[]])
        n = max(max(i, j) for i, j in omega_d.keys())
        M = np.zeros((n, n))
        for (i, j), val in omega_d.items():
            M[i - 1, j - 1] = val
            M[j - 1, i - 1] = val   # symmetrise
        return M

    def __repr__(self) -> str:
        return f"NONMEMParser({self.path.name})"
