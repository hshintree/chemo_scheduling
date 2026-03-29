"""
Real-data API clients for robust-adaptive-therapy.

Three sources are supported:

1. PharmacoDB  — in vitro drug sensitivity (IC50/AUC) across cancer cell lines.
   Public REST API, no key required.
   URL: https://pharmacodb.ca/api/v2
   Use: calibrate delta_plus bounds from IC50 values for prostate cell lines.

2. NCI GDC / TCGA  — clinical treatment records for TCGA-PRAD cohort.
   Public REST API, no key required.
   URL: https://api.gdc.cancer.gov
   Use: validate "standard-of-care" drug schedules, obtain baseline PSA
   distributions and overall survival as sanity checks on TTP predictions.

3. cBioPortal  — mCRPC cohort clinical and genomic data.
   Public REST API, no key required.
   URL: https://www.cbioportal.org/api
   Use: obtain PSA kinetics, treatment response, and outcomes for real mCRPC
   patients; create Patient objects directly when time-series data is present.

All network calls are cached as JSON / CSV in data/raw/<source>/ to minimise
repeated downloads.  The loaders work offline once the cache is populated.

Integration with existing codebase
-----------------------------------
All loaders have a `to_patient_list()` method that returns
List[src.patient.Patient], slotting directly into the fitting pipeline
(scripts/02_fit_patient_parameters.py) and simulation comparisons
(scripts/03_simulate_treatments.py).

When only summary statistics (baseline PSA, OS) are available (TCGA),
a Patient is created with `psa_data=None` and `params` holding the
summary, so it can be used for cohort-level calibration even without
a time series.
"""

from __future__ import annotations

import io
import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .patient import Patient

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_RAW_DIR      = _PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class APIError(RuntimeError):
    """Raised when an external API returns an unexpected / non-JSON response."""


def _parse_json_response(resp: "requests.Response") -> dict | list:
    """
    Safely parse a requests.Response as JSON.

    Raises APIError (not JSONDecodeError) with a human-readable message that
    includes the HTTP status code and the first 200 characters of the body so
    callers can distinguish 'server returned HTML' from 'server returned empty
    body' from 'endpoint not found'.
    """
    body = resp.text.strip()
    if not body:
        raise APIError(
            f"HTTP {resp.status_code} from {resp.url} — response body is empty. "
            "The endpoint may have moved or is temporarily down."
        )
    try:
        return resp.json()
    except Exception as exc:
        snippet = body[:200].replace("\n", " ")
        raise APIError(
            f"HTTP {resp.status_code} from {resp.url} — body is not JSON. "
            f"First 200 chars: {snippet!r}"
        ) from exc


def _get_json(url: str, cache_path: Path, params: Optional[dict] = None,
              timeout: int = 30) -> dict | list:
    """
    GET request with on-disk JSON cache.

    Returns cached result if cache_path exists; otherwise fetches and saves.
    Raises APIError (not JSONDecodeError) on non-JSON / empty responses.
    """
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    if not HAS_REQUESTS:
        raise ImportError(
            "The 'requests' library is required for live API calls. "
            "Install it with: pip install requests"
        )

    log.info("Fetching %s", url)
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = _parse_json_response(resp)
    _ensure_dir(cache_path.parent)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def _post_json(url: str, payload: dict, cache_path: Path,
               timeout: int = 30) -> dict | list:
    """POST request with on-disk JSON cache."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    if not HAS_REQUESTS:
        raise ImportError("The 'requests' library is required.")

    log.info("POST %s", url)
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = _parse_json_response(resp)
    _ensure_dir(cache_path.parent)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ---------------------------------------------------------------------------
# 1. PharmacoDB
# ---------------------------------------------------------------------------

class PharmacoDB:
    """
    Client for the PharmacoDB v2 REST API.
    https://pharmacodb.ca/api/v2

    Primary use: obtain IC50 / AUC_recomputed values for drugs relevant to
    mCRPC (abiraterone, enzalutamide, docetaxel, cabazitaxel) across prostate
    cancer cell lines (LNCaP, PC3, DU145, VCaP, 22Rv1).

    These IC50 values anchor the prior distribution on delta_plus in the LV
    model via the Hill-equation kill-rate model in src/pk_model.py.
    """

    BASE_URL  = "https://pharmacodb.ca/api/v2"
    CACHE_DIR = _RAW_DIR / "pharmacodb"

    # mCRPC-relevant drugs by PharmacoDB name
    DRUGS_OF_INTEREST = [
        "Abiraterone",
        "Enzalutamide",
        "Docetaxel",
        "Cabazitaxel",
        "Bicalutamide",
    ]

    # mCRPC-relevant cell lines
    PROSTATE_CELL_LINES = ["LNCaP", "PC3", "DU145", "VCaP", "22Rv1", "C4-2"]

    def __init__(self) -> None:
        _ensure_dir(self.CACHE_DIR)

    # ------------------------------------------------------------------
    # Drug search
    # ------------------------------------------------------------------

    def search_drug(self, name: str) -> list[dict]:
        """
        Return all PharmacoDB compound entries matching ``name``.

        Tries /compounds first (current v2 terminology), then /drugs as a
        legacy fallback.  Returns an empty list if the API is unreachable.
        """
        safe = name.lower().replace(" ", "_")
        for endpoint in ("compounds", "drugs"):
            cache = self.CACHE_DIR / f"drug_search_{safe}_{endpoint}.json"
            try:
                data = _get_json(f"{self.BASE_URL}/{endpoint}", cache,
                                 params={"name": name, "per_page": 50})
            except APIError as exc:
                log.debug("PharmacoDB /%s search failed: %s", endpoint, exc)
                if cache.exists():
                    cache.unlink()  # don't persist broken responses
                continue
            if isinstance(data, dict):
                return data.get("data", [])
            if isinstance(data, list):
                return data
        log.warning("PharmacoDB: could not reach /compounds or /drugs endpoint.")
        return []

    def list_all_drugs(self) -> pd.DataFrame:
        """Return a DataFrame of all compounds in PharmacoDB."""
        cache = self.CACHE_DIR / "all_compounds.json"
        try:
            data = _get_json(f"{self.BASE_URL}/compounds", cache,
                             params={"per_page": 2000})
        except APIError as exc:
            log.warning("PharmacoDB list_all_drugs failed: %s", exc)
            return pd.DataFrame()
        if isinstance(data, dict):
            records = data.get("data", [])
        else:
            records = data
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Sensitivity data
    # ------------------------------------------------------------------

    def fetch_prostate_sensitivity(
        self,
        drug_name: str,
        page_size: int = 2000,
    ) -> pd.DataFrame:
        """
        Download dose-response profiles for ``drug_name`` in prostate cell lines.

        Returns a DataFrame with columns:
            drug_name, cell_line, dataset, IC50_uM, AUC, AAC, source
        """
        cache = self.CACHE_DIR / f"sensitivity_{drug_name.lower()}.csv"
        if cache.exists():
            return pd.read_csv(cache)

        # Step 1: resolve compound ID
        compounds = self.search_drug(drug_name)
        if not compounds:
            log.warning("Drug '%s' not found in PharmacoDB.", drug_name)
            return pd.DataFrame()

        compound_id = compounds[0].get("id") or compounds[0].get("compound_id")
        if compound_id is None:
            log.warning("PharmacoDB returned a compound record with no 'id' for '%s'.", drug_name)
            return pd.DataFrame()

        records: list[dict] = []

        # Step 2: try /compounds/{id}/experiments, then /experiments?compound_id=
        for endpoint, ep_params in [
            (f"{self.BASE_URL}/compounds/{compound_id}/experiments",
             {"per_page": page_size}),
            (f"{self.BASE_URL}/experiments",
             {"compound_id": compound_id, "tissue_id": "prostate",
              "per_page": page_size}),
            (f"{self.BASE_URL}/experiments",
             {"drug_id": compound_id, "tissue_id": "prostate",
              "per_page": page_size}),
        ]:
            if records:
                break
            safe_id = str(compound_id).replace("/", "_")
            ep_key  = endpoint.split("/")[-1]
            exp_cache = self.CACHE_DIR / f"experiments_{drug_name.lower()}_{safe_id}_{ep_key}.json"
            try:
                data = _get_json(endpoint, exp_cache, params=ep_params)
            except APIError as exc:
                log.debug("Experiment endpoint %s failed: %s", endpoint, exc)
                if exp_cache.exists():
                    exp_cache.unlink()
                continue
            if isinstance(data, dict):
                records = data.get("data", [])
            elif isinstance(data, list):
                records = data

        if not records:
            log.warning("No sensitivity data found for '%s'.", drug_name)
            return pd.DataFrame()

        rows = []
        for rec in records:
            cell_line = (rec.get("cell_line") or {}).get("name", "")
            dataset   = (rec.get("dataset")   or {}).get("name", "")
            profile   = rec.get("dose_response") or {}
            rows.append({
                "drug_name": drug_name,
                "cell_line": cell_line,
                "dataset":   dataset,
                "IC50_uM":   rec.get("ic50", None),
                "AUC":       rec.get("auc", None),
                "AAC":       rec.get("aac", None),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_csv(cache, index=False)
        return df

    def estimate_delta_from_ic50(
        self,
        ic50_uM: float,
        drug_conc_uM: Optional[float] = None,
        E_max: float = 0.30,
        hill_n: float = 1.0,
    ) -> tuple[float, float]:
        """
        Convert IC50 (µM, in vitro) to a plausible range for delta_plus (day⁻¹).

        Uses the Hill equation:
            E(C) = E_max * C^n / (IC50^n + C^n)

        If drug_conc_uM is None, uses the typical steady-state abiraterone
        Cmax of ~120 ng/mL ≈ 0.31 µM.

        Parameters
        ----------
        ic50_uM : float
        drug_conc_uM : float, optional
        E_max : float
            Maximum fractional kill rate (day⁻¹); default 0.30 covers most
            published range for abiraterone in prostate cell lines.
        hill_n : float

        Returns
        -------
        (delta_low, delta_high) : tuple[float, float]
            Plausible range of delta_plus values for the LV model.
        """
        if drug_conc_uM is None:
            # Abiraterone Cmax ≈ 120 ng/mL, MW = 391 g/mol → 0.31 µM
            drug_conc_uM = 0.31

        frac = (drug_conc_uM ** hill_n) / (ic50_uM ** hill_n + drug_conc_uM ** hill_n)
        delta_nominal = E_max * frac

        # Uncertainty band: ±50% around nominal (IIV in typical PK studies)
        delta_low  = delta_nominal * 0.5
        delta_high = delta_nominal * 1.5
        return float(delta_low), float(delta_high)

    def summary_for_lv_model(self) -> pd.DataFrame:
        """
        Download IC50 data for all DRUGS_OF_INTEREST in prostate cell lines
        and return a summary table with delta_plus estimates.
        """
        rows = []
        for drug in self.DRUGS_OF_INTEREST:
            df = self.fetch_prostate_sensitivity(drug)
            if df.empty:
                continue
            prostate_df = df[df["cell_line"].isin(self.PROSTATE_CELL_LINES)]
            if prostate_df.empty:
                prostate_df = df   # fallback: use all cell lines

            ic50_vals = prostate_df["IC50_uM"].dropna()
            if ic50_vals.empty:
                continue

            ic50_median = float(ic50_vals.median())
            d_low, d_high = self.estimate_delta_from_ic50(ic50_median)
            rows.append({
                "drug":            drug,
                "n_cell_lines":    prostate_df["cell_line"].nunique(),
                "IC50_uM_median":  ic50_median,
                "IC50_uM_IQR_low": float(ic50_vals.quantile(0.25)),
                "IC50_uM_IQR_high": float(ic50_vals.quantile(0.75)),
                "delta_plus_low":  d_low,
                "delta_plus_high": d_high,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. GDC / TCGA
# ---------------------------------------------------------------------------

class GDCLoader:
    """
    Client for the NCI Genomic Data Commons (GDC) API.
    https://api.gdc.cancer.gov

    Downloads TCGA-PRAD (Prostate Adenocarcinoma) clinical data to provide:
    - Distribution of baseline PSA at treatment initiation
    - Treatment types and durations (standard-of-care patterns)
    - Overall survival as an external benchmark for LV model TTP predictions

    The TCGA clinical data does NOT contain longitudinal PSA time-series —
    it captures one or a few PSA measurements per patient.  Use cBioPortal
    for studies with richer clinical trajectories.
    """

    BASE_URL  = "https://api.gdc.cancer.gov"
    CACHE_DIR = _RAW_DIR / "tcga"
    PROJECT   = "TCGA-PRAD"

    def __init__(self) -> None:
        _ensure_dir(self.CACHE_DIR)

    # ------------------------------------------------------------------
    # Raw data download
    # ------------------------------------------------------------------

    def fetch_prad_cases(self, size: int = 1000) -> pd.DataFrame:
        """
        Download TCGA-PRAD case-level clinical data.

        Returns a DataFrame with one row per patient including:
        age, gender, race, tumor_stage, vital_status, days_to_death,
        days_to_last_follow_up, treatments_pharmaceutical, PSA fields.
        """
        cache = self.CACHE_DIR / f"prad_cases_{size}.json"

        payload = {
            "filters": {
                "op": "in",
                "content": {
                    "field": "project.project_id",
                    "value": [self.PROJECT],
                },
            },
            "fields": (
                "case_id,submitter_id,"
                "diagnoses.age_at_diagnosis,"
                "diagnoses.tumor_stage,"
                "diagnoses.days_to_last_follow_up,"
                "diagnoses.days_to_death,"
                "diagnoses.vital_status,"
                "diagnoses.days_to_recurrence,"
                "demographic.gender,demographic.race,"
                "treatments.treatment_type,"
                "treatments.therapeutic_agents,"
                "treatments.days_to_treatment_start,"
                "treatments.days_to_treatment_end,"
                "treatments.treatment_or_therapy"
            ),
            "format": "JSON",
            "size":   str(size),
        }

        data = _post_json(f"{self.BASE_URL}/cases", payload, cache)
        hits = data.get("data", {}).get("hits", []) if isinstance(data, dict) else []

        if not hits:
            log.warning("No TCGA-PRAD cases returned from GDC API.")
            return pd.DataFrame()

        rows = []
        for case in hits:
            base = {
                "case_id":      case.get("case_id", ""),
                "submitter_id": case.get("submitter_id", ""),
            }
            # Flatten first diagnosis
            dx = case.get("diagnoses", [{}])[0]
            base.update({
                "age_at_diagnosis":       dx.get("age_at_diagnosis", None),
                "tumor_stage":            dx.get("tumor_stage", ""),
                "days_to_last_followup":  dx.get("days_to_last_follow_up", None),
                "days_to_death":          dx.get("days_to_death", None),
                "vital_status":           dx.get("vital_status", ""),
                "days_to_recurrence":     dx.get("days_to_recurrence", None),
            })
            # Flatten demographics
            demo = case.get("demographic", {})
            base.update({
                "gender": demo.get("gender", ""),
                "race":   demo.get("race", ""),
            })
            # Treatments: one row per treatment event
            treatments = case.get("treatments", [])
            if treatments:
                for tx in treatments:
                    row = {**base}
                    row.update({
                        "treatment_type":    tx.get("treatment_type", ""),
                        "therapeutic_agent": tx.get("therapeutic_agents", ""),
                        "days_tx_start":     tx.get("days_to_treatment_start", None),
                        "days_tx_end":       tx.get("days_to_treatment_end", None),
                        "tx_response":       tx.get("treatment_or_therapy", ""),
                    })
                    rows.append(row)
            else:
                rows.append(base)

        df = pd.DataFrame(rows)
        df.to_csv(self.CACHE_DIR / "prad_clinical.csv", index=False)
        return df

    # ------------------------------------------------------------------
    # LV model integration helpers
    # ------------------------------------------------------------------

    def treatment_duration_summary(self) -> pd.DataFrame:
        """
        Summary of treatment types and durations in TCGA-PRAD.
        Used to validate drug schedule assumptions in the LV model.
        """
        df = self.fetch_prad_cases()
        if df.empty:
            return pd.DataFrame()

        if "days_tx_end" not in df.columns or "days_tx_start" not in df.columns:
            log.warning(
                "Treatment date columns not present in TCGA data "
                "(likely all cases lack treatment records). "
                "Returning empty summary."
            )
            return pd.DataFrame()

        df["duration_days"] = (
            pd.to_numeric(df["days_tx_end"], errors="coerce")
            - pd.to_numeric(df["days_tx_start"], errors="coerce")
        )
        summary = (
            df.groupby("therapeutic_agent")
            .agg(
                n=("case_id", "count"),
                mean_duration=("duration_days", "mean"),
                median_duration=("duration_days", "median"),
            )
            .round(1)
            .sort_values("n", ascending=False)
        )
        return summary

    def to_patient_list(self) -> list[Patient]:
        """
        Create Patient objects from TCGA-PRAD cases.

        Since TCGA does not have PSA time series, `psa_data` is None.
        Summary statistics are stored in `params` for cohort-level use.
        """
        df = self.fetch_prad_cases()
        if df.empty:
            return []

        patients = []
        for case_id, grp in df.groupby("case_id"):
            row = grp.iloc[0]
            summary_params = {
                "age_at_diagnosis":    row.get("age_at_diagnosis"),
                "vital_status":        row.get("vital_status"),
                "days_to_death":       row.get("days_to_death"),
                "days_to_last_followup": row.get("days_to_last_followup"),
                "tumor_stage":         row.get("tumor_stage"),
                "source":              "TCGA-PRAD",
            }
            patients.append(Patient(
                patient_id=str(case_id),
                psa_data=None,
                params=summary_params,
            ))
        return patients


# ---------------------------------------------------------------------------
# 3. cBioPortal
# ---------------------------------------------------------------------------

class CBioPortalLoader:
    """
    Client for the cBioPortal REST API.
    https://www.cbioportal.org/api

    mCRPC-relevant studies that may contain PSA kinetics data:

    Study ID               | N    | Notes
    -----------------------|------|----------------------------------------
    mcrpc_wcdt_2020        | 101  | WCDT mCRPC; clinical + WGS
    prad_su2c_2019         | 150  | SU2C mCRPC; clinical + WGS/RNA
    prad_mskcc_2014        | 150  | MSK CRPC; clinical + exome
    prad_mskcc_2010        | 181  | MSK localized; clinical + SNP array
    prad_broad_2019        | 1013 | Broad mCRPC; clinical + panel seq

    cBioPortal clinical data is structured as key-value attribute pairs
    per patient.  Common attributes for mCRPC studies:
        PSA_DIAGNOSIS, PSA_NADIR, TREATMENT_RESPONSE,
        PFS_MONTHS, OS_MONTHS, DRUG_TYPE, LINE_OF_THERAPY

    PSA time-series are NOT available in cBioPortal for most studies.
    Studies with longitudinal PSA (MSKCC / SU2C) may have:
        PSA_6MO, PSA_12MO, PSA_24MO
    which allow crude PSA trajectory reconstruction.
    """

    BASE_URL  = "https://www.cbioportal.org/api"
    CACHE_DIR = _RAW_DIR / "cbioportal"

    # Verified-working cBioPortal study IDs for metastatic / CRPC prostate cancer.
    # mcrpc_wcdt_2020 and prad_broad_2019 return 404 as of 2026.
    MCRPC_STUDIES = [
        "prad_su2c_2019",    # SU2C/PCF mCRPC, PNAS 2019  – 444 pts, WES + clinical
        "prad_mskcc_2014",   # MSK CRPC, PNAS 2014         – 104 pts, CNV + clinical
        "prad_su2c_2015",    # SU2C/PCF mCRPC, Cell 2015   – 150 pts, WES + clinical
        "prad_fhcrc",        # Fred Hutchinson disseminated, Nat Med 2016 – 176 pts
        "prad_mich",         # Michigan mCRPC, Nature 2012  – 61 pts, WES + clinical
        "mpcproject_broad_2021",  # Metastatic PC Project, 2021 – patient-driven
    ]

    def __init__(self) -> None:
        _ensure_dir(self.CACHE_DIR)

    # ------------------------------------------------------------------
    # Study discovery
    # ------------------------------------------------------------------

    def list_prad_studies(self) -> pd.DataFrame:
        """Return all prostate cancer studies in cBioPortal."""
        cache = self.CACHE_DIR / "prad_studies.json"
        data  = _get_json(f"{self.BASE_URL}/studies", cache,
                          params={"keyword": "prostate", "pageSize": 200})
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame(data.get("data", []) if isinstance(data, dict) else [])

    def fetch_clinical_attributes(self, study_id: str) -> list[dict]:
        """List all clinical attribute names for a study."""
        cache = self.CACHE_DIR / f"{study_id}_attributes.json"
        return _get_json(
            f"{self.BASE_URL}/studies/{study_id}/clinical-attributes", cache
        )

    # ------------------------------------------------------------------
    # Patient-level clinical data
    # ------------------------------------------------------------------

    def fetch_clinical_data(
        self,
        study_id: str,
        attribute_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch patient-level clinical data for a study.

        Parameters
        ----------
        study_id : str
            cBioPortal study identifier.
        attribute_ids : list[str], optional
            Specific clinical attribute IDs to fetch.  If None, fetches all.

        Returns
        -------
        pd.DataFrame with columns: [patient_id, attribute_id, value]
        """
        cache_name = f"{study_id}_clinical"
        if attribute_ids:
            cache_name += "_" + "_".join(sorted(attribute_ids))
        cache = self.CACHE_DIR / f"{cache_name}.json"

        url    = f"{self.BASE_URL}/studies/{study_id}/clinical-data"
        params = {"clinicalDataType": "PATIENT", "pageSize": 10000}
        data   = _get_json(url, cache, params=params)

        if isinstance(data, dict):
            records = data.get("data", [])
        else:
            records = data if isinstance(data, list) else []

        if not records:
            log.warning("No clinical data found for study '%s'.", study_id)
            return pd.DataFrame(columns=["patient_id", "attribute_id", "value"])

        df = pd.DataFrame(records)
        if "clinicalAttributeId" in df.columns:
            df = df.rename(columns={
                "clinicalAttributeId": "attribute_id",
                "patientId":           "patient_id",
            })

        if attribute_ids:
            df = df[df["attribute_id"].isin(attribute_ids)]

        df.to_csv(self.CACHE_DIR / f"{cache_name}.csv", index=False)
        return df

    def clinical_wide(self, study_id: str) -> pd.DataFrame:
        """
        Return clinical data pivoted to wide format (one row per patient).
        """
        long = self.fetch_clinical_data(study_id)
        if long.empty:
            return pd.DataFrame()
        return long.pivot_table(
            index="patient_id",
            columns="attribute_id",
            values="value",
            aggfunc="first",
        ).reset_index()

    # ------------------------------------------------------------------
    # PSA trajectory reconstruction
    # ------------------------------------------------------------------

    def extract_psa_trajectory(
        self,
        study_id: str,
        psa_attrs: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Reconstruct PSA time series from longitudinal clinical attributes.

        For most studies this produces a sparse series (PSA at 0, 6, 12 months).
        The result can be extended with WebPlotDigitizer-digitized values.

        Parameters
        ----------
        study_id : str
        psa_attrs : list[str], optional
            Clinical attribute IDs that contain PSA values at known time points.
            Auto-detected if None (looks for attrs containing 'PSA').

        Returns
        -------
        dict mapping patient_id → pd.DataFrame([day, psa, on_treatment])
        """
        wide = self.clinical_wide(study_id)
        if wide.empty:
            return {}

        if psa_attrs is None:
            psa_attrs = [c for c in wide.columns
                         if "PSA" in c.upper() and c != "patient_id"]

        # Auto-detect time points from attribute names (e.g. PSA_6MO → 180 days)
        month_re = __import__("re").compile(r"(\d+)\s*MO", __import__("re").IGNORECASE)
        trajectories: dict[str, pd.DataFrame] = {}

        for _, row in wide.iterrows():
            pid = str(row.get("patient_id", "unknown"))
            time_psa_pairs = []
            for attr in psa_attrs:
                val = row.get(attr)
                if pd.isna(val) or val == "":
                    continue
                try:
                    psa_val = float(val)
                except (ValueError, TypeError):
                    continue
                # Map attribute name to approximate day
                if "DIAGNO" in attr.upper() or "BASELINE" in attr.upper() or attr.upper().endswith("_0"):
                    day = 0
                elif "NADIR" in attr.upper():
                    day = None   # unknown timing
                else:
                    m = month_re.search(attr)
                    day = int(m.group(1)) * 30 if m else None
                if day is not None:
                    time_psa_pairs.append((day, psa_val))

            if len(time_psa_pairs) >= 2:
                time_psa_pairs.sort()
                df = pd.DataFrame(time_psa_pairs, columns=["day", "psa"])
                # on_treatment: default 1 (unknown schedule → conservative)
                df["on_treatment"] = 1
                trajectories[pid] = df

        return trajectories

    # ------------------------------------------------------------------
    # Patient list integration
    # ------------------------------------------------------------------

    def to_patient_list(
        self,
        study_id: str,
        min_psa_timepoints: int = 2,
    ) -> list[Patient]:
        """
        Create Patient objects from a cBioPortal study.

        Patients with ≥ ``min_psa_timepoints`` PSA values get a full
        time-series DataFrame.  Others get psa_data=None with summary
        statistics in params.

        Parameters
        ----------
        study_id : str
        min_psa_timepoints : int

        Returns
        -------
        list[Patient]
        """
        wide         = self.clinical_wide(study_id)
        trajectories = self.extract_psa_trajectory(study_id)
        patients     = []

        if wide.empty:
            return []

        for _, row in wide.iterrows():
            pid = str(row.get("patient_id", ""))
            traj_df = trajectories.get(pid)

            summary = {
                "source":       study_id,
                "os_months":    _try_float(row.get("OS_MONTHS")),
                "pfs_months":   _try_float(row.get("PFS_MONTHS")),
                "psa_baseline": _try_float(row.get("PSA_DIAGNOSIS",
                                           row.get("PSA_BASELINE", None))),
            }

            if traj_df is not None and len(traj_df) >= min_psa_timepoints:
                p = Patient(patient_id=pid, psa_data=traj_df, params=summary)
            else:
                p = Patient(patient_id=pid, psa_data=None, params=summary)
            patients.append(p)

        return patients


# ---------------------------------------------------------------------------
# 4. Cunningham / Zhang eLife 2022 Trial Data
# ---------------------------------------------------------------------------

class CunninghamTrialLoader:
    """
    Loader for the real mCRPC adaptive therapy trial data published in:

        Cunningham et al. (2022) "Evolution-based mathematical models significantly
        prolong response to abiraterone in metastatic castrate-resistant prostate
        cancer and identify strategies to further improve outcomes."
        eLife 11:e76284.  doi:10.7554/eLife.76284

    Data source (GitHub, CC-BY 4.0):
        https://github.com/cunninghamjj/Evolution-based-mathematical-models-
        significantly-prolong-response-to-Abiraterone-in-mCRPC

    The Excel file contains:
    - Sheet "Adaptive" : 17 patients receiving adaptive abiraterone (P1001–P1020)
    - Sheet "SOC"      : 15 patients on standard continuous dosing (C001–C015)

    Each patient's data has columns: Days, PSA (ng/mL), Abi (0/1), relPSA_Indi.

    This provides the only publicly available longitudinal PSA time-series from a
    prospective adaptive therapy RCT.  After loading, each patient is a full
    Patient object with psa_data = DataFrame[day, psa, on_treatment] ready for
    LV model fitting.
    """

    EXCEL_URL = (
        "https://raw.githubusercontent.com/cunninghamjj/"
        "Evolution-based-mathematical-models-significantly-prolong-"
        "response-to-Abiraterone-in-mCRPC/main/data/TrialPatientData.xlsx"
    )
    CACHE_DIR = _RAW_DIR / "cunningham_trial"

    def __init__(self) -> None:
        _ensure_dir(self.CACHE_DIR)

    # ------------------------------------------------------------------
    # Download / cache
    # ------------------------------------------------------------------

    def _download_excel(self, force: bool = False) -> bytes:
        cache = self.CACHE_DIR / "TrialPatientData.xlsx"
        if cache.exists() and not force:
            return cache.read_bytes()
        if not HAS_REQUESTS:
            raise ImportError("The 'requests' library is required.")
        log.info("Downloading Cunningham eLife 2022 trial data from GitHub ...")
        resp = requests.get(self.EXCEL_URL, timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
        log.info("  Saved → %s", cache)
        return resp.content

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_sheet(self, df_raw: pd.DataFrame, group: str) -> list[Patient]:
        """
        Parse a wide-format Excel sheet into a list of Patient objects.

        Layout (zero-indexed rows):
          Row 0 : patient IDs  (e.g. P1001, NaN, NaN, …, P1002, …)
          Row 1 : column names (Days, PSA, Abi, relPSA_Indi, NaN, …)
          Row 2+: numeric data
        """
        id_row   = df_raw.iloc[0]
        col_row  = df_raw.iloc[1]
        data     = df_raw.iloc[2:].reset_index(drop=True)
        n_cols   = df_raw.shape[1]

        patients: list[Patient] = []
        for start_col in range(n_cols - 3):
            pid = id_row.iloc[start_col]
            if pd.isna(pid) or not isinstance(pid, str):
                continue

            # Expect the next 4 columns to be Days, PSA, Abi, relPSA_Indi
            window = {
                str(col_row.iloc[start_col + offset]): start_col + offset
                for offset in range(4)
                if start_col + offset < n_cols
                   and isinstance(col_row.iloc[start_col + offset], str)
            }
            if not {"Days", "PSA", "Abi"}.issubset(window):
                continue

            sub = pd.DataFrame({
                "day": pd.to_numeric(data.iloc[:, window["Days"]], errors="coerce"),
                "psa": pd.to_numeric(data.iloc[:, window["PSA"]],  errors="coerce"),
                "on_treatment": (
                    pd.to_numeric(data.iloc[:, window["Abi"]], errors="coerce")
                    .fillna(0).astype(int)
                ),
            }).dropna(subset=["day", "psa"]).reset_index(drop=True)

            if len(sub) < 2:
                continue

            patients.append(Patient(
                patient_id=f"cunningham_{group}_{pid}",
                psa_data=sub,
                params={"source": f"Cunningham_eLife2022_{group}", "group": group},
            ))

        return patients

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_patients(
        self,
        groups: list[str] | None = None,
        force: bool = False,
    ) -> list[Patient]:
        """
        Load patients from the Cunningham trial Excel file.

        Parameters
        ----------
        groups : list of sheet names to load.  Default: ['Adaptive', 'SOC'].
        force  : re-download even if the cache exists.

        Returns
        -------
        list[Patient] – each has a full PSA time-series.
        """
        if groups is None:
            groups = ["Adaptive", "SOC"]

        # Check cache first
        cached = self._load_from_cache()
        wanted = {f"cunningham_{g}_" for g in groups}
        cached_for_groups = [p for p in cached
                             if any(p.patient_id.startswith(pfx) for pfx in wanted)]
        if cached_for_groups and not force:
            log.info("Cunningham trial: loaded %d patients from cache.",
                     len(cached_for_groups))
            return cached_for_groups

        try:
            raw_bytes = self._download_excel(force=force)
        except Exception as exc:
            log.warning("Could not download Cunningham trial data: %s", exc)
            return cached_for_groups  # return whatever is cached

        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "openpyxl is required to read .xlsx files. "
                "Install with: conda install openpyxl"
            )

        xf = pd.ExcelFile(io.BytesIO(raw_bytes))
        all_patients: list[Patient] = []

        for group in groups:
            if group not in xf.sheet_names:
                log.warning("Sheet '%s' not in trial Excel file.", group)
                continue
            df_raw   = pd.read_excel(xf, sheet_name=group, header=None)
            patients = self._parse_sheet(df_raw, group)
            log.info(
                "Cunningham trial (%s): parsed %d patients.", group, len(patients)
            )
            # Persist individual CSVs
            for p in patients:
                csv_path = self.CACHE_DIR / f"{p.patient_id}.csv"
                if p.psa_data is not None and not csv_path.exists():
                    p.psa_data.to_csv(csv_path, index=False)
            all_patients.extend(patients)

        return all_patients

    def _load_from_cache(self) -> list[Patient]:
        """Return previously downloaded patients from per-patient CSV files."""
        patients: list[Patient] = []
        for csv_file in sorted(self.CACHE_DIR.glob("cunningham_*.csv")):
            pid = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
                group = "Adaptive" if "_Adaptive_" in pid else "SOC"
                patients.append(Patient(
                    patient_id=pid,
                    psa_data=df,
                    params={"source": "Cunningham_eLife2022", "group": group},
                ))
            except Exception as exc:
                log.warning("Failed to load cached file %s: %s", csv_file, exc)
        return patients

    def summary(self) -> pd.DataFrame:
        """Return a summary table of patient IDs, n_obs, and PSA range."""
        patients = self.load_patients()
        rows = []
        for p in patients:
            df = p.psa_data
            rows.append({
                "patient_id": p.patient_id,
                "group":      (p.params or {}).get("group", ""),
                "n_obs":      len(df),
                "psa_min":    round(float(df["psa"].min()), 2),
                "psa_max":    round(float(df["psa"].max()), 2),
                "days_max":   int(df["day"].max()),
                "pct_on_tx":  round(float(df["on_treatment"].mean()) * 100, 1),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _try_float(val) -> Optional[float]:
    """Convert to float; return None on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def load_all_real_patients(
    min_psa_timepoints: int = 2,
    include_tcga: bool = True,
    include_cunningham: bool = True,
    cbioportal_studies: Optional[list[str]] = None,
) -> list[Patient]:
    """
    Convenience function: load all available real patients from all sources.

    Priority order (best longitudinal PSA data first):
    1. Cunningham / Zhang eLife 2022 trial  – 32 patients, full PSA time-series
    2. cBioPortal mCRPC studies             – 500–1000+ patients, summary only
    3. TCGA-PRAD (GDC)                      – 500 patients, summary only

    Parameters
    ----------
    min_psa_timepoints : int
        Minimum PSA observations for a cBioPortal patient to get a time-series.
    include_tcga : bool
        Include TCGA-PRAD summary patients (no PSA time-series).
    include_cunningham : bool
        Include the Cunningham eLife 2022 trial patients (full time-series).
    cbioportal_studies : list[str], optional
        Which cBioPortal studies to load.  Default: MCRPC_STUDIES.

    Returns
    -------
    list[Patient]
        Combined list.  patient.params["source"] records the origin.
    """
    patients: list[Patient] = []

    # ── Cunningham trial (best data: real longitudinal PSA) ──────────────────
    if include_cunningham:
        try:
            loader = CunninghamTrialLoader()
            pts    = loader.load_patients()
            patients.extend(pts)
            log.info("Cunningham eLife 2022 trial: %d patients (full PSA series)", len(pts))
        except Exception as exc:
            log.warning("Cunningham trial load failed: %s", exc)

    # ── cBioPortal ───────────────────────────────────────────────────────────
    if cbioportal_studies is None:
        cbioportal_studies = CBioPortalLoader.MCRPC_STUDIES

    cbio = CBioPortalLoader()
    for study_id in cbioportal_studies:
        log.info("Loading cBioPortal study: %s", study_id)
        try:
            pts = cbio.to_patient_list(study_id, min_psa_timepoints)
            patients.extend(pts)
            log.info("  → %d patients", len(pts))
        except Exception as exc:
            log.warning("Failed to load %s: %s", study_id, exc)

    # ── TCGA-PRAD ────────────────────────────────────────────────────────────
    if include_tcga:
        gdc = GDCLoader()
        log.info("Loading TCGA-PRAD clinical data...")
        try:
            pts = gdc.to_patient_list()
            patients.extend(pts)
            log.info("  → %d TCGA patients (summary only)", len(pts))
        except Exception as exc:
            log.warning("Failed to load TCGA-PRAD: %s", exc)

    return patients
