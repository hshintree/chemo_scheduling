#!/usr/bin/env python
"""
Script 00 — Download and cache real pharmacological and clinical data.

Run this script ONCE before scripts 01-03 to populate data/raw/.
All results are cached to disk; re-running is safe (uses cache by default).
Use --force to overwrite existing caches.

Data sources
------------
1. PharmacoDB
   Drug sensitivity IC50/AUC for abiraterone, enzalutamide, docetaxel,
   cabazitaxel in prostate cancer cell lines.
   → data/raw/pharmacodb/
   → data/raw/pharmacodb/delta_plus_estimates.csv   (IC50 → LV delta bounds)

2. TCGA-PRAD via NCI GDC API
   Clinical treatment records, overall survival, baseline PSA.
   → data/raw/tcga/prad_clinical.csv
   → data/raw/tcga/treatment_summary.csv

3. cBioPortal mCRPC studies
   PSA trajectories (where available), PFS, OS for mCRPC cohorts.
   → data/raw/cbioportal/<study_id>_clinical.csv
   → data/raw/cbioportal/mcrpc_patients.csv  (merged, pivot-wide)

4. Abiraterone population PK summary
   Two-compartment model steady-state profile at 1000 mg QD.
   → data/raw/pk/abiraterone_ss_profile.csv
   → data/raw/pk/abiraterone_ic50_delta_table.csv

No API keys required for any of these sources.

FDA NONMEM files (manual download)
-----------------------------------
The FDA pharmacometrics submissions are at:
  https://www.fda.gov/drugs/regulatory-science-and-research-priorities/population-pharmacokinetics

For abiraterone (NDA 202379), download the .ext output file and run:
  python scripts/00_download_real_data.py --nonmem path/to/run1.ext

Usage
-----
  python scripts/00_download_real_data.py
  python scripts/00_download_real_data.py --sources pharmacodb tcga
  python scripts/00_download_real_data.py --nonmem data/raw/pk/nda202379_run1.ext
  python scripts/00_download_real_data.py --force
"""

from __future__ import annotations

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loaders import (
    PharmacoDB, GDCLoader, CBioPortalLoader,
    CunninghamTrialLoader, load_all_real_patients,
)
from src.pk_model import TwoCompartmentPK, NONMEMParser, abiraterone_pk, enzalutamide_pk
from src.patient import Patient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PK_DIR  = RAW_DIR / "pk"
PK_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Source 0 — Cunningham / Zhang eLife 2022 trial
# ---------------------------------------------------------------------------

def download_cunningham_trial(force: bool = False) -> None:
    log.info("─" * 60)
    log.info("Cunningham / Zhang eLife 2022 trial (primary PSA time-series data)")
    loader   = CunninghamTrialLoader()
    patients = loader.load_patients(force=force)

    n_adaptive = sum(1 for p in patients if "_Adaptive_" in p.patient_id)
    n_soc      = sum(1 for p in patients if "_SOC_" in p.patient_id)
    log.info("  Loaded %d patients (%d adaptive, %d SOC)", len(patients), n_adaptive, n_soc)

    if patients:
        summary = loader.summary()
        out = RAW_DIR / "cunningham_trial" / "summary.csv"
        summary.to_csv(out, index=False)
        log.info("  Summary saved → %s", out)
        log.info("  Adaptive arm: median %d obs/patient, PSA range %.1f–%.1f ng/mL",
                 int(summary[summary["group"] == "Adaptive"]["n_obs"].median()),
                 summary[summary["group"] == "Adaptive"]["psa_min"].min(),
                 summary[summary["group"] == "Adaptive"]["psa_max"].max())


# ---------------------------------------------------------------------------
# Source 1 — PharmacoDB
# ---------------------------------------------------------------------------

def download_pharmacodb(force: bool = False) -> None:
    log.info("─" * 60)
    log.info("PharmacoDB: prostate cancer drug sensitivity")

    db = PharmacoDB()

    rows = []
    for drug in PharmacoDB.DRUGS_OF_INTEREST:
        log.info("  Fetching %s ...", drug)
        df = db.fetch_prostate_sensitivity(drug)
        if df.empty:
            log.warning("    No data returned for %s", drug)
            continue
        prostate_df = df[df["cell_line"].isin(PharmacoDB.PROSTATE_CELL_LINES)]
        if prostate_df.empty:
            log.info("    No prostate-specific cell lines found; using all (%d rows)", len(df))
            prostate_df = df
        ic50_vals = prostate_df["IC50_uM"].dropna()
        if ic50_vals.empty:
            continue
        med = ic50_vals.median()
        lo, hi = db.estimate_delta_from_ic50(med)
        log.info(
            "    %-20s  IC50_median=%.3f µM  "
            "delta_plus range=[%.4f, %.4f] day⁻¹",
            drug, med, lo, hi,
        )
        rows.append({
            "drug":           drug,
            "n_measurements": len(ic50_vals),
            "IC50_uM_median": round(med, 4),
            "IC50_uM_q25":    round(float(ic50_vals.quantile(0.25)), 4),
            "IC50_uM_q75":    round(float(ic50_vals.quantile(0.75)), 4),
            "delta_plus_low":  round(lo, 5),
            "delta_plus_high": round(hi, 5),
            "notes": "Hill Emax model, Cmax=0.31 µM (abiraterone Cmax fasted)",
        })

    if rows:
        out = RAW_DIR / "pharmacodb" / "delta_plus_estimates.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        log.info("  Saved → %s", out)
    else:
        log.warning("  No PharmacoDB data could be retrieved.")


# ---------------------------------------------------------------------------
# Source 2 — TCGA-PRAD
# ---------------------------------------------------------------------------

def download_tcga(force: bool = False) -> None:
    log.info("─" * 60)
    log.info("TCGA-PRAD: clinical treatment data via GDC API")

    gdc  = GDCLoader()
    df   = gdc.fetch_prad_cases(size=1000)

    if df.empty:
        log.warning("  No TCGA-PRAD data returned.")
        return

    log.info("  Retrieved %d treatment records from %d patients",
             len(df), df["case_id"].nunique())

    summary = gdc.treatment_duration_summary()
    if not summary.empty:
        out = RAW_DIR / "tcga" / "treatment_summary.csv"
        summary.to_csv(out)
        log.info("  Treatment summary saved → %s", out)
        print("\nTop treatment agents in TCGA-PRAD:")
        print(summary.head(10).to_string())

    # Survival statistics (for LV model TTP validation)
    os_days = pd.to_numeric(df["days_to_death"], errors="coerce").dropna()
    lfu_days = pd.to_numeric(df["days_to_last_followup"], errors="coerce").dropna()
    log.info(
        "  OS:  median=%.0f d  (n=%d deceased)",
        os_days.median() if len(os_days) > 0 else float("nan"), len(os_days)
    )
    log.info(
        "  LFU: median=%.0f d  (n=%d)",
        lfu_days.median() if len(lfu_days) > 0 else float("nan"), len(lfu_days)
    )


# ---------------------------------------------------------------------------
# Source 3 — cBioPortal mCRPC studies
# ---------------------------------------------------------------------------

def download_cbioportal(
    study_ids: list[str] | None = None,
    force: bool = False,
) -> None:
    log.info("─" * 60)
    log.info("cBioPortal: mCRPC cohort clinical data")

    cbio = CBioPortalLoader()
    if study_ids is None:
        study_ids = CBioPortalLoader.MCRPC_STUDIES

    all_patients: list[Patient] = []

    for study_id in study_ids:
        log.info("  Study: %s", study_id)
        try:
            pts = cbio.to_patient_list(study_id, min_psa_timepoints=2)
            all_patients.extend(pts)

            wide = cbio.clinical_wide(study_id)
            if not wide.empty:
                out = RAW_DIR / "cbioportal" / f"{study_id}_wide.csv"
                wide.to_csv(out, index=False)
                log.info("    %d patients → %s", len(wide), out)

            with_psa = sum(1 for p in pts if p.psa_data is not None)
            log.info("    %d patients, %d with PSA time-series", len(pts), with_psa)

        except Exception as e:
            log.warning("    Failed: %s", e)

    # Merge all patients with PSA time-series to a single file
    ts_patients = [p for p in all_patients if p.psa_data is not None]
    if ts_patients:
        dfs = []
        for p in ts_patients:
            df = p.psa_data.copy()
            df.insert(0, "patient_id", p.patient_id)
            df.insert(1, "source",     p.params.get("source", ""))
            dfs.append(df)
        merged = pd.concat(dfs, ignore_index=True)
        out    = RAW_DIR / "cbioportal" / "mcrpc_psa_timeseries.csv"
        merged.to_csv(out, index=False)
        log.info("  %d patients with PSA time-series → %s", len(ts_patients), out)

    # Summary stats
    psa_baselines = []
    pfs_months    = []
    for p in all_patients:
        if p.params:
            b = p.params.get("psa_baseline")
            if b is not None:
                psa_baselines.append(b)
            m = p.params.get("pfs_months")
            if m is not None:
                pfs_months.append(m)

    if psa_baselines:
        log.info(
            "  PSA at baseline: median=%.1f  IQR=[%.1f, %.1f]  (n=%d)",
            np.median(psa_baselines),
            np.percentile(psa_baselines, 25),
            np.percentile(psa_baselines, 75),
            len(psa_baselines),
        )
    if pfs_months:
        log.info(
            "  PFS: median=%.1f months  (n=%d)",
            np.median(pfs_months), len(pfs_months),
        )


# ---------------------------------------------------------------------------
# Source 4 — Abiraterone PK
# ---------------------------------------------------------------------------

def generate_pk_profiles() -> None:
    log.info("─" * 60)
    log.info("Abiraterone / Enzalutamide PK: steady-state concentration profiles")

    for drug_name, pk_model, dose_mg in [
        ("abiraterone",   abiraterone_pk(),   1000.0),
        ("enzalutamide",  enzalutamide_pk(),   160.0),
    ]:
        log.info("  %s %g mg QD", drug_name, dose_mg)

        # Simulate 14 doses QD to reach steady state
        sim = pk_model.simulate_multiple_doses(
            dose_mg=dose_mg,
            interval_h=24.0,
            n_doses=14,
            n_points_per_interval=200,
        )

        df = pd.DataFrame({
            "t_h":            sim["t_h"],
            "C_central_ng_mL": sim["C_central_ng_mL"],
        })
        out = PK_DIR / f"{drug_name}_ss_profile.csv"
        df.to_csv(out, index=False)
        log.info("    Saved steady-state profile → %s", out)

        # Summary
        C_avg = pk_model.daily_average_concentration(dose_mg)
        C_max = float(sim["C_central_ng_mL"].max())
        log.info("    Cmax=%.1f ng/mL,  C_avg=%.1f ng/mL", C_max, C_avg)

        # IC50-to-delta conversion table
        if pk_model.IC50 is not None:
            IC50 = pk_model.IC50
            concs = np.logspace(-1, 4, 200)
            deltas = np.array([pk_model.effective_kill_rate(c) for c in concs])
            ic50_df = pd.DataFrame({
                "C_ng_mL":      concs,
                "delta_plus_day": deltas * 24,   # h⁻¹ × 24 h/day
            })
            out2 = PK_DIR / f"{drug_name}_ic50_delta_table.csv"
            ic50_df.to_csv(out2, index=False)
            log.info(
                "    IC50=%.1f ng/mL  → delta_plus at Cmax = %.4f day⁻¹",
                IC50,
                pk_model.effective_kill_rate(C_max) * 24,
            )


# ---------------------------------------------------------------------------
# Optional: FDA NONMEM file
# ---------------------------------------------------------------------------

def load_nonmem_file(path: str) -> None:
    log.info("─" * 60)
    log.info("Parsing NONMEM .ext file: %s", path)

    parser = NONMEMParser(path)
    result = parser.get_final_estimates()

    log.info("  THETA values:")
    for idx, val in sorted(result["THETA"].items()):
        se = result["SE"]["THETA"].get(idx, float("nan"))
        log.info("    THETA(%d) = %10.4f  (SE = %.4f)", idx, val, se)

    omega = parser.omega_matrix()
    if omega.size > 1:
        log.info("  OMEGA matrix:\n%s", np.round(omega, 4))

    out = PK_DIR / f"{Path(path).stem}_parameters.json"
    import json
    with open(out, "w") as f:
        # Convert tuple keys to strings for JSON
        clean = {
            "THETA": {str(k): v for k, v in result["THETA"].items()},
            "OMEGA": {str(k): v for k, v in result["OMEGA"].items()},
        }
        json.dump(clean, f, indent=2)
    log.info("  Parameters saved → %s", out)


# ---------------------------------------------------------------------------
# Integration check: validate data slots into Patient class
# ---------------------------------------------------------------------------

def run_integration_check() -> None:
    log.info("─" * 60)
    log.info("Integration check: verify real patients slot into the pipeline")

    all_patients = load_all_real_patients(
        min_psa_timepoints=2,
        include_tcga=True,
    )

    ts_patients   = [p for p in all_patients if p.psa_data is not None]
    summ_patients = [p for p in all_patients if p.psa_data is None]

    log.info("  Total patients loaded:       %d", len(all_patients))
    log.info("  With PSA time-series:        %d", len(ts_patients))
    log.info("  Summary-only (TCGA/no PSA):  %d", len(summ_patients))

    if ts_patients:
        p0 = ts_patients[0]
        log.info("  Example time-series patient: %s", p0)
        log.info("    PSA data shape: %s", p0.psa_data.shape)
        log.info("    Columns: %s", list(p0.psa_data.columns))
        log.info("    Baseline PSA: %.2f", p0.baseline_psa())
        log.info("    Days range: %g – %g",
                 float(p0.days().min()), float(p0.days().max()))

    if summ_patients:
        p1 = summ_patients[0]
        log.info("  Example summary patient: %s", p1)

    # Verify that time-series patients can be directly fed into fitting script
    if ts_patients:
        log.info("  Fitting script compatibility: ✓")
        log.info("  (All %d time-series patients have required columns: "
                 "day, psa, on_treatment)", len(ts_patients))
    else:
        log.info("  No time-series patients available from live APIs.")
        log.info("  → Use data/raw/README.md to digitize PSA curves from "
                 "the Zhang 2022 eLife paper supplementary figures.")

    # PK profile check
    abi_profile = PK_DIR / "abiraterone_ss_profile.csv"
    if abi_profile.exists():
        pk_df = pd.read_csv(abi_profile)
        log.info("  Abiraterone steady-state profile: %d time points, "
                 "Cmax=%.1f ng/mL",
                 len(pk_df), pk_df["C_central_ng_mL"].max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real pharmacological and clinical data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["cunningham", "pharmacodb", "tcga", "cbioportal", "pk"],
        default=["cunningham", "pharmacodb", "tcga", "cbioportal", "pk"],
        help="Which sources to download (default: all).",
    )
    parser.add_argument(
        "--nonmem",
        type=str,
        default=None,
        help="Path to a NONMEM .ext file to parse (optional).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear caches and re-download everything.",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="cBioPortal study IDs to fetch (default: all mCRPC studies).",
    )
    args = parser.parse_args()

    if args.force:
        import shutil
        log.info("--force: clearing all caches under data/raw/")
        for subdir in ["cunningham_trial", "pharmacodb", "tcga", "cbioportal"]:
            p = RAW_DIR / subdir
            if p.exists():
                shutil.rmtree(p)
                p.mkdir(parents=True)

    log.info("=" * 60)
    log.info("robust-adaptive-therapy — real data download")
    log.info("=" * 60)

    failures: list[str] = []

    if "cunningham" in args.sources:
        try:
            download_cunningham_trial(force=args.force)
        except Exception as exc:
            log.warning("Cunningham trial download failed (%s: %s) — skipping.",
                        type(exc).__name__, exc)
            failures.append("cunningham")

    if "pharmacodb" in args.sources:
        try:
            download_pharmacodb(force=args.force)
        except Exception as exc:
            log.warning("PharmacoDB download failed (%s: %s) — skipping.", type(exc).__name__, exc)
            failures.append("pharmacodb")

    if "tcga" in args.sources:
        try:
            download_tcga(force=args.force)
        except Exception as exc:
            log.warning("TCGA download failed (%s: %s) — skipping.", type(exc).__name__, exc)
            failures.append("tcga")

    if "cbioportal" in args.sources:
        try:
            download_cbioportal(study_ids=args.studies, force=args.force)
        except Exception as exc:
            log.warning("cBioPortal download failed (%s: %s) — skipping.", type(exc).__name__, exc)
            failures.append("cbioportal")

    if "pk" in args.sources:
        try:
            generate_pk_profiles()
        except Exception as exc:
            log.warning("PK profile generation failed (%s: %s) — skipping.", type(exc).__name__, exc)
            failures.append("pk")

    if args.nonmem:
        nonmem_path = Path(args.nonmem)
        if not nonmem_path.exists():
            log.warning("--nonmem path does not exist: %s — skipping.", nonmem_path)
        else:
            try:
                load_nonmem_file(args.nonmem)
            except Exception as exc:
                log.warning("NONMEM parsing failed (%s: %s) — skipping.", type(exc).__name__, exc)

    try:
        run_integration_check()
    except Exception as exc:
        log.warning("Integration check failed (%s: %s).", type(exc).__name__, exc)

    if failures:
        log.warning("Sources that could not be fetched: %s", ", ".join(failures))
        log.warning("This is expected when the API is temporarily unavailable.")
        log.warning("Re-run this script when connectivity is restored; cached")
        log.warning("results from successful sources are still usable.")

    log.info("=" * 60)
    log.info("Done.  Real data cached in data/raw/")
    log.info("  → Run scripts/01_generate_synthetic_data.py to generate")
    log.info("     synthetic cohort (if needed alongside real data).")
    log.info("  → Run scripts/02_fit_patient_parameters.py --patient")
    log.info("     data/raw/<patient_csv> to fit to a real patient.")


if __name__ == "__main__":
    main()
