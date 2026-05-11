"""
Phase 3b: Build Evidence Database
===================================
Merges pico_results.csv + annotations.csv, normalizes drug names,
links ClinicalTrials.gov records (best-effort via PMID), and outputs:
  - glp1_database.csv
  - glp1_database.xlsx  (with filters pre-applied, one sheet per indication)

Run AFTER annotate.py is complete.

Usage:
    pip install pandas openpyxl requests
    python build_database.py
"""

import pandas as pd
import requests
import time
import logging
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

PICO_CSV        = "pico_results.csv"
ANNOTATIONS_CSV = "annotations.csv"
OUTPUT_CSV      = "glp1_database.csv"
OUTPUT_EXCEL    = "glp1_database.xlsx"
LOG_FILE        = "build_database_log.txt"

CT_REQUEST_DELAY = 0.5   # seconds between ClinicalTrials.gov API calls
CT_TIMEOUT       = 8     # seconds before giving up on a single lookup

# ── Drug name normalization map ────────────────────────────────────────────────
# Keys are lowercase substrings to match; values are canonical names.
# Order matters: more specific entries first.

DRUG_MAP = {
    "semaglutide":   "Semaglutide",
    "ozempic":       "Semaglutide",
    "wegovy":        "Semaglutide",
    "rybelsus":      "Semaglutide",
    "tirzepatide":   "Tirzepatide",
    "mounjaro":      "Tirzepatide",
    "zepbound":      "Tirzepatide",
    "liraglutide":   "Liraglutide",
    "victoza":       "Liraglutide",
    "saxenda":       "Liraglutide",
    "dulaglutide":   "Dulaglutide",
    "trulicity":     "Dulaglutide",
    "exenatide":     "Exenatide",
    "byetta":        "Exenatide",
    "bydureon":      "Exenatide",
    "albiglutide":   "Albiglutide",
    "tanzeum":       "Albiglutide",
    "lixisenatide":  "Lixisenatide",
    "adlyxin":       "Lixisenatide",
    "lyxumia":       "Lixisenatide",
    "efpeglenatide": "Efpeglenatide",
    "oral glp-1":    "Semaglutide",   # oral formulation
    "glp-1":         "GLP-1 (unspecified)",
}

# ── Indication normalization ───────────────────────────────────────────────────

INDICATION_MAP = {
    "type 2 diabetes": "Type 2 Diabetes",
    "t2dm":            "Type 2 Diabetes",
    "t2d":             "Type 2 Diabetes",
    "diabetes mellitus": "Type 2 Diabetes",
    "obesity":         "Obesity",
    "overweight":      "Obesity",
    "weight":          "Obesity",
    "cardiovascular":  "Cardiovascular Disease",
    "cvd":             "Cardiovascular Disease",
    "mace":            "Cardiovascular Disease",
    "heart failure":   "Heart Failure",
    "hfpef":           "Heart Failure",
    "hfref":           "Heart Failure",
    "nash":            "NASH/NAFLD",
    "nafld":           "NASH/NAFLD",
    "steatohepatitis": "NASH/NAFLD",
    "chronic kidney":  "Chronic Kidney Disease",
    "ckd":             "Chronic Kidney Disease",
    "renal":           "Chronic Kidney Disease",
    "sleep apnea":     "Sleep Apnea",
    "polycystic":      "PCOS",
    "pcos":            "PCOS",
    "alzheimer":       "Neurodegenerative Disease",
    "parkinson":       "Neurodegenerative Disease",
}

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Drug normalization ─────────────────────────────────────────────────────────

def normalize_drug(text: str) -> str:
    """Match drug name from intervention + title text."""
    if not text:
        return "Unknown"
    lower = text.lower()
    for key, canonical in DRUG_MAP.items():
        if key in lower:
            return canonical
    return "Other/Unknown"

def normalize_indication(text: str) -> str:
    if not text:
        return "Unknown"
    lower = text.lower()
    for key, canonical in INDICATION_MAP.items():
        if key in lower:
            return canonical
    # Return cleaned title-case version of raw value as fallback
    return text.strip().title()

# ── Enrollment + female_pct cleaning ──────────────────────────────────────────

def clean_int(val) -> str:
    """Convert to int string or empty."""
    try:
        return str(int(float(str(val).replace(",", "").strip())))
    except (ValueError, TypeError):
        return ""

def clean_float(val) -> str:
    """Round to 1 decimal or empty."""
    try:
        return str(round(float(str(val).strip()), 1))
    except (ValueError, TypeError):
        return ""

# ── ClinicalTrials.gov lookup ──────────────────────────────────────────────────

CT_BASE = "https://clinicaltrials.gov/api/v2/studies"

def lookup_ct_by_pmid(pmid: str) -> dict:
    """
    Query ClinicalTrials.gov API v2 for studies linked to a PubMed ID.
    Returns dict with nct_id, ct_title, ct_status, ct_phase, ct_enrollment
    or empty strings if not found.
    """
    empty = {"nct_id": "", "ct_status": "", "ct_phase": "", "ct_enrollment": ""}
    try:
        params = {
            "query.term": f"AREA[ReferencePMID]{pmid}",
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,EnrollmentCount",
            "pageSize": 1,
            "format": "json",
        }
        resp = requests.get(CT_BASE, params=params, timeout=CT_TIMEOUT)
        resp.raise_for_status()
        data  = resp.json()
        studies = data.get("studies", [])
        if not studies:
            return empty

        s       = studies[0]
        proto   = s.get("protocolSection", {})
        id_mod  = proto.get("identificationModule", {})
        stat_mod = proto.get("statusModule", {})
        design  = proto.get("designModule", {})

        return {
            "nct_id":        id_mod.get("nctId", ""),
            "ct_status":     stat_mod.get("overallStatus", ""),
            "ct_phase":      ", ".join(design.get("phases", [])),
            "ct_enrollment": str(design.get("enrollmentInfo", {}).get("count", "")),
        }

    except Exception as e:
        log.debug(f"  CT lookup failed for PMID {pmid}: {e}")
        return empty

# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info(f"Phase 3b: Build Database -- {datetime.now().isoformat()}")
    log.info("=" * 60)

    # ── Load inputs ──────────────────────────────────────────────────────────
    log.info("Loading PICO results...")
    pico = pd.read_csv(PICO_CSV, dtype=str).fillna("")
    log.info(f"  {len(pico)} rows in pico_results.csv")

    log.info("Loading annotations...")
    ann  = pd.read_csv(ANNOTATIONS_CSV, dtype=str).fillna("")
    log.info(f"  {len(ann)} rows in annotations.csv")

    # ── Merge on PMID ────────────────────────────────────────────────────────
    df = pico.merge(ann, on="pmid", how="left", suffixes=("", "_ann"))
    log.info(f"  Merged: {len(df)} rows")

    # ── Normalize drug names ──────────────────────────────────────────────────
    log.info("Normalizing drug names...")
    df["drug_name"] = (df["intervention"].fillna("") + " " + df["title"].fillna("")).apply(normalize_drug)

    # ── Normalize indications ─────────────────────────────────────────────────
    log.info("Normalizing indications...")
    df["indication_clean"] = df["indication"].apply(normalize_indication)

    # ── Clean numeric fields ──────────────────────────────────────────────────
    df["enrollment_size"] = df["enrollment_size"].apply(clean_int)
    df["female_pct"]      = df["female_pct"].apply(clean_float)

    # ── Comparator type cleanup ───────────────────────────────────────────────
    def clean_comparator(val: str) -> str:
        v = val.lower().strip()
        if "placebo" in v:                        return "Placebo"
        if "active" in v or "comparator" in v:    return "Active Comparator"
        if "dose" in v:                           return "Dose Comparison"
        if v in ("none", "nr", ""):               return "NR"
        return val.strip().title()

    df["comparator_type"] = df["comparator_type"].apply(clean_comparator)

    # ── ClinicalTrials.gov linkage ────────────────────────────────────────────
    log.info("Linking ClinicalTrials.gov (best-effort, this may take a few minutes)...")
    ct_records = []
    for i, pmid in enumerate(df["pmid"], 1):
        if i % 100 == 0:
            log.info(f"  CT lookup: {i}/{len(df)}")
        ct_records.append(lookup_ct_by_pmid(pmid))
        time.sleep(CT_REQUEST_DELAY)

    ct_df = pd.DataFrame(ct_records)
    df = pd.concat([df.reset_index(drop=True), ct_df.reset_index(drop=True)], axis=1)

    linked = (df["nct_id"] != "").sum()
    log.info(f"  Linked {linked}/{len(df)} papers to ClinicalTrials.gov records")

    # ── Final column order ────────────────────────────────────────────────────
    final_cols = [
        # Identifiers
        "pmid", "nct_id", "year", "journal", "title",
        # Drug + indication
        "drug_name", "indication_clean",
        # Trial characteristics
        "trial_phase", "enrollment_size", "female_pct", "follow_up_duration",
        "comparator_type", "ct_status", "ct_phase", "ct_enrollment",
        # PICO fields
        "population", "intervention", "comparison",
        "outcomes_primary", "outcomes_secondary",
        # Sex flags
        "women_only_trial", "sex_stratified_outcomes", "female_enrollment_reported",
        # Notes + status
        "extraction_notes", "annotation_notes",
        "extraction_status", "annotation_status",
    ]
    # Keep only columns that exist
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"Saved CSV -> {OUTPUT_CSV} ({len(df)} rows)")

    # ── Save Excel ────────────────────────────────────────────────────────────
    log.info("Building Excel workbook...")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

        # Sheet 1: Full database with auto-filter
        df.to_excel(writer, sheet_name="All Trials", index=False)
        ws = writer.sheets["All Trials"]
        ws.auto_filter.ref = ws.dimensions

        # Freeze top row
        ws.freeze_panes = "A2"

        # Set column widths
        col_widths = {
            "pmid": 12, "nct_id": 14, "year": 6, "journal": 25, "title": 50,
            "drug_name": 16, "indication_clean": 22, "trial_phase": 12,
            "enrollment_size": 12, "female_pct": 10, "follow_up_duration": 16,
            "comparator_type": 18, "ct_status": 16, "ct_phase": 12,
            "ct_enrollment": 14, "population": 40, "intervention": 40,
            "comparison": 30, "outcomes_primary": 40, "outcomes_secondary": 40,
            "women_only_trial": 14, "sex_stratified_outcomes": 22,
            "female_enrollment_reported": 24, "extraction_notes": 30,
            "annotation_notes": 30, "extraction_status": 16, "annotation_status": 16,
        }
        for col_idx, col_name in enumerate(df.columns, 1):
            width = col_widths.get(col_name, 16)
            ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = width

        # Sheet 2: Sex-specific data only
        sex_df = df[
            (df["women_only_trial"].str.lower() == "yes") |
            (df["sex_stratified_outcomes"].str.lower() == "yes") |
            (df["female_enrollment_reported"].str.lower() == "yes")
        ].copy()
        sex_df.to_excel(writer, sheet_name="Sex Data Available", index=False)
        ws2 = writer.sheets["Sex Data Available"]
        ws2.auto_filter.ref = ws2.dimensions
        ws2.freeze_panes = "A2"

        # Sheet 3+: One sheet per indication
        indications = df["indication_clean"].dropna().unique()
        indications = sorted([i for i in indications if i and i != "Unknown"])

        for indication in indications:
            ind_df = df[df["indication_clean"] == indication].copy()
            # Excel sheet name max 31 chars; strip invalid characters: / \ ? * [ ] :
            sheet_name = indication.replace("/", "-").replace("\\", "-").replace("?", "").replace("*", "").replace("[", "").replace("]", "").replace(":", "")[:31]
            ind_df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws_ind = writer.sheets[sheet_name]
            ws_ind.auto_filter.ref = ws_ind.dimensions
            ws_ind.freeze_panes = "A2"

        # Summary sheet
        summary = df.groupby("indication_clean").agg(
            trials=("pmid", "count"),
            drugs=("drug_name", lambda x: ", ".join(sorted(x.dropna().unique()))),
            avg_enrollment=("enrollment_size", lambda x: round(
                pd.to_numeric(x, errors="coerce").mean(), 0)),
            avg_female_pct=("female_pct", lambda x: round(
                pd.to_numeric(x, errors="coerce").mean(), 1)),
            sex_stratified=("sex_stratified_outcomes", lambda x: (x.str.lower() == "yes").sum()),
            ct_linked=("nct_id", lambda x: (x != "").sum()),
        ).reset_index().rename(columns={"indication_clean": "Indication"})

        summary.columns = [
            "Indication", "N Trials", "Drugs Studied",
            "Avg Enrollment", "Avg Female %",
            "Sex-Stratified Outcomes", "CT.gov Linked"
        ]
        summary.to_excel(writer, sheet_name="Summary", index=False)
        ws_sum = writer.sheets["Summary"]
        ws_sum.freeze_panes = "A2"
        for col_idx in range(1, len(summary.columns) + 1):
            ws_sum.column_dimensions[ws_sum.cell(1, col_idx).column_letter].width = 22

    log.info(f"Saved Excel -> {OUTPUT_EXCEL}")
    log.info(f"  Sheets: All Trials | Sex Data Available | Summary | {len(indications)} indication sheets")

    log.info("\n" + "=" * 60)
    log.info("DATABASE SUMMARY")
    log.info(f"  Total trials:             {len(df)}")
    log.info(f"  Unique drugs:             {df['drug_name'].nunique()}")
    log.info(f"  Unique indications:       {df['indication_clean'].nunique()}")
    log.info(f"  CT.gov linked:            {linked} ({100*linked//len(df)}%)")
    log.info(f"  Women-only trials:        {(df['women_only_trial'].str.lower()=='yes').sum()}")
    log.info(f"  Sex-stratified outcomes:  {(df['sex_stratified_outcomes'].str.lower()=='yes').sum()}")
    log.info(f"  Female % reported:        {(df['female_enrollment_reported'].str.lower()=='yes').sum()}")
    log.info("=" * 60)
    log.info("\nPhase 3 complete.")

if __name__ == "__main__":
    run()
