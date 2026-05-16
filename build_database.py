"""
Phase 3b: Build Database (Updated)
====================================
Changes from original:
  - Incorporates new subgroup fields from annotate.py:
    subgroup_elderly, subgroup_pediatric, subgroup_racial_ethnic,
    subgroup_sex_specific, subgroup_renal_hepatic, subgroup_other
  - Incorporates new outcome fields:
    patient_centered_outcomes, surrogate_outcomes, patient_centeredness_flag
  - New Excel sheet: "Underrepresented Populations" — trials with any subgroup gap
  - New Excel sheet: "Surrogate-Heavy Trials" — trials flagged mostly/all surrogate
  - Updated Summary sheet to include subgroup + patient-centeredness columns
  - Excel sheet name sanitization (strips invalid characters)

Merges:
  glp1_corpus.csv     (Phase 1)
  pico_results.csv    (Phase 2)
  annotations.csv     (Phase 3a)

Outputs:
  glp1_database.csv   — flat database, one row per trial
  glp1_database.xlsx  — Excel workbook with multiple analysis sheets

Usage:
    pip install pandas openpyxl requests
    python build_database.py
"""

import csv
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ── Configuration ──────────────────────────────────────────────────────────────

CORPUS_CSV      = "glp1_corpus.csv"
PICO_CSV        = "pico_results.csv"
ANNOTATIONS_CSV = "annotations.csv"
OUTPUT_CSV      = "glp1_database.csv"
OUTPUT_EXCEL    = "glp1_database.xlsx"
LOG_FILE        = "build_database_log.txt"

CT_REQUEST_DELAY = 0.5   # seconds between ClinicalTrials.gov API calls

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Drug normalization ─────────────────────────────────────────────────────────

DRUG_MAP = {
    # Semaglutide
    "semaglutide": "Semaglutide", "ozempic": "Semaglutide",
    "wegovy": "Semaglutide", "rybelsus": "Semaglutide",
    # Liraglutide
    "liraglutide": "Liraglutide", "victoza": "Liraglutide", "saxenda": "Liraglutide",
    # Tirzepatide
    "tirzepatide": "Tirzepatide", "mounjaro": "Tirzepatide", "zepbound": "Tirzepatide",
    # Dulaglutide
    "dulaglutide": "Dulaglutide", "trulicity": "Dulaglutide",
    # Exenatide
    "exenatide": "Exenatide", "byetta": "Exenatide",
    "bydureon": "Exenatide", "bydureon bcise": "Exenatide",
    # Albiglutide
    "albiglutide": "Albiglutide", "tanzeum": "Albiglutide", "eperzan": "Albiglutide",
    # Lixisenatide
    "lixisenatide": "Lixisenatide", "adlyxin": "Lixisenatide", "lyxumia": "Lixisenatide",
    # Efpeglenatide
    "efpeglenatide": "Efpeglenatide",
    # Oral semaglutide
    "oral semaglutide": "Semaglutide (oral)",
}


def normalize_drug(text: str) -> str:
    text_lower = text.lower()
    found = []
    for keyword, canonical in DRUG_MAP.items():
        if keyword in text_lower and canonical not in found:
            found.append(canonical)
    return "; ".join(found) if found else "Other/NR"

# ── Indication normalization ───────────────────────────────────────────────────

INDICATION_MAP = {
    "type 2 diabetes": "Type 2 Diabetes",
    "t2dm": "Type 2 Diabetes",
    "t2d": "Type 2 Diabetes",
    "diabetes mellitus": "Type 2 Diabetes",
    "obesity": "Obesity",
    "overweight": "Obesity",
    "weight management": "Obesity",
    "cardiovascular": "Cardiovascular Disease",
    "heart failure": "Heart Failure",
    "nafld": "NASH-NAFLD",
    "nash": "NASH-NAFLD",
    "chronic kidney": "Chronic Kidney Disease",
    "ckd": "Chronic Kidney Disease",
    "renal": "Chronic Kidney Disease",
    "polycystic ovary": "PCOS",
    "pcos": "PCOS",
    "alzheimer": "Neurodegenerative Disease",
    "parkinson": "Neurodegenerative Disease",
}


def normalize_indication(text: str) -> str:
    text_lower = str(text).lower()
    found = []
    for keyword, canonical in INDICATION_MAP.items():
        if keyword in text_lower and canonical not in found:
            found.append(canonical)
    if not found:
        return "Other/NR"
    return "; ".join(found)

# ── Numeric cleaning ───────────────────────────────────────────────────────────

def clean_int(val) -> str:
    try:
        return str(int(float(str(val).replace(",", "").strip())))
    except Exception:
        return ""


def clean_float(val) -> str:
    try:
        return str(round(float(str(val).strip()), 1))
    except Exception:
        return ""

# ── Comparator normalization ───────────────────────────────────────────────────

def clean_comparator(val: str) -> str:
    v = str(val).lower().strip()
    if "placebo" in v:                      return "Placebo"
    if "active" in v or "comparator" in v:  return "Active Comparator"
    if "dose" in v:                         return "Dose Comparison"
    if "multiple" in v:                     return "Multiple"
    if v in ("none", "nr", ""):             return "NR"
    return str(val).strip().title()

# ── Patient-centeredness normalization ─────────────────────────────────────────

CENTEREDNESS_ORDER = [
    "all_patient_centered",
    "mostly_patient_centered",
    "mixed",
    "mostly_surrogate",
    "all_surrogate",
]


def is_surrogate_heavy(flag: str) -> bool:
    return str(flag).lower() in ("mostly_surrogate", "all_surrogate")


def is_patient_centered(flag: str) -> bool:
    return str(flag).lower() in ("all_patient_centered", "mostly_patient_centered")

# ── Subgroup flag helpers ──────────────────────────────────────────────────────

SUBGROUP_COLS = [
    "subgroup_elderly", "subgroup_pediatric", "subgroup_racial_ethnic",
    "subgroup_sex_specific", "subgroup_renal_hepatic",
]


def has_any_subgroup(row) -> bool:
    """True if any subgroup field is 'yes'."""
    return any(str(row.get(col, "")).lower() == "yes" for col in SUBGROUP_COLS)


def missing_subgroups(row) -> str:
    """Return comma-separated list of subgroup dimensions with no/unclear."""
    missing = []
    labels = {
        "subgroup_elderly": "Elderly",
        "subgroup_pediatric": "Pediatric",
        "subgroup_racial_ethnic": "Racial/Ethnic",
        "subgroup_sex_specific": "Sex-specific",
        "subgroup_renal_hepatic": "Renal/Hepatic",
    }
    for col, label in labels.items():
        if str(row.get(col, "")).lower() in ("no", "unclear", "nr", ""):
            missing.append(label)
    return "; ".join(missing)

# ── ClinicalTrials.gov linkage ─────────────────────────────────────────────────

CT_API = "https://clinicaltrials.gov/api/v2/studies"


def lookup_ct_by_pmid(pmid: str) -> dict:
    """Best-effort lookup of a ClinicalTrials.gov record by PMID."""
    empty = {"nct_id": "", "ct_status": "", "ct_phase": "", "ct_enrollment": ""}
    try:
        params = {
            "query.term": f"PMID{pmid}",
            "fields": "NCTId,OverallStatus,Phase,EnrollmentCount",
            "pageSize": 1,
        }
        resp = requests.get(CT_API, params=params, timeout=10)
        resp.raise_for_status()
        data    = resp.json()
        studies = data.get("studies", [])
        if not studies:
            return empty
        s = studies[0].get("protocolSection", {})
        return {
            "nct_id":        s.get("identificationModule", {}).get("nctId", ""),
            "ct_status":     s.get("statusModule", {}).get("overallStatus", ""),
            "ct_phase":      "; ".join(s.get("designModule", {}).get("phases", [])),
            "ct_enrollment": str(s.get("designModule", {}).get("enrollmentInfo", {}).get("count", "")),
        }
    except Exception:
        return empty

# ── Excel helpers ──────────────────────────────────────────────────────────────

def safe_sheet_name(name: str) -> str:
    """Sanitize string for use as Excel sheet name (max 31 chars, no / \\ ? * [ ] :)."""
    name = re.sub(r'[/\\?*\[\]:]', '-', name)
    return name[:31]


def write_sheet(writer, df: pd.DataFrame, sheet_name: str) -> None:
    name = safe_sheet_name(sheet_name)
    df.to_excel(writer, sheet_name=name, index=False)
    ws = writer.sheets[name]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_idx in range(1, len(df.columns) + 1):
        ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = 22

# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info(f"Phase 3b: Build Database — {datetime.now().isoformat()}")
    log.info("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────────
    log.info("Loading input files...")
    corpus      = pd.read_csv(CORPUS_CSV,      dtype=str).fillna("")
    pico        = pd.read_csv(PICO_CSV,        dtype=str).fillna("")
    annotations = pd.read_csv(ANNOTATIONS_CSV, dtype=str).fillna("")

    log.info(f"  Corpus:      {len(corpus)} rows")
    log.info(f"  PICO:        {len(pico)} rows")
    log.info(f"  Annotations: {len(annotations)} rows")

    # Deduplicate — keep last run's result for each PMID
    corpus      = corpus.drop_duplicates(subset="pmid", keep="last")
    pico        = pico.drop_duplicates(subset="pmid", keep="last")
    annotations = annotations.drop_duplicates(subset="pmid", keep="last")

    corpus      = corpus[corpus["pmid"].str.strip() != ""]
    pico        = pico[pico["pmid"].str.strip() != ""]
    annotations = annotations[annotations["pmid"].str.strip() != ""]

    log.info(f"  After dedup — Corpus: {len(corpus)} | PICO: {len(pico)} | Annotations: {len(annotations)}")
    # ── Merge ─────────────────────────────────────────────────────────────────
    log.info("Merging...")

    
    # Drop duplicate metadata columns from pico and annotations before merging
    # Drop overlapping columns before merging to avoid duplicates
    pico_clean = pico.drop(columns=[c for c in pico.columns 
                                    if c in corpus.columns and c != "pmid"])
    ann_clean  = annotations.drop(columns=[c for c in annotations.columns 
                                            if c in corpus.columns and c != "pmid"])
    ann_clean  = ann_clean.drop(columns=[c for c in ann_clean.columns 
                                        if c in pico_clean.columns and c != "pmid"])

    
    df = (corpus
    .merge(pico_clean,  on="pmid", how="left")
    .merge(ann_clean,   on="pmid", how="left"))

    df = df.fillna("")

    # Remove empty rows after merge
    before = len(df)
    df = df[df["pmid"].str.strip() != ""]
    log.info(f"  Removed {before - len(df)} empty rows after merge")
    log.info(f"  Merged: {len(df)} rows, {len(df.columns)} columns")
    

    # ── Normalize ─────────────────────────────────────────────────────────────
    log.info("Normalizing drug names and indications...")
    df["drug_name"]       = (df.get("intervention", "") + " " + df["title"]).apply(normalize_drug)
    df["indication_clean"] = df["indication"].apply(normalize_indication)
    df["enrollment_size"] = df["enrollment_size"].apply(clean_int)
    df["female_pct"]      = df["female_pct"].apply(clean_float)
    df["comparator_type"] = df["comparator_type"].apply(clean_comparator)

    # ── Subgroup gap columns ──────────────────────────────────────────────────
    log.info("Computing subgroup gap flags...")
    df["has_any_subgroup"]  = df.apply(has_any_subgroup, axis=1)
    df["missing_subgroups"] = df.apply(missing_subgroups, axis=1)

    # ── ClinicalTrials.gov linkage ────────────────────────────────────────────
    log.info("Linking ClinicalTrials.gov (best-effort)...")
    ct_records = []
    for i, pmid in enumerate(df["pmid"], 1):
        if i % 100 == 0:
            log.info(f"  CT lookup: {i}/{len(df)}")
        ct_records.append(lookup_ct_by_pmid(pmid))
        time.sleep(CT_REQUEST_DELAY)

    ct_df = pd.DataFrame(ct_records)
    df    = pd.concat([df.reset_index(drop=True), ct_df.reset_index(drop=True)], axis=1)

    linked = (df["nct_id"] != "").sum()
    log.info(f"  Linked {linked}/{len(df)} papers to ClinicalTrials.gov records")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"Saved CSV -> {OUTPUT_CSV} ({len(df)} rows)")

    # ── Build Excel ───────────────────────────────────────────────────────────
    log.info("Building Excel workbook...")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

        # Sheet 1: All Trials
        write_sheet(writer, df, "All Trials")

        # Sheet 2: Sex Data Available
        sex_df = df[
            (df["sex_stratified_outcomes"].str.lower() == "yes") |
            (df["women_only_trial"].str.lower() == "yes") |
            (df["female_enrollment_reported"].str.lower() == "yes")
        ]
        write_sheet(writer, sex_df, "Sex Data Available")

        # Sheet 3: Underrepresented Populations (NEW)
        # Trials missing at least 3 subgroup dimensions (i.e. under-studied populations)
        under_df = df[df["missing_subgroups"].str.count(";") >= 2].copy()
        write_sheet(writer, under_df, "Underrepresented Populations")

        # Sheet 4: Surrogate-Heavy Trials (NEW)
        surrogate_df = df[df["patient_centeredness_flag"].apply(is_surrogate_heavy)]
        write_sheet(writer, surrogate_df, "Surrogate-Heavy Trials")

        # Sheet 5: Patient-Centered Trials (NEW)
        pc_df = df[df["patient_centeredness_flag"].apply(is_patient_centered)]
        write_sheet(writer, pc_df, "Patient-Centered Trials")

        # Sheet 6: Summary by indication
        indications = sorted(df["indication_clean"].unique())
        summary = (
            df.groupby("indication_clean")
            .agg(
                n_trials        = ("pmid", "count"),
                drugs_studied   = ("drug_name", lambda x: "; ".join(sorted(x.dropna().unique()))),
                avg_enrollment  = ("enrollment_size", lambda x:
                                   round(pd.to_numeric(x, errors="coerce").mean(), 0)),
                avg_female_pct  = ("female_pct", lambda x:
                                   round(pd.to_numeric(x, errors="coerce").mean(), 1)),
                sex_stratified  = ("sex_stratified_outcomes",
                                   lambda x: (x.str.lower() == "yes").sum()),
                ct_linked       = ("nct_id", lambda x: (x != "").sum()),
                # NEW summary columns
                has_subgroup    = ("has_any_subgroup",
                                   lambda x: x.sum()),
                surrogate_heavy = ("patient_centeredness_flag",
                                   lambda x: x.apply(is_surrogate_heavy).sum()),
                patient_centered= ("patient_centeredness_flag",
                                   lambda x: x.apply(is_patient_centered).sum()),
            )
            .reset_index()
            .rename(columns={"indication_clean": "Indication"})
        )
        summary.columns = [
            "Indication", "N Trials", "Drugs Studied",
            "Avg Enrollment", "Avg Female %",
            "Sex-Stratified Outcomes", "CT.gov Linked",
            "Has Subgroup Data", "Surrogate-Heavy", "Patient-Centered",
        ]
        write_sheet(writer, summary, "Summary")

        # Per-indication sheets
        for indication in indications:
            ind_df = df[df["indication_clean"] == indication]
            write_sheet(writer, ind_df, indication)

    log.info(f"Saved Excel -> {OUTPUT_EXCEL}")
    log.info(f"  Sheets: All Trials | Sex Data Available | "
             f"Underrepresented Populations | Surrogate-Heavy Trials | "
             f"Patient-Centered Trials | Summary | {len(indications)} indication sheets")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("DATABASE SUMMARY")
    log.info(f"  Total trials:                  {len(df)}")
    log.info(f"  Unique drugs:                  {df['drug_name'].nunique()}")
    log.info(f"  Unique indications:            {df['indication_clean'].nunique()}")
    log.info(f"  CT.gov linked:                 {linked} ({100*linked//len(df)}%)")
    log.info(f"  Women-only trials:             {(df['women_only_trial'].str.lower()=='yes').sum()}")
    log.info(f"  Sex-stratified outcomes:       {(df['sex_stratified_outcomes'].str.lower()=='yes').sum()}")
    log.info(f"  Female % reported:             {(df['female_enrollment_reported'].str.lower()=='yes').sum()}")
    log.info(f"  Trials with subgroup data:     {df['has_any_subgroup'].sum()}")
    log.info(f"  Surrogate-heavy trials:        {df['patient_centeredness_flag'].apply(is_surrogate_heavy).sum()}")
    log.info(f"  Patient-centered trials:       {df['patient_centeredness_flag'].apply(is_patient_centered).sum()}")
    log.info("=" * 60)
    log.info("\nPhase 3b complete. Next: run map_outcomes.py (SNOMED-CT mapping)")


if __name__ == "__main__":
    run()