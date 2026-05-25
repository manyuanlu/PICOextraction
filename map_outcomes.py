"""
map_outcomes.py — Phase 4b: SNOMED-CT Outcome Mapping
======================================================
GLP-1 RCT Evidence Pipeline | Dr. Chunhua Weng Lab

PURPOSE
-------
Maps free-text primary and secondary outcome strings from glp1_database.csv
to canonical SNOMED-CT concepts using the Claude API, then:
  1. Produces a per-trial mapping table (snomed_trial_outcomes.csv)
  2. Produces a concept-level frequency table (snomed_concept_frequency_{sheet}.csv)
  3. Appends snomed_primary_code and snomed_primary_label columns to
     glp1_database_snomed.csv for downstream use in Phase 4 notebooks

APPROACH
--------
Each unique outcome string is sent to Claude once (deduplicated), which returns:
  - snomed_code      : SNOMED-CT concept ID (e.g. 43396009)
  - snomed_label     : Canonical preferred term (e.g. "HbA1c measurement")
  - domain           : Clinical domain (Metabolic | Cardiovascular | Renal |
                       Patient-Reported | Anthropometric | Safety | Other)
  - patient_centered : yes | no | partial
  - confidence       : high | medium | low

Results are cached to snomed_cache.json so the script is resumable —
re-running only maps new/uncached strings.

USAGE
-----
  python map_outcomes.py [--db glp1_database.csv] [--out-dir outputs/]
                         [--batch-size 50] [--dry-run]

REQUIREMENTS
------------
  pip install anthropic pandas tqdm
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd
from tqdm import tqdm


ANTHROPIC_API_KEY = "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"
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL            = "claude-sonnet-4-20250514"
MAX_TOKENS       = 12000
RETRY_LIMIT      = 3
RETRY_DELAY      = 5          # seconds between retries
BATCH_SIZE       = 50         # unique strings per API call
CACHE_FILE       = "snomed_cache.json"

# SNOMED-CT concepts for the most common GLP-1 trial outcomes
# Used as few-shot examples in the prompt
FEW_SHOT_EXAMPLES = [
    {"text": "change in HbA1c from baseline",
     "snomed_code": "43396009", "snomed_label": "HbA1c measurement",
     "domain": "Metabolic", "patient_centered": "no", "confidence": "high"},
    {"text": "body weight reduction",
     "snomed_code": "27113001", "snomed_label": "Body weight",
     "domain": "Anthropometric", "patient_centered": "no", "confidence": "high"},
    {"text": "major adverse cardiovascular events",
     "snomed_code": "276746005", "snomed_label": "Cardiovascular event",
     "domain": "Cardiovascular", "patient_centered": "yes", "confidence": "high"},
    {"text": "fasting plasma glucose",
     "snomed_code": "33747003", "snomed_label": "Glucose measurement, fasting",
     "domain": "Metabolic", "patient_centered": "no", "confidence": "high"},
    {"text": "health-related quality of life",
     "snomed_code": "363870007", "snomed_label": "Quality of life",
     "domain": "Patient-Reported", "patient_centered": "yes", "confidence": "high"},
    {"text": "systolic blood pressure",
     "snomed_code": "271649006", "snomed_label": "Systolic blood pressure",
     "domain": "Cardiovascular", "patient_centered": "no", "confidence": "high"},
    {"text": "all-cause mortality",
     "snomed_code": "419620001", "snomed_label": "Death",
     "domain": "Patient-Reported", "patient_centered": "yes", "confidence": "high"},
    {"text": "eGFR",
     "snomed_code": "80274001", "snomed_label": "Glomerular filtration rate",
     "domain": "Renal", "patient_centered": "no", "confidence": "high"},
    {"text": "waist circumference",
     "snomed_code": "276361009", "snomed_label": "Waist circumference",
     "domain": "Anthropometric", "patient_centered": "no", "confidence": "high"},
    {"text": "nausea",
     "snomed_code": "422587007", "snomed_label": "Nausea",
     "domain": "Safety", "patient_centered": "yes", "confidence": "high"},
]

SYSTEM_PROMPT = """You are a clinical informatics expert specialising in SNOMED-CT terminology.
Your task is to map free-text outcome strings from randomised controlled trial records
to the single best-fitting SNOMED-CT concept.

Rules:
1. Return ONLY a JSON array — no prose, no markdown fences, no explanation.
2. Each element must have exactly these keys:
   text, snomed_code, snomed_label, domain, patient_centered, confidence
3. snomed_code: a real SNOMED-CT concept ID (numeric string). If genuinely uncertain,
   use "000000000" and confidence "low".
4. snomed_label: the SNOMED-CT preferred term in English.
5. domain: one of Metabolic | Cardiovascular | Renal | Patient-Reported |
   Anthropometric | Safety | Other
6. patient_centered: "yes" if the outcome matters directly to patients
   (death, hospitalisation, QoL, symptoms, function); "no" for biomarkers/surrogates;
   "partial" for composites that mix both.
7. confidence: "high" | "medium" | "low"
8. Map each string individually; do not merge or skip any.
9. For compound strings (e.g. "HbA1c and body weight"), map to the PRIMARY concept only.
"""


def build_user_prompt(texts: list[str]) -> str:
    examples_str = json.dumps(FEW_SHOT_EXAMPLES, indent=2)
    texts_str    = json.dumps(texts, indent=2)
    return (
        f"Examples of correct mappings:\n{examples_str}\n\n"
        f"Now map each of the following {len(texts)} outcome strings "
        f"using the same format:\n{texts_str}"
    )


# ── Claude API call ───────────────────────────────────────────────────────────
def call_claude(client: anthropic.Anthropic, texts: list[str]) -> list[dict]:
    """Call Claude and return parsed list of mapping dicts."""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(texts)}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            mappings = json.loads(raw)
            # Validate length matches
            if len(mappings) != len(texts):
                log.warning(
                    f"Length mismatch: sent {len(texts)}, got {len(mappings)} — padding with unknowns"
                )
                existing = {m["text"]: m for m in mappings if "text" in m}
                mappings = [
                    existing.get(t, _unknown(t)) for t in texts
                ]
            return mappings
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Parse error on attempt {attempt}/{RETRY_LIMIT}: {e}")
        except anthropic.RateLimitError:
            log.warning(f"Rate limit — sleeping {RETRY_DELAY * attempt}s")
            time.sleep(RETRY_DELAY * attempt)
        except anthropic.APIError as e:
            log.error(f"API error: {e}")
            time.sleep(RETRY_DELAY)
        if attempt < RETRY_LIMIT:
            time.sleep(RETRY_DELAY)
    log.error(f"All {RETRY_LIMIT} attempts failed for batch of {len(texts)}")
    return [_unknown(t) for t in texts]


def _unknown(text: str) -> dict:
    return {
        "text": text,
        "snomed_code": "000000000",
        "snomed_label": "UNMAPPED",
        "domain": "Other",
        "patient_centered": "no",
        "confidence": "low",
    }


# ── Cache helpers ─────────────────────────────────────────────────────────────
def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


# ── Text normalisation ────────────────────────────────────────────────────────
def normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip trailing punctuation."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(".,;:")


def extract_outcome_strings(df: pd.DataFrame) -> list[str]:
    """Pull unique, non-null outcome strings from primary and secondary columns."""
    strings: set[str] = set()
    for col in ["outcomes_primary", "outcomes_secondary"]:
        if col not in df.columns:
            continue
        for val in df[col].dropna():
            # Outcomes may be semicolon- or comma-separated lists
            for part in re.split(r"[;|]+", str(val)):
                n = normalise(part)
                if n and len(n) > 3:
                    strings.add(n)
    return sorted(strings)


# ── Main ──────────────────────────────────────────────────────────────────────
def run(
    db_path:    str  = "glp1_database.csv",
    out_dir:    str  = "outputs",
    batch_size: int  = BATCH_SIZE,
    dry_run:    bool = False,
) -> None:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load database ──────────────────────────────────────────────────────────
    log.info(f"Loading database: {db_path}")
    df = pd.read_csv(db_path, low_memory=False)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    log.info(f"  {len(df):,} trials loaded")

    # ── Extract unique outcome strings ─────────────────────────────────────────
    unique_strings = extract_outcome_strings(df)
    log.info(f"  {len(unique_strings):,} unique outcome strings to map")

    if dry_run:
        log.info("Dry-run mode — showing first 10 strings, no API calls.")
        for s in unique_strings[:10]:
            print(f"  {s}")
        return

    # ── Load cache ─────────────────────────────────────────────────────────────
    cache_path = str(out / CACHE_FILE)
    cache      = load_cache(cache_path)
    log.info(f"  Cache: {len(cache):,} already mapped")

    unmapped = [s for s in unique_strings if s not in cache]
    log.info(f"  {len(unmapped):,} strings need API calls")

    # ── Initialise Claude client ───────────────────────────────────────────────
    api_key = ANTHROPIC_API_KEY
    if not api_key and unmapped:
        log.error("ANTHROPIC_API_KEY not set. Export it and re-run.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # ── Batch API calls ────────────────────────────────────────────────────────
    batches = [unmapped[i:i + batch_size] for i in range(0, len(unmapped), batch_size)]
    log.info(f"  Processing {len(batches)} batches of ≤{batch_size} strings each")

    for i, batch in enumerate(tqdm(batches, desc="Mapping outcomes", unit="batch")):
        results = call_claude(client, batch)
        for mapping in results:
            key = normalise(mapping.get("text", ""))
            cache[key] = mapping
        # Save after every batch so restarts are cheap
        save_cache(cache, cache_path)
        # Polite pause to avoid rate limits
        if i < len(batches) - 1:
            time.sleep(1)

    log.info(f"Mapping complete. Cache size: {len(cache):,}")

    # ── Build per-trial output ─────────────────────────────────────────────────
    log.info("Building per-trial mapping table...")
    rows = []
    for _, trial in df.iterrows():
        pmid = trial.get("pmid", "")
        for col_key, outcome_type in [
            ("outcomes_primary",   "primary"),
            ("outcomes_secondary", "secondary"),
        ]:
            if col_key not in df.columns:
                continue
            raw = trial.get(col_key, "")
            if not isinstance(raw, str) or not raw.strip():
                continue
            for part in re.split(r"[;|]+", raw):
                n = normalise(part)
                if not n or len(n) <= 3:
                    continue
                mapping = cache.get(n, _unknown(n))
                rows.append({
                    "pmid":            pmid,
                    "drug_name":       trial.get("drug_name", ""),
                    "indication_clean":trial.get("indication_clean", ""),
                    "outcome_type":    outcome_type,
                    "outcome_text_raw":part.strip(),
                    "outcome_text_norm":n,
                    "snomed_code":     mapping.get("snomed_code", ""),
                    "snomed_label":    mapping.get("snomed_label", ""),
                    "domain":          mapping.get("domain", ""),
                    "patient_centered":mapping.get("patient_centered", ""),
                    "confidence":      mapping.get("confidence", ""),
                })

    trial_df = pd.DataFrame(rows)
    trial_csv = out / "snomed_trial_outcomes.csv"
    trial_df.to_csv(trial_csv, index=False)
    log.info(f"  Saved: {trial_csv}  ({len(trial_df):,} rows)")

    # ── Concept frequency table ────────────────────────────────────────────────
    log.info("Building concept frequency table...")
    concept_df = (
        trial_df.groupby(
            ["snomed_code", "snomed_label", "domain", "patient_centered", "confidence"],
            dropna=False,
        )
        .agg(
            n_trials   =("pmid", "nunique"),
            n_mappings =("pmid", "count"),
            pct_of_corpus=("pmid", lambda x: round(x.nunique() / len(df) * 100, 2)),
        )
        .reset_index()
        .sort_values("n_trials", ascending=False)
    )

    # ── Per-domain breakdown ───────────────────────────────────────────────────
    domain_df = (
        trial_df.groupby("domain")
        .agg(
            n_mappings =("pmid", "count"),
            n_trials   =("pmid", "nunique"),
            pct_trials =("pmid", lambda x: round(x.nunique() / len(df) * 100, 1)),
        )
        .reset_index()
        .sort_values("n_trials", ascending=False)
    )

    # ── Patient-centeredness breakdown ─────────────────────────────────────────
    pco_df = (
        trial_df[trial_df["outcome_type"] == "primary"]
        .groupby("patient_centered")
        .agg(
            n_mappings=("pmid", "count"),
            n_trials  =("pmid", "nunique"),
        )
        .reset_index()
        .sort_values("n_trials", ascending=False)
    )

    # ── Rare outcomes (appear in ≤2 trials) ────────────────────────────────────
    rare_df = concept_df[concept_df["n_trials"] <= 2].copy()

    # ── Unmapped outcomes ──────────────────────────────────────────────────────
    unmapped_df = concept_df[concept_df["snomed_code"] == "000000000"].copy()

    # ── Gap analysis: outcomes never or rarely seen ────────────────────────────
    # Outcomes that appear in fewer than 1% of trials are evidence gaps
    threshold = max(1, round(len(df) * 0.01))
    gap_df = concept_df[
        (concept_df["patient_centered"].isin(["yes", "partial"])) &
        (concept_df["n_trials"] < threshold)
    ].copy()
    gap_df["gap_severity"] = gap_df["n_trials"].apply(
        lambda n: "Critical" if n <= 5 else ("High" if n <= 20 else "Moderate")
    )

    # ── Write Excel output ─────────────────────────────────────────────────────
    csv_sheets = {
        "all_concepts":        concept_df,
        "by_domain":           domain_df,
        "patient_centeredness":pco_df,
        "rare_outcomes":       rare_df,
        "pco_gaps":            gap_df,
        "unmapped":            unmapped_df,
    }
    saved_csvs = []
    for name, frame in csv_sheets.items():
        path = out / f"snomed_concept_frequency_{name}.csv"
        frame.to_csv(path, index=False)
        saved_csvs.append(path)
        log.info(f"  Saved: {path}")

    # ── Append SNOMED columns to main database ─────────────────────────────────
    log.info("Appending SNOMED columns to database copy...")

    # Primary outcome mapping only (one code per trial)
    primary_map = (
        trial_df[trial_df["outcome_type"] == "primary"]
        .groupby("pmid")
        .first()
        .reset_index()
        [["pmid", "snomed_code", "snomed_label", "domain", "patient_centered", "confidence"]]
        .rename(columns={
            "snomed_code":     "snomed_primary_code",
            "snomed_label":    "snomed_primary_label",
            "domain":          "snomed_primary_domain",
            "patient_centered":"snomed_primary_pco",
            "confidence":      "snomed_primary_confidence",
        })
    )

    df["pmid"] = df["pmid"].astype(str)
    primary_map["pmid"] = primary_map["pmid"].astype(str)
    df_enriched = df.merge(primary_map, on="pmid", how="left")

    enriched_path = out / "glp1_database_snomed.csv"
    df_enriched.to_csv(enriched_path, index=False)
    log.info(f"  Saved: {enriched_path}  ({len(df_enriched):,} rows, "
             f"{len(df_enriched.columns)} columns)")

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("SNOMED-CT MAPPING SUMMARY")
    log.info(f"  Total trials:               {len(df):,}")
    log.info(f"  Unique outcome strings:     {len(unique_strings):,}")
    log.info(f"  Concepts mapped:            {len(concept_df):,}")
    log.info(f"  Unmapped (low confidence):  {len(unmapped_df):,}")
    log.info(f"  Rare outcomes (≤2 trials):  {len(rare_df):,}")
    log.info(f"  PCO gap outcomes:           {len(gap_df):,}")
    log.info("")
    log.info("  Domain breakdown:")
    for _, row in domain_df.iterrows():
        log.info(f"    {row['domain']:<22}  {row['n_trials']:>4,} trials  ({row['pct_trials']:.1f}%)")
    log.info("")
    log.info("  Patient-centeredness (primary outcomes):")
    for _, row in pco_df.iterrows():
        log.info(f"    {row['patient_centered']:<10}  {row['n_trials']:>4,} trials")
    log.info("=" * 65)
    log.info("Phase 4b complete.")
    log.info(f"  snomed_trial_outcomes.csv      → {trial_csv}")
    for p in saved_csvs:
        log.info(f"  {p.name:<45} → {p}")
    log.info(f"  glp1_database_snomed.csv       → {enriched_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4b: SNOMED-CT outcome mapping")
    parser.add_argument("--db",         default="glp1_database.csv",
                        help="Path to glp1_database.csv (default: glp1_database.csv)")
    parser.add_argument("--out-dir",    default="outputs",
                        help="Output directory (default: outputs/)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Strings per API call (default: {BATCH_SIZE})")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print unique outcome strings without calling the API")
    args = parser.parse_args()

    run(
        db_path    = args.db,
        out_dir    = args.out_dir,
        batch_size = args.batch_size,
        dry_run    = args.dry_run,
    )
