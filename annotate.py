"""
Phase 3a: Annotation Extraction
=================================
Changes from original:
  - Uses full_text instead of abstract (same strategy as pico_extraction.py)
  - New subgroup fields: elderly, pediatric, racial_ethnic_minority,
    sex_specific, other_subgroups — to identify underrepresented populations
  - New outcome fields: patient_centered_outcomes, patient_centeredness_flag
    — to assess whether outcomes are patient-relevant
Reads glp1_corpus.csv (Phase 1 output), sends each abstract to Claude,
and extracts structured annotation fields:
  - indication, trial_phase, enrollment_size, female_pct,
    follow_up_duration, comparator_type
  - sex flags: women_only_trial, sex_stratified_outcomes, female_enrollment_reported

Output: annotations.csv (one row per paper)
Checkpoint: annotation_checkpoint.json (resume-safe)

Usage:
    pip install anthropic
    Set ANTHROPIC_API_KEY in script or as environment variable
    python annotate.py
    Run again to continue next batch.
"""

import anthropic
import csv
import json
import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = ""  # or paste key here

INPUT_CSV      = "glp1_corpus.csv"
OUTPUT_CSV     = "annotations.csv"
CHECKPOINT     = "annotation_checkpoint.json"
LOG_FILE       = "annotation_log.txt"

BATCH_SIZE     = 791       # papers per run; increase once validated
REQUEST_DELAY  = 1.0      # seconds between API calls
MAX_RETRIES    = 3
RETRY_BACKOFF  = 5

MODEL          = "claude-opus-4-5"
MAX_TOKENS     = 1500

FULLTEXT_CHAR_LIMIT = 60000

# ── Output fields ──────────────────────────────────────────────────────────────

OUTPUT_FIELDS = [
    # Identifiers
    "pmid", "title", "year", "journal", "text_source",

    "indication",               # e.g. Type 2 Diabetes, Obesity, NASH, CVD
    "trial_phase",              # Phase 2, Phase 3, Phase 4, NR (not reported)
    "enrollment_size",          # integer or empty
    "female_pct",               # 0-100 or empty if not reported
    "follow_up_duration",       # e.g. "52 weeks", "2 years", "NR"
    "comparator_type",          # placebo, active comparator, dose comparison, none
    "women_only_trial",         # yes / no
    "sex_stratified_outcomes",  # yes / no / unclear
    "female_enrollment_reported", # yes / no
    "subgroup_elderly",          # yes / no / unclear
    "subgroup_pediatric",       # yes / no / unclear
    "subgroup_racial_ethnic",
    "subgroup_sex_specific",
    "subgroup_renal_hepatic",
    "subgroup_other",
    "patient_centered_outcomes",
    "surrogate_outcomes",
    "patient_centeredness_flag",         # flags, ambiguities
    "annotation_status",        # success / failed
]

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical research assistant extracting structured metadata from RCT publications for systematic review.
 
Extract the following fields from the provided text. Use ONLY information explicitly stated — do not infer.
 
Return a JSON object with exactly these keys:
 
{
  "indication": "Primary clinical indication studied (e.g. Type 2 Diabetes, Obesity, Cardiovascular Disease, or combinations)",
  "trial_phase": "Phase 2 / Phase 3 / Phase 4 / NR",
  "enrollment_size": "Total number of participants randomized as integer, or NR",
  "female_pct": "Percentage of female participants as number only (e.g. 45.2), or NR",
  "follow_up_duration": "Follow-up duration as stated (e.g. 56 weeks, 2 years), or NR",
  "comparator_type": "Placebo / Active Comparator / Dose Comparison / Multiple / NR",
  "women_only_trial": "yes / no",
  "sex_stratified_outcomes": "yes / no / unclear",
  "female_enrollment_reported": "yes / no",
 
  "subgroup_elderly": "yes / no / unclear — was there a subgroup or focused enrolment of elderly patients (≥65 years)?",
  "subgroup_pediatric": "yes / no / unclear — were patients under 18 included or specifically studied?",
  "subgroup_racial_ethnic": "yes / no / unclear — was race or ethnicity reported or was a specific racial/ethnic group targeted?",
  "subgroup_sex_specific": "yes / no / unclear — was there a sex-specific subgroup analysis beyond overall female %?",
  "subgroup_renal_hepatic": "yes / no / unclear — were patients with renal or hepatic impairment specifically included or studied?",
  "subgroup_other": "Free text — any other notable subgroup explicitly studied (e.g. heart failure, CKD stage, BMI subgroup). Write 'none' if not applicable.",
 
  "patient_centered_outcomes": "Comma-separated list of outcomes that are patient-centered (e.g. quality of life, symptoms, mortality, hospitalization, functional status, patient-reported outcomes). Write 'none' if not present.",
  "surrogate_outcomes": "Comma-separated list of surrogate or biomarker outcomes (e.g. HbA1c, body weight, blood pressure, lipid levels, eGFR). Write 'none' if not present.",
  "patient_centeredness_flag": "One of: all_patient_centered / mostly_patient_centered / mixed / mostly_surrogate / all_surrogate — based on the balance of primary and secondary outcomes"
}
 
Patient-centered outcomes include: mortality, hospitalization, quality of life, functional status, symptoms, patient-reported outcomes (PROs), major adverse cardiovascular events (MACE), stroke, MI.
Surrogate outcomes include: HbA1c, fasting glucose, body weight, BMI, blood pressure, lipids, eGFR, biomarkers.
 
Return only valid JSON, no preamble or explanation."""

USER_TEMPLATE = """Extract annotation fields from this RCT text:

Title: {title}
Year: {year}
Journal: {journal}
Source: {text_source}

{text_label}:
{text}"""

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Full-text preprocessing (same logic as pico_extraction.py) ─────────────────
 
PRIORITY_SECTIONS = [
    "methods", "patients and methods", "study design", "participants",
    "eligibility", "inclusion", "exclusion", "randomization",
    "interventions", "treatment", "outcomes", "endpoints",
    "results", "primary outcome", "secondary outcome",
]
 
SKIP_SECTIONS = [
    "references", "acknowledgements", "acknowledgments",
    "funding", "conflicts of interest", "disclosures",
    "abbreviations", "supplementary", "appendix",
]

def extract_relevant_sections(full_text: str, text_source: str) -> tuple[str, str]:
    """Same strategy as pico_extraction.py — whole text if fits, sections if too long."""
    if not full_text:
        return "", "empty"
 
    if text_source != "fulltext":
        return full_text, "abstract"
 
    if len(full_text) <= FULLTEXT_CHAR_LIMIT:
        return full_text, "fulltext_complete"
 
    # Too long: extract priority sections
    lines = full_text.split("\n")
    sections = {}
    current_section = "preamble"
    current_lines = []
 
    for line in lines:
        stripped = line.strip()
        stripped_lower = stripped.lower().rstrip(":")
        is_header = (
            len(stripped) < 80
            and stripped_lower in PRIORITY_SECTIONS + SKIP_SECTIONS
        )
        if is_header:
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = stripped_lower
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()
 
    ordered_parts = []
    if "preamble" in sections and sections["preamble"]:
        ordered_parts.append(sections["preamble"])
    for key in PRIORITY_SECTIONS:
        if key in sections and sections[key]:
            ordered_parts.append(f"{key.upper()}:\n{sections[key]}")
 
    combined = "\n\n".join(ordered_parts)
    if len(combined) > FULLTEXT_CHAR_LIMIT:
        combined = combined[:FULLTEXT_CHAR_LIMIT] + "\n\n[truncated]"
 
    return combined, "fulltext_truncated"

# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if not Path(CHECKPOINT).exists():
        return set()
    with open(CHECKPOINT, encoding="utf-8") as f:
        return set(json.load(f).get("completed_pmids", []))

def save_checkpoint(completed: set):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"completed_pmids": list(completed)}, f)

# ── CSV helpers ────────────────────────────────────────────────────────────────
 
def init_output_csv() -> None:
    if not Path(OUTPUT_CSV).exists():
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=OUTPUT_FIELDS).writeheader()
 
 
def append_row(row: dict) -> None:
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writerow(row)

# ── API call with retry ────────────────────────────────────────────────────────

ANNOTATION_KEYS = [
    "indication", "trial_phase", "enrollment_size", "female_pct",
    "follow_up_duration", "comparator_type", "women_only_trial",
    "sex_stratified_outcomes", "female_enrollment_reported",
    "subgroup_elderly", "subgroup_pediatric", "subgroup_racial_ethnic",
    "subgroup_sex_specific", "subgroup_renal_hepatic", "subgroup_other",
    "patient_centered_outcomes", "surrogate_outcomes", "patient_centeredness_flag",
]

def annotate(client: anthropic.Anthropic, record: dict) -> dict:
    text_source = record.get("text_source", "abstract_only")
    raw_text = record.get("full_text", "").strip() or record.get("abstract", "").strip()
 
    processed_text, strategy = extract_relevant_sections(raw_text, text_source)
    log.info(f"    strategy: {strategy} ({len(processed_text):,} chars)")
 
    text_label = {
        "abstract":           "Abstract",
        "fulltext_complete":  "Full Text",
        "fulltext_truncated": "Full Text (priority sections, truncated)",
        "empty":              "No text available",
    }.get(strategy, "Text")
 
    prompt = USER_TEMPLATE.format(
        title       = record.get("title", ""),
        year        = record.get("year", ""),
        journal     = record.get("journal", ""),
        text_source = text_source,
        text_label  = text_label,
        text        = processed_text,
    )
 
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model      = MODEL,
                max_tokens = MAX_TOKENS,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
 
            parsed = json.loads(raw)
            for key in ANNOTATION_KEYS:
                parsed.setdefault(key, "NR")
            parsed["annotation_status"] = "success"
            return parsed
 
        except json.JSONDecodeError as e:
            log.warning(f"    JSON parse error (attempt {attempt}): {e}")
        except anthropic.RateLimitError:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(f"    Rate limit (attempt {attempt}) — waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            log.warning(f"    API error (attempt {attempt}): {e}")
            time.sleep(RETRY_BACKOFF)
 
    return {k: "NR" for k in ANNOTATION_KEYS} | {"annotation_status": "failed"}


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info(f"Phase 3a: Annotation Extraction -- {datetime.now().isoformat()}")
    log.info(f"Batch size: {BATCH_SIZE}")
    log.info("=" * 60)

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    with open(INPUT_CSV, encoding="utf-8") as f:
        corpus = list(csv.DictReader(f))
    log.info(f"Corpus: {len(corpus)} papers")

    fulltext_count = sum(1 for r in corpus if r.get("text_source") == "fulltext")
    log.info(f"Text sources: {fulltext_count} full-text | "
             f"{len(corpus) - fulltext_count} abstract-only")
    
    completed = load_checkpoint()
    log.info(f"Already annotated: {len(completed)}")

    remaining = [r for r in corpus if r["pmid"] not in completed]
    batch     = remaining[:BATCH_SIZE]
    log.info(f"Processing {len(batch)} papers this run\n")

    if not batch:
        log.info("All papers annotated. Phase 3a complete!")
        return

    init_output_csv()
 
    success_count = 0
    failed_count  = 0
 
    for i, record in enumerate(batch, 1):
        pmid        = record["pmid"]
        text_source = record.get("text_source", "abstract_only")
        log.info(f"[{i}/{len(batch)}] PMID {pmid} ({text_source})")
 
        result = annotate(client, record)
 
        row = {
            "pmid":        pmid,
            "title":       record.get("title", ""),
            "year":        record.get("year", ""),
            "journal":     record.get("journal", ""),
            "text_source": text_source,
            **result,
        }
 
        append_row(row)
        completed.add(pmid)
        save_checkpoint(completed)
 
        if result["annotation_status"] == "success":
            success_count += 1
            log.info(f"    ✓ annotated")
        else:
            failed_count += 1
            log.warning(f"    ✗ failed")
 
        time.sleep(REQUEST_DELAY)


    log.info("\n" + "=" * 60)
    log.info("BATCH SUMMARY")
    log.info(f"  Processed:       {len(batch)}")
    log.info(f"  Successful:      {success_count}")
    log.info(f"  Failed:          {failed_count}")
    log.info(f"  Total annotated: {len(completed)} / {len(corpus)}")
    log.info(f"  Remaining:       {len(corpus) - len(completed)}")
    log.info("=" * 60)

    if len(completed) < len(corpus):
        log.info(f"\nRun again to process the next {BATCH_SIZE} papers.")
    else:
        log.info("\nAll papers annotated. Run build_database.py to build the final database.")

if __name__ == "__main__":
    run()