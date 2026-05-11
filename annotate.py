"""
Phase 3a: Annotation Extraction
=================================
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
import time
import logging
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = "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"  # or paste key here

INPUT_CSV      = "glp1_corpus.csv"
OUTPUT_CSV     = "annotations.csv"
CHECKPOINT     = "annotation_checkpoint.json"
LOG_FILE       = "annotation_log.txt"

BATCH_SIZE     = 241       # papers per run; increase once validated
REQUEST_DELAY  = 1.0      # seconds between API calls
MAX_RETRIES    = 3
RETRY_BACKOFF  = 5

MODEL          = "claude-opus-4-5"
MAX_TOKENS     = 800

# ── Output fields ──────────────────────────────────────────────────────────────

OUTPUT_FIELDS = [
    "pmid",
    "indication",               # e.g. Type 2 Diabetes, Obesity, NASH, CVD
    "trial_phase",              # Phase 2, Phase 3, Phase 4, NR (not reported)
    "enrollment_size",          # integer or empty
    "female_pct",               # 0-100 or empty if not reported
    "follow_up_duration",       # e.g. "52 weeks", "2 years", "NR"
    "comparator_type",          # placebo, active comparator, dose comparison, none
    "women_only_trial",         # yes / no
    "sex_stratified_outcomes",  # yes / no / unclear
    "female_enrollment_reported", # yes / no
    "annotation_notes",         # flags, ambiguities
    "annotation_status",        # success / failed
]

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical research assistant performing structured data extraction for a systematic review of GLP-1 receptor agonist trials.

Extract annotation fields from the abstract below using ONLY information explicitly stated. Do NOT infer or assume.

Rules:
- indication: the primary disease/condition being treated (e.g. "Type 2 Diabetes", "Obesity", "Cardiovascular Disease", "NASH", "Heart Failure"). Use standardized labels.
- trial_phase: report as "Phase 2", "Phase 3", "Phase 4", or "NR" if not reported.
- enrollment_size: total number of participants randomized as an integer. Empty string if not stated.
- female_pct: percentage of female participants as a number 0-100. Empty string if not reported.
- follow_up_duration: duration as stated (e.g. "52 weeks", "2 years", "6 months"). "NR" if not reported.
- comparator_type: one of "placebo", "active comparator", "dose comparison", "none", or "NR".
- women_only_trial: "yes" if all participants are female, otherwise "no".
- sex_stratified_outcomes: "yes" if results are reported separately by sex, "no" if not, "unclear" if ambiguous.
- female_enrollment_reported: "yes" if female percentage or count is stated anywhere, "no" if not.
- annotation_notes: brief notes on any ambiguities, missing fields, or flags. Empty string if none.

Respond ONLY with a valid JSON object in this exact format, no preamble, no markdown:
{
  "indication": "...",
  "trial_phase": "...",
  "enrollment_size": "...",
  "female_pct": "...",
  "follow_up_duration": "...",
  "comparator_type": "...",
  "women_only_trial": "...",
  "sex_stratified_outcomes": "...",
  "female_enrollment_reported": "...",
  "annotation_notes": "..."
}"""

USER_TEMPLATE = """Extract annotation fields from this RCT abstract:

Title: {title}

Abstract: {abstract}"""

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

# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if not Path(CHECKPOINT).exists():
        return set()
    with open(CHECKPOINT, encoding="utf-8") as f:
        return set(json.load(f).get("completed_pmids", []))

def save_checkpoint(completed: set):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"completed_pmids": list(completed)}, f)

# ── API call with retry ────────────────────────────────────────────────────────

ANNOTATION_KEYS = [
    "indication", "trial_phase", "enrollment_size", "female_pct",
    "follow_up_duration", "comparator_type", "women_only_trial",
    "sex_stratified_outcomes", "female_enrollment_reported", "annotation_notes"
]

def extract_annotation(client, pmid, title, abstract) -> dict:
    user_msg = USER_TEMPLATE.format(title=title, abstract=abstract)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}]
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)
            for k in ANNOTATION_KEYS:
                result.setdefault(k, "")
            result["annotation_status"] = "success"
            return result

        except json.JSONDecodeError as e:
            log.warning(f"  PMID {pmid} attempt {attempt}: JSON parse error - {e}")
            if attempt == MAX_RETRIES:
                return _failed(f"JSON parse error: {e}")

        except anthropic.RateLimitError:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(f"  PMID {pmid} attempt {attempt}: rate limited, waiting {wait}s")
            time.sleep(wait)
            if attempt == MAX_RETRIES:
                return _failed("Rate limit - max retries exceeded")

        except anthropic.APIError as e:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(f"  PMID {pmid} attempt {attempt}: API error {e}, waiting {wait}s")
            time.sleep(wait)
            if attempt == MAX_RETRIES:
                return _failed(f"API error: {e}")

    return _failed("Unknown failure")

def _failed(reason: str) -> dict:
    return {k: "" for k in ANNOTATION_KEYS} | {
        "annotation_notes": reason,
        "annotation_status": "failed"
    }

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

    completed = load_checkpoint()
    log.info(f"Already annotated: {len(completed)}")

    remaining = [r for r in corpus if r["pmid"] not in completed]
    batch     = remaining[:BATCH_SIZE]
    log.info(f"Processing {len(batch)} papers this run\n")

    if not batch:
        log.info("All papers annotated. Phase 3a complete!")
        return

    file_exists = Path(OUTPUT_CSV).exists()
    out_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDS)
    if not file_exists:
        writer.writeheader()

    success_count = fail_count = 0

    for i, paper in enumerate(batch, 1):
        pmid     = paper["pmid"]
        title    = paper.get("title", "")
        abstract = paper.get("abstract", "")

        log.info(f"[{i}/{len(batch)}] PMID {pmid}: {title[:60]}...")

        ann = extract_annotation(client, pmid, title, abstract)

        row = {"pmid": pmid} | {k: ann[k] for k in OUTPUT_FIELDS if k != "pmid"}
        writer.writerow(row)
        out_file.flush()

        if ann["annotation_status"] == "success":
            completed.add(pmid)
            save_checkpoint(completed)
            success_count += 1
            log.info(f"  -> success | {ann['indication']} | enroll={ann['enrollment_size']} | female={ann['female_pct']}%")
        else:
            fail_count += 1
            log.warning(f"  -> FAILED: {ann['annotation_notes']}")

        time.sleep(REQUEST_DELAY)

    out_file.close()

    log.info("\n" + "=" * 60)
    log.info("BATCH SUMMARY")
    log.info(f"  Processed:       {len(batch)}")
    log.info(f"  Successful:      {success_count}")
    log.info(f"  Failed:          {fail_count}")
    log.info(f"  Total annotated: {len(completed)} / {len(corpus)}")
    log.info(f"  Remaining:       {len(corpus) - len(completed)}")
    log.info("=" * 60)

    if len(completed) < len(corpus):
        log.info(f"\nRun again to process the next {BATCH_SIZE} papers.")
    else:
        log.info("\nAll papers annotated. Run build_database.py to build the final database.")

if __name__ == "__main__":
    run()
