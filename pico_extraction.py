"""
Phase 2: Batch LLM PICO Extraction
====================================
Reads glp1_corpus.csv (output of Phase 1), sends each abstract to
Claude via the Anthropic API using Prompt 3, and extracts structured
PICO fields. Saves results to pico_results.csv.

Features:
  - Configurable batch size (set BATCH_SIZE below)
  - Checkpoint file: resumes from last completed paper after any failure
  - Per-paper retry with exponential backoff on API errors
  - Structured JSON output parsed into individual PICO columns

Usage:
    pip install anthropic
    python pico_extraction.py

Set your Anthropic API key:
    Windows:  set ANTHROPIC_API_KEY=your_key_here
    Mac/Linux: export ANTHROPIC_API_KEY=your_key_here
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

INPUT_CSV      = "glp1_corpus.csv"       # output from Phase 1
OUTPUT_CSV     = "pico_results.csv"      # final results
CHECKPOINT     = "checkpoint.json"       # tracks last completed PMID
LOG_FILE       = "extraction_log.txt"

BATCH_SIZE     = 41      # how many papers to process per run (adjust as needed)
                          # run the script multiple times to process more
REQUEST_DELAY  = 1.0     # seconds between API calls (be gentle on rate limits)
MAX_RETRIES    = 3       # retries per paper on API failure
RETRY_BACKOFF  = 5       # seconds for first retry; doubles each attempt

MODEL          = "claude-opus-4-5"
MAX_TOKENS     = 1000

# ── Output fields ──────────────────────────────────────────────────────────────

OUTPUT_FIELDS = [
    "pmid", "title", "year", "journal",
    "population", "intervention", "comparison", "outcomes_primary",
    "outcomes_secondary", "extraction_notes", "extraction_status"
]

# ── Prompt 3 (your validated best prompt from the pilot) ──────────────────────

SYSTEM_PROMPT = """You are a clinical research assistant performing structured data extraction for systematic review.

Extract PICO elements using ONLY information explicitly stated in the abstract. Do NOT infer, interpret, or include results or numerical findings.

Rules:
- Population: include only eligibility-defining characteristics; do not include baseline statistics unless they define inclusion criteria
- Intervention: include drug name, dose, route, frequency, and duration
- Comparison: list all comparator groups explicitly
- Outcomes: list primary and secondary outcomes separately as definitions only — NOT results or numerical findings

Respond ONLY with a valid JSON object in this exact format, no preamble, no markdown:
{
  "population": "...",
  "intervention": "...",
  "comparison": "...",
  "outcomes_primary": "...",
  "outcomes_secondary": "...",
  "extraction_notes": "any flags, ambiguities, or missing fields (or empty string)"
}"""

USER_TEMPLATE = """Extract PICO elements from this RCT abstract:

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
    """Return set of PMIDs already successfully extracted."""
    if not Path(CHECKPOINT).exists():
        return set()
    with open(CHECKPOINT, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("completed_pmids", []))


def save_checkpoint(completed: set):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"completed_pmids": list(completed)}, f)


# ── API call with retry ────────────────────────────────────────────────────────

def extract_pico(client: anthropic.Anthropic, pmid: str, title: str, abstract: str) -> dict:
    """
    Call Claude API and parse JSON response into PICO fields.
    Returns a dict with PICO keys + extraction_status.
    """
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

            # Strip markdown fences if model adds them despite instructions
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            pico = json.loads(raw)

            # Validate expected keys present
            for key in ["population", "intervention", "comparison",
                        "outcomes_primary", "outcomes_secondary", "extraction_notes"]:
                pico.setdefault(key, "")

            pico["extraction_status"] = "success"
            return pico

        except json.JSONDecodeError as e:
            log.warning(f"  PMID {pmid} attempt {attempt}: JSON parse failed — {e}")
            if attempt == MAX_RETRIES:
                return _failed(f"JSON parse error after {MAX_RETRIES} attempts: {e}")

        except anthropic.RateLimitError:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(f"  PMID {pmid} attempt {attempt}: rate limited, waiting {wait}s")
            time.sleep(wait)
            if attempt == MAX_RETRIES:
                return _failed("Rate limit — max retries exceeded")

        except anthropic.APIError as e:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(f"  PMID {pmid} attempt {attempt}: API error {e}, waiting {wait}s")
            time.sleep(wait)
            if attempt == MAX_RETRIES:
                return _failed(f"API error after {MAX_RETRIES} attempts: {e}")

    return _failed("Unknown failure")


def _failed(reason: str) -> dict:
    return {
        "population": "", "intervention": "", "comparison": "",
        "outcomes_primary": "", "outcomes_secondary": "",
        "extraction_notes": reason,
        "extraction_status": "failed"
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info(f"Phase 2: PICO Extraction -- started {datetime.now().isoformat()}")
    log.info(f"Batch size: {BATCH_SIZE} papers per run")
    log.info("=" * 60)

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set. Add it to the script or set as environment variable.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Load corpus
    with open(INPUT_CSV, encoding="utf-8") as f:
        corpus = list(csv.DictReader(f))
    log.info(f"Corpus loaded: {len(corpus)} papers")

    # Load checkpoint
    completed = load_checkpoint()
    log.info(f"Already completed: {len(completed)} papers (from previous runs)")

    # Filter to papers not yet done
    remaining = [r for r in corpus if r["pmid"] not in completed]
    log.info(f"Remaining: {len(remaining)} papers")

    # Take this run's batch
    batch = remaining[:BATCH_SIZE]
    log.info(f"Processing {len(batch)} papers this run\n")

    if not batch:
        log.info("Nothing left to process. Extraction complete!")
        return

    # Prepare output CSV — append if file exists, write header if new
    file_exists = Path(OUTPUT_CSV).exists()
    out_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDS)
    if not file_exists:
        writer.writeheader()

    # Process
    success_count = 0
    fail_count    = 0

    for i, paper in enumerate(batch, 1):
        pmid     = paper["pmid"]
        title    = paper.get("title", "")
        abstract = paper.get("abstract", "")
        year     = paper.get("year", "")
        journal  = paper.get("journal", "")

        log.info(f"[{i}/{len(batch)}] PMID {pmid}: {title[:60]}...")

        pico = extract_pico(client, pmid, title, abstract)

        row = {
            "pmid":    pmid,
            "title":   title,
            "year":    year,
            "journal": journal,
            **{k: pico[k] for k in [
                "population", "intervention", "comparison",
                "outcomes_primary", "outcomes_secondary",
                "extraction_notes", "extraction_status"
            ]}
        }
        writer.writerow(row)
        out_file.flush()  # write immediately so no data lost on crash

        if pico["extraction_status"] == "success":
            completed.add(pmid)
            save_checkpoint(completed)
            success_count += 1
            log.info(f"  -> success")
        else:
            fail_count += 1
            log.warning(f"  -> FAILED: {pico['extraction_notes']}")

        time.sleep(REQUEST_DELAY)

    out_file.close()

    log.info("\n" + "=" * 60)
    log.info("BATCH SUMMARY")
    log.info(f"  Processed this run:  {len(batch)}")
    log.info(f"  Successful:          {success_count}")
    log.info(f"  Failed:              {fail_count}")
    log.info(f"  Total completed:     {len(completed)} / {len(corpus)}")
    log.info(f"  Remaining:           {len(corpus) - len(completed)}")
    log.info(f"  Results appended to: {OUTPUT_CSV}")
    log.info("=" * 60)

    if len(completed) < len(corpus):
        log.info(f"\nRun the script again to process the next {BATCH_SIZE} papers.")
    else:
        log.info("\nAll papers processed. Phase 2 complete!")


if __name__ == "__main__":
    run()
