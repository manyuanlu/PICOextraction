"""
Phase 2: Batch LLM PICO Extraction
====================================
Changes from original:
  - Reads 'full_text' column instead of 'abstract'
  - For abstract-only papers, behavior is unchanged

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
    set ANTHROPIC_API_KEY=your_key_here
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

ANTHROPIC_API_KEY = "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" 

INPUT_CSV      = "glp1_corpus.csv"       # output from Phase 1
OUTPUT_CSV     = "pico_results.csv"      # final results
CHECKPOINT     = "checkpoint.json"       # tracks last completed PMID
LOG_FILE       = "extraction_log.txt"

BATCH_SIZE     = 2391      # how many papers to process per run (adjust as needed)
                          # run the script multiple times to process more
REQUEST_DELAY  = 1.0     # seconds between API calls (be gentle on rate limits)
MAX_RETRIES    = 3       # retries per paper on API failure
RETRY_BACKOFF  = 5       # seconds for first retry; doubles each attempt

MODEL          = "claude-opus-4-5"
MAX_TOKENS     = 1500

# Approximate character limit before we switch to section extraction.
# Claude's context window is large, but we leave headroom for prompt + output.
# ~60,000 chars ≈ ~15,000 tokens — comfortably fits most full-text articles whole.
# Papers longer than this get section-extracted instead.
FULLTEXT_CHAR_LIMIT = 60000

# ── Output fields ──────────────────────────────────────────────────────────────

OUTPUT_FIELDS = [
    "pmid", "title", "year", "journal", "text_source", 
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

USER_TEMPLATE = """Title: {title}
Journal: {journal}
Source: {text_source}

{text_label}:
{text}

Extract the PICO elements from the text above."""

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Full-text preprocessing ─────────────────────────────────────────────────────────
# Section headers that are most relevant for PICO extraction
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
    """
    Returns (processed_text, strategy_used).
 
    Strategy:
      - If text fits within FULLTEXT_CHAR_LIMIT → send whole text as-is
      - If text is too long → extract priority sections and trim
      - Abstract-only text → always sent as-is (already short)
    """
    if not full_text:
        return "", "empty"
 
    # Abstract-only: always fits, send as-is
    if text_source != "fulltext":
        return full_text, "abstract"
 
    # Full text within limit: send whole article
    if len(full_text) <= FULLTEXT_CHAR_LIMIT:
        return full_text, "fulltext_complete"
 
    # Full text too long: extract priority sections
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
 
    # Final safety trim
    if len(combined) > FULLTEXT_CHAR_LIMIT:
        combined = combined[:FULLTEXT_CHAR_LIMIT] + "\n\n[truncated]"
 
    return combined, "fulltext_truncated"

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

def extract_pico(client: anthropic.Anthropic, record: dict) -> dict:
    """
    Call Claude API and parse JSON response into PICO fields.
    Returns a dict with PICO keys + extraction_status.
    """
    text_source = record.get("text_source", "abstract_only")
    raw_text = record.get("full_text") or record.get("abstract", "")

    processed_text, strategy = extract_relevant_sections(raw_text, text_source)
    log.info(f"    strategy: {strategy} ({len(processed_text):,} chars)")
 
    text_label = {
        "abstract":          "Abstract",
        "fulltext_complete": "Full Text",
        "fulltext_truncated": "Full Text (priority sections, truncated)",
        "empty":             "No text available",
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
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
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
            log.warning(f"    JSON parse error (attempt {attempt}): {e}")
        except anthropic.RateLimitError:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(f"    Rate limit (attempt {attempt}) — waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            log.warning(f"    API error (attempt {attempt}): {e}")
            time.sleep(RETRY_BACKOFF)

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

    # if not ANTHROPIC_API_KEY:
    #     raise ValueError("ANTHROPIC_API_KEY is not set. Add it to the script or set as environment variable.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Load corpus
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        corpus = list(csv.DictReader(f))
    log.info(f"Corpus loaded: {len(corpus)} papers")

    # Log text source breakdown
    fulltext_count = sum(1 for r in corpus if r.get("text_source") == "fulltext")
    log.info(f"Text sources: {fulltext_count} full-text | "
             f"{len(corpus) - fulltext_count} abstract-only")
    
    # Load checkpoint
    completed = load_checkpoint()
    # Filter to papers not yet done
    remaining = [r for r in corpus if r["pmid"] not in completed]
    log.info(f"Already completed: {len(completed)} papers (from previous runs)")

    if not remaining:
        log.info("All papers already processed. Nothing to do.")
        return
 
    init_output_csv()
    
    log.info(f"Remaining: {len(remaining)} papers")

    # Take this run's batch
    batch = remaining[:BATCH_SIZE]
    success_count = 0
    failed_count  = 0
 
    for i, record in enumerate(batch, 1):
        pmid        = record["pmid"]
        text_source = record.get("text_source", "abstract_only")
        log.info(f"[{i}/{len(batch)}] PMID {pmid} ({text_source})")
 
        result = extract_pico(client, record)
 
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
 
        if result["extraction_status"] == "success":
            success_count += 1
            log.info(f"    ✓ extracted")
        else:
            failed_count += 1
            log.warning(f"    ✗ failed")
 
        time.sleep(REQUEST_DELAY)
 
    # Summary
    total_done = len(completed)
    total_left = len(corpus) - total_done
 
    log.info("\n" + "=" * 60)
    log.info("BATCH SUMMARY")
    log.info(f"  Processed this run:  {len(batch)}")
    log.info(f"  Successful:          {success_count}")
    log.info(f"  Failed:              {failed_count}")
    log.info(f"  Total completed:     {total_done} / {len(corpus)}")
    log.info(f"  Remaining:           {total_left}")
    log.info(f"  Results appended to: {OUTPUT_CSV}")
    log.info("=" * 60)
 
    if total_left > 0:
        log.info(f"\n{total_left} papers remaining — run the script again to continue.")
    else:
        log.info("\nAll papers processed. Phase 2 complete!")

if __name__ == "__main__":
    run()
