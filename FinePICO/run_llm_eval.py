"""
run_llm_eval.py
===============
Runs your LLM PICO extraction pipeline on eval_set.csv
using the exact same prompt and API logic as pico_extraction.py.

Output: llm_outputs.json  (same structure as finepico_outputs.json
        for easy side-by-side comparison)

Usage:
    python run_llm_eval.py --input eval_set.csv --output llm_outputs.json
    
Set your Anthropic API key first:
    Windows:  set ANTHROPIC_API_KEY=your_key_here
    Mac/Linux: export ANTHROPIC_API_KEY=your_key_here
"""

import anthropic
import argparse
import json
import os
import re
import time
import logging
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL          = "claude-opus-4-5-20251101"
MAX_TOKENS     = 1500
REQUEST_DELAY  = 1.0
MAX_RETRIES    = 3
RETRY_BACKOFF  = 5

# ── Prompt (identical to pico_extraction.py) ───────────────────────────────────
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
Year: {year}
Journal: {journal}
Source: abstract_only

Abstract:
{text}

Extract the PICO elements from the text above."""

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── API call ───────────────────────────────────────────────────────────────────
def extract_pico(client, record: dict) -> dict:
    prompt = USER_TEMPLATE.format(
        title   = record.get("title", ""),
        year    = record.get("year", ""),
        journal = record.get("journal", ""),
        text    = record.get("abstract", ""),
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
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            pico = json.loads(raw)
            for key in ["population", "intervention", "comparison",
                        "outcomes_primary", "outcomes_secondary", "extraction_notes"]:
                pico.setdefault(key, "")
            pico["extraction_status"] = "success"
            return pico

        except json.JSONDecodeError as e:
            log.warning(f"    JSON parse error (attempt {attempt}): {e}")
        except anthropic.RateLimitError:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(f"    Rate limit — waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            log.warning(f"    API error (attempt {attempt}): {e}")
            time.sleep(RETRY_BACKOFF)

    return {
        "population": "", "intervention": "", "comparison": "",
        "outcomes_primary": "", "outcomes_secondary": "",
        "extraction_notes": "failed after all retries",
        "extraction_status": "failed"
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,  help="eval_set.csv (columns: pmid, abstract)")
    parser.add_argument("--output", default="llm_outputs.json")
    args = parser.parse_args()

    api_key = "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"
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    # Load eval set — merge with full metadata if eval_set_full.csv is present
    df = pd.read_csv(args.input, dtype=str).fillna("")
    full_path = args.input.replace("eval_set.csv", "eval_set_full.csv")
    if os.path.exists(full_path):
        full_df = pd.read_csv(full_path, dtype=str).fillna("")
        df = df.merge(full_df[["pmid", "title", "year", "indication_hint"]],
                      on="pmid", how="left").fillna("")
        log.info("Merged with eval_set_full.csv for title/year metadata.")

    log.info(f"Loaded {len(df)} abstracts from {args.input}")
    log.info(f"Model: {MODEL}")
    log.info("=" * 50)

    results = []
    success, failed = 0, 0
    t0 = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM extraction"):
        pmid = row["pmid"]
        record = row.to_dict()

        pico = extract_pico(client, record)

        # Build output in same structure as finepico_outputs.json
        # pico_summary mirrors FinePICO's coarse P/I/C/O lists
        if pico["extraction_status"] == "success":
            pico_summary = {
                "Population":    [pico["population"]]    if pico["population"]    else [],
                "Intervention":  [pico["intervention"]]  if pico["intervention"]  else [],
                "Comparison":    [pico["comparison"]]    if pico["comparison"]    else [],
                "Outcome":       [pico["outcomes_primary"], pico["outcomes_secondary"]] if pico["outcomes_primary"] else [],
            }
            success += 1
        else:
            pico_summary = {"Population": [], "Intervention": [], "Comparison": [], "Outcome": []}
            failed += 1

        results.append({
            "pmid":             pmid,
            "indication":       row.get("indication_hint", ""),
            "error":            None if pico["extraction_status"] == "success" else pico["extraction_notes"],
            "extraction_status": pico["extraction_status"],
            # Full structured fields (LLM-specific, richer than FinePICO)
            "population":       pico["population"],
            "intervention":     pico["intervention"],
            "comparison":       pico["comparison"],
            "outcomes_primary": pico["outcomes_primary"],
            "outcomes_secondary": pico["outcomes_secondary"],
            "extraction_notes": pico["extraction_notes"],
            # Coarse summary for comparison with FinePICO
            "pico_summary":     pico_summary,
        })

        time.sleep(REQUEST_DELAY)

    elapsed = time.time() - t0

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Coverage summary (mirrors FinePICO output)
    log.info("\n" + "=" * 50)
    log.info(f"Done. {success}/{len(results)} successful  |  {failed} failed")
    log.info(f"Total time: {elapsed:.1f}s  ({elapsed/len(results):.1f}s per abstract)")
    log.info(f"Results saved to: {args.output}")
    log.info("\nEntity coverage (% abstracts with at least one span extracted):")
    for cat in ["Population", "Intervention", "Comparison", "Outcome"]:
        count = sum(1 for r in results if r["pico_summary"].get(cat))
        log.info(f"  {cat}: {count}/{len(results)} ({100*count/len(results):.1f}%)")

if __name__ == "__main__":
    main()
