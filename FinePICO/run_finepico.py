"""
run_finepico.py
===============
Runs FinePICO inference on a CSV of GLP-1 abstracts and saves
structured JSON output for comparison against your LLM pipeline.

Usage:
    python run_finepico.py \
        --input  eval_set.csv \
        --model  ./model_checkpoint \
        --output finepico_outputs.json

Input CSV must have columns: pmid, abstract
"""

import argparse
import json
import time
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# ── FinePICO entity groups → coarse PICO mapping ──────────────────────────────
# Adjust this mapping once you confirm the full label list from the checkpoint.
# Run: from transformers import AutoModelForTokenClassification
#      m = AutoModelForTokenClassification.from_pretrained("./model_checkpoint")
#      print(list(m.config.id2label.values()))
ENTITY_TO_PICO = {
    # Population
    "Recruit_participant_conditions": "Population",
    "Total_sample":                   "Population",
    "INT_sample":                     "Population",
    "Control_sample":                 "Population",
    "Overall_age":                    "Population",
    "Sex":                            "Population",
    "Other_Demographics":             "Population",
    # Intervention
    "INT_name":                       "Intervention",
    "INT_bin_percent":                "Intervention",
    "INT_bin_abs":                    "Intervention",
    "INT_cont_mean":                  "Intervention",
    "INT_cont_sd":                    "Intervention",
    "INT_cont_median":                "Intervention",
    "INT_cont_others":                "Intervention",
    # Comparison / Control
    "Control_name":                   "Comparison",
    "Control_bin_percent":            "Comparison",
    "Control_bin_abs":                "Comparison",
    "Control_cont_mean":              "Comparison",
    "Control_cont_sd":                "Comparison",
    "Control_cont_median":            "Comparison",
    "Control_cont_others":            "Comparison",
    # Outcome
    "Outcome_measurements":           "Outcome",
}

def map_pico(entity_group: str) -> str:
    """Return coarse PICO label, or 'Other' if not in mapping."""
    return ENTITY_TO_PICO.get(entity_group, "Other")


def run_inference(model_path: str, abstracts: list[dict], batch_size: int = 8) -> list[dict]:
    """
    Load FinePICO pipeline and run inference over all abstracts.

    Each item in `abstracts` is {"pmid": str, "abstract": str}.
    Returns list of result dicts.
    """
    print(f"Loading model from: {model_path}")
    classifier = pipeline(
        "token-classification",
        model=model_path,
        aggregation_strategy="simple",
        device=-1,          # CPU; change to device=0 for GPU
    )
    print("Model loaded.\n")

    results = []
    failed = []

    for item in tqdm(abstracts, desc="Running FinePICO"):
        pmid     = item["pmid"]
        abstract = item["abstract"]

        if not isinstance(abstract, str) or abstract.strip() == "":
            failed.append(pmid)
            results.append({"pmid": pmid, "error": "empty abstract", "entities": []})
            continue

        try:
            raw_entities = classifier(abstract)
        except Exception as e:
            failed.append(pmid)
            results.append({"pmid": pmid, "error": str(e), "entities": []})
            continue

        # Structure each entity
        entities = []
        for ent in raw_entities:
            entities.append({
                "entity_group": ent["entity_group"],
                "pico_category": map_pico(ent["entity_group"]),
                "word":          ent["word"],
                "score":         round(float(ent["score"]), 4),
                "start":         ent["start"],
                "end":           ent["end"],
            })

        # Also produce a flat PICO summary (one list of spans per category)
        pico_summary = {"Population": [], "Intervention": [], "Comparison": [], "Outcome": [], "Other": []}
        for ent in entities:
            pico_summary[ent["pico_category"]].append(ent["word"])

        results.append({
            "pmid":         pmid,
            "error":        None,
            "entities":     entities,           # full fine-grained output
            "pico_summary": pico_summary,       # coarse P/I/C/O lists for comparison
        })

    print(f"\nDone. {len(results) - len(failed)}/{len(results)} abstracts processed successfully.")
    if failed:
        print(f"Failed PMIDs: {failed}")

    return results


def print_label_list(model_path: str):
    """Helper: print all entity labels the model can predict."""
    from transformers import AutoModelForTokenClassification
    m = AutoModelForTokenClassification.from_pretrained(model_path)
    labels = list(m.config.id2label.values())
    print("Model label list:")
    for label in labels:
        print(f"  {label}")


def main():
    parser = argparse.ArgumentParser(description="Run FinePICO on GLP-1 eval set")
    parser.add_argument("--input",      required=True,  help="Path to eval_set.csv (columns: pmid, abstract)")
    parser.add_argument("--model",      required=True,  help="Path to downloaded FinePICO model checkpoint folder")
    parser.add_argument("--output",     default="finepico_outputs.json", help="Output JSON path")
    parser.add_argument("--labels",     action="store_true", help="Print model label list and exit")
    args = parser.parse_args()

    # Optionally just print labels and exit
    if args.labels:
        print_label_list(args.model)
        return

    # Load input
    df = pd.read_csv(args.input)
    assert "pmid" in df.columns and "abstract" in df.columns, \
        "Input CSV must have 'pmid' and 'abstract' columns"

    abstracts = df[["pmid", "abstract"]].dropna(subset=["pmid"]).to_dict(orient="records")
    print(f"Loaded {len(abstracts)} abstracts from {args.input}")

    # Run inference
    t0 = time.time()
    results = run_inference(args.model, abstracts)
    elapsed = time.time() - t0

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")
    print(f"Total time: {elapsed:.1f}s  ({elapsed/len(abstracts):.2f}s per abstract)")

    # Quick coverage summary
    coverage = {cat: 0 for cat in ["Population", "Intervention", "Comparison", "Outcome"]}
    for r in results:
        if r.get("pico_summary"):
            for cat in coverage:
                if r["pico_summary"].get(cat):
                    coverage[cat] += 1
    print("\nEntity coverage (% abstracts with at least one span extracted):")
    for cat, count in coverage.items():
        print(f"  {cat}: {count}/{len(results)} ({100*count/len(results):.1f}%)")


if __name__ == "__main__":
    main()
