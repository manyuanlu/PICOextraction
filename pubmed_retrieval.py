"""
Phase 1: PubMed Retrieval Pipeline for GLP-1 RCTs
===================================================
Retrieves all relevant GLP-1 RCT abstracts from PubMed using
NCBI E-utilities (ESearch + EFetch), filters, deduplicates,
and saves a structured corpus CSV ready for Phase 2 LLM extraction.

Usage:
    python pubmed_retrieval.py

Requirements:
    pip install requests

Optional but recommended:
    Set your NCBI API key as an environment variable:
        export NCBI_API_KEY="your_key_here"
    Get a free key at: https://www.ncbi.nlm.nih.gov/account/
"""

import requests
import xml.etree.ElementTree as ET
import csv
import json
import time
import os
import logging
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")  # Set via env var

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

BATCH_SIZE      = 100   # PMIDs per EFetch call (NCBI max recommended)
REQUEST_DELAY   = 0.4   # seconds between requests (stay under rate limit)
                         # With API key: 10 req/sec allowed; without: 3 req/sec

OUTPUT_CSV  = "glp1_corpus.csv"
OUTPUT_JSON = "glp1_corpus.json"
LOG_FILE    = "retrieval_log.txt"

# ─── Search Queries ────────────────────────────────────────────────────────────
# Multiple targeted queries, then deduplicate by PMID.
# Each query uses PubMed field tags:
#   [tiab]  = title/abstract
#   [mh]    = MeSH heading
#   [pt]    = publication type
#   [la]    = language
#   [sh]    = MeSH subheading

SEARCH_QUERIES = [
    # Core drug names — broad RCT filter
    (
        '(semaglutide[tiab] OR liraglutide[tiab] OR dulaglutide[tiab] '
        'OR tirzepatide[tiab] OR exenatide[tiab] OR albiglutide[tiab] '
        'OR lixisenatide[tiab]) '
        'AND "Randomized Controlled Trial"[pt] '
        'AND "humans"[mh] '
        'AND English[la]'
    ),
    # GLP-1 receptor agonist MeSH term
    (
        '"Glucagon-Like Peptide-1 Receptor"[mh] '
        'AND "Randomized Controlled Trial"[pt] '
        'AND "humans"[mh] '
        'AND English[la]'
    ),
    # Broader GLP-1 class term
    (
        '"Glucagon-Like Peptides"[mh] '
        'AND "Randomized Controlled Trial"[pt] '
        'AND "humans"[mh] '
        'AND English[la]'
    ),
]

# ─── Logging Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── Helper: base params ───────────────────────────────────────────────────────

def base_params():
    p = {"retmode": "json"}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p

# ─── Step 1: ESearch — get PMIDs ──────────────────────────────────────────────

def esearch(query: str) -> list[str]:
    """Return all PMIDs matching the query (handles pagination)."""
    pmids = []
    retstart = 0
    retmax = 500  # fetch 500 at a time from ESearch

    while True:
        params = base_params() | {
            "db": "pubmed",
            "term": query,
            "retstart": retstart,
            "retmax": retmax,
            "usehistory": "n",
        }
        resp = requests.get(ESEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        result  = data.get("esearchresult", {})
        ids     = result.get("idlist", [])
        total   = int(result.get("count", 0))
        pmids.extend(ids)

        log.info(f"  ESearch page: retstart={retstart}, got {len(ids)}, total={total}")
        time.sleep(REQUEST_DELAY)

        retstart += retmax
        if retstart >= total:
            break

    return pmids

# ─── Step 2: EFetch — get full records in batches ─────────────────────────────

def efetch_batch(pmids: list[str]) -> str:
    """Fetch XML for a batch of PMIDs."""
    params = base_params() | {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    # override retmode for XML
    params["retmode"] = "xml"

    resp = requests.get(EFETCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp.text

# ─── Step 3: Parse XML record ─────────────────────────────────────────────────

def parse_abstract(abstract_node) -> str:
    """
    Extract abstract text, handling both:
    - Structured abstracts (multiple <AbstractText Label="BACKGROUND"> etc.)
    - Unstructured abstracts (single <AbstractText>)
    Returns a clean string.
    """
    if abstract_node is None:
        return ""

    parts = []
    for at in abstract_node.findall("AbstractText"):
        label = at.get("Label", "")
        text  = (at.text or "").strip()
        if label:
            parts.append(f"{label}: {text}")
        elif text:
            parts.append(text)

    return " | ".join(parts)


def parse_pub_types(article_node) -> list[str]:
    """Extract publication type list."""
    types = []
    for pt in article_node.findall(".//PublicationType"):
        if pt.text:
            types.append(pt.text.strip())
    return types


def parse_mesh_terms(medline_node) -> list[str]:
    """Extract MeSH heading names."""
    terms = []
    for mh in medline_node.findall(".//MeshHeading/DescriptorName"):
        if mh.text:
            terms.append(mh.text.strip())
    return terms


def parse_record(article_xml) -> dict | None:
    """
    Parse a single <PubmedArticle> XML element into a flat dict.
    Returns None if the record lacks a PMID or abstract.
    """
    try:
        medline   = article_xml.find("MedlineCitation")
        if medline is None:
            return None

        pmid_node = medline.find("PMID")
        pmid      = pmid_node.text.strip() if pmid_node is not None else ""
        if not pmid:
            return None

        article   = medline.find("Article")
        if article is None:
            return None

        # Title
        title_node = article.find("ArticleTitle")
        title      = (title_node.text or "").strip() if title_node is not None else ""

        # Abstract
        abstract_node = article.find("Abstract")
        abstract      = parse_abstract(abstract_node)

        # Skip if no abstract
        if not abstract:
            return None

        # Journal
        journal_node = article.find(".//Journal/Title")
        journal      = (journal_node.text or "").strip() if journal_node is not None else ""

        # Year
        year = ""
        for year_path in [
            ".//PubDate/Year",
            ".//PubDate/MedlineDate",
            ".//ArticleDate/Year",
        ]:
            y_node = article.find(year_path)
            if y_node is not None and y_node.text:
                year = y_node.text[:4]
                break

        # Publication types
        pub_types = parse_pub_types(article)

        # MeSH terms
        mesh_terms = parse_mesh_terms(medline)

        # Authors (last name list, semicolon-separated)
        authors = []
        for author in article.findall(".//AuthorList/Author"):
            ln = author.find("LastName")
            fn = author.find("ForeName")
            if ln is not None and ln.text:
                name = ln.text
                if fn is not None and fn.text:
                    name += f" {fn.text[0]}"  # last + first initial
                authors.append(name)

        return {
            "pmid":       pmid,
            "title":      title,
            "abstract":   abstract,
            "year":       year,
            "journal":    journal,
            "authors":    "; ".join(authors),
            "pub_types":  "; ".join(pub_types),
            "mesh_terms": "; ".join(mesh_terms),
        }

    except Exception as e:
        log.warning(f"  Parse error on record: {e}")
        return None

# ─── Step 4: Filter ───────────────────────────────────────────────────────────

def is_rct(record: dict) -> bool:
    """
    Confirm the record is an RCT.
    ESearch already filters by [pt], but EFetch results can include
    slightly mismatched records — double-check publication type field.
    """
    pub_types = record.get("pub_types", "").lower()
    return "randomized controlled trial" in pub_types

# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info(f"Phase 1: PubMed Retrieval — started {datetime.now().isoformat()}")
    log.info(f"API key: {'set' if NCBI_API_KEY else 'NOT SET (rate-limited to 3 req/sec)'}")
    log.info("=" * 60)

    # ── 1. Collect all PMIDs across queries ──────────────────────────────────
    all_pmids = set()
    for i, query in enumerate(SEARCH_QUERIES, 1):
        log.info(f"\nQuery {i}/{len(SEARCH_QUERIES)}:\n  {query}")
        pmids = esearch(query)
        log.info(f"  → {len(pmids)} PMIDs returned")
        all_pmids.update(pmids)

    pmid_list = sorted(all_pmids)
    log.info(f"\nTotal unique PMIDs after deduplication: {len(pmid_list)}")

    # ── 2. Fetch + parse in batches ──────────────────────────────────────────
    records = []
    skipped_no_abstract = 0
    skipped_not_rct     = 0
    parse_errors        = 0

    for batch_start in range(0, len(pmid_list), BATCH_SIZE):
        batch = pmid_list[batch_start : batch_start + BATCH_SIZE]
        log.info(f"Fetching batch {batch_start // BATCH_SIZE + 1}: PMIDs {batch_start+1}–{batch_start+len(batch)}")

        try:
            xml_text = efetch_batch(batch)
            root     = ET.fromstring(xml_text)
        except Exception as e:
            log.error(f"  EFetch/parse failed for batch: {e}")
            parse_errors += len(batch)
            continue

        for article_xml in root.findall("PubmedArticle"):
            rec = parse_record(article_xml)

            if rec is None:
                skipped_no_abstract += 1
                continue

            if not is_rct(rec):
                skipped_not_rct += 1
                log.debug(f"  Non-RCT filtered: PMID {rec.get('pmid')} | {rec.get('pub_types')}")
                continue

            records.append(rec)

    # ── 3. Final dedup by PMID (safety net) ─────────────────────────────────
    seen  = set()
    deduped = []
    for r in records:
        if r["pmid"] not in seen:
            seen.add(r["pmid"])
            deduped.append(r)

    log.info("\n" + "=" * 60)
    log.info("RETRIEVAL SUMMARY")
    log.info(f"  Unique PMIDs retrieved:        {len(pmid_list)}")
    log.info(f"  Records parsed successfully:   {len(records)}")
    log.info(f"  Skipped (no abstract):         {skipped_no_abstract}")
    log.info(f"  Skipped (not RCT after check): {skipped_not_rct}")
    log.info(f"  Parse/fetch errors:            {parse_errors}")
    log.info(f"  Final corpus size:             {len(deduped)} papers")
    log.info("=" * 60)

    # ── 4. Save outputs ──────────────────────────────────────────────────────
    if not deduped:
        log.warning("No records to save.")
        return

    # CSV
    fieldnames = ["pmid", "title", "abstract", "year", "journal", "authors", "pub_types", "mesh_terms"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)
    log.info(f"Saved CSV  -> {OUTPUT_CSV}")

    # JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    log.info(f"Saved JSON -> {OUTPUT_JSON}")

    log.info("\nPhase 1 complete. Pass glp1_corpus.csv to Phase 2 for LLM extraction.")


if __name__ == "__main__":
    run()
