"""
Phase 1: PubMed Retrieval Pipeline for GLP-1 RCTs
===================================================
 - Fetches full text from PubMed Central (PMC) when a PMCID is available
 - Falls back to abstract-only for papers not in PMC
 - Adds `text_source` column: "fulltext" | "abstract"
 - Adds `pmcid` column for downstream traceability
 - Stores full text in `full_text` column; keeps `abstract` column intact

Outputs:
    glp1_corpus.csv                       - one row per paper, with full text + abstract
    retrieval_log.txt                     - run log
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
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")  # Set via env var

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"  # same endpoint as EFETCH_URL, used for PMC full-text (db=pmc)

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
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
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

def delay():
    rate = 0.11 if NCBI_API_KEY else REQUEST_DELAY
    time.sleep(rate)

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
        delay()

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
    delay()
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

GLP1_DRUGS = [
    "semaglutide", "liraglutide", "dulaglutide", "tirzepatide",
    "exenatide", "albiglutide", "lixisenatide",
    "ozempic", "wegovy", "rybelsus", "victoza", "saxenda",
    "trulicity", "mounjaro", "zepbound", "bydureon", "byetta",
]
 
def detect_indication_hint(text: str) -> str:
    text_lower = text.lower()
    found = [d for d in GLP1_DRUGS if d in text_lower]
    return "; ".join(found)

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

        # Journal
        journal_node = article.find(".//Journal/Title")
        journal      = (journal_node.text or "").strip() if journal_node is not None else ""

        # Year
        year = ""
        for year_path in [
            ".//PubDate/Year",
            ".//PubDate/MedlineDate",
            ".//ArticleDate/Year",
            "Journal/JournalIssue/PubDate/Year",
            "Journal/JournalIssue/PubDate/MedlineDate"
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

        # PMCID (from PubmedData > ArticleIdList)
        pmcid = ""
        for aid in article_xml.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "pmc":
                pmcid = aid.text.strip() if aid.text else ""
                break

        indication_hint = detect_indication_hint(title + " " + abstract)
        
        return {
            "pmid":       pmid,
            "pmcid":      pmcid,
            "title":      title,
            "abstract":   abstract,
            "full_text":  "",
            "year":       year,
            "journal":    journal,
            "authors":    "; ".join(authors),
            "pub_types":  "; ".join(pub_types),
            "mesh_terms": "; ".join(mesh_terms),
            "indication_hint": indication_hint
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

def fetch_pmc_fulltext(pmcid: str) -> str:
    """
    Fetch full text for a paper from PMC using EFetch with rettype=full.
    Returns plain text (sections joined), or empty string on failure.
    Only open-access PMC articles are available this way.
    """
    if not pmcid:
        return ""
 
    # Normalise: PMC1234567 → 1234567
    numeric_id = pmcid.replace("PMC", "").strip()
 
    params = {
        "db":      "pmc",
        "id":      numeric_id,
        "rettype": "full",
        "retmode": "xml",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
 
    try:
        resp = requests.get(PMC_FETCH_URL, params=params, timeout=60)
        resp.raise_for_status()
        delay()
 
        root = ET.fromstring(resp.text)
 
        # PMC full-text XML has body sections under <body><sec>
        # We extract all <p> (paragraph) text, preserving section order
        sections = []
 
        # Try to get title
        title_node = root.find(".//article-title")
        if title_node is not None:
            sections.append("TITLE: " + "".join(title_node.itertext()).strip())
 
        # Abstract
        for abs_node in root.findall(".//abstract"):
            abs_text = " ".join("".join(p.itertext()).strip() for p in abs_node.findall(".//p"))
            if abs_text:
                sections.append("ABSTRACT: " + abs_text)
 
        # Body sections
        for sec in root.findall(".//body//sec"):
            # Section title
            sec_title_node = sec.find("title")
            sec_title = ("".join(sec_title_node.itertext()).strip() + ": ") if sec_title_node is not None else ""
 
            # Paragraphs
            paragraphs = []
            for p in sec.findall("p"):
                p_text = "".join(p.itertext()).strip()
                if p_text:
                    paragraphs.append(p_text)
 
            if paragraphs:
                sections.append(sec_title + " ".join(paragraphs))
 
        full_text = "\n\n".join(sections).strip()
        return full_text
 
    except Exception as e:
        log.warning(f"    PMC fetch failed for {pmcid}: {e}")
        return ""
 
 
def enrich_with_fulltext(records: list[dict]) -> None:
    """
    In-place: for every record that has a PMCID, attempt to fetch full text.
    Updates full_text and text_source fields.
    """
    pmc_eligible = [r for r in records if r["pmcid"]]
    log.info(f"\nPMC full-text fetch: {len(pmc_eligible)} / {len(records)} papers have a PMCID")
 
    success = 0
    failed  = 0
 
    for i, record in enumerate(pmc_eligible, 1):
        pmcid = record["pmcid"]
        log.info(f"  [{i}/{len(pmc_eligible)}] Fetching {pmcid} (PMID {record['pmid']})")
 
        full_text = fetch_pmc_fulltext(pmcid)
 
        if full_text and len(full_text) > 500:   # sanity check: real full text is long
            record["full_text"]   = full_text
            record["text_source"] = "fulltext"
            success += 1
            log.info(f"    ✓ {len(full_text):,} chars")
        else:
            # Keep abstract as fallback
            record["full_text"]   = record["abstract"]
            record["text_source"] = "abstract_only"
            failed += 1
            log.info(f"    ✗ No full text available — keeping abstract")
 
    # Papers without PMCID also get abstract as full_text fallback
    for record in records:
        if not record["pmcid"]:
            record["full_text"]   = record["abstract"]
            record["text_source"] = "abstract_only"
 
    log.info(f"\nFull-text summary: {success} fulltext | {failed} fallback to abstract | "
             f"{len(records) - len(pmc_eligible)} no PMCID")
    
CSV_FIELDS = [
    "pmid", "pmcid", "title", "abstract", "full_text", "text_source",
    "year", "journal", "authors", "pub_types", "mesh_terms",
    "indication_hint",   # drug names detected in title/abstract
]

def save_csv(records: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info(f"Saved {len(records)} records → {path}")
# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info(f"Phase 1: PubMed + PMC Retrieval — started {datetime.now().isoformat()}")
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
    skipped_parse_error = 0
    skipped_not_rct     = 0
    parse_errors        = 0

    for batch_start in range(0, len(pmid_list), BATCH_SIZE):
        batch = pmid_list[batch_start : batch_start + BATCH_SIZE]
        log.info(f"Fetching PubMed batch {batch_start // BATCH_SIZE + 1}: "
                 f"records {batch_start + 1}–{batch_start + len(batch)}")

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
                skipped_parse_error += 1
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
            
    log.info(f"\nPubMed pass complete: {len(deduped)} RCTs retained "
             f"(skipped: {skipped_parse_error} unparseable records, "
             f"{skipped_not_rct} non-RCT, {parse_errors} batch errors)")
 
    # 3. PMC full-text enrichment
    enrich_with_fulltext(deduped)

    # 4. Summary + save
    ft_count  = sum(1 for r in deduped if r["text_source"] == "fulltext")
    abs_count = len(deduped) - ft_count
 
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY")
    log.info(f"  Total papers:          {len(deduped)}")
    log.info(f"  Full text (PMC):       {ft_count}")
    log.info(f"  Abstract only:         {abs_count}")
    log.info(f"  PMC coverage:          {ft_count / len(deduped) * 100:.1f}%")
    log.info("=" * 60)
 
    save_csv(deduped, OUTPUT_CSV)
    log.info("\nPhase 1 complete. Next: run pico_extraction.py")
 
 
if __name__ == "__main__":
    run()
