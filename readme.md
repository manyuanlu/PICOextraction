# GLP-1 RCT Evidence Pipeline

A three-phase computational pipeline for systematic retrieval, structured extraction, and database construction of GLP-1 receptor agonist randomized controlled trial evidence from PubMed.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Total RCTs retrieved | 2,341 |
| Unique GLP-1 drugs identified | 10 |
| Extraction failure rate | 0% |
| ClinicalTrials.gov linkage | 1,201 / 2,341 (51%) |
| Women-only trials | 77 (3.3%) |
| Sex-stratified outcomes reported | 34 (1.5%) |
| Female enrollment % reported | 435 (18.6%) |

---

## Pipeline Overview

```
PubMed (3 queries)
       ↓
Phase 1: Literature Retrieval        → 2,390 PMIDs → 2,341 abstracts
       ↓
Phase 2: PICO Extraction (LLM)       → 2,341 structured PICO records
       ↓
Phase 3: Database Construction       → glp1_database.csv + .xlsx
```

---

## Phase 1 — Literature Retrieval

**Script:** `pubmed_retrieval.py`

Queries PubMed via the NCBI E-utilities API (ESearch + EFetch) using three complementary strategies:

| Query | Strategy | Results |
|-------|----------|---------|
| 1 | Drug names in title/abstract (`semaglutide`, `liraglutide`, `dulaglutide`, `tirzepatide`, `exenatide`, `albiglutide`, `lixisenatide`) | 1,425 PMIDs |
| 2 | MeSH: `"Glucagon-Like Peptide-1 Receptor"` | 197 PMIDs |
| 3 | MeSH: `"Glucagon-Like Peptides"` | 1,866 PMIDs |

All queries filtered to: `"Randomized Controlled Trial"[pt]` + `"humans"[mh]` + `English[la]`

**Results:** 2,390 unique PMIDs after deduplication → **2,341 papers** after excluding 47 records with no abstract and 2 non-RCT records on post-retrieval verification.

### Setup

```bash
pip install requests
```

```python
# pubmed_retrieval.py — line 30
NCBI_API_KEY = "your_key_here"   # free key at ncbi.nlm.nih.gov/account
```

```bash
python pubmed_retrieval.py
```

**Outputs:** `glp1_corpus.csv`, `glp1_corpus.json`, `retrieval_log.txt`

---

## Phase 2 — PICO Extraction

**Script:** `pico_extraction.py`

Sends each abstract to Claude (`claude-opus-4-5`) via the Anthropic API and extracts structured PICO fields as JSON. Runs in configurable batches with checkpoint-and-resume — safe to interrupt at any time.

**Extracted fields:**

| Field | Description |
|-------|-------------|
| `population` | Eligibility-defining patient characteristics |
| `intervention` | Drug name, dose, route, frequency, duration |
| `comparison` | All comparator arms explicitly stated |
| `outcomes_primary` | Primary outcome definitions (not results) |
| `outcomes_secondary` | Secondary outcome definitions |
| `extraction_notes` | Flags for ambiguity or missing fields |

**Results:** 2,341 / 2,341 papers extracted successfully. **0 failures.**

### Setup

```bash
pip install anthropic
```

```python
# pico_extraction.py — line 38
ANTHROPIC_API_KEY = "your_key_here"   # console.anthropic.com → API Keys

BATCH_SIZE = 50   # increase after validating first batch
```

```bash
python pico_extraction.py   # run repeatedly until complete
```

**Outputs:** `pico_results.csv`, `checkpoint.json`, `extraction_log.txt`

---

## Phase 3 — Database Construction

Two scripts: annotate first, then build.

### 3a — Annotation Extraction

**Script:** `annotate.py`

Second LLM pass over all abstracts to extract structured metadata fields:

| Field | Description |
|-------|-------------|
| `indication` | Clinical condition (e.g. Type 2 Diabetes, Obesity, CVD) |
| `trial_phase` | Phase 2 / 3 / 4 / NR |
| `enrollment_size` | Total randomized participants |
| `female_pct` | Female enrollment percentage |
| `follow_up_duration` | Duration as stated in abstract |
| `comparator_type` | Placebo / Active Comparator / Dose Comparison / NR |
| `women_only_trial` | yes / no |
| `sex_stratified_outcomes` | yes / no / unclear |
| `female_enrollment_reported` | yes / no |

Same batch/checkpoint pattern as Phase 2.

```bash
python annotate.py   # run repeatedly until complete
```

**Outputs:** `annotations.csv`, `annotation_checkpoint.json`, `annotation_log.txt`

### 3b — Build Database

**Script:** `build_database.py`

Merges PICO results + annotations, normalizes drug names and indications, links ClinicalTrials.gov records by PMID, and builds the final database.

```bash
pip install pandas openpyxl requests
python build_database.py
```

**Drug normalization** — 10 canonical drug names mapped from brand names and synonyms (e.g. Ozempic → Semaglutide, Mounjaro → Tirzepatide).

**ClinicalTrials.gov linkage** — best-effort PMID lookup via CT.gov API v2. Appends: NCT ID, study status, registered phase, registered enrollment.

**Outputs:**

| File | Description |
|------|-------------|
| `glp1_database.csv` | Flat 2,341-row database, 24 columns |
| `glp1_database.xlsx` | Excel workbook with auto-filter + frozen headers |

Excel sheets:

| Sheet | Contents |
|-------|----------|
| All Trials | Full database |
| Sex Data Available | Trials with any sex-specific data |
| Summary | Per-indication aggregate statistics |
| *(one per indication)* | Filtered subsets |

---

## Key Findings

### Sex-Disaggregated Reporting Gap

Of 2,341 GLP-1 RA RCTs:
- Only **34 trials (1.5%)** reported sex-stratified outcomes
- Only **77 trials (3.3%)** were women-only
- **435 trials (18.6%)** reported female enrollment percentage at all

This gap motivates the downstream sex-stratified simulation analysis this pipeline was built to support.

### CT.gov Coverage

51% of papers (1,201/2,341) were successfully linked to ClinicalTrials.gov records. The remaining 49% could not be automatically matched, likely due to incomplete PMID registration in the CT.gov database.

---

## Repository Structure

```
PICOextraction/
├── pubmed_retrieval.py       # Phase 1: PubMed retrieval
├── pico_extraction.py        # Phase 2: PICO extraction
├── annotate.py               # Phase 3a: annotation extraction
├── build_database.py         # Phase 3b: database builder
│
├── glp1_corpus.csv           # Phase 1 output
├── glp1_corpus.json          # Phase 1 output (JSON)
├── pico_results.csv          # Phase 2 output
├── annotations.csv           # Phase 3a output
├── glp1_database.csv         # Final database (CSV)
├── glp1_database.xlsx        # Final database (Excel)
│
├── checkpoint.json           # Phase 2 resume checkpoint
├── annotation_checkpoint.json # Phase 3a resume checkpoint
│
├── retrieval_log.txt
├── extraction_log.txt
├── annotation_log.txt
└── build_database_log.txt
```

---

## Requirements

```
requests
anthropic
pandas
openpyxl
```

```bash
pip install requests anthropic pandas openpyxl
```

Python 3.10+ required. Developed and tested on Python 3.14 / Windows 11.

---

## API Keys

| Key | Where to get it | Where to set it |
|-----|----------------|-----------------|
| NCBI API key | [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) | `pubmed_retrieval.py` line 30 |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) → API Keys | `pico_extraction.py` line 38 and `annotate.py` line 38 |

> **Cost estimate:** Approximately $5–15 USD in Anthropic API credits for the full 2,341-paper extraction (Phase 2 + Phase 3a combined).

---

## Limitations

- Search restricted to PubMed/MEDLINE; Embase and CENTRAL not included
- Extraction based on abstract text only; full-text data not captured
- LLM extraction may introduce systematic errors on ambiguous abstracts
- CT.gov linkage achieved 51% coverage; 49% unmatched
- Indication normalization from 356 raw LLM labels involved automated string matching; multi-indication trials may be miscategorized
