# GLP-1 RCT Evidence Pipeline

A four-phase computational pipeline for systematic retrieval, structured extraction, annotation, and database construction of GLP-1 receptor agonist randomized controlled trial evidence from PubMed.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Total RCTs retrieved | 2,392 |
| Unique GLP-1 drugs identified | 27 |
| Unique indications | 28 |
| Extraction failure rate | 0% |
| ClinicalTrials.gov linkage | 1,219 / 2,392 (50%) |
| Women-only trials | 80 (3.3%) |
| Sex-stratified outcomes reported | 100 (4.2%) |
| Female enrollment % reported | 862 (36.0%) |
| Trials with subgroup data | 618 (25.8%) |
| Surrogate-heavy trials | 1,888 (78.9%) |
| Patient-centered trials | 179 (7.5%) |

---

## Pipeline Overview

```
PubMed (3 queries)
       ↓
Phase 1: Literature Retrieval        → 2,392 papers (957 full-text, 1,435 abstract-only)
       ↓
Phase 2: PICO Extraction (LLM)       → 2,392 structured PICO records
       ↓
Phase 3a: Annotation Extraction (LLM) → 2,392 structured annotation records
       ↓
Phase 3b: Database Construction      → glp1_database.csv + .xlsx
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

For papers with a PubMed Central ID (PMCID), full text is retrieved via the NCBI EFetch API (`db=pmc`). Papers without open-access full text fall back to abstract only.

**Results:** 2,392 papers — 957 full-text (40%), 1,435 abstract-only (60%).

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

Sends each paper's full text (or abstract where full text is unavailable) to Claude (`claude-opus-4-5-20251101`) via the Anthropic API and extracts structured PICO fields as JSON. Runs in configurable batches with checkpoint-and-resume — safe to interrupt at any time.

**Extracted fields:**

| Field | Description |
|-------|-------------|
| `population` | Eligibility-defining patient characteristics |
| `intervention` | Drug name, dose, route, frequency, duration |
| `comparison` | All comparator arms explicitly stated |
| `outcomes_primary` | Primary outcome definitions (not results) |
| `outcomes_secondary` | Secondary outcome definitions |
| `extraction_notes` | Flags for ambiguity or missing fields |

**Results:** 2,392 / 2,392 papers extracted successfully. **0 failures.**

### Setup

```bash
pip install anthropic
```

```python
# pico_extraction.py — line 38
ANTHROPIC_API_KEY = "your_key_here"   # console.anthropic.com → API Keys

BATCH_SIZE = 800   # script resumes via checkpoint; run repeatedly until complete
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

Second LLM pass (`claude-opus-4-5-20251101`) over all papers to extract structured metadata fields:

| Field | Description |
|-------|-------------|
| `indication` | Clinical condition (e.g. Type 2 Diabetes, Obesity, CVD) |
| `trial_phase` | Phase 2 / 3 / 4 / NR |
| `enrollment_size` | Total randomized participants |
| `female_pct` | Female enrollment percentage |
| `follow_up_duration` | Duration as stated in text |
| `comparator_type` | Placebo / Active Comparator / Dose Comparison / NR |
| `women_only_trial` | yes / no |
| `sex_stratified_outcomes` | yes / no / unclear |
| `female_enrollment_reported` | yes / no |
| `subgroup_elderly` | yes / no / unclear |
| `subgroup_pediatric` | yes / no / unclear |
| `subgroup_racial_ethnic` | yes / no / unclear |
| `subgroup_sex_specific` | yes / no / unclear |
| `subgroup_renal_hepatic` | yes / no / unclear |
| `subgroup_other` | Free text or none |
| `patient_centered_outcomes` | Comma-separated list |
| `surrogate_outcomes` | Comma-separated list |
| `patient_centeredness_flag` | all_patient_centered / mixed / all_surrogate / etc. |

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

**Drug normalization** — 27 canonical drug names mapped from brand names and synonyms (e.g. Ozempic → Semaglutide, Mounjaro → Tirzepatide, Rybelsus → Semaglutide (oral)).

**ClinicalTrials.gov linkage** — best-effort PMID lookup via CT.gov API v2 (`AREA[ReferencePMID]`). Appends: NCT ID, study status, registered phase, registered enrollment.

**Outputs:**

| File | Description |
|------|-------------|
| `glp1_database.csv` | Flat 2,392-row database |
| `glp1_database.xlsx` | Excel workbook with auto-filter + frozen headers |

Excel sheets:

| Sheet | Contents |
|-------|----------|
| All Trials | Full database |
| Sex Data Available | Trials with any sex-specific data |
| Underrepresented Populations | Trials missing ≥3 subgroup dimensions |
| Summary | Per-indication aggregate statistics |
| *(one per indication)* | Filtered subsets |

---

## Key Findings

### Sex-Disaggregated Reporting Gap

Of 2,392 GLP-1 RA RCTs:
- Only **100 trials (4.2%)** reported sex-stratified outcomes
- Only **80 trials (3.3%)** were women-only
- **862 trials (36.0%)** reported female enrollment percentage at all

This gap motivates the downstream sex-stratified simulation analysis this pipeline was built to support.

### Outcome Type Distribution

- **1,888 trials (78.9%)** were surrogate-heavy (primary endpoints: HbA1c, body weight, biomarkers)
- Only **179 trials (7.5%)** were patient-centered (mortality, MACE, hospitalization, QoL)

### CT.gov Coverage

50% of papers (1,219/2,392) were successfully linked to ClinicalTrials.gov records via PMID. The remaining 50% could not be automatically matched, likely due to incomplete PMID registration in the CT.gov database.

---

## Repository Structure

```
PICOextraction/
├── pubmed_retrieval.py        # Phase 1: PubMed retrieval + PMC full-text
├── pico_extraction.py         # Phase 2: PICO extraction
├── annotate.py                # Phase 3a: annotation extraction
├── build_database.py          # Phase 3b: database builder
│
├── glp1_corpus.csv            # Phase 1 output
├── glp1_corpus.json           # Phase 1 output (JSON)
├── pico_results.csv           # Phase 2 output
├── annotations.csv            # Phase 3a output
├── glp1_database.csv          # Final database (CSV)
├── glp1_database.xlsx         # Final database (Excel)
│
├── checkpoint.json            # Phase 2 resume checkpoint
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

Python 3.10+ required. Developed and tested on Python 3.12 / Windows 11.

---

## API Keys

| Key | Where to get it | Where to set it |
|-----|----------------|-----------------|
| NCBI API key | [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) | `pubmed_retrieval.py` line 30 |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) → API Keys | `pico_extraction.py` line 38 and `annotate.py` line 38 |

> **Cost estimate:** Approximately $5–15 USD in Anthropic API credits for the full 2,392-paper extraction (Phase 2 + Phase 3a combined).

---

## Limitations

- Search restricted to PubMed/MEDLINE; Embase and CENTRAL not included
- Full text available for 40% of papers via PMC open access; remaining 60% use abstract only
- LLM extraction may introduce systematic errors on ambiguous abstracts
- CT.gov linkage achieved 50% coverage; 50% unmatched due to incomplete PMID registration
- Indication normalization involved automated string matching; multi-indication trials may be miscategorized
