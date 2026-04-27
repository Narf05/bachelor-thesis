# Bachelor Thesis — Implementation Plan

## Goal
Estimate and compare vocabulary richness across individuals. The three new corpora (BNC, IMSDb, SBCorpus) currently contain speech from mixed speakers and must be split into per-speaker files before any analysis. The pipeline then needs to be generalised so any speaker file can be preprocessed and cached in the existing SQLite database.

---

## Overview of Current State

| Component | Status |
|---|---|
| `preprocessing/loaders/shakespeare.py` | Shakespeare-specific; extracts clean literary text |
| `preprocessing/loaders/transcript.py` | Stub; only `load_plain_text` (no parsing) |
| `preprocessing/tokenizer.py` | General; reusable as-is |
| `preprocessing/frequencies.py` | General; reusable as-is |
| `db/cache.py` | Caches word+freq counts; stores metadata but **not** per-count cap |
| `data/shakespeare-dataset-main/` | 43 works, fully processed |
| `data/IMSDb/movie_scripts.parquet` | 108 MB parquet; mixed speakers; not integrated |
| `data/bnc/` | 29 HTML files; mixed speakers; not integrated |
| `data/SBCorpus/TRN/` | 60 `.trn` files; mixed speakers; not integrated |

---

## Task List

---

### TASK 1 — Explore and document raw corpus formats

Before writing any parser, read a representative sample of each format to understand its structure exactly.

- [ ] **1.1** Open and inspect 2–3 BNC HTML files (`data/bnc/*.htm`). Document: tag structure, how speaker turns are marked, what metadata surrounds the speech.
- [ ] **1.2** Open and inspect 2–3 SBCorpus `.trn` files (`data/SBCorpus/TRN/SBC001.trn`, etc.). Document: line format, speaker ID format, pause/noise markers, overlapping speech markers.
- [ ] **1.3** Open `data/IMSDb/movie_scripts.parquet` with pandas and inspect the `Script` column. Document: how characters are identified inside the screenplay text, what non-dialogue lines look like (scene descriptions, action lines, parentheticals).

---

### TASK 2 — Per-speaker extraction (new data → per-person text files)

Each corpus needs a dedicated extractor that reads the raw source and writes one plain-text file per speaker. Only the words spoken (no metadata, no stage directions, no noise markers) should end up in each file.

#### 2.1 — BNC extractor

- [ ] Write `code/preprocessing/extractors/bnc.py`
  - Parse each HTML file using BeautifulSoup (or lxml).
  - Identify speaker-turn elements and the speaker ID attribute.
  - Strip all non-speech content (headers, metadata, overlap markers, etc.).
  - Concatenate all turns for the same speaker (across all 29 files) into one string.
  - Write output: `data/speakers/bnc/<SPEAKER_ID>.txt`
- [ ] Create output directory `data/speakers/bnc/`

#### 2.2 — SBCorpus extractor

- [ ] Write `code/preprocessing/extractors/sbcorpus.py`
  - Parse each `.trn` file.
  - Identify speaker IDs and extract only their spoken turns.
  - Strip pause markers, overlap markers, non-verbal annotations, timestamps.
  - Concatenate all turns per speaker (across all 60 files) into one string.
  - Write output: `data/speakers/sbcorpus/<SPEAKER_ID>.txt`
- [ ] Create output directory `data/speakers/sbcorpus/`

#### 2.3 — IMSDb extractor

- [ ] Write `code/preprocessing/extractors/imsdb.py`
  - Load `movie_scripts.parquet` with pandas.
  - For each movie, parse the `Script` column:
    - Identify character names (usually all-caps lines or a known screenplay formatting pattern).
    - Extract only the dialogue lines spoken by each character.
    - Strip scene headings, action lines, parentheticals.
  - Write output: `data/speakers/imsdb/<MOVIE_TITLE>__<CHARACTER_NAME>.txt`
- [ ] Create output directory `data/speakers/imsdb/`

**Note:** Speaker granularity may vary. For BNC and SBCorpus use the raw speaker ID. For IMSDb, using `<MOVIE>__<CHARACTER>` avoids collisions between characters with the same name across films.

---

### TASK 3 — Generalise the preprocessing pipeline

#### 3.1 — Replace corpus-specific loaders with one general loader

The Shakespeare-specific loader logic is no longer the right abstraction because:
- New speaker files are already clean plain text after extraction (Task 2).
- The Shakespeare parser strips metadata that is specific to the Folger format; that logic is not needed for speaker files.

- [ ] **Delete** `code/preprocessing/loaders/shakespeare.py` and `code/preprocessing/loaders/transcript.py`.
- [ ] **Delete** `code/preprocessing/loaders/__init__.py` (or repurpose it — see below).
- [ ] **Evaluate** whether the `loaders/` sub-package is still needed at all. If the only remaining operation is `open(path).read()`, inline it directly in `cache.py` or `frequencies.py` and remove the sub-package entirely.
- [ ] If any Shakespeare-specific loading is still needed (e.g., stripping Folger metadata), move it to `code/preprocessing/extractors/shakespeare.py` alongside the other extractors, with the same interface: input = raw file, output = plain-text speaker file written to `data/speakers/shakespeare/<WORK_TITLE>.txt`. This makes Shakespeare just another source that goes through the same pipeline.

#### 3.2 — Unified preprocessing entry point

- [ ] Write (or heavily refactor) `code/preprocessing/pipeline.py` (or keep using `frequencies.py`) so the full pipeline is:
  1. Read plain-text file from `data/speakers/<corpus>/<id>.txt`.
  2. Tokenize (reuse `tokenizer.py` unchanged).
  3. Compute word counts and frequency counts (reuse `frequencies.py` unchanged).
  4. Store in database (see Task 4).

  No loader abstraction needed beyond `open(path).read()`.

---

### TASK 4 — Update the database schema and caching logic

#### 4.1 — Schema additions

The current `corpora` table stores `n_tokens` and `s_obs` but is missing explicit source corpus tagging. Extend it:

- [ ] Add column `corpus_source TEXT` to `corpora` (values: `'shakespeare'`, `'bnc'`, `'sbcorpus'`, `'imsdb'`).
- [ ] Add column `speaker_id TEXT` to `corpora` (the raw speaker/character identifier from the source corpus).
- [ ] Confirm `freq_counts` already has a primary key on `(corpus_id, k)` — it does; no change needed there.

#### 4.2 — Cap frequency storage at k ≤ 20

The current schema stores all `(k, f_k)` pairs. To keep the database lean and focused:

- [ ] In the caching write path (`cache.py`), filter `freq_counts` before inserting: only store rows where `k <= 20`.
- [ ] Add a note in the schema comment that `freq_counts` contains only `k = 1..20`.
- [ ] Verify that estimators that need higher-k values (ACE uses k up to 10 by default; breakaway uses up to 20 ratios) still have access to the needed counts. Options:
  - Compute higher counts on the fly from `word_counts` (already stored in full).
  - Alternatively store up to k=max_needed (e.g., 100) — decide based on estimator requirements.

  **Decision needed:** The breakaway estimator uses up to 20 ratio terms (k up to ~21). Storing up to k=20 is sufficient. ACE uses k ≤ 10. Store `k <= 20` for now.

#### 4.3 — Pre-insertion existence check

- [ ] Before processing a file, `cache.py` must query `corpora` by `(source_path, source_mtime)` (already done for Shakespeare — verify it works correctly for the new speaker file paths).
- [ ] If a record exists and `source_mtime` matches, skip processing and return cached data.
- [ ] If `force=True`, delete the old record and reprocess.

#### 4.4 — Migrate existing cached data (optional)

- [ ] The 44 existing Shakespeare entries do not have `corpus_source` or `speaker_id`. Either:
  - Backfill them in a migration script, or
  - Accept `NULL` for old entries and only populate for new ones.
  - Recommended: Write a one-off migration that sets `corpus_source='shakespeare'` for all existing rows where `loader='shakespeare'` or `loader='corpus'`.

---

### TASK 5 — Consolidate duplicate / redundant files

- [ ] **`preprocessing/__init__.py`**: Check if it exports anything; if not, clear it (keep the file for package resolution only).
- [ ] **`preprocessing/loaders/__init__.py`**: Remove once loaders are deleted (Task 3.1).
- [ ] **`data_test.ipynb`**: This notebook is a scratch exploration of the parquet format. Once the IMSDb extractor (Task 2.3) is written, this notebook has no value. Delete it or archive it.
- [ ] Review `code/analysis/shakespeare_analysis.ipynb`: confirm it calls `db/cache.py` via `get_or_process()` and will continue to work after the loader deletion. Update any direct imports of `load_shakespeare` if present.

---

### TASK 6 — Run extraction and preprocessing for all new corpora

After Tasks 1–5 are complete:

- [ ] Run `bnc.py` extractor → writes files to `data/speakers/bnc/`.
- [ ] Run `sbcorpus.py` extractor → writes files to `data/speakers/sbcorpus/`.
- [ ] Run `imsdb.py` extractor → writes files to `data/speakers/imsdb/`.
- [ ] Run preprocessing pipeline on all speaker files → populates `processed.db` with new entries.
- [ ] Verify database entries: run `list_corpora()` and confirm row counts, `n_tokens`, `s_obs` look reasonable.

---

### TASK 7 — (Future) Apply estimators to new corpora

Not in scope for this phase, but noted here so the schema and pipeline are built with this in mind:

- Compute Chao1, iChao1, ACE, Jackknife, Breakaway for each speaker entry.
- Store estimates in `corpora` table or a separate `estimates` table.
- Compare vocabulary richness across speakers, corpora types, and Shakespeare.

---

## File Map After All Tasks Complete

```
code/
  preprocessing/
    extractors/
      bnc.py             # NEW — BNC HTML → per-speaker .txt
      sbcorpus.py        # NEW — .trn → per-speaker .txt
      imsdb.py           # NEW — parquet → per-character .txt
      shakespeare.py     # NEW (if kept) — Folger .txt → clean .txt per work
    tokenizer.py         # UNCHANGED
    frequencies.py       # UNCHANGED (or minor: cap at k<=20 moved to cache.py)
    pipeline.py          # NEW or REFACTORED — unified entry point
    loaders/             # DELETED (or emptied)
  db/
    cache.py             # UPDATED — new schema columns, k<=20 cap, existence check
  estimators/            # UNCHANGED
  analysis/
    shakespeare_analysis.ipynb  # UPDATED — remove direct loader imports if any

data/
  speakers/
    shakespeare/         # NEW — one .txt per Shakespeare work
    bnc/                 # NEW — one .txt per BNC speaker
    sbcorpus/            # NEW — one .txt per SBCorpus speaker
    imsdb/               # NEW — one .txt per movie character
  shakespeare-dataset-main/   # UNCHANGED (source)
  IMSDb/                      # UNCHANGED (source)
  bnc/                        # UNCHANGED (source)
  SBCorpus/                   # UNCHANGED (source)
  processed.db                # UPDATED schema + new rows
```

---

## Implementation Order

1. **Task 1** (format exploration) — no code, just reading and documenting findings.
2. **Task 4.1 + 4.2 + 4.3** (DB schema) — update schema before any new data is written.
3. **Task 3** (generalise pipeline / delete loaders) — clean up before adding new code.
4. **Task 2** (extractors) — one extractor per corpus, in order of complexity: SBCorpus → BNC → IMSDb.
5. **Task 5** (cleanup) — remove dead files after extractors are verified.
6. **Task 6** (run everything) — populate database with all new speaker data.
7. **Task 7** (estimators) — future phase.
