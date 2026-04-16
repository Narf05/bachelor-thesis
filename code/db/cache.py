"""
SQLite cache for preprocessed corpus data.

Stores word counts and frequency counts so text files are only tokenised
and counted once.  The cache is automatically invalidated when the source
file (or any file in a corpus directory) is newer than the stored entry.

Database location: data/processed.db  (relative to the repo root)

Public API
----------
get_or_process(name, source, loader, ...)  -> (word_counts, freq_counts)
list_corpora()                             -> pandas DataFrame of cached entries
clear_corpus(name)                         -> remove one entry
clear_all()                                -> drop everything
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Paths & types
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH    = _REPO_ROOT / "data" / "processed.db"

Loader = Literal["shakespeare", "corpus", "plain_text"]

# ---------------------------------------------------------------------------
# Schema — initialised once per process
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT    UNIQUE NOT NULL,
    source_path        TEXT    NOT NULL,
    loader             TEXT    NOT NULL,
    n_tokens           INTEGER,
    s_obs              INTEGER,
    coverage_turing    REAL,
    coverage_chao_jost REAL,
    source_mtime       REAL    NOT NULL,
    processed_at       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS word_counts (
    corpus_id  INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    word       TEXT    NOT NULL,
    count      INTEGER NOT NULL,
    PRIMARY KEY (corpus_id, word)
);

CREATE TABLE IF NOT EXISTS freq_counts (
    corpus_id  INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    k          INTEGER NOT NULL,
    f_k        INTEGER NOT NULL,
    PRIMARY KEY (corpus_id, k)
);

CREATE INDEX IF NOT EXISTS idx_wc_corpus ON word_counts(corpus_id);
CREATE INDEX IF NOT EXISTS idx_fc_corpus ON freq_counts(corpus_id);
"""

_schema_applied = False


def _ensure_schema() -> None:
    """Run schema creation exactly once per process."""
    global _schema_applied
    if _schema_applied:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    _schema_applied = True


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------

@contextmanager
def _db():
    """Yield an open, schema-ready connection; commit and close on exit."""
    _ensure_schema()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Loader registry — avoids if-elif chain in _load_and_process
# ---------------------------------------------------------------------------

def _get_loaders() -> dict:
    from preprocessing.loaders import load_shakespeare, load_corpus, load_plain_text
    return {
        "shakespeare": load_shakespeare,
        "corpus":      load_corpus,
        "plain_text":  load_plain_text,
    }


def _load_and_process(
    source: Path,
    loader: Loader,
    remove_fillers: bool,
    lemmatize: bool,
) -> tuple[dict[str, int], dict[int, int]]:
    """Run the full preprocessing pipeline and return (word_counts, freq_counts)."""
    from preprocessing.frequencies import pipeline

    loaders = _get_loaders()
    if loader not in loaders:
        raise ValueError(f"Unknown loader: {loader!r}. "
                         f"Choose from {list(loaders)!r}.")
    text = loaders[loader](source)
    return pipeline(text, remove_fillers=remove_fillers, lemmatize=lemmatize)


def _load_from_db(
    conn: sqlite3.Connection, corpus_id: int
) -> tuple[dict[str, int], dict[int, int]]:
    """Read word_counts and freq_counts for *corpus_id* from the database."""
    wc = dict(conn.execute(
        "SELECT word, count FROM word_counts WHERE corpus_id = ?",
        (corpus_id,)
    ).fetchall())
    fc = {k: fk for k, fk in conn.execute(
        "SELECT k, f_k FROM freq_counts WHERE corpus_id = ? ORDER BY k",
        (corpus_id,)
    ).fetchall()}
    return wc, fc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_process(
    name: str,
    source: str | Path,
    loader: Loader = "shakespeare",
    *,
    remove_fillers: bool = False,
    lemmatize: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> tuple[dict[str, int], dict[int, int]]:
    """
    Return (word_counts, freq_counts), loading from cache when possible.

    On first call the source is tokenised and stored in the SQLite database.
    Subsequent calls return the stored data instantly.  If the source file is
    newer than the cached entry the cache is refreshed automatically.

    Parameters
    ----------
    name : str
        Unique identifier, e.g. 'hamlet' or 'full_corpus'.
    source : str or Path
        Path to a single .txt file, or a directory when loader='corpus'.
    loader : 'shakespeare' | 'corpus' | 'plain_text'
    remove_fillers : bool
        Strip filler words (for speech transcripts). Default False.
    lemmatize : bool
        Lemmatize tokens with NLTK. Default False.
    force : bool
        Ignore the cache and reprocess unconditionally.
    verbose : bool
        Print a one-line status message. Default True.
    """
    source = Path(source)
    current_mtime = _source_mtime(source, loader)

    with _db() as conn:
        # ---- Try cache ----------------------------------------------------
        if not force:
            row = conn.execute(
                "SELECT id, source_mtime FROM corpora WHERE name = ?", (name,)
            ).fetchone()

            if row is not None:
                corpus_id, cached_mtime = row
                if cached_mtime >= current_mtime:
                    if verbose:
                        print(f"[cache] loaded '{name}' from database.")
                    return _load_from_db(conn, corpus_id)
                if verbose:
                    print(f"[cache] '{name}' is stale — reprocessing.")

        # ---- Process ------------------------------------------------------
        if verbose:
            print(f"[cache] processing '{name}' from {source} ...")

        wc, fc = _load_and_process(source, loader, remove_fillers, lemmatize)

        from estimators.coverage import coverage_turing, coverage_chao_jost
        n_tok  = sum(k * fk for k, fk in fc.items())  # derived from fc, not wc
        s_obs  = sum(fc.values())
        cov_t  = coverage_turing(fc)
        cov_cj = coverage_chao_jost(fc)
        now    = datetime.now(timezone.utc).isoformat()

        # ---- Store --------------------------------------------------------
        conn.execute("DELETE FROM corpora WHERE name = ?", (name,))
        cursor = conn.execute(
            """INSERT INTO corpora
                   (name, source_path, loader, n_tokens, s_obs,
                    coverage_turing, coverage_chao_jost, source_mtime, processed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, str(source), loader, n_tok, s_obs,
             cov_t, cov_cj, current_mtime, now),
        )
        corpus_id = cursor.lastrowid

        conn.executemany(
            "INSERT INTO word_counts (corpus_id, word, count) VALUES (?, ?, ?)",
            [(corpus_id, w, c) for w, c in wc.items()],
        )
        conn.executemany(
            "INSERT INTO freq_counts (corpus_id, k, f_k) VALUES (?, ?, ?)",
            [(corpus_id, k, fk) for k, fk in fc.items()],
        )

        if verbose:
            print(f"[cache] '{name}' stored  (n={n_tok:,}, S_obs={s_obs:,}).")
        return wc, fc


def list_corpora() -> "pd.DataFrame":
    """Return a DataFrame of all cached corpora and their metadata."""
    import pandas as pd
    with _db() as conn:
        rows = conn.execute(
            """SELECT name, loader, n_tokens, s_obs,
                      coverage_turing, coverage_chao_jost,
                      processed_at, source_path
               FROM corpora ORDER BY name"""
        ).fetchall()
    cols = ["name", "loader", "n_tokens", "s_obs",
            "coverage_turing", "coverage_chao_jost",
            "processed_at", "source_path"]
    return pd.DataFrame(rows, columns=cols)


def clear_corpus(name: str) -> None:
    """Remove one cached corpus (cascade-deletes word/freq counts)."""
    with _db() as conn:
        conn.execute("DELETE FROM corpora WHERE name = ?", (name,))
    print(f"[cache] '{name}' removed.")


def clear_all() -> None:
    """Remove all cached data."""
    with _db() as conn:
        conn.execute("DELETE FROM corpora")
    print("[cache] all entries removed.")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _source_mtime(source: Path, loader: Loader) -> float:
    if loader == "corpus":
        txt_files = list(source.glob("*.txt"))
        return max((p.stat().st_mtime for p in txt_files), default=0.0)
    return source.stat().st_mtime if source.exists() else 0.0
