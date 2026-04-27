"""
Preprocessing runner — extract per-speaker files and populate the database.

Run from the code/ directory:
    python preprocess.py [--corpus all|shakespeare|bnc|sbcorpus|imsdb]
                         [--force]
                         [--fillers]
                         [--min-words N|none]

Steps per corpus
----------------
1. Extract per-speaker .txt files into data/speakers/<corpus>/
2. Call get_or_process() on each file to tokenise and cache in processed.db

The Shakespeare step (1) re-extracts from the Folger source files so the
per-work files in data/speakers/shakespeare/ stay in sync with the DB.
"""

import argparse
import sys
from pathlib import Path

# Make sure code/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from db.cache import clear_missing_corpora, get_or_process

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data"
SPEAKERS    = DATA_DIR / "speakers"


# ---------------------------------------------------------------------------
# Step 1 helpers — run extractors
# ---------------------------------------------------------------------------

def extract_shakespeare(min_words: int | None = 400):
    from preprocessing.extractors.shakespeare import extract_corpus
    src = DATA_DIR / "shakespeare-dataset-main" / "text"
    out = SPEAKERS / "shakespeare"
    print(f"\n[extract] Shakespeare: {src} → {out}")
    print(f"[extract] minimum words per file: {'none' if min_words is None else min_words}")
    paths = extract_corpus(src, out, min_words=min_words)
    print(f"[extract] {len(paths)} files written.")
    return paths


def extract_bnc(min_words: int | None = 400):
    from preprocessing.extractors.bnc import extract
    src = DATA_DIR / "bnc"
    out = SPEAKERS / "bnc"
    print(f"\n[extract] BNC: {src} → {out}")
    print(f"[extract] minimum words per file: {'none' if min_words is None else min_words}")
    return extract(src, out, min_words=min_words)


def extract_sbcorpus(min_words: int | None = 400):
    from preprocessing.extractors.sbcorpus import extract
    src = DATA_DIR / "SBCorpus" / "TRN"
    out = SPEAKERS / "sbcorpus"
    print(f"\n[extract] SBCorpus: {src} → {out}")
    print(f"[extract] minimum words per file: {'none' if min_words is None else min_words}")
    return extract(src, out, min_words=min_words)


def extract_imsdb(force: bool = False, min_words: int | None = 400):
    from preprocessing.extractors.imsdb import extract
    src = DATA_DIR / "IMSDb" / "movie_scripts.parquet"
    out = SPEAKERS / "imsdb"
    print(f"\n[extract] IMSDb: {src} → {out}")
    min_words_label = "none" if min_words is None else str(min_words)
    print(f"[extract] IMSDb minimum words per character: {min_words_label}")
    return extract(src, out, min_words=min_words, force=force)


# ---------------------------------------------------------------------------
# Step 2 helper — process speaker files into DB
# ---------------------------------------------------------------------------

def _already_processed() -> set[str]:
    """Return the set of corpus names already stored in the database."""
    try:
        from db.cache import list_corpora
        return set(list_corpora()["name"].tolist())
    except Exception:
        return set()


def process_speaker_dir(
    corpus_source: str,
    speaker_dir: Path,
    *,
    remove_fillers: bool = False,
    force: bool = False,
):
    """
    Call get_or_process() for every .txt file in *speaker_dir*.

    The cache name is  <corpus_source>__<stem>  (e.g. 'bnc__PS1DA').

    Already-cached files are skipped entirely so a crashed run can be
    resumed from where it left off without re-touching the database.
    """
    txt_files = sorted(speaker_dir.glob("*.txt"))
    if not txt_files:
        print(f"[process] No .txt files found in {speaker_dir} — run extraction first.")
        return

    done = set() if force else _already_processed()

    pending = [f for f in txt_files
               if f"{corpus_source}__{f.stem}" not in done]
    skipped = len(txt_files) - len(pending)

    print(f"\n[process] {corpus_source}: {len(txt_files)} files total — "
          f"{skipped} already done, {len(pending)} to process.")

    for i, txt_path in enumerate(pending, 1):
        name = f"{corpus_source}__{txt_path.stem}"
        print(f"  [{i}/{len(pending)}]", end=" ")
        get_or_process(
            name,
            txt_path,
            loader="plain_text",
            corpus_source=corpus_source,
            speaker_id=txt_path.stem,
            remove_fillers=remove_fillers,
            force=force,
            verbose=True,
        )        


def sync_cache_to_speaker_dir(corpus_source: str, speaker_dir: Path) -> None:
    """
    Remove cached entries for *corpus_source* that no longer have a .txt file.
    """
    valid_names = {f"{corpus_source}__{txt_path.stem}"
                   for txt_path in speaker_dir.glob("*.txt")}
    clear_missing_corpora(corpus_source, valid_names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CORPORA = ["shakespeare", "bnc", "sbcorpus", "imsdb"]

EXTRACTORS = {
    "shakespeare": extract_shakespeare,
    "bnc":         extract_bnc,
    "sbcorpus":    extract_sbcorpus,
    "imsdb":       extract_imsdb,
}


def main():
    def _parse_min_words(value: str) -> int | None:
        if value.lower() in {"none", "no", "off"}:
            return None
        parsed = int(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("--min-words must be >= 0, or 'none'.")
        return parsed

    parser = argparse.ArgumentParser(description="Extract and preprocess corpora.")
    parser.add_argument(
        "--corpus", default="all",
        choices=["all"] + CORPORA,
        help="Which corpus to process (default: all).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess even if a cache entry already exists.",
    )
    parser.add_argument(
        "--fillers", action="store_true",
        help="Strip filler words (um, uh, …) during tokenisation.",
    )
    parser.add_argument(
        "--min-words", type=_parse_min_words, default="none",
        help="Minimum words required to write a corpus speaker/work file (default: 400). Use 'none' to disable the cutoff.",
    )
    args = parser.parse_args()

    targets = CORPORA if args.corpus == "all" else [args.corpus]

    for corpus in targets:
        # Step 1: extract
        extractor = EXTRACTORS[corpus]
        if corpus == "imsdb":
            extractor(force=args.force, min_words=args.min_words)
        else:
            extractor(min_words=args.min_words)

        sync_cache_to_speaker_dir(corpus, SPEAKERS / corpus)

        # Step 2: process into DB
        process_speaker_dir(
            corpus,
            SPEAKERS / corpus,
            remove_fillers=args.fillers,
            force=args.force,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
