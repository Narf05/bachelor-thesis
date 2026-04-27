r"""
Shakespeare extractor — Folger plain-text files.

Two usage modes
---------------
1. Text cleaning (used by the cache for backwards-compatible cache misses):
   clean_text(filepath)    -> str   (clean literary text)
   clean_corpus(directory) -> str   (all works concatenated)

2. Per-work extraction to speaker files (used to populate data/speakers/):
   extract_work(filepath, output_dir)    -> Path  (output .txt path)
   extract_corpus(directory, output_dir) -> list[Path]

Folger file structure
---------------------
    <title>
    by William Shakespeare
    ...
    Created on ...

    ACT 1          <- plays: content starts here
    =====
    Scene 1
    SPEAKER NAME
    Dialogue line.
    SPEAKER  Inline dialogue.

    ---OR (sonnets/poems)---
    1              <- sonnet number
    From fairest creatures ...

Lines stripped: file header, dividers (=====), act/scene headings,
sonnet numbers, full-line stage directions, inline stage directions,
speaker-only lines, speaker-prefixed dialogue (name kept only as speaker).
"""

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_SPEAKER_INLINE = re.compile(r"^([A-Z][A-Z\s,'\-\.]+)\s{2,}(.+)$")
_FULL_STAGE     = re.compile(r"^\[.*\]$")
_INLINE_STAGE   = re.compile(r"\[.*?\]")
_HTML_TAG       = re.compile(r"<[^>]+>")


def _is_all_caps(line: str) -> bool:
    alpha = [c for c in line if c.isalpha()]
    return bool(alpha) and all(c.isupper() for c in alpha)


# ---------------------------------------------------------------------------
# Core cleaning logic (single file)
# ---------------------------------------------------------------------------

def clean_text(filepath: str | Path) -> str:
    """
    Read a Folger Shakespeare plain-text file and return only the literary
    text (dialogue for plays, verse for poems/sonnets).
    """
    filepath = Path(filepath)
    raw   = filepath.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    # Pass 1 — find content start
    is_play = False
    content_start = 8

    for i, ln in enumerate(lines):
        if re.match(r"^ACT\s+\d+", ln.strip()):
            is_play = True
            content_start = i
            break

    if not is_play:
        for i, ln in enumerate(lines):
            if re.match(r"^Created on", ln.strip()):
                content_start = i + 1
                break

    # Pass 2 — line-by-line filtering
    kept = []
    for ln in lines[content_start:]:
        stripped = ln.strip()

        if not stripped:
            continue
        if re.match(r"^=+$", stripped):
            continue
        if re.match(r"^(ACT|Scene)\s", stripped):
            continue
        if re.match(r"^\d+$", stripped):
            continue
        if _FULL_STAGE.match(stripped):
            continue

        stripped = _HTML_TAG.sub("", stripped).strip()
        if not stripped:
            continue

        stripped = _INLINE_STAGE.sub("", stripped).strip()
        if not stripped:
            continue

        m = _SPEAKER_INLINE.match(stripped)
        if m:
            dialogue = m.group(2).strip()
            if dialogue:
                kept.append(dialogue)
            continue

        if _is_all_caps(stripped):
            continue

        kept.append(stripped)

    return "\n".join(kept)


def clean_corpus(directory: str | Path) -> str:
    """Clean and concatenate all Shakespeare .txt files in *directory*."""
    directory = Path(directory)
    return "\n".join(clean_text(p) for p in sorted(directory.glob("*.txt")))


def _word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Extraction to speaker files
# ---------------------------------------------------------------------------

def extract_work(
    filepath: str | Path,
    output_dir: str | Path,
    min_words: int | None = 400,
) -> Path | None:
    """
    Clean one Folger .txt file and write the result to
    *output_dir/<stem>.txt*.  Returns the output path.
    """
    filepath   = Path(filepath)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a readable title from the filename
    stem = (filepath.stem
            .replace("_TXT_FolgerShakespeare", "")
            .replace("-", "_"))
    out_path = output_dir / f"{stem}.txt"
    text = clean_text(filepath)
    if min_words is not None and _word_count(text) < min_words:
        if out_path.exists():
            out_path.unlink()
        return None
    out_path.write_text(text, encoding="utf-8")
    return out_path


def extract_corpus(
    directory: str | Path,
    output_dir: str | Path,
    min_words: int | None = 400,
) -> list[Path]:
    """
    Run extract_work on every .txt file in *directory*.
    Returns the list of written output paths.
    """
    directory = Path(directory)
    return [
        out_path
        for p in sorted(directory.glob("*.txt"))
        if (out_path := extract_work(p, output_dir, min_words=min_words)) is not None
    ]
