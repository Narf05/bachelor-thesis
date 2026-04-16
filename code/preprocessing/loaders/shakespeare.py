r"""
Loaders for Shakespeare Folger plain-text files.

Each file has the structure:

    <title>
    by William Shakespeare
    Edited by ...
    Folger Shakespeare Library
    https://...
    Created on ...

    Characters in the Play        <- plays only
    ======================
    ...

    ACT 1                         <- plays: content starts here
    =====
    Scene 1
    =======
    [stage direction]

    SPEAKER NAME
    Dialogue line.

    SPEAKER  Inline dialogue.

    ---OR (for sonnets/poems)---

    1                             <- sonnet number
    From fairest creatures ...

Lines to strip
--------------
- The file header (everything before the first content line)
- Dividers:          ^=+$
- Act headers:       ^ACT\s
- Scene headers:     ^Scene\s
- Sonnet numbers:    ^\d+$
- Full stage dirs:   ^\[.*\]$
- Inline stage dirs: \[.*?\]  (removed from line)
- Speaker-only:      lines where every alphabetic character is uppercase
- Speaker + text:    "SPEAKER NAME  dialogue"  ->  keep "dialogue" only
"""

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_all_caps(line: str) -> bool:
    """Return True if every alphabetic character in *line* is uppercase."""
    alpha = [c for c in line if c.isalpha()]
    return bool(alpha) and all(c.isupper() for c in alpha)


# Matches "SPEAKER NAME  rest of line" (2+ spaces separate name from text)
_SPEAKER_INLINE = re.compile(r"^([A-Z][A-Z\s,'\-\.]+)\s{2,}(.+)$")

# Matches a complete stage direction on its own line
_FULL_STAGE = re.compile(r"^\[.*\]$")

# Matches inline stage directions to be removed
_INLINE_STAGE = re.compile(r"\[.*?\]")

# HTML tags sometimes present in Folger files (e.g. <title>…</title>)
_HTML_TAG = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_shakespeare(filepath: str | Path) -> str:
    """
    Read a Folger Shakespeare plain-text file and return only the
    literary text (dialogue for plays, verse for poems/sonnets).

    All metadata, character lists, stage directions, speaker labels,
    act/scene headings, and dividers are removed.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    str
        Clean literary text, one dialogue / verse line per line.
    """
    filepath = Path(filepath)
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    # ------------------------------------------------------------------ #
    # Pass 1 — find where literary content begins                          #
    # ------------------------------------------------------------------ #
    # Check for ACT marker first (plays); fall back to "Created on" (poems).
    # Two short early-terminating loops — files are ~200 lines.
    is_play = False
    content_start = 8  # fallback for poems without a "Created on" line

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

    # ------------------------------------------------------------------ #
    # Pass 2 — line-by-line filtering                                      #
    # ------------------------------------------------------------------ #
    kept = []
    for ln in lines[content_start:]:
        ln = ln.rstrip("\n")
        stripped = ln.strip()

        # Blank lines
        if not stripped:
            continue

        # Dividers (=====)
        if re.match(r"^=+$", stripped):
            continue

        # Act / Scene headers
        if re.match(r"^(ACT|Scene)\s", stripped):
            continue

        # Standalone numbers (sonnet / section numbers)
        if re.match(r"^\d+$", stripped):
            continue

        # Full-line stage direction
        if _FULL_STAGE.match(stripped):
            continue

        # Remove HTML tags (some Folger files use <title>…</title>)
        stripped = _HTML_TAG.sub("", stripped).strip()
        if not stripped:
            continue

        # Remove inline stage directions [Exit.] [Aside.] etc.
        stripped = _INLINE_STAGE.sub("", stripped).strip()
        if not stripped:
            continue

        # "SPEAKER  dialogue" — strip the name, keep the dialogue
        m = _SPEAKER_INLINE.match(stripped)
        if m:
            dialogue = m.group(2).strip()
            if dialogue:
                kept.append(dialogue)
            continue

        # Speaker-only line (all caps, name alone)
        if _is_all_caps(stripped):
            continue

        # Everything else is dialogue or verse
        kept.append(stripped)

    return "\n".join(kept)


def load_corpus(directory: str | Path) -> str:
    """
    Load all Shakespeare .txt files in *directory* and concatenate
    their cleaned texts into a single string.

    Parameters
    ----------
    directory : str or Path

    Returns
    -------
    str
    """
    directory = Path(directory)
    texts = []
    for path in sorted(directory.glob("*.txt")):
        texts.append(load_shakespeare(path))
    return "\n".join(texts)
