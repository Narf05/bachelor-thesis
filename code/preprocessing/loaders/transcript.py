"""
Loaders for spoken-language transcripts.

Planned formats
---------------
- Plain .txt transcripts (e.g. manual or Whisper output)
- Whisper .vtt / .srt subtitle files (timestamp lines stripped)

Filler words ("um", "uh", …) are NOT stripped here — that is handled
downstream by the tokenizer so the choice stays configurable.
"""

import re
from pathlib import Path


def load_plain_text(filepath: str | Path) -> str:
    """
    Load a plain-text transcript and return its contents as-is.

    No structural stripping is performed — the file is assumed to
    contain only the words to be analysed (no speaker labels,
    timestamps, or metadata).

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    str
    """
    return Path(filepath).read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# TODO: Whisper VTT / SRT loader
# ---------------------------------------------------------------------------
# Whisper output looks like:
#
#   WEBVTT
#
#   00:00:00.000 --> 00:00:03.140
#   So the first thing I want to talk about today is...
#
#   00:00:03.140 --> 00:00:06.020
#   um, the idea of species richness.
#
# The loader should:
#   1. Skip the WEBVTT header line.
#   2. Skip timestamp lines (pattern: HH:MM:SS.mmm --> HH:MM:SS.mmm).
#   3. Skip blank lines.
#   4. Concatenate the remaining text lines.
#
# def load_whisper_vtt(filepath: str | Path) -> str:
#     ...
