"""
BNC extractor — British National Corpus spoken HTML files.

Each HTML file (e.g. D92.html) contains a <table> with alternating rows:
    <tr>
        <td valign="top">(SPEAKER_ID)</td>
        <td>[1] Utterance text.<br>[2] Another sentence.<br></td>
    </tr>

The speaker-metadata table has class="dramper" and is skipped.
Utterances are stripped of [N] numbering and [...] redaction markers.

Usage
-----
    from preprocessing.extractors.bnc import extract
    extract("data/bnc", "data/speakers/bnc")
"""

import re
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError as e:
    raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4") from e


_SPEAKER_ID   = re.compile(r"\(([A-Z0-9]+)\)")
_UTT_NUMBER   = re.compile(r"^\s*\[\d+\]\s*")   # leading [N] (may have leading spaces)
_REDACTION    = re.compile(r"\[\s*\.\.\.\s*\]")  # [...] unclear content
_EXTRA_SPACE  = re.compile(r"\s{2,}")

# Files that are not transcript data
_SKIP_FILES = {"index.html", "PraatSearch.html"}


def _word_count(text: str) -> int:
    return len(text.split())


def _parse_html(filepath: Path) -> dict[str, list[str]]:
    """
    Parse one BNC HTML file.
    Returns {speaker_id: [utterance_text, ...]} for all speakers in the file.

    BNC files use different table class names across documents
    (no class, class="dialog", etc.).  We detect speaker-turn rows by the
    presence of <td valign="top"> in the first cell, which is consistent
    across all BNC HTML files.
    """
    html = filepath.read_text(encoding="iso-8859-1", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    speaker_utterances: dict[str, list[str]] = {}
    current_speaker = None

    # Find all <tr> elements that have a <td valign="top"> as their first cell
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 2:
            continue
        if tds[0].get("valign", "").lower() != "top":
            continue

        # First <td valign="top">: contains (SPEAKER_ID)
        speaker_cell = tds[0].get_text()
        m = _SPEAKER_ID.search(speaker_cell)
        if m:
            current_speaker = m.group(1)

        if current_speaker is None:
            continue

        # Second <td>: utterances separated by <br>
        utt_cell = tds[1]
        for br in utt_cell.find_all("br"):
            br.replace_with("\n")
        raw_text = utt_cell.get_text()

        for line in raw_text.splitlines():
            line = _UTT_NUMBER.sub("", line)   # strip [N] prefix
            line = _REDACTION.sub(" ", line)   # strip [...] redaction
            line = _EXTRA_SPACE.sub(" ", line).strip()
            if line:
                speaker_utterances.setdefault(current_speaker, []).append(line)

    return speaker_utterances


def extract(
    input_dir: str | Path,
    output_dir: str | Path,
    min_words: int | None = 400,
) -> dict[str, Path]:
    """
    Parse all BNC HTML files in *input_dir* and write one .txt file per
    speaker to *output_dir*.

    Speakers with fewer than *min_words* total words are skipped. If
    *min_words* is None, no minimum is applied.

    Returns {speaker_id: output_path}.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate utterances per speaker across all files
    all_utterances: dict[str, list[str]] = {}
    for html_file in sorted(input_dir.glob("*.html")):
        if html_file.name in _SKIP_FILES:
            continue
        for speaker_id, utterances in _parse_html(html_file).items():
            all_utterances.setdefault(speaker_id, []).extend(utterances)

    output_paths: dict[str, Path] = {}
    for speaker_id, utterances in sorted(all_utterances.items()):
        out_path = output_dir / f"{speaker_id}.txt"
        text = "\n".join(utterances)
        if min_words is not None and _word_count(text) < min_words:
            if out_path.exists():
                out_path.unlink()
            continue
        out_path.write_text(text, encoding="utf-8")
        output_paths[speaker_id] = out_path

    print(f"[bnc] wrote {len(output_paths)} speaker files to {output_dir}")
    return output_paths
