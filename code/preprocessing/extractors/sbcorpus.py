"""
SBCorpus extractor — Santa Barbara Corpus of Spoken American English.

Each .trn file uses a tab-separated format:
    start_time end_time <TAB> SPEAKER: <TAB> utterance text
    start_time end_time <TAB> (spaces)  <TAB> continuation text

Annotation markers stripped before writing:
    (H), (Hx), (H)=, (TSK), (N), etc.  — breath / noise in parentheses
    <X text X>  — unclear speech (content removed)
    <@ text @>  — laughing quality (content removed)
    <TAG text TAG>  — other quality markers (content kept, tags removed)
    [2 ... 2], [3 ... 3]  — overlap brackets (content kept, brackets removed)
    [text]  — transcriber notes (content kept, brackets removed)
    ...  ..  — pause markers
    =  — vowel elongation marker (e.g. ti=me -> time)
    ~  — word truncation marker
    --  — false start / interruption
    %  — incomplete word marker
    XX  — unintelligible

Usage
-----
    from preprocessing.extractors.sbcorpus import extract
    extract("data/SBCorpus/TRN", "data/speakers/sbcorpus")
"""

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Annotation-cleaning regexes (applied in order)
# ---------------------------------------------------------------------------

# Breath / noise markers: (H), (Hx), (H)=, (TSK), (N), (TSK)=, (Hx)=, etc.
_PAREN_MARKER  = re.compile(r"\([A-Za-z]+[=]?\)")

# Quality markers with code letters: <X text X>, <@ text @>
# These contain distorted/unclear speech — remove content entirely
_QUALITY_BLOCK = re.compile(r"<[X@]\s.*?\s[X@]>", re.DOTALL)

# Other angle-bracket annotations: strip opening/closing tags, keep words
_ANGLE_OPEN    = re.compile(r"<[A-Z]+\s")   # e.g. <YWN
_ANGLE_CLOSE   = re.compile(r"\s[A-Z]+>")   # e.g. YWN>

# Overlap brackets [2...2], [3...3] — remove numbers, keep text
_OVERLAP_NUM   = re.compile(r"\[(\d+)(.*?)\1\]")

# Remaining brackets [text] — keep content, remove brackets
_BRACKET       = re.compile(r"\[(.*?)\]")

# Pause markers (two or more dots)
_PAUSE         = re.compile(r"\.{2,}")

# False starts / interruptions
_FALSE_START   = re.compile(r"\s*--\s*")

# Elongation (= inside/end of word), truncation (~), incomplete (%)
_ELONGATION    = re.compile(r"=")
_TRUNCATION    = re.compile(r"~")
_INCOMPLETE    = re.compile(r"%")

# Unintelligible marker XX
_UNCLEAR       = re.compile(r"\bX{2,}\b")

# Collapse multiple spaces
_MULTI_SPACE   = re.compile(r"\s{2,}")


def _word_count(text: str) -> int:
    return len(text.split())


def _clean(text: str) -> str:
    text = _QUALITY_BLOCK.sub(" ", text)
    text = _PAREN_MARKER.sub(" ", text)
    text = _ANGLE_OPEN.sub(" ", text)
    text = _ANGLE_CLOSE.sub(" ", text)
    text = _OVERLAP_NUM.sub(r" \2 ", text)
    text = _BRACKET.sub(r" \1 ", text)
    text = _PAUSE.sub(" ", text)
    text = _FALSE_START.sub(" ", text)
    text = _ELONGATION.sub("", text)
    text = _TRUNCATION.sub("", text)
    text = _INCOMPLETE.sub("", text)
    text = _UNCLEAR.sub(" ", text)
    return _MULTI_SPACE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_trn(filepath: Path) -> dict[str, list[str]]:
    """
    Parse one .trn file.
    Returns {speaker_id: [cleaned_text_segment, ...]} for speakers in the file.
    """
    speaker_segments: dict[str, list[str]] = {}
    current_speaker: str | None = None

    for raw_line in filepath.read_text(encoding="utf-8", errors="replace").splitlines():
        # Lines have at least two tab characters
        parts = raw_line.split("\t", 2)
        if len(parts) < 3:
            continue

        _timestamps, speaker_field, text_field = parts
        speaker_field = speaker_field.strip()
        text_field    = text_field.strip()

        # Non-blank speaker field means a new speaker turn
        if speaker_field:
            # Remove trailing colon(s) and whitespace: "LYNNE:  " -> "LYNNE"
            candidate = re.sub(r":+$", "", speaker_field).strip().upper()
            # Skip non-speaker tracks (e.g. >ENV) and reject anything that
            # doesn't look like a real speaker name (only word chars + hyphens,
            # max 30 chars) to avoid misidentifying dialogue as a speaker ID.
            if (candidate
                    and not candidate.startswith(">")
                    and re.fullmatch(r"[A-Z0-9][A-Z0-9\-_]{0,29}", candidate)):
                current_speaker = candidate

        if current_speaker is None or not text_field:
            continue

        cleaned = _clean(text_field)
        if cleaned:
            speaker_segments.setdefault(current_speaker, []).append(cleaned)

    return speaker_segments


def extract(
    input_dir: str | Path,
    output_dir: str | Path,
    min_words: int | None = 400,
) -> dict[str, Path]:
    """
    Parse all .trn files in *input_dir* and write one .txt file per speaker
    to *output_dir*.  Speakers appearing in multiple files are merged.

    Speakers with fewer than *min_words* total words are skipped. If
    *min_words* is None, no minimum is applied.

    Returns {speaker_id: output_path}.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_segments: dict[str, list[str]] = {}
    for trn_file in sorted(input_dir.glob("*.trn")):
        for speaker_id, segments in _parse_trn(trn_file).items():
            all_segments.setdefault(speaker_id, []).extend(segments)

    output_paths: dict[str, Path] = {}
    for speaker_id, segments in sorted(all_segments.items()):
        out_path = output_dir / f"{speaker_id}.txt"
        text = "\n".join(segments)
        if min_words is not None and _word_count(text) < min_words:
            if out_path.exists():
                out_path.unlink()
            continue
        out_path.write_text(text, encoding="utf-8")
        output_paths[speaker_id] = out_path

    print(f"[sbcorpus] wrote {len(output_paths)} speaker files to {output_dir}")
    return output_paths
