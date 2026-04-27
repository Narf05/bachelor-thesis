"""
IMSDb extractor — Internet Movie Script Database parquet file.

The parquet file has columns: Movie, Script (and others).
The Script column contains raw screenplay text using \\r\\n line endings.

Screenplay structure detected heuristically:
    CHARACTER NAME       <- ALL CAPS, not a scene heading, moderate length
    (parenthetical)      <- starts with "(" — skipped
    Dialogue line.       <- follows the character name until blank line

Scene headings are ALL CAPS but contain location/time patterns:
    INT., EXT., INT/EXT, " - DAY", " - NIGHT", " - MORNING", etc.

Output filename: data/speakers/imsdb/<MOVIE_TITLE>__<CHARACTER>.txt
  - Movie title is sanitised (non-alphanumeric → underscore, lowercase)
  - Character name is sanitised the same way

Usage
-----
    from preprocessing.extractors.imsdb import extract
    extract("data/IMSDb/movie_scripts.parquet", "data/speakers/imsdb")
"""

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Screenplay line classification
# ---------------------------------------------------------------------------

_SCENE_MARKER = re.compile(
    r"\b(INT\.|EXT\.|INT/EXT\.|I/E\.)"
    r"|(\s+-\s+(DAY|NIGHT|MORNING|EVENING|DUSK|DAWN|LATER|CONTINUOUS))\b",
    re.IGNORECASE,
)

_MULTI_SPACE = re.compile(r"\s{2,}")


def _is_scene_heading(stripped: str) -> bool:
    return bool(_SCENE_MARKER.search(stripped))


def _is_character_cue(stripped: str) -> bool:
    """True if line looks like a character name before dialogue."""
    if not stripped:
        return False
    # Must be ALL CAPS (only alphabetic chars considered)
    alpha = [c for c in stripped if c.isalpha()]
    if not alpha or not all(c.isupper() for c in alpha):
        return False
    # Scene headings also all-caps — exclude them
    if _is_scene_heading(stripped):
        return False
    # Very long lines are action or headings, not character names
    if len(stripped) > 50:
        return False
    return True


def _sanitise(name: str) -> str:
    """Make a string safe for use in a filename."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_").lower()


def _word_count(text: str) -> int:
    """Count whitespace-delimited words in a dialogue block."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Per-script parser
# ---------------------------------------------------------------------------

def _parse_script(script: str) -> dict[str, list[str]]:
    """
    Extract {character: [dialogue_line, ...]} from one screenplay string.
    """
    lines = script.replace("\r\n", "\n").replace("\r", "\n").splitlines()

    character_dialogue: dict[str, list[str]] = {}
    current_char: str | None = None
    in_dialogue = False

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            in_dialogue = False
            current_char = None
            continue

        if in_dialogue:
            # Parentheticals inside dialogue block — skip, stay in dialogue
            if stripped.startswith("(") and stripped.endswith(")"):
                continue
            # Another ALL CAPS line while in dialogue = new character cue
            if _is_character_cue(stripped):
                current_char = stripped
                character_dialogue.setdefault(current_char, [])
                # don't add cue line as dialogue
                continue
            # Regular dialogue line
            if current_char is not None:
                cleaned = _MULTI_SPACE.sub(" ", stripped)
                if cleaned:
                    character_dialogue[current_char].append(cleaned)
        else:
            if _is_character_cue(stripped):
                current_char = stripped
                character_dialogue.setdefault(current_char, [])
                in_dialogue = True

    return character_dialogue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(
    parquet_path: str | Path,
    output_dir: str | Path,
    min_words: int | None = 400,
    force: bool = False,
) -> dict[str, Path]:
    """
    Parse every movie script in *parquet_path* and write one .txt file per
    character to *output_dir*.

    Characters with fewer than *min_words* total words are skipped to avoid
    extremely sparse files. If *min_words* is None, no minimum is applied.

    If *force* is False (default), movies whose output files already exist are
    skipped so a crashed run can be resumed without re-parsing from scratch.

    Output filename: <movie_title>__<character>.txt

    Returns {key: output_path} where key = '<movie>__<character>'.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required: pip install pandas pyarrow") from e

    parquet_path = Path(parquet_path)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build index of already-extracted files for fast lookup
    existing_files: set[str] = (
        set() if force
        else {p.stem for p in output_dir.glob("*.txt")}
    )

    df = pd.read_parquet(parquet_path, columns=["Movie", "Script"])
    total = len(df)

    output_paths: dict[str, Path] = {}
    skipped_small   = 0
    skipped_existing = 0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        movie_title = str(row["Movie"])
        script      = str(row["Script"]) if row["Script"] else ""
        if not script.strip():
            continue

        movie_slug    = _sanitise(movie_title)
        char_dialogue = _parse_script(script)

        for character, lines in char_dialogue.items():
            char_slug = _sanitise(character)
            key       = f"{movie_slug}__{char_slug}"
            out_path  = output_dir / f"{key}.txt"

            text = "\n".join(lines)
            if min_words is not None and _word_count(text) < min_words:
                if out_path.exists():
                    out_path.unlink()
                    existing_files.discard(key)
                skipped_small += 1
                continue

            if not force and key in existing_files:
                output_paths[key] = out_path
                skipped_existing += 1
                continue

            out_path.write_text(text, encoding="utf-8")
            output_paths[key] = out_path
            existing_files.add(key)

        if i % 100 == 0 or i == total:
            print(f"  [imsdb] {i}/{total} movies processed …", flush=True)

    print(
        f"[imsdb] {len(output_paths)} character files in {output_dir} "
        f"({skipped_existing} already existed, {skipped_small} too small)"
    )
    return output_paths
