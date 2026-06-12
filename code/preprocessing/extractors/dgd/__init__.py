"""
DGD extractor — Datenbank für Gesprochenes Deutsch (Zwirner corpus).

Each .fln file is XML with word-by-word speech contributions per speaker:

    <contribution speaker-reference="S1" ...>
        <w pos="ADV" lemma="so">So</w>
        <w pos="ADV" lemma="nun">nun</w>
        <p>.</p>
    </contribution>

Speaker IDs (S1, S2, ...) restart from S1 in every file, so output files
are keyed as <file_stem>__<speaker_id> to keep speakers distinct:
    zw___e_04704_se_01_t_01__s1.txt

Usage
-----
    from preprocessing.extractors.dgd import extract
    extract("~/Downloads/DGD_ZW/fln_files", "data/speakers/dgd")
"""

import re
from pathlib import Path
from xml.etree import ElementTree as ET


_MULTI_SPACE = re.compile(r"\s{2,}")


def _sanitise(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_").lower()


def _word_count(text: str) -> int:
    return len(text.split())


def _contribution_text(contribution: ET.Element) -> str:
    """Reconstruct plain text from <w> and <p> children."""
    tokens: list[tuple[str, str]] = []
    for elem in contribution:
        tag = elem.tag.rpartition("}")[-1]
        if tag == "w":
            text = (elem.text or "").strip()
            if text:
                tokens.append(("word", text))
        elif tag == "p":
            text = (elem.text or "").strip()
            if text:
                tokens.append(("punct", text))

    if not tokens:
        return ""

    parts = [tokens[0][1]]
    for typ, text in tokens[1:]:
        parts.append(text if typ == "punct" else " " + text)

    return _MULTI_SPACE.sub(" ", "".join(parts)).strip()


def _parse_fln(filepath: Path) -> dict[str, list[str]]:
    """Parse one .fln file. Returns {speaker_id: [utterance, ...]}."""
    try:
        tree = ET.parse(filepath)
    except ET.ParseError as exc:
        print(f"  [dgd] Warning: skipping {filepath.name} — {exc}")
        return {}

    root = tree.getroot()
    speaker_utterances: dict[str, list[str]] = {}

    for contribution in root.iter("contribution"):
        speaker = (contribution.get("speaker-reference") or "").strip()
        if not speaker:
            continue
        utterance = _contribution_text(contribution)
        if utterance:
            speaker_utterances.setdefault(speaker, []).append(utterance)

    return speaker_utterances


def extract(
    input_dir: str | Path,
    output_dir: str | Path,
    min_words: int | None = 400,
    force: bool = False,
) -> dict[str, Path]:
    """
    Parse all .fln files in input_dir and write one .txt per speaker per file.

    Output filename: <file_stem>__<speaker_id>.txt

    Returns {key: output_path}.
    """
    input_dir  = Path(input_dir).expanduser()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing: set[str] = (
        set() if force
        else {p.stem for p in output_dir.glob("*.txt")}
    )

    fln_files = sorted(input_dir.glob("*.fln"))
    if not fln_files:
        print(f"[dgd] No .fln files found in {input_dir}")
        return {}

    output_paths:   dict[str, Path] = {}
    skipped_small    = 0
    skipped_existing = 0
    total = len(fln_files)

    for i, fln_file in enumerate(fln_files, 1):
        file_slug = _sanitise(fln_file.stem)
        for speaker_id, utterances in _parse_fln(fln_file).items():
            key      = f"{file_slug}__{speaker_id.lower()}"
            out_path = output_dir / f"{key}.txt"
            text     = "\n".join(utterances)

            if min_words is not None and _word_count(text) < min_words:
                if out_path.exists():
                    out_path.unlink()
                skipped_small += 1
                continue

            if not force and key in existing:
                output_paths[key] = out_path
                skipped_existing += 1
                continue

            out_path.write_text(text, encoding="utf-8")
            output_paths[key] = out_path

        if i % 50 == 0 or i == total:
            print(f"  [dgd] {i}/{total} files processed …", flush=True)

    print(
        f"[dgd] {len(output_paths)} speaker files in {output_dir} "
        f"({skipped_existing} already existed, {skipped_small} too small)"
    )
    return output_paths
