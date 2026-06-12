"""
CLAPI extractor — Claire-Dialogue-French-0.1 (Hugging Face).

Dataset: OpenLLM-France/Claire-Dialogue-French-0.1
Each row has a `text` field containing one conversation.
Each line in a conversation is one speech turn:

    [SpeakerName:] utterance text
    [speaker001:] utterance text   <- anonymous speaker

Named speakers (e.g. [Paul:], [François Mitterrand:]) are aggregated
globally — all their utterances across the corpus go into one file.

Anonymous speakers (matching speakerNNN) are scoped per conversation to
avoid mixing different real people under the same label:
    conv_00001__speaker001.txt

Special markers stripped from utterances:
    [PII], [NOISE], [LAUGHTER], and any other [...] annotations.

Usage
-----
    from preprocessing.extractors.clapi import extract
    extract("data/speakers/clapi")
"""

import re
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError("datasets is required: pip install datasets") from e


_TURN_RE     = re.compile(r"^\[(.+?):\]\s*(.*)")
_ANON_RE     = re.compile(r"^speaker\d+$", re.IGNORECASE)
_ANNOT_RE    = re.compile(r"\[[^\]]*\]")
_MULTI_SPACE = re.compile(r"\s{2,}")


def _sanitise(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _word_count(text: str) -> int:
    return len(text.split())


def _clean(text: str) -> str:
    text = _ANNOT_RE.sub(" ", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def _parse_conversation(text: str) -> list[tuple[str, str]]:
    """Return [(speaker_id, cleaned_utterance), ...] for one conversation."""
    turns = []
    for line in text.splitlines():
        m = _TURN_RE.match(line.strip())
        if m:
            speaker   = m.group(1).strip()
            utterance = _clean(m.group(2))
            if utterance:
                turns.append((speaker, utterance))
    return turns


def extract(
    output_dir: str | Path,
    split: str = "train",
    min_words: int | None = 400,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download Claire-Dialogue-French-0.1 and write one .txt per speaker.

    Named speakers are merged across all conversations.
    Anonymous speakers (speakerNNN) are scoped per conversation.

    Returns {key: output_path}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing: set[str] = (
        set() if force
        else {p.stem for p in output_dir.glob("*.txt")}
    )

    print(f"[clapi] loading split='{split}' from Hugging Face …", flush=True)
    ds = load_dataset("OpenLLM-France/Claire-Dialogue-French-0.1", split=split)

    named_utts: dict[str, list[str]] = {}
    anon_utts:  dict[str, list[str]] = {}

    total = len(ds)
    for i, row in enumerate(ds):
        text = row.get("text") or row.get("dialogue") or ""
        if not text.strip():
            continue

        conv_key = f"conv_{i:05d}"
        for speaker, utterance in _parse_conversation(text):
            if _ANON_RE.match(speaker):
                key = f"{conv_key}__{speaker.lower()}"
                anon_utts.setdefault(key, []).append(utterance)
            else:
                key = _sanitise(speaker)
                named_utts.setdefault(key, []).append(utterance)

        if (i + 1) % 2000 == 0 or (i + 1) == total:
            print(f"  [clapi] {i + 1}/{total} conversations parsed …", flush=True)

    output_paths:  dict[str, Path] = {}
    skipped_small    = 0
    skipped_existing = 0

    def _write(key: str, utterances: list[str]) -> None:
        nonlocal skipped_small, skipped_existing
        out_path = output_dir / f"{key}.txt"
        text = "\n".join(utterances)
        if min_words is not None and _word_count(text) < min_words:
            if out_path.exists():
                out_path.unlink()
            skipped_small += 1
            return
        if not force and key in existing:
            output_paths[key] = out_path
            skipped_existing += 1
            return
        out_path.write_text(text, encoding="utf-8")
        output_paths[key] = out_path

    for key, utterances in sorted(named_utts.items()):
        _write(key, utterances)
    for key, utterances in sorted(anon_utts.items()):
        _write(key, utterances)

    print(
        f"[clapi] {len(output_paths)} speaker files in {output_dir} "
        f"({skipped_existing} already existed, {skipped_small} too small)"
    )
    return output_paths
