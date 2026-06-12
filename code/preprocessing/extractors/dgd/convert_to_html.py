"""
Convert DGD .fln transcript files to HTML.

.fln files are XML-based transcriptions from the Datenbank für Gesprochenes
Deutsch (DGD).  Each file holds one recording with word-by-word contributions
per speaker:

    <contribution speaker-reference="S1">
        <w pos="ADV" lemma="so">So</w>
        <w pos="ADV" lemma="nun">nun</w>
        <p>.</p>
    </contribution>

Output: one .html file per .fln file — a readable table with one row per
speech turn.

Usage
-----
    python code/preprocessing/extractors/dgd/convert_to_html.py
    python code/preprocessing/extractors/dgd/convert_to_html.py \
        --source ~/Downloads/DGD_ZW/fln_files \
        --output data/dgd/html
"""

from __future__ import annotations

import argparse
import html as html_mod
from pathlib import Path
from xml.etree import ElementTree as ET

DEFAULT_SOURCE = Path.home() / "Downloads" / "DGD_PF" / "fln_files"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[4] / "data" / "dgd" / "html"


def _build_utterance(contribution: ET.Element) -> str:
    """Reconstruct plain text from <w> and <p> children of a contribution."""
    tokens: list[tuple[str, str]] = []
    for elem in contribution:
        tag = elem.tag.rpartition("}")[-1]   # strip namespace prefix if any
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

    return "".join(parts).strip()


def fln_to_html(fln_path: Path) -> str:
    """Return an HTML string for one .fln file."""
    try:
        tree = ET.parse(fln_path)
    except ET.ParseError as exc:
        return f"<p>Parse error in {html_mod.escape(fln_path.name)}: {exc}</p>"

    root = tree.getroot()
    rows: list[str] = []

    for contribution in root.iter("contribution"):
        speaker = (contribution.get("speaker-reference") or "").strip()
        if not speaker:
            continue
        utterance = _build_utterance(contribution)
        if utterance:
            rows.append(
                f"  <tr>\n"
                f"    <td class=\"sp\">{html_mod.escape(speaker)}</td>\n"
                f"    <td>{html_mod.escape(utterance)}</td>\n"
                f"  </tr>"
            )

    title = html_mod.escape(fln_path.stem)
    table_rows = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    body  {{ font-family: sans-serif; font-size: 14px; max-width: 900px; margin: 2em auto; }}
    h2   {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td   {{ padding: 4px 10px; vertical-align: top; border-bottom: 1px solid #f0f0f0; }}
    .sp  {{ font-weight: bold; color: #555; width: 4em; white-space: nowrap; }}
  </style>
</head>
<body>
<h2>{title}</h2>
<table>
{table_rows}
</table>
</body>
</html>"""


def convert_all(
    source_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> int:
    """Convert all .fln files in source_dir to HTML in output_dir."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for fln_file in sorted(source_dir.glob("*.fln")):
        out_path = output_dir / (fln_file.stem + ".html")
        if not force and out_path.exists():
            continue
        html_content = fln_to_html(fln_file)
        out_path.write_text(html_content, encoding="utf-8")
        print(f"  {fln_file.name} -> {out_path.name}")
        converted += 1

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DGD .fln transcript files to HTML."
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help=f"Folder containing .fln files. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Folder for .html output. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert files that already have an HTML output.",
    )
    args = parser.parse_args()

    print(f"[dgd] Converting .fln → HTML\n  source: {args.source}\n  output: {args.output}")
    count = convert_all(args.source, args.output, force=args.force)
    print(f"[dgd] Converted {count} file(s).")


if __name__ == "__main__":
    main()
