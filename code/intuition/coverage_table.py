"""
Create an intuition table for sample coverage.

The script reads the processed SQLite cache and writes a compact table image
showing representative examples of Turing sample coverage.

Run from the repository root:
    python code/intuition/coverage_table.py

Output:
    code/intuition/sample_coverage_examples.png

To add a new data source, add its corpus_source name to CORPUS_SPECS below.
The script will automatically choose low-, median-, and high-coverage examples
from that source if enough cached entries exist.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "processed.db"
OUT_PATH = Path(__file__).with_name("sample_coverage_examples.png")


@dataclass(frozen=True)
class CorpusSpec:
    """Selection rules for one corpus/source in the example table."""

    source: str | None
    label: str
    min_tokens: int = 400
    include_quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    max_rows: int = 3


# Add future data sources here, for example:
#     CorpusSpec("coca", "COCA spoken", min_tokens=400),
CORPUS_SPECS = [
    CorpusSpec("imsdb", "IMSDb character"),
    CorpusSpec("bnc", "BNC speaker"),
    CorpusSpec("sbcorpus", "SBCorpus speaker"),
]

# Manually pin especially interpretable examples. These are added when present.
PINNED_NAMES = {
    "shakespeare__hamlet": "Hamlet",
    "shakespeare_corpus": "Full Shakespeare corpus",
    "full_corpus": "Full Shakespeare corpus",
}


def load_corpora() -> pd.DataFrame:
    """Load cached corpus metadata from processed.db."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                c.name,
                c.corpus_source,
                c.speaker_id,
                c.n_tokens,
                c.s_obs,
                c.coverage_turing,
                COALESCE(f1.f_k, 0) AS f1,
                COALESCE(f2.f_k, 0) AS f2
            FROM corpora AS c
            LEFT JOIN freq_counts AS f1
              ON c.id = f1.corpus_id AND f1.k = 1
            LEFT JOIN freq_counts AS f2
              ON c.id = f2.corpus_id AND f2.k = 2
            WHERE c.n_tokens IS NOT NULL
              AND c.s_obs IS NOT NULL
              AND c.coverage_turing IS NOT NULL
            """,
            conn,
        )

    # Older untagged rows in this project are Shakespeare cache entries.
    df["corpus_source"] = df["corpus_source"].fillna("shakespeare_legacy")
    return df


def example_label(row: pd.Series, source_label: str | None = None) -> str:
    """Return a compact human-readable row label."""
    if row["name"] in PINNED_NAMES:
        return PINNED_NAMES[row["name"]]

    speaker = row["speaker_id"]
    if pd.isna(speaker) or not str(speaker).strip():
        speaker = row["name"]

    prefix = source_label or row["corpus_source"]
    return f"{prefix}: {speaker}"


def anonymize_non_shakespeare_labels(examples: pd.DataFrame) -> pd.DataFrame:
    """Replace non-Shakespeare labels by corpus-level numbered examples."""
    examples = examples.copy()
    counters: dict[str, int] = {}

    for idx, row in examples.iterrows():
        source = row["corpus_source"]
        if source in {"shakespeare", "shakespeare_legacy"}:
            continue

        label_root = next(
            (spec.label for spec in CORPUS_SPECS if spec.source == source),
            str(source),
        )
        counters[label_root] = counters.get(label_root, 0) + 1
        examples.at[idx, "example"] = f"{label_root} {counters[label_root]}"

    return examples


def select_quantile_examples(df: pd.DataFrame, spec: CorpusSpec) -> pd.DataFrame:
    """Select examples nearest requested coverage quantiles for one corpus."""
    subset = df.loc[
        (df["corpus_source"] == spec.source) & (df["n_tokens"] >= spec.min_tokens)
    ].copy()
    if subset.empty:
        return subset

    chosen_indices: list[int] = []
    for q in spec.include_quantiles:
        target = subset["coverage_turing"].quantile(q)
        distances = (subset["coverage_turing"] - target).abs()
        idx = distances.sort_values().index[0]
        if idx not in chosen_indices:
            chosen_indices.append(idx)
        if len(chosen_indices) >= spec.max_rows:
            break

    selected = subset.loc[chosen_indices].copy()
    selected["example"] = [
        example_label(row, spec.label) for _, row in selected.iterrows()
    ]
    return selected


def select_pinned_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Select manually pinned examples when they exist in the database."""
    pinned = df.loc[df["name"].isin(PINNED_NAMES)].copy()
    if pinned.empty:
        return pinned

    # Prefer the tagged full Shakespeare entry over the older untagged alias.
    if "shakespeare_corpus" in set(pinned["name"]):
        pinned = pinned.loc[pinned["name"] != "full_corpus"]

    pinned["example"] = [example_label(row) for _, row in pinned.iterrows()]
    return pinned


def build_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Build the final example table from pinned and corpus-level selections."""
    selected = [select_pinned_examples(df)]
    selected.extend(select_quantile_examples(df, spec) for spec in CORPUS_SPECS)

    examples = pd.concat([part for part in selected if not part.empty], ignore_index=True)
    examples = examples.drop_duplicates(subset=["name"])
    examples = examples.sort_values(["coverage_turing", "n_tokens"])
    return anonymize_non_shakespeare_labels(examples)


def format_table(examples: pd.DataFrame) -> pd.DataFrame:
    """Format values for display in the thesis figure."""
    table = examples[
        [
            "example",
            "n_tokens",
            "s_obs",
            "f1",
            "f2",
            "coverage_turing",
        ]
    ].copy()
    table.columns = [
        "Example",
        r"Tokens $n$",
        r"Observed types $S_{\mathrm{obs}}$",
        r"$f_1$",
        r"$f_2$",
        r"Turing $\hat{C}$",
    ]

    table[r"Tokens $n$"] = table[r"Tokens $n$"].map(lambda x: f"{int(x):,}")
    table[r"Observed types $S_{\mathrm{obs}}$"] = table[
        r"Observed types $S_{\mathrm{obs}}$"
    ].map(lambda x: f"{int(x):,}")
    table[r"$f_1$"] = table[r"$f_1$"].map(lambda x: f"{int(x):,}")
    table[r"$f_2$"] = table[r"$f_2$"].map(lambda x: f"{int(x):,}")
    table[r"Turing $\hat{C}$"] = table[r"Turing $\hat{C}$"].map(
        lambda x: f"{x:.4f}"
    )
    return table


def save_table_image(table: pd.DataFrame) -> None:
    """Save a publication-friendly table as a PNG image."""
    row_count = len(table)
    fig_height = max(2.2, 0.42 * row_count + 1.1)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis("off")

    mpl_table = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.36, 0.14, 0.22, 0.10, 0.10, 0.14],
    )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    mpl_table.scale(1.0, 1.35)

    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#e9eef3")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
        else:
            cell.set_facecolor("white")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")

    ax.set_title(
        "Examples of Sample Coverage in the Thesis Corpora",
        fontsize=12,
        weight="bold",
        pad=12,
    )
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    corpora = load_corpora()
    examples = build_examples(corpora)
    if examples.empty:
        raise RuntimeError("No suitable coverage examples found in processed.db.")

    table = format_table(examples)
    save_table_image(table)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
