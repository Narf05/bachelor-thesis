"""
Create intuition figures for sample-size rarefaction and extrapolation.

The script reads one real corpus item from data/processed.db and saves a
thesis-ready curve with:
    - solid rarefaction curve for m < n
    - observed point at (n, S_obs)
    - dashed extrapolation curve for n < m <= 2n
    - faint vertical reference line at n

Run from the repository root:
    python code/intuition/rarefaction_extrapolation.py

Outputs:
    code/intuition/rarefaction_extrapolation_hamlet.png
    code/intuition/rarefaction_extrapolation_medium_example.png

To change the examples, edit EXAMPLES below.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "processed.db"
OUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExampleSpec:
    """Configuration for one rarefaction/extrapolation figure."""

    output_stem: str
    title: str
    preferred_name: str | None = None
    fallback_source: str | None = None
    fallback_min_tokens: int = 1_500
    fallback_max_tokens: int = 5_000


EXAMPLES = [
    ExampleSpec(
        output_stem="rarefaction_extrapolation_hamlet",
        title="Hamlet",
        preferred_name="shakespeare__hamlet",
        fallback_source="shakespeare",
        fallback_min_tokens=20_000,
        fallback_max_tokens=35_000,
    ),
    ExampleSpec(
        output_stem="rarefaction_extrapolation_medium_example",
        title="Medium-sized speaker example",
        preferred_name=None,
        fallback_source=None,
        fallback_min_tokens=1_500,
        fallback_max_tokens=5_000,
    ),
]


def load_corpora() -> pd.DataFrame:
    """Load corpus metadata from the processed database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT id, name, corpus_source, speaker_id, n_tokens, s_obs
            FROM corpora
            WHERE n_tokens IS NOT NULL
              AND s_obs IS NOT NULL
            """,
            conn,
        )


def choose_corpus(corpora: pd.DataFrame, spec: ExampleSpec) -> pd.Series:
    """Choose the configured corpus, falling back to a medium-sized entry."""
    if spec.preferred_name is not None:
        preferred = corpora.loc[corpora["name"] == spec.preferred_name]
        if not preferred.empty:
            return preferred.iloc[0]

    candidates = corpora.loc[
        (corpora["n_tokens"] >= spec.fallback_min_tokens)
        & (corpora["n_tokens"] <= spec.fallback_max_tokens)
    ].copy()

    if spec.fallback_source is not None:
        candidates = candidates.loc[candidates["corpus_source"] == spec.fallback_source]

    if candidates.empty:
        raise RuntimeError(f"No fallback corpus found for {spec.output_stem}.")

    target = (spec.fallback_min_tokens + spec.fallback_max_tokens) / 2
    distances = (candidates["n_tokens"] - target).abs()
    return candidates.loc[distances.sort_values().index[0]]


def load_word_counts(corpus_id: int) -> np.ndarray:
    """Return the observed species abundances for one cached corpus."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT count
            FROM word_counts
            WHERE corpus_id = ?
            """,
            (int(corpus_id),),
        ).fetchall()

    counts = np.array([count for (count,) in rows], dtype=int)
    if counts.size == 0:
        raise RuntimeError(f"No word counts found for corpus_id={corpus_id}.")
    return counts


def log_choose(n: np.ndarray | int, k: np.ndarray | int) -> np.ndarray:
    """Compute log binomial coefficients using lgamma, scalar or vectorized."""
    n_arr = np.asarray(n, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    lgamma = np.vectorize(math.lgamma)
    return lgamma(n_arr + 1) - lgamma(k_arr + 1) - lgamma(n_arr - k_arr + 1)


def rarefaction_curve(counts: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    """
    Expected observed richness for m <= n.

    For a type observed x times in n tokens, the probability that it appears
    at least once in a rarefied sample of size m is:
        1 - C(n - x, m) / C(n, m).
    """
    n = int(counts.sum())
    richness = []

    for m in m_values:
        m = int(m)
        absent = np.zeros_like(counts, dtype=float)
        possible = n - counts >= m
        if possible.any():
            absent[possible] = np.exp(
                log_choose(n - counts[possible], m) - log_choose(n, m)
            )
        richness.append(float(np.sum(1.0 - absent)))

    return np.array(richness)


def frequency_counts(counts: np.ndarray) -> dict[int, int]:
    """Return {k: f_k} from observed word-count abundances."""
    values, freqs = np.unique(counts, return_counts=True)
    return {int(k): int(v) for k, v in zip(values, freqs)}


def chao1_unseen(freq: dict[int, int], n: int) -> float:
    """Finite-sample Chao1 estimate of f_0."""
    f1 = freq.get(1, 0)
    f2 = freq.get(2, 0)
    correction = (n - 1) / n
    if f2 > 0:
        return correction * f1**2 / (2 * f2)
    return correction * f1 * (f1 - 1) / 2


def extrapolation_curve(counts: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    """
    Richness extrapolation for n <= m <= 2n.

    This uses the common Chao-style abundance extrapolation:
        S(m) = S_obs + f0_hat * (1 - (1 - f1/(n*f0_hat)) ** (m - n)).
    The curve is only meant as an intuition figure, so the script limits it
    to doubling the original sample size.
    """
    n = int(counts.sum())
    s_obs = int(counts.size)
    freq = frequency_counts(counts)
    f1 = freq.get(1, 0)
    f0_hat = chao1_unseen(freq, n)

    if f1 == 0 or f0_hat <= 0:
        return np.full_like(m_values, fill_value=s_obs, dtype=float)

    extra = m_values - n
    detection_rate = f1 / (n * f0_hat)
    return s_obs + f0_hat * (1.0 - (1.0 - detection_rate) ** extra)


def make_figure(row: pd.Series, counts: np.ndarray, spec: ExampleSpec) -> None:
    """Create and save one rarefaction/extrapolation figure."""
    n = int(counts.sum())
    s_obs = int(counts.size)

    m_rare = np.linspace(max(1, n // 50), n, 120, dtype=int)
    m_extra = np.linspace(n, 2 * n, 80, dtype=int)

    s_rare = rarefaction_curve(counts, m_rare)
    s_extra = extrapolation_curve(counts, m_extra)

    milestone_m = np.array(
        [n // 4, n // 2, 3 * n // 4, n, 3 * n // 2, 2 * n],
        dtype=int,
    )
    milestone_s = np.concatenate(
        [
            rarefaction_curve(counts, milestone_m[:4]),
            extrapolation_curve(counts, milestone_m[4:]),
        ]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        m_rare,
        s_rare,
        color="#2F6B9A",
        linewidth=2.2,
        label="Rarefaction",
    )
    ax.plot(
        m_extra,
        s_extra,
        color="#C65A1E",
        linewidth=2.2,
        linestyle="--",
        label="Extrapolation",
    )
    ax.scatter(
        [n],
        [s_obs],
        color="#222222",
        s=48,
        zorder=5,
        label=r"Observed $(n, S_{\mathrm{obs}})$",
    )
    ax.scatter(
        milestone_m[:3],
        milestone_s[:3],
        color="#2F6B9A",
        s=28,
        zorder=4,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.scatter(
        milestone_m[4:],
        milestone_s[4:],
        color="#C65A1E",
        s=28,
        zorder=4,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.axvline(n, color="#777777", linewidth=1.0, alpha=0.35)

    label_specs = [
        (milestone_m[0], milestone_s[0], r"$0.25n$", -8, 10),
        (milestone_m[1], milestone_s[1], r"$0.5n$", -5, 10),
        (milestone_m[2], milestone_s[2], r"$0.75n$", -5, 10),
        (n, s_obs, r"$n$", 6, -16),
        (milestone_m[4], milestone_s[4], r"$1.5n$", 5, 10),
        (milestone_m[5], milestone_s[5], r"$2n$", -18, 10),
    ]
    for x, y, label, dx, dy in label_specs:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color="#333333",
        )

    ax.set_xlabel(r"Sample size $m$", fontsize=11)
    ax.set_ylabel("Expected observed richness", fontsize=11)
    ax.set_title(spec.title, fontsize=12, weight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, axis="both", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_xlim(0, 2 * n)
    ax.set_ylim(0, max(s_extra.max(), s_obs) * 1.08)

    subtitle = f"n = {n:,} tokens, S_obs = {s_obs:,} observed types"
    ax.text(
        0.02,
        0.96,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )

    out_path = OUT_DIR / f"{spec.output_stem}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    corpora = load_corpora()
    for spec in EXAMPLES:
        row = choose_corpus(corpora, spec)
        counts = load_word_counts(int(row["id"]))
        make_figure(row, counts, spec)
        print(f"Wrote {OUT_DIR / (spec.output_stem + '.png')}")


if __name__ == "__main__":
    main()
