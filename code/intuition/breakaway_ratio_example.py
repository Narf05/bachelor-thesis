"""
Create a Breakaway frequency-ratio illustration.

The script reads one real corpus item from data/processed.db, plots observed
frequency ratios f_{j+1}/f_j, and overlays three rational fits using increasing
numbers of ratios. The fitted value at j = 0 gives beta_0, hence:
    f0_hat = f1 / beta_0
    S_hat  = S_obs + f0_hat

Run from the repository root:
    python code/intuition/breakaway_ratio_example.py

Output:
    code/intuition/breakaway_ratio_example.png

To change the example, edit SELECTED_CORPUS_NAME or the fallback settings below.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "processed.db"
OUT_PATH = Path(__file__).with_name("breakaway_ratio_example.png")

# Set this to a cache name such as "shakespeare__hamlet" to force one example.
# Leave as None to choose a medium-sized corpus item automatically.
SELECTED_CORPUS_NAME: str | None = None

# Fallback selection: prefer a medium-sized non-Shakespeare item with enough
# consecutive nonzero ratios to support all requested fits.
FALLBACK_MIN_TOKENS = 1_500
FALLBACK_MAX_TOKENS = 8_000
FALLBACK_EXCLUDE_SOURCES = {"shakespeare", "shakespeare_legacy"}

# Increasing numbers of ratios to fit. A fit with J uses observed ratios
# f_2/f_1, ..., f_{J+1}/f_J.
FIT_J_VALUES = (6, 12, 20)

# Pedagogic filters for the automatically selected example. Near-zero beta0
# values imply enormous f0 estimates and make the intercept visually collapse
# onto the x-axis, which is useful as a failure case but not as an intuition
# figure for the method.
MIN_BETA0_FOR_FIGURE = 0.03
MAX_F0_TO_SOBS_RATIO = 8.0


def rational_curve(j: np.ndarray, beta0: float, beta1: float, beta2: float, beta3: float):
    """Breakaway-style rational frequency-ratio curve."""
    return (beta0 + beta1 * j) / (1.0 + beta2 * j + beta3 * j**2)


def load_corpora() -> pd.DataFrame:
    """Load corpus metadata from processed.db."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT id, name, corpus_source, speaker_id, n_tokens, s_obs
            FROM corpora
            WHERE n_tokens IS NOT NULL
              AND s_obs IS NOT NULL
            """,
            conn,
        )
    df["corpus_source"] = df["corpus_source"].fillna("shakespeare_legacy")
    return df


def load_word_counts(corpus_id: int) -> np.ndarray:
    """Return observed word abundances for one cached corpus."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT count FROM word_counts WHERE corpus_id = ?",
            (int(corpus_id),),
        ).fetchall()

    counts = np.array([count for (count,) in rows], dtype=int)
    if counts.size == 0:
        raise RuntimeError(f"No word counts found for corpus_id={corpus_id}.")
    return counts


def frequency_counts(counts: np.ndarray) -> dict[int, int]:
    """Return {k: f_k} from observed word-count abundances."""
    values, freqs = np.unique(counts, return_counts=True)
    return {int(k): int(v) for k, v in zip(values, freqs)}


def consecutive_ratio_data(freq: dict[int, int], max_j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build consecutive ratios up to max_j.

    Returns j, ratio, and weights f_j for all consecutive nonzero ratios.
    Stops at the first missing denominator or numerator, matching the simple
    Breakaway implementation in code/estimators/breakaway.py.
    """
    js: list[float] = []
    ratios: list[float] = []
    weights: list[float] = []

    for j in range(1, max_j + 1):
        fj = freq.get(j, 0)
        fj1 = freq.get(j + 1, 0)
        if fj == 0 or fj1 == 0:
            break
        js.append(float(j))
        ratios.append(fj1 / fj)
        weights.append(float(fj))

    return np.array(js), np.array(ratios), np.array(weights)


def has_enough_ratios(corpus_id: int, required_j: int) -> bool:
    """Check whether a corpus supports the largest requested fit."""
    counts = load_word_counts(corpus_id)
    freq = frequency_counts(counts)
    js, _, _ = consecutive_ratio_data(freq, required_j)
    return len(js) >= required_j


def choose_corpus(corpora: pd.DataFrame) -> pd.Series:
    """Choose the configured corpus or a suitable medium-sized fallback."""
    if SELECTED_CORPUS_NAME is not None:
        selected = corpora.loc[corpora["name"] == SELECTED_CORPUS_NAME]
        if selected.empty:
            raise RuntimeError(f"Corpus not found: {SELECTED_CORPUS_NAME}")
        return selected.iloc[0]

    candidates = corpora.loc[
        (corpora["n_tokens"] >= FALLBACK_MIN_TOKENS)
        & (corpora["n_tokens"] <= FALLBACK_MAX_TOKENS)
        & (~corpora["corpus_source"].isin(FALLBACK_EXCLUDE_SOURCES))
    ].copy()

    valid_indices = [
        idx for idx, row in candidates.iterrows() if supports_requested_fits(int(row["id"]))
    ]
    if not valid_indices:
        raise RuntimeError(
            "No medium-sized fallback corpus has stable positive Breakaway fits. "
            "Try lowering FIT_J_VALUES or widening the token range."
        )

    candidates = candidates.loc[valid_indices].copy()
    target = (FALLBACK_MIN_TOKENS + FALLBACK_MAX_TOKENS) / 2
    distances = (candidates["n_tokens"] - target).abs()
    return candidates.loc[distances.sort_values().index[0]]


def fit_ratio_curve(
    js: np.ndarray,
    ratios: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit the rational ratio curve and return beta, f0_hat, S_hat later."""
    if len(js) < 4:
        raise RuntimeError("Need at least four ratios for the four-parameter fit.")

    p0 = [max(ratios[0], 1e-4), 0.0, 0.0, 0.0]
    beta, _ = curve_fit(
        rational_curve,
        js,
        ratios,
        p0=p0,
        sigma=1.0 / np.maximum(weights, 1.0),
        absolute_sigma=True,
        bounds=([1e-10, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=20_000,
    )

    beta0 = float(beta[0])
    if beta0 <= 0:
        raise RuntimeError(f"Fitted beta0 is non-positive: {beta0:.4g}")
    return beta, beta0, float(ratios[0])


def supports_requested_fits(corpus_id: int) -> bool:
    """True if all requested ratio fits converge with positive beta0."""
    counts = load_word_counts(corpus_id)
    freq = frequency_counts(counts)
    js_all, ratios_all, weights_all = consecutive_ratio_data(freq, max(FIT_J_VALUES))
    if len(js_all) < max(FIT_J_VALUES):
        return False

    f1 = freq.get(1, 0)
    s_obs = int(counts.size)
    min_beta0 = max(MIN_BETA0_FOR_FIGURE, f1 / (MAX_F0_TO_SOBS_RATIO * s_obs))

    for j_max in FIT_J_VALUES:
        mask = js_all <= j_max
        try:
            beta, beta0, _ = fit_ratio_curve(
                js_all[mask],
                ratios_all[mask],
                weights_all[mask],
            )
        except Exception:
            return False

        y_grid = rational_curve(np.linspace(0, j_max, 100), *beta)
        if not np.all(np.isfinite(y_grid)):
            return False
        if np.nanmax(np.abs(y_grid)) > 10:
            return False
        if beta0 < min_beta0:
            return False

    return True


def make_label(j_max: int, beta0: float, f1: int, s_obs: int) -> str:
    """Legend label with implied unseen and total richness estimates."""
    f0_hat = f1 / beta0
    s_hat = s_obs + f0_hat
    return rf"$J={j_max}$: $\hat f_0={f0_hat:,.0f}$, $\hat S={s_hat:,.0f}$"


def display_title(row: pd.Series) -> str:
    """Create an anonymous but informative title."""
    source = row["corpus_source"]
    if source == "imsdb":
        return "IMSDb character example"
    if source == "bnc":
        return "BNC speaker example"
    if source == "sbcorpus":
        return "SBCorpus speaker example"
    if row["name"] == "shakespeare__hamlet":
        return "Hamlet"
    return str(row["name"])


def make_figure(row: pd.Series, counts: np.ndarray) -> None:
    """Create and save the Breakaway ratio illustration."""
    freq = frequency_counts(counts)
    max_j = max(FIT_J_VALUES)
    js_all, ratios_all, weights_all = consecutive_ratio_data(freq, max_j)

    if len(js_all) < max_j:
        raise RuntimeError(
            f"Selected corpus has only {len(js_all)} consecutive ratios; "
            f"need {max_j}."
        )

    f1 = freq.get(1, 0)
    s_obs = int(counts.size)
    n_tokens = int(counts.sum())

    colors = ["#6A8FBF", "#D08A3E", "#4F8A5B"]
    x_grid = np.linspace(0, max_j, 300)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    ax.scatter(
        js_all,
        ratios_all,
        color="#222222",
        s=34,
        zorder=5,
        label=r"Observed ratios $f_{j+1}/f_j$",
    )

    for j_max, color in zip(FIT_J_VALUES, colors):
        mask = js_all <= j_max
        beta, beta0, _ = fit_ratio_curve(
            js_all[mask],
            ratios_all[mask],
            weights_all[mask],
        )
        y_grid = rational_curve(x_grid, *beta)
        ax.plot(
            x_grid,
            y_grid,
            color=color,
            linewidth=2.0,
            label=make_label(j_max, beta0, f1, s_obs),
        )
        ax.scatter(
            [0],
            [beta0],
            color=color,
            s=46,
            zorder=6,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.axvline(0, color="#777777", linewidth=1.0, alpha=0.35)
    ax.set_xlabel(r"Frequency-ratio index $j$", fontsize=11)
    ax.set_ylabel(r"Observed ratio $f_{j+1}/f_j$", fontsize=11)
    ax.set_title(display_title(row), fontsize=12, weight="bold")

    subtitle = f"n = {n_tokens:,} tokens, S_obs = {s_obs:,}, f1 = {f1:,}"
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

    ax.set_xlim(-0.35, max_j + 0.5)
    y_max = max(float(np.nanmax(ratios_all[:max_j])), ax.get_ylim()[1])
    ax.set_ylim(0, y_max * 1.15)
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    corpora = load_corpora()
    row = choose_corpus(corpora)
    counts = load_word_counts(int(row["id"]))
    make_figure(row, counts)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
