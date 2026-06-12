#!/usr/bin/env python3
"""
Generate all tables and figures for thesis Sections 6, 7, and 8.

Usage (from repo root or from results/):
    python results/generate_results.py

Outputs
-------
results/tables/
    shakespeare_estimator_table.csv
    language_corpus_overview.csv
    corpus_overview.csv
    per_language_estimator_summary.csv
    cross_language_estimator_summary.csv
    estimator_failure_summary.csv

results/figures/
    shakespeare_estimator_comparison.png
    tokens_by_language.png
    sobs_by_corpus.png
    coverage_by_language.png
    estimator_boxplots_by_language.png
    rarefaction_extrapolation_shakespeare.png
    rarefaction_extrapolation_by_language.png
    coverage_standardized_language_comparison.png
    estimator_correlation_heatmap.png
    estimator_ratio_to_chao1.png
"""

from __future__ import annotations

import sqlite3
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# --- Repo layout -------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

DB_PATH = ROOT / "data" / "processed.db"
TABLES  = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# --- Estimator imports --------------------------------------------------------

from estimators.classical import chao1, ichao1, ace, jackknife1, jackknife2
from estimators.breakaway import breakaway
from estimators.coverage  import coverage_turing, coverage_chao_jost

# --- Constants ----------------------------------------------------------------

CORPUS_LANGUAGE: dict[str, str] = {
    "shakespeare": "Shakespeare",
    "imsdb":       "English",
    "bnc":         "English",
    "sbcorpus":    "English",
    "clapi":       "French",
    "dgd":         "German",
}

CORPUS_DISPLAY: dict[str, str] = {
    "shakespeare": "Shakespeare",
    "imsdb":       "IMSDb",
    "bnc":         "BNC",
    "sbcorpus":    "SBCorpus",
    "clapi":       "CLAPI",
    "dgd":         "DGD",
}

SPOKEN_CORPORA = {"imsdb", "bnc", "sbcorpus", "clapi", "dgd"}
LANGUAGE_ORDER = ["English", "French", "German"]
ESTIMATOR_ORDER = ["Chao1", "iChao1", "ACE", "Jackknife1", "Jackknife2", "Breakaway"]

PALETTE_LANG = {
    "English":    "#2196F3",
    "French":     "#E91E63",
    "German":     "#4CAF50",
    "Shakespeare": "#FF9800",
}
PALETTE_EST = {
    est: c for est, c in zip(
        ESTIMATOR_ORDER,
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    )
}

# Thesis-quality plot defaults
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# --- Data loading -------------------------------------------------------------

def load_metadata() -> pd.DataFrame:
    """Return the corpora metadata table with language and spoken tags."""
    from db.cache import list_corpora
    df = list_corpora()
    df["language"]  = df["corpus_source"].map(CORPUS_LANGUAGE)
    df["is_spoken"] = df["corpus_source"].isin(SPOKEN_CORPORA)
    df["corpus_display"] = df["corpus_source"].map(CORPUS_DISPLAY)
    return df


def load_freq_counts_all(meta: pd.DataFrame) -> dict[str, dict[int, int]]:
    """Load stored freq_counts (k = 1..20) for every unit in *meta*."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT c.name, fc.k, fc.f_k
           FROM freq_counts fc
           JOIN corpora c ON fc.corpus_id = c.id
           ORDER BY c.name, fc.k"""
    ).fetchall()
    conn.close()

    fc_all: dict[str, dict[int, int]] = defaultdict(dict)
    for name, k, fk in rows:
        fc_all[name][k] = fk
    return dict(fc_all)


def load_word_counts_for_names(names: list[str]) -> dict[str, dict[str, int]]:
    """Load word->count dicts for the given unit names."""
    conn = sqlite3.connect(DB_PATH)
    result: dict[str, dict[str, int]] = {}
    for name in names:
        row = conn.execute(
            "SELECT id FROM corpora WHERE name = ?", (name,)
        ).fetchone()
        if row:
            wc = dict(conn.execute(
                "SELECT word, count FROM word_counts WHERE corpus_id = ?",
                (row[0],)
            ).fetchall())
            result[name] = wc
    conn.close()
    return result


def load_merged_word_counts(corpus_sources: list[str]) -> dict[str, int]:
    """
    Aggregate word counts across all units belonging to *corpus_sources*.

    Uses a single SQL GROUP BY query for efficiency.
    """
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(corpus_sources))
    rows = conn.execute(
        f"""SELECT w.word, SUM(w.count) as total
            FROM word_counts w
            JOIN corpora c ON w.corpus_id = c.id
            WHERE c.corpus_source IN ({placeholders})
            GROUP BY w.word""",
        corpus_sources,
    ).fetchall()
    conn.close()
    return dict(rows)


def wc_to_fc(wc: dict[str, int]) -> dict[int, int]:
    """Convert word->count mapping to frequency-count dict {k: f_k}."""
    fc: dict[int, int] = defaultdict(int)
    for count in wc.values():
        fc[count] += 1
    return dict(fc)


# --- Estimators --------------------------------------------------------------

def run_estimators(fc: dict[int, int]) -> dict[str, float | None]:
    """
    Run all estimators on *fc*.  Returns None for any that raise.
    """
    out: dict[str, float | None] = {}
    for label, fn in [
        ("Chao1",     chao1),
        ("iChao1",    ichao1),
        ("ACE",       ace),
        ("Jackknife1", jackknife1),
        ("Jackknife2", jackknife2),
    ]:
        try:
            out[label] = float(fn(fc))
        except Exception:
            out[label] = None

    try:
        br = breakaway(fc)
        out["Breakaway"] = float(br["S_hat"])
    except Exception:
        out["Breakaway"] = None

    return out


def estimator_dataframe(meta: pd.DataFrame, fc_all: dict[str, dict[int, int]]) -> pd.DataFrame:
    """
    Run all estimators for every unit in *meta*.

    Returns a long-format DataFrame with columns:
        name, corpus_source, language, n_tokens, s_obs,
        coverage_turing, estimator, S_hat, f0_hat, ratio
    """
    records = []
    for _, row in meta.iterrows():
        fc = fc_all.get(row["name"], {})
        if not fc:
            continue
        S_obs = row["s_obs"]
        n     = row["n_tokens"]
        f1    = fc.get(1, 0)
        f2    = fc.get(2, 0)
        cov   = row["coverage_turing"]

        for est_name, S_hat in run_estimators(fc).items():
            failed = S_hat is None or not np.isfinite(S_hat)
            if failed:
                S_hat = None
                f0_hat = None
                ratio  = None
            else:
                f0_hat = max(S_hat - S_obs, 0.0)
                ratio  = S_hat / S_obs if S_obs > 0 else None

            records.append({
                "name":         row["name"],
                "corpus_source": row["corpus_source"],
                "corpus_display": row["corpus_display"],
                "language":     row["language"],
                "is_spoken":    row["is_spoken"],
                "n_tokens":     n,
                "s_obs":        S_obs,
                "f1":           f1,
                "f2":           f2,
                "coverage":     cov,
                "estimator":    est_name,
                "S_hat":        S_hat,
                "f0_hat":       f0_hat,
                "ratio":        ratio,
                "failed":       failed,
            })

    return pd.DataFrame(records)


# --- Rarefaction / extrapolation ---------------------------------------------

def _log_binom_ratio(n: int, k: int, m_arr: np.ndarray) -> np.ndarray:
    """
    log( C(n-k, m) / C(n, m) ) for each m in m_arr.

    Returns -inf where C(n-k, m) = 0 (i.e., m > n-k).
    """
    m_arr = np.asarray(m_arr, dtype=float)
    result = np.full(len(m_arr), -np.inf)
    valid = m_arr <= (n - k)
    mv = m_arr[valid]
    if len(mv) == 0:
        return result
    log_r = (
        gammaln(n - k + 1) - gammaln(n - k - mv + 1)
        - gammaln(n + 1)   + gammaln(n - mv + 1)
    )
    result[valid] = log_r
    return result


def rarefaction_curve(
    wc: dict[str, int],
    n_points: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the rarefaction curve E[S(m)] for m in [1, n].

    Uses the exact hypergeometric formula:
        E[S(m)] = sum_k  f_k * (1 - C(n-k, m) / C(n, m))

    Returns (m_values, E_S_values).
    """
    fc = wc_to_fc(wc)
    n  = sum(wc.values())
    m_arr = np.unique(np.round(np.linspace(1, n, n_points)).astype(int))

    result = np.zeros(len(m_arr))
    for k, fk in fc.items():
        if fk == 0 or k == 0:
            continue
        log_r = _log_binom_ratio(n, k, m_arr.astype(float))
        ratio  = np.where(np.isfinite(log_r), np.exp(log_r), 0.0)
        result += fk * (1.0 - ratio)

    return m_arr.astype(float), result


def extrapolation_curve(
    fc: dict[int, int],
    n: int,
    extrap_factor: float = 2.0,
    n_points: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrapolate expected richness from n to extrap_factor * n.

    Uses the asymptotic approach:
        E[S(n+t)] = S_chao - (S_chao - S_obs) * r^t
    where  r = 1 - f1 / (n * S_chao)  and  S_chao = chao1(fc).

    Returns (m_values, E_S_values) with m starting at n.
    """
    S_obs = sum(fc.values())
    f1    = fc.get(1, 0)
    try:
        S_chao = chao1(fc)
    except Exception:
        return np.array([float(n)]), np.array([float(S_obs)])

    f0 = max(S_chao - S_obs, 0.0)
    if f0 == 0 or S_chao <= 0:
        t_arr = np.linspace(0, int(n * (extrap_factor - 1)), n_points)
        return (n + t_arr), np.full(n_points, float(S_obs))

    rate = 1.0 - f1 / (n * S_chao + 1e-12)
    rate = np.clip(rate, 0.0, 1.0)

    t_arr = np.linspace(0, int(n * (extrap_factor - 1)), n_points)
    E_S   = S_chao - f0 * rate ** t_arr
    return n + t_arr, E_S


# --- Helpers -----------------------------------------------------------------

def _save_fig(fig: plt.Figure, stem: str) -> None:
    path = FIGURES / f"{stem}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path.relative_to(ROOT)}")


def _save_csv(df: pd.DataFrame, stem: str) -> None:
    path = TABLES / f"{stem}.csv"
    df.to_csv(path, index=False)
    print(f"  saved -> {path.relative_to(ROOT)}")


def _fmt_int(x: float) -> str:
    return f"{int(round(x)):,}" if pd.notna(x) else "--"


# ===========================================================================
# Section 6.1 -- Shakespeare replication baseline
# ===========================================================================

def section_6_1_shakespeare(meta: pd.DataFrame, fc_all: dict[str, dict[int, int]]) -> None:
    print("\n--- Section 6.1 Shakespeare baseline ---")

    shk_meta = meta[meta["corpus_source"] == "shakespeare"].copy()
    if shk_meta.empty:
        print("  No Shakespeare data found in DB -- skipping 6.1.")
        return

    # --- Per-play estimator results ----------------------------------------
    # When individual play word counts are merged by summation, most words
    # appear in more than one play, collapsing f1 to 0.  Per-play analysis
    # preserves meaningful singleton/doubleton counts and is the correct
    # unit of analysis for estimator validation.

    play_rows = []
    for _, row in shk_meta.iterrows():
        fc = fc_all.get(row["name"], {})
        if not fc:
            continue
        S_obs_play = row["s_obs"]
        n_play     = row["n_tokens"]
        f1_play    = fc.get(1, 0)
        f2_play    = fc.get(2, 0)
        cov_play   = row["coverage_turing"]
        for est_name, S_hat in run_estimators(fc).items():
            failed = S_hat is None or not np.isfinite(S_hat)
            f0_hat = (S_hat - S_obs_play) if (not failed and S_hat is not None) else None
            ratio  = (S_hat / S_obs_play) if (not failed and S_obs_play > 0 and S_hat is not None) else None
            play_rows.append({
                "Dataset":        row["name"],
                "Tokens n":       n_play,
                "Observed types": S_obs_play,
                "f1":             f1_play,
                "f2":             f2_play,
                "Coverage":       round(cov_play, 4),
                "Estimator":      est_name,
                "f0_hat":         round(f0_hat, 1) if f0_hat is not None else None,
                "S_hat":          round(S_hat, 1) if S_hat is not None else None,
                "S_hat / S_obs":  round(ratio, 4) if ratio is not None else None,
            })

    play_table = pd.DataFrame(play_rows)
    _save_csv(play_table, "shakespeare_estimator_table")

    # --- Summary per estimator (median across plays) -----------------------
    summary_rows = []
    for est in ESTIMATOR_ORDER:
        sub = play_table[play_table["Estimator"] == est].dropna(subset=["S_hat"])
        n_fail = play_table[play_table["Estimator"] == est]["S_hat"].isna().sum()
        summary_rows.append({
            "Dataset":          "Shakespeare (per play, median)",
            "Tokens n":         shk_meta["n_tokens"].median().round(0),
            "Observed types":   shk_meta["s_obs"].median().round(0),
            "f1":               shk_meta.apply(lambda r: fc_all.get(r["name"], {}).get(1, 0), axis=1).median(),
            "f2":               shk_meta.apply(lambda r: fc_all.get(r["name"], {}).get(2, 0), axis=1).median(),
            "Coverage":         shk_meta["coverage_turing"].median().round(4),
            "Estimator":        est,
            "f0_hat":           sub["f0_hat"].median().round(1) if not sub.empty else None,
            "S_hat":            sub["S_hat"].median().round(1) if not sub.empty else None,
            "S_hat / S_obs":    sub["S_hat / S_obs"].median().round(4) if not sub.empty else None,
        })

    summary_table = pd.DataFrame(summary_rows)
    _save_csv(summary_table, "shakespeare_estimator_summary")

    # -- Plot: median S_hat per estimator across plays ---------------------
    valid = summary_table.dropna(subset=["S_hat"])
    median_sobs = shk_meta["s_obs"].median()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [PALETTE_EST.get(e, "steelblue") for e in valid["Estimator"]]
    ax.bar(valid["Estimator"], valid["S_hat"], color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(median_sobs, color="black", linewidth=1.4, linestyle="--",
               label=f"Median S_obs = {median_sobs:,.0f}")

    # Efron & Thisted (1976): on the full Works (~884k tokens, ~31.5k observed types)
    # they estimated ~35,000 additional unseen types (S_hat ~ 66,500).
    # Our units are individual plays with far fewer tokens -- not directly comparable.
    # Uncomment and adjust the line below if you obtain a whole-corpus estimate:
    # ax.axhline(66_500, color="red", linewidth=1.0, linestyle=":",
    #            label="Efron & Thisted (1976) full-corpus ~ 66,500")

    ax.set_xlabel("Estimator")
    ax.set_ylabel("Median $\\hat{S}$ across plays")
    ax.set_title("Shakespeare: median estimated richness per play by estimator")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=8)
    plt.xticks(rotation=20, ha="right")
    _save_fig(fig, "shakespeare_estimator_comparison")

    # -- Rarefaction for a representative Shakespeare play -----------------
    # Pick the play closest to the median token count
    median_n = shk_meta["n_tokens"].median()
    rep_idx  = (shk_meta["n_tokens"] - median_n).abs().idxmin()
    rep_name = shk_meta.loc[rep_idx, "name"]
    print(f"  Representative play for rarefaction: '{rep_name}' ...")
    rep_wc_dict = load_word_counts_for_names([rep_name])
    rep_wc = rep_wc_dict.get(rep_name, {})
    rep_fc = fc_all.get(rep_name, wc_to_fc(rep_wc))
    rep_n  = shk_meta.loc[rep_idx, "n_tokens"]
    rep_sobs = shk_meta.loc[rep_idx, "s_obs"]

    if rep_wc:
        print(f"  Computing rarefaction curve (this may take a moment) ...")
        m_rare, ES_rare = rarefaction_curve(rep_wc, n_points=50)
        m_extrap, ES_extrap = extrapolation_curve(rep_fc, rep_n, extrap_factor=2.0, n_points=30)
    else:
        m_rare = np.array([rep_n])
        ES_rare = np.array([rep_sobs])
        m_extrap, ES_extrap = np.array([rep_n]), np.array([rep_sobs])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(m_rare, ES_rare, color="#1f77b4", linewidth=2, label="Rarefaction")
    ax.plot(m_extrap, ES_extrap, color="#1f77b4", linewidth=2, linestyle="--",
            label="Extrapolation (Chao1 asymptote)")
    ax.axvline(rep_n, color="grey", linewidth=1.0, linestyle=":", label="Observed n")
    ax.axhline(rep_sobs, color="grey", linewidth=0.8, linestyle="-.", alpha=0.6,
               label=f"S_obs = {rep_sobs:,}")

    chao1_est = run_estimators(rep_fc).get("Chao1")
    if chao1_est and np.isfinite(chao1_est):
        ax.axhline(chao1_est, color="#ff7f0e", linewidth=0.9, linestyle=":",
                   alpha=0.7, label=f"Chao1 = {chao1_est:,.0f}")

    ax.set_xlabel("Number of tokens sampled")
    ax.set_ylabel("Expected distinct types $E[S(m)]$")
    ax.set_title("Shakespeare: rarefaction and extrapolation curve")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=8)
    _save_fig(fig, "rarefaction_extrapolation_shakespeare")


# ===========================================================================
# Section 6.2 -- Corpus and language overview
# ===========================================================================

def section_6_2_overview(meta: pd.DataFrame) -> None:
    print("\n-- Section 6.2 corpus/language overview --")

    # -- Language-level summary ---------------------------------------------
    spoken = meta[meta["is_spoken"]]

    lang_rows = []
    for lang in LANGUAGE_ORDER:
        sub = spoken[spoken["language"] == lang]
        if sub.empty:
            continue
        corpora = ", ".join(sorted(sub["corpus_source"].unique()))
        lang_rows.append({
            "Language":             lang,
            "Corpora included":     corpora,
            "Number of units":      len(sub),
            "Total tokens":         sub["n_tokens"].sum(),
            "Total observed types": None,  # requires merge -- expensive; note in table
            "Mean tokens":          sub["n_tokens"].mean().round(0),
            "Median tokens":        sub["n_tokens"].median().round(0),
            "Mean S_obs":           sub["s_obs"].mean().round(0),
            "Mean coverage":        sub["coverage_turing"].mean().round(4),
        })

    lang_table = pd.DataFrame(lang_rows)
    _save_csv(lang_table, "language_corpus_overview")

    # -- Corpus-level summary -----------------------------------------------
    corpus_rows = []
    for src, grp in meta.groupby("corpus_source"):
        lang = CORPUS_LANGUAGE.get(src, "Unknown")
        corpus_rows.append({
            "Corpus":           CORPUS_DISPLAY.get(src, src),
            "Language":         lang,
            "Type":             "Written" if src == "shakespeare" else "Spoken/Dialogue",
            "Number of units":  len(grp),
            "Total tokens":     grp["n_tokens"].sum(),
            "Median tokens":    grp["n_tokens"].median().round(0),
            "Min tokens":       grp["n_tokens"].min(),
            "Max tokens":       grp["n_tokens"].max(),
            "Median S_obs":     grp["s_obs"].median().round(0),
            "Median f1/n":      (grp["s_obs"].median() / grp["n_tokens"].median()).round(4)
                                 if grp["n_tokens"].median() > 0 else None,
            "Median coverage":  grp["coverage_turing"].median().round(4),
        })

    corpus_table = pd.DataFrame(corpus_rows)
    _save_csv(corpus_table, "corpus_overview")

    # -- Frequency spectrum by language ------------------------------------
    # Raw f_k values are useful, but not directly comparable because the
    # language corpora have very different total sizes.  The normalized
    # columns express each f_k relative to observed vocabulary size and token
    # count, making cross-language comparison more meaningful.
    language_sources = {
        "English": ["bnc", "sbcorpus", "imsdb"],
        "French":  ["clapi"],
        "German":  ["dgd"],
    }
    spectrum_rows = []
    spectrum_summary_rows = []
    max_k = 50

    for lang, sources in language_sources.items():
        available_sources = [s for s in sources if s in set(meta["corpus_source"])]
        if not available_sources:
            continue
        wc_lang = load_merged_word_counts(available_sources)
        fc_lang = wc_to_fc(wc_lang)
        total_tokens = sum(wc_lang.values())
        s_obs = len(wc_lang)

        spectrum_summary_rows.append({
            "Language": lang,
            "Total tokens": total_tokens,
            "Observed types": s_obs,
            "Singletons f1": fc_lang.get(1, 0),
            "Doubletons f2": fc_lang.get(2, 0),
            "f1 / S_obs": fc_lang.get(1, 0) / s_obs if s_obs else None,
            "f1 / tokens": fc_lang.get(1, 0) / total_tokens if total_tokens else None,
        })

        for k in range(1, max_k + 1):
            fk = fc_lang.get(k, 0)
            spectrum_rows.append({
                "Language": lang,
                "k": k,
                "f_k": fk,
                "Total tokens": total_tokens,
                "Observed types S_obs": s_obs,
                "f_k / S_obs": fk / s_obs if s_obs else None,
                "f_k per 1000 observed types": 1000 * fk / s_obs if s_obs else None,
                "Token mass k*f_k": k * fk,
                "k*f_k / total tokens": (k * fk) / total_tokens if total_tokens else None,
            })

    spectrum_df = pd.DataFrame(spectrum_rows)
    _save_csv(spectrum_df, "language_frequency_spectrum")
    _save_csv(pd.DataFrame(spectrum_summary_rows), "language_frequency_spectrum_summary")

    wide_parts = []
    for lang in language_sources:
        sub = spectrum_df[spectrum_df["Language"] == lang][
            ["k", "f_k", "f_k per 1000 observed types", "k*f_k / total tokens"]
        ].copy()
        if sub.empty:
            continue
        sub = sub.rename(columns={
            "f_k": f"{lang} f_k",
            "f_k per 1000 observed types": f"{lang} f_k per 1000 types",
            "k*f_k / total tokens": f"{lang} token share",
        })
        wide_parts.append(sub)

    if wide_parts:
        wide_df = wide_parts[0]
        for part in wide_parts[1:]:
            wide_df = wide_df.merge(part, on="k", how="outer")
        _save_csv(wide_df, "language_frequency_spectrum_comparable")

    # -- Plot: token-count distribution by language -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Histogram/KDE by language (spoken only)
    ax = axes[0]
    for lang in LANGUAGE_ORDER:
        sub = spoken[spoken["language"] == lang]["n_tokens"]
        if sub.empty:
            continue
        sub_log = np.log10(sub.clip(lower=1))
        sub_log.plot.hist(ax=ax, bins=30, alpha=0.5, color=PALETTE_LANG[lang],
                          label=lang, density=True)
    ax.set_xlabel("log10(tokens per unit)")
    ax.set_ylabel("Density")
    ax.set_title("Token count distribution by language (log scale)")
    ax.legend()

    # Boxplot by corpus
    ax = axes[1]
    order = [CORPUS_DISPLAY[s] for s in ["bnc", "sbcorpus", "imsdb", "clapi", "dgd"]
             if s in meta["corpus_source"].values]
    plot_df = meta[meta["is_spoken"]].copy()
    plot_df["log_tokens"] = np.log10(plot_df["n_tokens"].clip(lower=1))
    sns.boxplot(data=plot_df, x="corpus_display", y="log_tokens", order=order,
                palette="Set2", ax=ax, fliersize=2)
    ax.set_xlabel("Corpus")
    ax.set_ylabel("log10(tokens per unit)")
    ax.set_title("Token counts by corpus")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    _save_fig(fig, "tokens_by_language")

    # -- Plot: S_obs distribution by corpus --------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=plot_df, x="corpus_display", y="s_obs", order=order,
                palette="Set2", ax=ax, fliersize=2)
    ax.set_xlabel("Corpus")
    ax.set_ylabel("Observed types $S_{obs}$")
    ax.set_title("Observed vocabulary size by corpus")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=20, ha="right")
    _save_fig(fig, "sobs_by_corpus")

    # -- Plot: coverage distribution by language ----------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=spoken, x="language", y="coverage_turing",
                order=LANGUAGE_ORDER, palette=PALETTE_LANG, ax=ax, fliersize=2)
    ax.set_xlabel("Language")
    ax.set_ylabel("Sample coverage (Turing)")
    ax.set_title("Sample coverage distribution by language")
    ax.set_ylim(0, 1.05)
    _save_fig(fig, "coverage_by_language")


# ===========================================================================
# Section 6.3 -- Per-language estimator results
# ===========================================================================

def section_6_3_per_language(est_df: pd.DataFrame) -> None:
    print("\n-- Section 6.3 per-language estimator results --")

    spoken = est_df[est_df["is_spoken"]]

    # -- Summary table -----------------------------------------------------
    rows = []
    for lang in LANGUAGE_ORDER:
        for est in ESTIMATOR_ORDER:
            sub = spoken[(spoken["language"] == lang) & (spoken["estimator"] == est)]
            valid = sub.dropna(subset=["S_hat"])
            n_fail = sub["failed"].sum()
            rows.append({
                "Language":      lang,
                "Estimator":     est,
                "Mean S_hat":    valid["S_hat"].mean().round(1) if not valid.empty else None,
                "Median S_hat":  valid["S_hat"].median().round(1) if not valid.empty else None,
                "IQR S_hat":     (valid["S_hat"].quantile(0.75) - valid["S_hat"].quantile(0.25)).round(1)
                                  if not valid.empty else None,
                "Mean f0_hat":   valid["f0_hat"].mean().round(1) if not valid.empty else None,
                "Median S_hat/S_obs": valid["ratio"].median().round(4) if not valid.empty else None,
                "Failed / undefined": int(n_fail),
            })

    summary = pd.DataFrame(rows)
    _save_csv(summary, "per_language_estimator_summary")

    # -- Boxplots: S_hat by estimator, one panel per language --------------
    fig, axes = plt.subplots(1, len(LANGUAGE_ORDER), figsize=(14, 5), sharey=False)
    for ax, lang in zip(axes, LANGUAGE_ORDER):
        sub = spoken[spoken["language"] == lang].dropna(subset=["S_hat"])
        if sub.empty:
            ax.set_title(lang)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        sns.boxplot(data=sub, x="estimator", y="S_hat", order=ESTIMATOR_ORDER,
                    palette=PALETTE_EST, ax=ax, fliersize=2)
        ax.set_title(lang)
        ax.set_xlabel("Estimator")
        ax.set_ylabel("$\\hat{S}$" if ax is axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    fig.suptitle("Estimated total vocabulary $\\hat{S}$ by estimator and language", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "estimator_boxplots_by_language")

    # -- Scatter: coverage vs S_hat/S_obs, coloured by estimator (English) -
    for lang in LANGUAGE_ORDER:
        sub = spoken[(spoken["language"] == lang)].dropna(subset=["ratio"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for est in ESTIMATOR_ORDER:
            esub = sub[sub["estimator"] == est]
            if esub.empty:
                continue
            ax.scatter(esub["coverage"], esub["ratio"],
                       color=PALETTE_EST.get(est, "grey"),
                       alpha=0.5, s=15, label=est)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Sample coverage (Turing)")
        ax.set_ylabel("$\\hat{S} / S_{obs}$")
        ax.set_title(f"{lang}: coverage vs. richness-ratio by estimator")
        ax.legend(fontsize=7, ncol=2)
        _save_fig(fig, f"coverage_vs_ratio_{lang.lower()}")


# ===========================================================================
# Section 6.4 -- Completely monotone / k-monotone model selection
# ===========================================================================

def section_6_4_cm(meta: pd.DataFrame, fc_all: dict[str, dict[int, int]],
                   sample_per_language: int = 15, k_max: int = 12) -> None:
    print("\n--- Section 6.4 CM / k-monotone model selection ---")

    from estimators.monotone import select_k, cm_estimator

    spoken = meta[meta["is_spoken"]].copy()

    # Take the top-N by token count per language (larger units are more stable)
    sample_rows = []
    for lang in LANGUAGE_ORDER:
        sub = spoken[spoken["language"] == lang]
        sub = sub[sub["s_obs"] >= 80]          # skip very small units
        picked = sub.nlargest(sample_per_language, "n_tokens")
        sample_rows.append(picked)

    # Add all Shakespeare plays
    shk = meta[meta["corpus_source"] == "shakespeare"]
    shk = shk[shk["s_obs"] >= 80]
    analysis_df = pd.concat(sample_rows + [shk], ignore_index=True).drop_duplicates("name")

    print(f"  Running k-selection on {len(analysis_df)} units (k_max={k_max}) ...")

    records = []
    bic_curves: dict[str, pd.DataFrame] = {}   # saved for plotting

    for _, row in analysis_df.iterrows():
        fc = fc_all.get(row["name"], {})
        if not fc:
            continue
        try:
            res = select_k(fc, k_max=k_max, n_profile=15, s_max_factor=5.0)
            records.append({
                "name":           row["name"],
                "corpus_source":  row["corpus_source"],
                "language":       row["language"],
                "n_tokens":       row["n_tokens"],
                "s_obs":          row["s_obs"],
                "k_star":         res["k_star"],
                "S_hat_kstar":    res["S_hat_kstar"],
                "S_hat_cm":       res["S_hat_cm"],
                "diff":           (res["S_hat_cm"] - res["S_hat_kstar"])
                                  if (res["S_hat_cm"] and res["S_hat_kstar"]) else None,
            })
            bic_curves[row["name"]] = res["results_df"]
            lang_tag = row.get("language", row["corpus_source"])
            print(f"    {row['name'][:40]:<40}  k*={res['k_star']}  "
                  f"S_hat_cm={res['S_hat_cm']:.0f}" if res["S_hat_cm"] else
                  f"    {row['name'][:40]:<40}  k*={res['k_star']}  S_hat_cm=nan")
        except Exception as e:
            print(f"  WARNING {row['name'][:30]}: {e}")

    if not records:
        print("  No results -- skipping 6.4 plots.")
        return

    table = pd.DataFrame(records)
    _save_csv(table, "cm_kmonotone_selection")

    # -- Plot 1: BIC vs k for one representative unit per language/corpus ----
    rep_names: dict[str, str] = {}
    for lang in LANGUAGE_ORDER:
        sub = table[table["language"] == lang].dropna(subset=["k_star"])
        if sub.empty:
            continue
        median_n = sub["n_tokens"].median()
        rep_names[lang] = sub.loc[(sub["n_tokens"] - median_n).abs().idxmin(), "name"]

    shk_sub = table[table["corpus_source"] == "shakespeare"].dropna(subset=["k_star"])
    if not shk_sub.empty:
        median_shk = shk_sub["n_tokens"].median()
        rep_names["Shakespeare"] = shk_sub.loc[
            (shk_sub["n_tokens"] - median_shk).abs().idxmin(), "name"]

    panels = {k: v for k, v in rep_names.items() if v in bic_curves}
    n_panels = len(panels)
    if n_panels:
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=False)
        if n_panels == 1:
            axes = [axes]
        for ax, (label, uname) in zip(axes, panels.items()):
            df_bic = bic_curves[uname].dropna(subset=["BIC"])
            color = PALETTE_LANG.get(label, "#888888")
            ax.plot(df_bic["k"], df_bic["BIC"], marker="o", color=color, linewidth=1.8)
            k_star_val = table.loc[table["name"] == uname, "k_star"].values
            if len(k_star_val) and not np.isnan(k_star_val[0]):
                ax.axvline(k_star_val[0], color="red", linestyle="--",
                           linewidth=1.2, label=f"k* = {int(k_star_val[0])}")
                ax.legend(fontsize=8)
            ax.set_title(f"{label}\n{uname.split('__')[-1][:25]}", fontsize=9)
            ax.set_xlabel("k (mixture components)")
            ax.set_ylabel("BIC" if ax is axes[0] else "")

        fig.suptitle("BIC vs k (representative units)", y=1.02)
        fig.tight_layout()
        _save_fig(fig, "cm_k_selection_bic")

    # -- Plot 2: histogram of k* by language ----------------------------------
    valid_table = table.dropna(subset=["k_star"])
    if not valid_table.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = range(1, k_max + 2)
        for lang in LANGUAGE_ORDER + ["Shakespeare"]:
            sub = valid_table[valid_table["language"] == lang]["k_star"] \
                  if lang != "Shakespeare" \
                  else valid_table[valid_table["corpus_source"] == "shakespeare"]["k_star"]
            if sub.empty:
                continue
            ax.hist(sub, bins=bins, alpha=0.5, label=lang,
                    color=PALETTE_LANG.get(lang, "#888888"), density=True)
        ax.set_xlabel("BIC-selected k")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of selected k by language/corpus")
        ax.legend(fontsize=8)
        _save_fig(fig, "cm_k_selection_histogram")

    # -- Plot 3: CM vs k* scatter plot ----------------------------------------
    valid2 = table.dropna(subset=["S_hat_cm", "S_hat_kstar"])
    if not valid2.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        for lang in LANGUAGE_ORDER + ["Shakespeare"]:
            if lang == "Shakespeare":
                sub = valid2[valid2["corpus_source"] == "shakespeare"]
            else:
                sub = valid2[valid2["language"] == lang]
            if sub.empty:
                continue
            ax.scatter(sub["S_hat_kstar"], sub["S_hat_cm"],
                       color=PALETTE_LANG.get(lang, "#888"), alpha=0.65,
                       s=20, label=lang)
        lim = max(valid2[["S_hat_cm", "S_hat_kstar"]].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.9)
        ax.set_xlabel("$\\hat{S}$ (k* monotone)")
        ax.set_ylabel("$\\hat{S}$ (CM, fine grid)")
        ax.set_title("CM estimate vs k*-monotone estimate")
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        _save_fig(fig, "cm_vs_kmonotone_scatter")

    # -- Summary table by language ---------------------------------------------
    summary_rows = []
    for lang in LANGUAGE_ORDER + ["Shakespeare"]:
        if lang == "Shakespeare":
            sub = valid_table[valid_table["corpus_source"] == "shakespeare"]
        else:
            sub = valid_table[valid_table["language"] == lang]
        if sub.empty:
            continue
        summary_rows.append({
            "Language/Corpus": lang,
            "N units":         len(sub),
            "Median k*":       sub["k_star"].median(),
            "Mean k*":         sub["k_star"].mean().round(1),
            "k* = k_max (%)":  (100 * (sub["k_star"] == k_max).mean()).round(1),
            "Median S_hat_cm": sub["S_hat_cm"].median().round(1) if "S_hat_cm" in sub else None,
            "Median S_hat_kstar": sub["S_hat_kstar"].median().round(1),
        })
    if summary_rows:
        _save_csv(pd.DataFrame(summary_rows), "cm_kmonotone_summary")


# ===========================================================================
# Section 6.5 -- Rarefaction and extrapolation curves
# ===========================================================================

def section_6_5_rarefaction(meta: pd.DataFrame, fc_all: dict[str, dict[int, int]]) -> None:
    print("\n-- Section 6.5 rarefaction/extrapolation --")

    spoken_meta = meta[meta["is_spoken"]]

    # For each language pick the unit closest to the median token count
    rep_units: dict[str, str] = {}
    for lang in LANGUAGE_ORDER:
        sub = spoken_meta[spoken_meta["language"] == lang]
        if sub.empty:
            continue
        median_n = sub["n_tokens"].median()
        idx = (sub["n_tokens"] - median_n).abs().idxmin()
        rep_units[lang] = sub.loc[idx, "name"]

    print(f"  Representative units: {rep_units}")
    wc_dict = load_word_counts_for_names(list(rep_units.values()))

    # -- Per-language combined rarefaction (full aggregated corpus) ---------
    # Also load merged word counts for each language
    lang_sources = {
        "English": ["bnc", "sbcorpus", "imsdb"],
        "French":  ["clapi"],
        "German":  ["dgd"],
    }

    fig, axes = plt.subplots(1, len(LANGUAGE_ORDER), figsize=(14, 4.5), sharey=False)

    for ax, lang in zip(axes, LANGUAGE_ORDER):
        color = PALETTE_LANG[lang]

        # Representative unit curve
        rep_name = rep_units.get(lang)
        if rep_name and rep_name in wc_dict:
            wc = wc_dict[rep_name]
            fc = fc_all.get(rep_name, wc_to_fc(wc))
            n  = sum(wc.values())

            print(f"  {lang}: rarefaction for '{rep_name}' (n={n:,}) ...")
            m_rare, ES_rare = rarefaction_curve(wc, n_points=40)
            m_ex,  ES_ex   = extrapolation_curve(fc, n, extrap_factor=2.0, n_points=30)

            ax.plot(m_rare / 1000, ES_rare, color=color, linewidth=2, label="Rarefaction")
            ax.plot(m_ex   / 1000, ES_ex,   color=color, linewidth=2, linestyle="--",
                    label="Extrapolation")
            ax.axvline(n / 1000, color="grey", linewidth=0.9, linestyle=":", alpha=0.7)

            S_obs_unit = sum(fc.values())
            ax.scatter([n / 1000], [S_obs_unit], color=color, s=40, zorder=5)

        ax.set_title(f"{lang}\n(representative unit)")
        ax.set_xlabel("Tokens (thousands)")
        ax.set_ylabel("$E[S(m)]$" if ax is axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=7)

    fig.suptitle("Rarefaction and extrapolation curves by language (representative unit)",
                 y=1.01)
    fig.tight_layout()
    _save_fig(fig, "rarefaction_extrapolation_by_language")

    # -- Combined language curves (aggregated word counts) ------------------
    print("  Computing combined language rarefaction (loading merged word counts) ...")
    fig, ax = plt.subplots(figsize=(8, 5))

    for lang in LANGUAGE_ORDER:
        sources = lang_sources.get(lang, [])
        available = [s for s in sources if s in meta["corpus_source"].values]
        if not available:
            continue
        color = PALETTE_LANG[lang]
        print(f"  {lang}: merging {available} ...")
        wc_lang = load_merged_word_counts(available)
        if not wc_lang:
            continue
        fc_lang = wc_to_fc(wc_lang)
        n_lang  = sum(wc_lang.values())

        print(f"    n={n_lang:,}, S_obs={len(wc_lang):,} -- computing rarefaction ...")
        m_rare, ES_rare = rarefaction_curve(wc_lang, n_points=50)
        m_ex,  ES_ex    = extrapolation_curve(fc_lang, n_lang, extrap_factor=1.3, n_points=20)

        ax.plot(m_rare / 1e6, ES_rare / 1e3, color=color, linewidth=2.0, label=lang)
        ax.plot(m_ex   / 1e6, ES_ex   / 1e3, color=color, linewidth=2.0, linestyle="--",
                alpha=0.7)
        ax.axvline(n_lang / 1e6, color=color, linewidth=0.8, linestyle=":", alpha=0.5)

    ax.set_xlabel("Tokens (millions)")
    ax.set_ylabel("Expected distinct types (thousands)")
    ax.set_title("Combined language rarefaction / extrapolation")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}k"))
    _save_fig(fig, "rarefaction_language_combined")


# ===========================================================================
# Section 6.6 -- Cross-language comparison
# ===========================================================================

def section_6_6_cross_language(est_df: pd.DataFrame) -> None:
    print("\n-- Section 6.6 cross-language comparison --")

    spoken = est_df[est_df["is_spoken"]]

    # -- Summary table -----------------------------------------------------
    rows = []
    for lang in LANGUAGE_ORDER:
        for est in ESTIMATOR_ORDER:
            sub = spoken[(spoken["language"] == lang) & (spoken["estimator"] == est)]
            valid = sub.dropna(subset=["S_hat"])
            rows.append({
                "Language":          lang,
                "Estimator":         est,
                "Median S_obs":      sub["s_obs"].median().round(1) if not sub.empty else None,
                "Median S_hat":      valid["S_hat"].median().round(1) if not valid.empty else None,
                "Median f0_hat":     valid["f0_hat"].median().round(1) if not valid.empty else None,
                "Median coverage":   sub["coverage"].median().round(4) if not sub.empty else None,
                "Median S_hat/S_obs": valid["ratio"].median().round(4) if not valid.empty else None,
            })

    cross_table = pd.DataFrame(rows)
    _save_csv(cross_table, "cross_language_estimator_summary")

    # -- Grouped bar chart: median S_hat by language and estimator ---------
    pivot = cross_table.pivot(index="Estimator", columns="Language", values="Median S_hat")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax,
               color=[PALETTE_LANG[l] for l in pivot.columns if l in PALETTE_LANG],
               alpha=0.85, edgecolor="white")
    ax.set_xlabel("Estimator")
    ax.set_ylabel("Median $\\hat{S}$")
    ax.set_title("Median estimated richness by estimator and language\n(spoken/dialogue corpora only)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="Language")
    _save_fig(fig, "cross_language_median_Shat")

    # -- Heatmap: median S_hat/S_obs by language x estimator ---------------
    heat_pivot = cross_table.pivot(index="Language", columns="Estimator",
                                   values="Median S_hat/S_obs")
    heat_pivot = heat_pivot.reindex(index=LANGUAGE_ORDER, columns=ESTIMATOR_ORDER)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.heatmap(heat_pivot.astype(float), annot=True, fmt=".2f",
                cmap="YlOrRd", ax=ax, linewidths=0.4, cbar_kws={"label": "$\\hat{S}/S_{obs}$"})
    ax.set_title("Median $\\hat{S}/S_{obs}$ by language and estimator")
    ax.set_xlabel("Estimator")
    ax.set_ylabel("Language")
    plt.xticks(rotation=20, ha="right")
    _save_fig(fig, "coverage_standardized_language_comparison")

    # -- Boxplot: S_hat/S_obs by language and estimator --------------------
    valid_spoken = spoken.dropna(subset=["ratio"])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=valid_spoken, x="estimator", y="ratio", hue="language",
                order=ESTIMATOR_ORDER, hue_order=LANGUAGE_ORDER,
                palette=PALETTE_LANG, ax=ax, fliersize=2)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Estimator")
    ax.set_ylabel("$\\hat{S} / S_{obs}$")
    ax.set_title("Richness ratio $\\hat{S}/S_{obs}$ by estimator and language")
    ax.legend(title="Language", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    _save_fig(fig, "cross_language_ratio_boxplot")


# ===========================================================================
# Section 6.7 -- Estimator comparison across all data
# ===========================================================================

def section_6_7_estimator_comparison(est_df: pd.DataFrame) -> None:
    print("\n-- Section 6.7 estimator comparison across all data --")

    spoken = est_df[est_df["is_spoken"]]

    # -- Failure summary ----------------------------------------------------
    fail_rows = []
    n_units = spoken["name"].nunique()
    for est in ESTIMATOR_ORDER:
        sub = spoken[spoken["estimator"] == est]
        n_fail = sub["failed"].sum()
        fail_rows.append({
            "Estimator":     est,
            "Total units":   n_units,
            "Failed / undefined": int(n_fail),
            "Failure rate (%)": round(100.0 * n_fail / n_units, 1) if n_units > 0 else None,
        })

    fail_table = pd.DataFrame(fail_rows)
    _save_csv(fail_table, "estimator_failure_summary")

    # -- Global summary table -----------------------------------------------
    summary_rows = []
    for est in ESTIMATOR_ORDER:
        sub = spoken[spoken["estimator"] == est].dropna(subset=["ratio"])
        n_fail = spoken[spoken["estimator"] == est]["failed"].sum()
        summary_rows.append({
            "Estimator":         est,
            "Mean S_hat/S_obs":  sub["ratio"].mean().round(4) if not sub.empty else None,
            "Median S_hat/S_obs": sub["ratio"].median().round(4) if not sub.empty else None,
            "IQR":               (sub["ratio"].quantile(0.75) - sub["ratio"].quantile(0.25)).round(4)
                                  if not sub.empty else None,
            "Failure rate":      round(100.0 * n_fail / n_units, 1) if n_units > 0 else None,
            "Typical behavior":  _typical_behavior(est),
        })

    summary_table = pd.DataFrame(summary_rows)
    _save_csv(summary_table, "estimator_summary_global")

    # -- Estimator correlation heatmap --------------------------------------
    wide = spoken.pivot_table(index="name", columns="estimator", values="S_hat")
    wide = wide.reindex(columns=ESTIMATOR_ORDER).dropna(how="all")

    corr = wide.corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mask, ax=ax, linewidths=0.3,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Estimator correlation (Pearson r, spoken corpora)")
    plt.xticks(rotation=20, ha="right")
    _save_fig(fig, "estimator_correlation_heatmap")

    # -- Difference plot: estimator - Chao1 --------------------------------
    diff_data = []
    for est in ESTIMATOR_ORDER:
        if est == "Chao1":
            continue
        merged = wide[[est, "Chao1"]].dropna()
        diff = merged[est] - merged["Chao1"]
        diff_data.append(pd.DataFrame({"Estimator": est, "Difference": diff}))

    if diff_data:
        diff_df = pd.concat(diff_data, ignore_index=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=diff_df, x="Estimator", y="Difference",
                    order=[e for e in ESTIMATOR_ORDER if e != "Chao1"],
                    palette=PALETTE_EST, ax=ax, fliersize=2)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Estimator")
        ax.set_ylabel("$\\hat{S}_{est} - \\hat{S}_{Chao1}$")
        ax.set_title("Estimator difference relative to Chao1")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.xticks(rotation=20, ha="right")
        _save_fig(fig, "estimator_ratio_to_chao1")

    # -- Ratio plot: estimator / Chao1 -------------------------------------
    ratio_data = []
    for est in ESTIMATOR_ORDER:
        if est == "Chao1":
            continue
        merged = wide[[est, "Chao1"]].dropna()
        merged = merged[merged["Chao1"] > 0]
        ratio = merged[est] / merged["Chao1"]
        ratio_data.append(pd.DataFrame({"Estimator": est, "Ratio to Chao1": ratio}))

    if ratio_data:
        ratio_df = pd.concat(ratio_data, ignore_index=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=ratio_df, x="Estimator", y="Ratio to Chao1",
                    order=[e for e in ESTIMATOR_ORDER if e != "Chao1"],
                    palette=PALETTE_EST, ax=ax, fliersize=2)
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Estimator")
        ax.set_ylabel("$\\hat{S}_{est} / \\hat{S}_{Chao1}$")
        ax.set_title("Estimator ratio relative to Chao1")
        plt.xticks(rotation=20, ha="right")
        _save_fig(fig, "estimator_ratio_over_chao1")


def _typical_behavior(est: str) -> str:
    behaviors = {
        "Chao1":     "Lower bound via f1/f2 ratio",
        "iChao1":    "Tighter lower bound using f3/f4",
        "ACE":       "Heterogeneity-corrected lower bound",
        "Jackknife1": "Delete-one jackknife; may overshoot",
        "Jackknife2": "Delete-two jackknife; more aggressive",
        "Breakaway": "Ratio-regression; can be unstable for small n",
    }
    return behaviors.get(est, "")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("Loading metadata from DB ...")
    meta = load_metadata()
    print(f"  {len(meta):,} corpus units found.")
    print(f"  Sources: {meta['corpus_source'].value_counts().to_dict()}")

    if meta.empty:
        print("ERROR: No data in DB. Run preprocessing first.")
        print("  python code/preprocessing/preprocess.py --corpus all")
        sys.exit(1)

    print("\nLoading freq_counts (k=1..20) ...")
    fc_all = load_freq_counts_all(meta)
    print(f"  Loaded freq_counts for {len(fc_all):,} units.")

    print("\nRunning all estimators on each unit ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est_df = estimator_dataframe(meta, fc_all)
    print(f"  Estimator table: {len(est_df):,} rows.")

    section_6_1_shakespeare(meta, fc_all)
    section_6_2_overview(meta)
    section_6_3_per_language(est_df)

    try:
        section_6_4_cm(meta, fc_all, sample_per_language=15, k_max=12)
    except Exception as e:
        print(f"  WARNING: section 6.4 CM failed -- {e}")

    print("\n-- Section 6.5 rarefaction/extrapolation --")
    try:
        section_6_5_rarefaction(meta, fc_all)
    except Exception as e:
        print(f"  WARNING: rarefaction failed -- {e}")

    section_6_6_cross_language(est_df)
    section_6_7_estimator_comparison(est_df)

    print("\nOK All outputs written.")
    print(f"  Tables : {TABLES}")
    print(f"  Figures: {FIGURES}")


if __name__ == "__main__":
    main()
