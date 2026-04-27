"""
Convert token lists into the frequency-count structures used by the estimators.

Terminology (from thesis Section 2)
------------------------------------
word_counts  : {word: n_i}   — how many times each distinct word appears
freq_counts  : {k: f_k}      — how many words appear exactly k times

The estimators in `estimators/` all expect a `freq_counts` dict.
"""

from collections import Counter
from typing import Iterable

from .tokenizer import tokenize


# ---------------------------------------------------------------------------
# Step 1 — tokens → word counts
# ---------------------------------------------------------------------------

def word_counts(tokens: Iterable[str]) -> dict[str, int]:
    """
    Count occurrences of each distinct word token.

    Parameters
    ----------
    tokens : iterable of str
        Normalised tokens (output of ``tokenize``).

    Returns
    -------
    dict[str, int]
        {word: count}, sorted descending by count.
    """
    counts = Counter(tokens)
    return dict(counts.most_common())


# ---------------------------------------------------------------------------
# Step 2 — word counts → frequency counts
# ---------------------------------------------------------------------------

def freq_counts(wc: dict[str, int]) -> dict[int, int]:
    """
    Build the abundance frequency-count table from word counts.

    f_k = number of distinct words that appear exactly k times.

    Parameters
    ----------
    wc : dict[str, int]
        Output of ``word_counts``.

    Returns
    -------
    dict[int, int]
        {k: f_k} for all k >= 1 that appear in the data.
        Keys are sorted in ascending order.
    """
    fc: dict[int, int] = {}
    for count in wc.values():
        fc[count] = fc.get(count, 0) + 1
    return dict(sorted(fc.items()))


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def pipeline(
    text: str,
    *,
    remove_fillers: bool = False,
    lemmatize: bool = False,
) -> tuple[dict[str, int], dict[int, int]]:
    """
    Tokenize a pre-loaded text string and return (word_counts, freq_counts).

    The caller is responsible for loading and cleaning the raw text using
    whichever loader is appropriate (``loaders.shakespeare``,
    ``loaders.transcript``, etc.).  This keeps the pipeline format-agnostic.

    Parameters
    ----------
    text : str
        Clean literary or transcript text (output of any loader).
    remove_fillers : bool
        If True, strip filler words before counting. Useful for speech
        transcripts; typically False for written text.
    lemmatize : bool
        If True, lemmatize tokens with NLTK before counting.

    Returns
    -------
    wc : dict[str, int]
        Word counts  {word: n_i}.
    fc : dict[int, int]
        Frequency counts  {k: f_k}.

    Example
    -------
    >>> text = Path("data/speakers/shakespeare/hamlet.txt").read_text()
    >>> wc, fc = pipeline(text)
    >>> print(f"n={sum(wc.values())}, S_obs={len(wc)}, f1={fc.get(1, 0)}")
    """
    tokens = tokenize(text, remove_fillers=remove_fillers, lemmatize=lemmatize)
    wc = word_counts(tokens)
    fc = freq_counts(wc)
    return wc, fc
