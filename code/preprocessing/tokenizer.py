"""
Tokenizer for literary text.

Design choices (matching thesis Section 5 discussion)
------------------------------------------------------
- Lowercase: yes (always)
- Punctuation: stripped; internal apostrophes kept so contractions and
  possessives stay as single types (ne'er, that's, hamlet's).
- Leading / trailing apostrophes stripped ('Tis → tis).
- Purely non-alphabetic tokens discarded.
- Lemmatization: off by default (configurable). When enabled, requires
  the 'nltk' package with the WordNet lemmatizer.
- Filler words ("um", "uh", "hmm", ...): filtered via a configurable
  set. Relevant for speech transcripts; not needed for Shakespeare.
"""

import re
from typing import Iterable


# ---------------------------------------------------------------------------
# Default filler-word list (speech transcripts)
# ---------------------------------------------------------------------------

DEFAULT_FILLERS: frozenset[str] = frozenset(
    {
        "um", "uh", "hmm", "hm", "mhm", "uhm", "er", "eh",
        "ah", "oh", "mm", "mmm", "huh", "yeah", "yep",
    }
)

# Regex that captures letter sequences with optional internal apostrophes
# Examples: "ne'er", "that's", "hamlet's", "tis", "o'er"
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']*[a-zA-Z]|[a-zA-Z]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tokenize(
    text: str,
    *,
    remove_fillers: bool = False,
    fillers: Iterable[str] = DEFAULT_FILLERS,
    lemmatize: bool = False,
) -> list[str]:
    """
    Convert a string of text into a list of normalised word tokens.

    Steps
    -----
    1. Extract word-like sequences (letters + internal apostrophes).
    2. Lowercase.
    3. Strip leading/trailing apostrophes from each token.
    4. Drop empty or purely-apostrophe tokens.
    5. Optionally remove filler words.
    6. Optionally lemmatize with NLTK WordNetLemmatizer.

    Parameters
    ----------
    text : str
    remove_fillers : bool
        If True, remove tokens in *fillers* (default False).
    fillers : iterable of str
        Set of filler words to remove (default DEFAULT_FILLERS).
    lemmatize : bool
        If True, lemmatize tokens using NLTK. Requires `nltk` and the
        'wordnet' corpus (``nltk.download('wordnet')``). Default False.

    Returns
    -------
    list of str
    """
    tokens = [m.group(0).lower().strip("'") for m in _WORD_RE.finditer(text)]
    tokens = [t for t in tokens if t]  # drop empty strings

    if remove_fillers:
        filler_set = frozenset(fillers)
        tokens = [t for t in tokens if t not in filler_set]

    if lemmatize:
        tokens = _lemmatize(tokens)

    return tokens


def _lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize *tokens* using NLTK's WordNetLemmatizer."""
    try:
        from nltk.stem import WordNetLemmatizer
    except ImportError as e:
        raise ImportError(
            "lemmatize=True requires the 'nltk' package. "
            "Install it with: pip install nltk"
        ) from e

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]
