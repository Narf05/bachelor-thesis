"""
Sample coverage estimators.

References
----------
Good (1953) — Turing's estimator (eq. 2 in thesis)
Chao & Jost (2012) — bias-corrected estimator (eq. 3 in thesis)
"""


def coverage_turing(freq_counts: dict) -> float:
    """
    Turing's sample coverage estimator.

    C_hat = 1 - f1 / n

    Parameters
    ----------
    freq_counts : dict
        Mapping k -> f_k (number of species seen exactly k times), k >= 1.

    Returns
    -------
    float
        Estimated sample coverage in [0, 1].
    """
    f1 = freq_counts.get(1, 0)
    n = sum(k * v for k, v in freq_counts.items())
    if n == 0:
        raise ValueError("Sample size n = 0.")
    return 1.0 - f1 / n


def coverage_chao_jost(freq_counts: dict) -> float:
    """
    Bias-corrected sample coverage estimator (Chao & Jost 2012).

    C_hat = 1 - (f1/n) * ((n-1)*f1 / ((n-1)*f1 + 2*f2))

    Falls back to Turing's estimator when f2 = 0 (the correction
    factor is undefined; the raw f1/n term is returned as-is).

    Parameters
    ----------
    freq_counts : dict
        Mapping k -> f_k (number of species seen exactly k times), k >= 1.

    Returns
    -------
    float
        Estimated sample coverage in [0, 1].
    """
    f1 = freq_counts.get(1, 0)
    f2 = freq_counts.get(2, 0)
    n = sum(k * v for k, v in freq_counts.items())
    if n == 0:
        raise ValueError("Sample size n = 0.")
    if f2 == 0:
        # Correction undefined; fall back to Turing
        return 1.0 - f1 / n
    denominator = (n - 1) * f1 + 2 * f2
    return 1.0 - (f1 / n) * ((n - 1) * f1 / denominator)
