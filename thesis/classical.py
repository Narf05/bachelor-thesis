"""
Classical nonparametric species-richness estimators.

All estimators accept a frequency-count dict:
    freq_counts[k] = f_k = number of species observed exactly k times, k >= 1

References (eq. numbers refer to the thesis)
---------------------------------------------
Good (1953)          — Good-Turing (eq. 4)
Chao (1984)          — Chao1       (eq. 6)
Chiu et al. (2014)   — iChao1      (eq. 8)
Chao & Lee (1992)    — ACE         (eq. 10-11)
Burnham & Overton    — Jackknife   (eq. 12-13)
"""


def _basics(freq_counts: dict) -> tuple:
    """Return (S_obs, n, f1, f2) from a frequency-count dict."""
    S_obs = sum(freq_counts.values())
    n = sum(k * v for k, v in freq_counts.items())
    f1 = freq_counts.get(1, 0)
    f2 = freq_counts.get(2, 0)
    return S_obs, n, f1, f2


# ---------------------------------------------------------------------------
# Good-Turing
# ---------------------------------------------------------------------------

def good_turing_p0(freq_counts: dict) -> float:
    """
    Estimated total probability mass of all unseen species.

    p_hat_0 = f1 / n

    This is NOT the count of undetected species; it is the complement of
    Turing's sample coverage (1 - C_hat).

    Returns
    -------
    float
        Probability mass on unseen species, in [0, 1].
    """
    _, n, f1, _ = _basics(freq_counts)
    if n == 0:
        raise ValueError("Sample size n = 0.")
    return f1 / n


# ---------------------------------------------------------------------------
# Chao1 / iChao1
# ---------------------------------------------------------------------------

def chao1(freq_counts: dict) -> float:
    """
    Chao1 lower bound for species richness (Chao 1984, eq. 6).

    S_hat = S_obs + (n-1)/n * f1^2 / (2*f2)       if f2 > 0
    S_hat = S_obs + (n-1)/n * f1*(f1-1) / 2        if f2 = 0

    Returns
    -------
    float
        Estimated total species richness.
    """
    S_obs, n, f1, f2 = _basics(freq_counts)
    if n == 0:
        raise ValueError("Sample size n = 0.")
    correction = (n - 1) / n
    if f2 > 0:
        return S_obs + correction * f1 ** 2 / (2 * f2)
    else:
        return S_obs + correction * f1 * (f1 - 1) / 2


def ichao1(freq_counts: dict) -> float:
    """
    Improved Chao1 lower bound (Chiu et al. 2014, eq. 8).

    Extends Chao1 by incorporating f3 and f4 to tighten the bound.

    Falls back to Chao1 when f3 = 0 or f4 = 0 (the extra term vanishes).

    Returns
    -------
    float
        Estimated total species richness.
    """
    S_obs, n, f1, f2 = _basics(freq_counts)
    f3 = freq_counts.get(3, 0)
    f4 = freq_counts.get(4, 0)
    base = chao1(freq_counts)
    if f3 == 0 or f4 == 0:
        return base
    extra_term = (n - 3) / n * f3 / (4 * f4)
    inner = f1 - (n - 3) / (n - 1) * f2 * f3 / (2 * f4)
    return base + extra_term * max(inner, 0.0)


# ---------------------------------------------------------------------------
# ACE
# ---------------------------------------------------------------------------

def ace(freq_counts: dict, k_cutoff: int = 10) -> float:
    """
    Abundance-based Coverage Estimator (Chao & Lee 1992, eq. 10-11).

    Species with abundance <= k_cutoff are treated as 'rare'; the rest as
    'abundant'. The rare group drives the heterogeneity correction via the
    squared CV of its abundances.

    Parameters
    ----------
    freq_counts : dict
        Mapping k -> f_k, k >= 1.
    k_cutoff : int
        Threshold separating rare from abundant species (default 10).

    Returns
    -------
    float
        Estimated total species richness.
    """
    f1 = freq_counts.get(1, 0)

    S_abun = S_rare = n_rare = sum_i_im1_fi = 0
    for k, v in freq_counts.items():
        if k > k_cutoff:
            S_abun += v
        else:
            S_rare += v
            n_rare += k * v
            sum_i_im1_fi += k * (k - 1) * v

    if n_rare == 0:
        return S_abun

    C_rare = 1.0 - f1 / n_rare

    if C_rare <= 0:
        return float("inf")

    gamma2 = (S_rare / C_rare) * sum_i_im1_fi / (n_rare * (S_rare - 1)) - 1 if S_rare > 1 else 0.0
    gamma2 = max(gamma2, 0.0)

    return S_abun + S_rare / C_rare + f1 / C_rare * gamma2


# ---------------------------------------------------------------------------
# Jackknife
# ---------------------------------------------------------------------------

def jackknife1(freq_counts: dict) -> float:
    """
    First-order jackknife estimator (eq. 12).

    S_hat = S_obs + (n-1)/n * f1  ≈  S_obs + f1

    Returns
    -------
    float
    """
    S_obs, n, f1, _ = _basics(freq_counts)
    if n == 0:
        raise ValueError("Sample size n = 0.")
    return S_obs + (n - 1) / n * f1


def jackknife2(freq_counts: dict) -> float:
    """
    Second-order jackknife estimator (eq. 13).

    S_hat = S_obs + (2n-3)/n * f1 - (n-2)^2 / (n*(n-1)) * f2
          ≈ S_obs + 2*f1 - f2

    Returns
    -------
    float
    """
    S_obs, n, f1, f2 = _basics(freq_counts)
    if n <= 1:
        raise ValueError("Sample size n must be > 1 for jackknife2.")
    return S_obs + (2 * n - 3) / n * f1 - (n - 2) ** 2 / (n * (n - 1)) * f2
