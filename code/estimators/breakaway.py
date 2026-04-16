"""
Breakaway: frequency-ratio regression estimator (Willis & Bunge 2015).

Fits a rational function to the observed frequency ratios f_{j+1}/f_j and
extrapolates to j=0 to estimate f_0, the number of unseen species.

Rational model (eq. in thesis, Section 4.2):
    f_{j+1}/f_j = (beta0 + beta1*j) / (1 + beta2*j + beta3*j^2)

Extrapolating to j=0 gives  f_1/f_0 = beta0, so:
    f_0_hat = f_1 / beta0
    S_hat   = S_obs + f_0_hat

Heteroscedastic weights: w_j = f_j  (larger counts are more reliable).

Reference
---------
Willis & Bunge (2015) — "Estimating diversity via frequency ratios"
"""

import numpy as np
from scipy.optimize import curve_fit


def _rational(j, beta0, beta1, beta2, beta3):
    """Rational function for the frequency ratio at position j."""
    return (beta0 + beta1 * j) / (1.0 + beta2 * j + beta3 * j ** 2)


def breakaway(freq_counts: dict, max_ratio_terms: int = 20) -> dict:
    """
    Breakaway species-richness estimator.

    Parameters
    ----------
    freq_counts : dict
        Mapping k -> f_k (number of species seen exactly k times), k >= 1.
    max_ratio_terms : int
        Maximum number of consecutive frequency ratios to include in the
        regression (default 20). Ratios beyond this are noisy and omitted.

    Returns
    -------
    dict with keys:
        'S_hat'   : float  — estimated total richness
        'f0_hat'  : float  — estimated number of unseen species
        'beta'    : array  — fitted parameters [beta0, beta1, beta2, beta3]
        'S_obs'   : int    — observed species count
    """
    S_obs = sum(freq_counts.values())
    f1 = freq_counts.get(1, 0)

    # Build consecutive ratio pairs (j, ratio) where ratio = f_{j+1}/f_j
    js, ratios, weights = [], [], []
    for j in range(1, max_ratio_terms + 1):
        fj = freq_counts.get(j, 0)
        fj1 = freq_counts.get(j + 1, 0)
        if fj == 0:
            break
        if fj1 == 0:
            # Zero in numerator — stop; can't form further ratios
            break
        js.append(float(j))
        ratios.append(fj1 / fj)
        weights.append(float(fj))  # heteroscedastic weights

    if len(js) < 4:
        raise ValueError(
            f"Not enough non-zero consecutive frequency counts to fit the "
            f"rational model (need at least 4, got {len(js)})."
        )

    js = np.array(js)
    ratios = np.array(ratios)
    weights = np.array(weights)

    # Initial parameter guess: constant ratio ~ ratios[0], small corrections
    r0 = ratios[0]
    p0 = [r0, 0.0, 0.0, 0.0]

    beta, _ = curve_fit(
        _rational,
        js,
        ratios,
        p0=p0,
        sigma=1.0 / weights,   # inverse-weight: down-weight noisy ratios
        absolute_sigma=True,
        maxfev=10_000,
    )

    beta0 = beta[0]
    if beta0 <= 0:
        raise ValueError(
            f"Fitted beta0 = {beta0:.4f} <= 0; cannot estimate f0. "
            "The rational model may be misspecified for this dataset."
        )

    f0_hat = f1 / beta0
    S_hat = S_obs + f0_hat

    return {
        "S_hat": S_hat,
        "f0_hat": f0_hat,
        "beta": beta,
        "S_obs": S_obs,
    }
