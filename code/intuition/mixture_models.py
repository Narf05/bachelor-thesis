"""
Parametric Poisson-mixture models for vocabulary estimation.

Three models are fitted to the zero-truncated word-frequency distribution
of Shakespeare's complete works (Folger Digital Texts):

  1. Gamma–Poisson         (Negative Binomial)
  2. Log-Normal–Poisson
  3. Poisson–Inverse Gaussian  (exact PMF via generating-function recurrence)

Each model has the form  X | Lambda ~ Poisson(Lambda),  Lambda ~ G(theta),
and is fitted by maximum likelihood on the zero-truncated distribution.

Run this file to fit and print the summary table.
Import it (from mixture_models import *) to access fitted parameters from
another script.
"""

import math
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.optimize import differential_evolution
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "code"))

from db.cache import get_or_process

# ============================================================
# CONFIGURATION
# ============================================================

# Cache key identifying the full Shakespeare corpus in the SQLite database.
CORPUS_KEY = "shakespeare_corpus"
CORPUS_DIR = _ROOT / "data" / "shakespeare-dataset-main" / "text"

# Differential-evolution optimiser settings.
DE_SEED    = 42       # reproducibility
DE_MAXITER = 2000     # maximum optimiser iterations
DE_TOL     = 1e-14    # convergence tolerance

# Parameter search bounds  (lower, upper) for each model.
#   Gamma-Poisson:          (r, mu)       — r: shape, mu: mean rate
#   Log-Normal-Poisson:     (log_mu, sigma) — log-mean and std of log-rate
#   Poisson-Inv. Gaussian:  (mu, phi)     — mean and shape of InvGauss mixing
BOUNDS_NB  = [(0.005, 15),  (0.01,  50)]
BOUNDS_LN  = [(-8,    3),   (0.1,    5)]
BOUNDS_PIG = [(0.01, 200),  (0.01, 200)]

# Number of Gauss-Hermite quadrature nodes for the Log-Normal-Poisson PMF.
LNP_QUAD_NODES = 150

# ============================================================
# 1. DATA — load Shakespeare word counts from the database
# ============================================================
print("Loading Shakespeare corpus from database ...")
word_counts_dict, _ = get_or_process(
    CORPUS_KEY,
    CORPUS_DIR,
    loader="corpus",
    corpus_source="shakespeare",
)

# Rebuild the full frequency-of-frequency table from word counts.
# (The database only caches f_k for k <= 20; word_counts are stored in full.)
freq_of_freq = Counter(word_counts_dict.values())

S_obs    = len(word_counts_dict)                  # observed vocabulary size
N_tokens = sum(word_counts_dict.values())         # total token count
x_vals   = np.array(sorted(freq_of_freq.keys()))  # distinct abundance values
f_vals   = np.array([freq_of_freq[x] for x in x_vals])

print(f"  Tokens : {N_tokens:,}")
print(f"  Types  : {S_obs:,}  (S_obs)")
print(f"  Hapax  : {freq_of_freq.get(1, 0):,}  (f_1, proportion {freq_of_freq.get(1,0)/S_obs:.2%})")

# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

# ------------------------------------------------------------
# Model 1: Gamma–Poisson  (Negative Binomial)
#
#   Lambda ~ Gamma(r, r/mu)   =>   marginal X ~ NegBin(r, mu)
#   p(x; r, mu) = Gamma(r+x) / (Gamma(r) x!) * p^r * (1-p)^x,   p = r/(r+mu)
# ------------------------------------------------------------

def negbin_pmf(x, r, mu):
    p = r / (r + mu)
    return np.exp(
        gammaln(r + x) - gammaln(r) - gammaln(x + 1)
        + r * np.log(p) + x * np.log(1.0 - p)
    )

def negbin_trunc(x, r, mu):
    """Zero-truncated Negative Binomial PMF."""
    return negbin_pmf(x, r, mu) / (1.0 - (r / (r + mu)) ** r)

def negbin_nll(params, x_vals, f_vals):
    r, mu = params
    if r <= 0.001 or mu <= 0:
        return 1e10
    try:
        probs = negbin_trunc(x_vals, r, mu)
        if np.any(probs <= 0):
            return 1e10
        return -np.sum(f_vals * np.log(probs))
    except Exception:
        return 1e10


# ------------------------------------------------------------
# Model 2: Log-Normal–Poisson
#
#   Lambda ~ LogNormal(mu, sigma^2)
#   p(x; mu, sigma) = E_{Lambda}[Pois(x; Lambda)]
#   Computed via Gauss-Hermite quadrature on the log scale.
# ------------------------------------------------------------

def lnp_pmf(x_arr, mu, sigma, nq=LNP_QUAD_NODES):
    nd, wt = np.polynomial.hermite.hermgauss(nq)
    out    = np.zeros(len(x_arr))
    for i, xi in enumerate(x_arr):
        lam    = np.exp(mu + sigma * np.sqrt(2.0) * nd)
        out[i] = np.sum(
            wt * np.exp(int(xi) * np.log(lam + 1e-300) - lam - gammaln(int(xi) + 1))
        ) / np.sqrt(np.pi)
    return out

def lnp_trunc(x_arr, mu, sigma):
    """Zero-truncated Log-Normal-Poisson PMF."""
    p0 = lnp_pmf(np.array([0]), mu, sigma)[0]
    return lnp_pmf(x_arr, mu, sigma) / (1.0 - p0)

def lnp_nll(params, x_vals, f_vals):
    mu, sigma = params
    if sigma <= 0.05:
        return 1e10
    try:
        probs = lnp_trunc(x_vals, mu, sigma)
        if np.any(probs <= 0):
            return 1e10
        return -np.sum(f_vals * np.log(probs))
    except Exception:
        return 1e10


# ------------------------------------------------------------
# Model 3: Poisson–Inverse Gaussian
#
#   Lambda ~ InvGaussian(mu, phi)
#   p(x; mu, phi) = E_{Lambda}[Pois(x; Lambda)]
#
#   Computed exactly via the Taylor-coefficient recurrence of the PGF:
#
#     G(z) = exp((phi/mu)(1 - sqrt(1 + 2*mu^2*(1-z)/phi)))
#
#   Define c[x] = x! * p(x) = G^(x)(0).  Leibniz rule on G'(z) = G(z)*mu/F(z)
#   (where F(z) = sqrt(1+2*mu^2*(1-z)/phi)) yields:
#
#     c[x+1] = (mu/F0) * sum_{k=0}^{x} C(x,k) * c[k] * (2(x-k)-1)!! * r^{x-k}
#
#   with  F0 = sqrt(1 + 2*mu^2/phi)  and  r = mu^2/(phi + 2*mu^2) in (0, 1).
#   Because r < 1 the series converges and no quadrature is needed.
# ------------------------------------------------------------

def pig_logpmf(x_arr, mu, phi):
    """Log PMF for the Poisson-Inverse-Gaussian distribution."""
    x_arr = np.asarray(x_arr, dtype=int)
    x_max = int(np.max(x_arr))

    F0 = math.sqrt(1.0 + 2.0 * mu**2 / phi)
    log_p = np.empty(x_max + 1)
    log_p[0] = (phi / mu) * (1.0 - F0)

    if x_max >= 1:
        beta  = 1.0 + phi / (2.0 * mu**2)
        gamma = phi / 2.0
        log_p[1] = math.log(mu / F0) + log_p[0]

        for x in range(1, x_max):
            prev = math.log((x - 0.5) / (beta * (x + 1.0))) + log_p[x]
            tail = math.log(gamma / (beta * x * (x + 1.0))) + log_p[x - 1]
            log_p[x + 1] = np.logaddexp(prev, tail)

    return log_p[x_arr]


def pig_pmf(x_arr, mu, phi):
    """Exact Poisson-Inverse-Gaussian PMF (stable recurrence)."""
    return np.exp(pig_logpmf(x_arr, mu, phi))

def pig_trunc(x_arr, mu, phi):
    """Zero-truncated Poisson-Inverse-Gaussian PMF."""
    p0 = pig_pmf(np.array([0]), mu, phi)[0]
    return pig_pmf(x_arr, mu, phi) / (1.0 - p0)

def pig_nll(params, x_vals, f_vals):
    mu, phi = params
    if mu <= 0.01 or phi <= 0.01:
        return 1e10
    try:
        p0 = pig_pmf(np.array([0]), mu, phi)[0]
        log_probs = pig_logpmf(x_vals, mu, phi) - math.log1p(-p0)
        if np.any(~np.isfinite(log_probs)):
            return 1e10
        return -np.sum(f_vals * log_probs)
    except Exception:
        return 1e10


# ============================================================
# 3. FIT ALL THREE MODELS
# ============================================================

def _de(nll_fn, bounds):
    return differential_evolution(
        nll_fn, bounds=bounds, args=(x_vals, f_vals),
        seed=DE_SEED, maxiter=DE_MAXITER, tol=DE_TOL,
    )

print("\nFitting Gamma-Poisson (Negative Binomial) ...")
res_nb       = _de(negbin_nll, BOUNDS_NB)
r_hat, mu_nb = res_nb.x
p0_nb        = (r_hat / (r_hat + mu_nb)) ** r_hat
N_nb         = int(np.floor(S_obs / (1.0 - p0_nb)))

print("Fitting Log-Normal-Poisson ...")
res_ln          = _de(lnp_nll, BOUNDS_LN)
mu_hat, sig_hat = res_ln.x
p0_ln           = lnp_pmf(np.array([0]), mu_hat, sig_hat)[0]
N_ln            = int(np.floor(S_obs / (1.0 - p0_ln)))

print("Fitting Poisson-Inverse Gaussian ...")
res_pig           = _de(pig_nll, BOUNDS_PIG)
mu_pig, phi_pig   = res_pig.x
p0_pig            = pig_pmf(np.array([0]), mu_pig, phi_pig)[0]
N_pig             = int(np.floor(S_obs / (1.0 - p0_pig)))

# ============================================================
# 4. SUMMARY TABLE
# ============================================================

aic_nb  = 4.0 + 2.0 * res_nb.fun
aic_ln  = 4.0 + 2.0 * res_ln.fun
aic_pig = 4.0 + 2.0 * res_pig.fun

print(f"\n{'='*76}")
print(f"{'Model':26s} {'p(0)':>7s} {'N_hat':>9s} {'NLL':>10s} {'AIC':>10s}")
print(f"{'-'*76}")
print(f"{'Gamma-Poisson (NB)':26s} {p0_nb:>7.4f} {N_nb:>9,} {res_nb.fun:>10.1f} {aic_nb:>10.1f}")
print(f"{'Log-Normal-Poisson':26s} {p0_ln:>7.4f} {N_ln:>9,} {res_ln.fun:>10.1f} {aic_ln:>10.1f}")
print(f"{'Poisson-Inv. Gaussian':26s} {p0_pig:>7.4f} {N_pig:>9,} {res_pig.fun:>10.1f} {aic_pig:>10.1f}")
print(f"{'='*76}")
print(f"S_obs = {S_obs:,}  |  N_tokens = {N_tokens:,}")

# ============================================================
# 5. EXPORT — available when this module is imported
# ============================================================

# Fitted parameters and results, keyed by model name.
fitted = {
    "Gamma-Poisson": {
        "params":  (r_hat, mu_nb),
        "p0":      p0_nb,
        "N_hat":   N_nb,
        "nll":     res_nb.fun,
        "aic":     aic_nb,
        "trunc_fn": negbin_trunc,
        "pmf_fn":   negbin_pmf,
    },
    "LogNormal-Poisson": {
        "params":  (mu_hat, sig_hat),
        "p0":      p0_ln,
        "N_hat":   N_ln,
        "nll":     res_ln.fun,
        "aic":     aic_ln,
        "trunc_fn": lnp_trunc,
        "pmf_fn":   lnp_pmf,
    },
    "Poisson-InvGaussian": {
        "params":  (mu_pig, phi_pig),
        "p0":      p0_pig,
        "N_hat":   N_pig,
        "nll":     res_pig.fun,
        "aic":     aic_pig,
        "trunc_fn": pig_trunc,
        "pmf_fn":   pig_pmf,
    },
}
