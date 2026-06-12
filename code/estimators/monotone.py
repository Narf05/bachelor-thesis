"""
Completely monotone (CM) and k-component geometric mixture estimators.

A completely monotone sequence {p_j}_{j>=0} can be represented as:
    p_j = sum_m w_m * (1 - t_m) * t_m^j   (geometric mixture)
with w_m >= 0, sum w_m = 1, t_m in (0,1).

For k-component model: k atoms on a uniform grid.
For the CM estimator: fine grid (default 25 atoms).

BIC over k identifies the minimal adequate complexity.
The CM limit is the fine-grid model (k -> large).

References: Chee & Wang (2014, 2016), Balabdaoui & Kulagina (2020).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def _basis(grid: np.ndarray, K: int) -> np.ndarray:
    """A[j, m] = (1 - t_m) * t_m^j, shape (K+1, M)."""
    j = np.arange(K + 1)[:, None]
    return (1.0 - grid) * (grid ** j)


def _fit_weights(A: np.ndarray, f_vec: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Maximise  sum_j f_j * log(p_j)  s.t. p = A @ w, w >= 0, sum(w) = 1.
    Uses softmax parameterisation.
    Returns (loglik, w_opt).
    """
    M = A.shape[1]
    mask = f_vec > 0

    def obj(v):
        v = v - v.max()
        w = np.exp(v); w /= w.sum()
        p = np.clip(A @ w, 1e-15, None)
        return -float(np.sum(f_vec[mask] * np.log(p[mask])))

    res = minimize(obj, np.zeros(M), method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-9})
    v = res.x - res.x.max()
    w = np.exp(v); w /= w.sum()
    return -res.fun, w


def _profile_one(fc: dict, S_hat: float, grid: np.ndarray, K: int) -> float:
    S_obs = sum(fc.values())
    f0 = S_hat - S_obs
    if f0 <= 0:
        return -np.inf
    f_vec = np.array([f0] + [fc.get(j, 0) for j in range(1, K + 1)], dtype=float)
    loglik, _ = _fit_weights(_basis(grid, K), f_vec)
    return loglik


def cm_estimator(fc: dict, grid_size: int = 25, n_profile: int = 30,
                 s_max_factor: float = 5.0) -> dict:
    """
    Completely monotone estimator (fine-grid geometric mixture).

    Returns dict: S_hat, f0_hat, S_obs, loglik, S_hat_grid, profile_logliks.
    """
    S_obs = sum(fc.values())
    K = min(max(fc.keys()), 15)
    grid = np.linspace(0.02, 0.98, grid_size)
    S_grid = np.linspace(S_obs + 1, S_obs * s_max_factor, n_profile)
    logliks = np.array([_profile_one(fc, s, grid, K) for s in S_grid])

    valid = np.isfinite(logliks)
    if not valid.any():
        return dict(S_hat=np.nan, f0_hat=np.nan, S_obs=S_obs,
                    loglik=np.nan, S_hat_grid=S_grid, profile_logliks=logliks)
    best = np.argmax(logliks[valid])
    S_hat = S_grid[valid][best]
    return dict(S_hat=S_hat, f0_hat=S_hat - S_obs, S_obs=S_obs,
                loglik=logliks[valid][best],
                S_hat_grid=S_grid, profile_logliks=logliks)


def kmonotone_estimator(fc: dict, k: int, n_profile: int = 20,
                        s_max_factor: float = 5.0) -> dict:
    """
    k-component geometric mixture estimator.

    Returns dict: S_hat, f0_hat, S_obs, loglik, BIC, AIC, k.
    """
    S_obs = sum(fc.values())
    K = min(max(fc.keys()), 15)
    grid = np.linspace(0.05, 0.95, max(k, 2))
    S_grid = np.linspace(S_obs + 1, S_obs * s_max_factor, n_profile)
    logliks = np.array([_profile_one(fc, s, grid, K) for s in S_grid])

    valid = np.isfinite(logliks)
    if not valid.any():
        return dict(S_hat=np.nan, f0_hat=np.nan, S_obs=S_obs,
                    loglik=np.nan, BIC=np.nan, AIC=np.nan, k=k)
    best = np.argmax(logliks[valid])
    S_hat = S_grid[valid][best]
    loglik = logliks[valid][best]
    n_p = max(k - 1, 1)
    return dict(S_hat=S_hat, f0_hat=S_hat - S_obs, S_obs=S_obs,
                loglik=loglik,
                BIC=-2 * loglik + n_p * np.log(max(S_obs, 2)),
                AIC=-2 * loglik + 2 * n_p, k=k)


def select_k(fc: dict, k_max: int = 12, n_profile: int = 20,
             s_max_factor: float = 5.0) -> dict:
    """
    BIC-based k selection + CM estimate.

    Returns dict: k_star, S_hat_kstar, S_hat_cm, results_df.
    """
    import pandas as pd

    rows = [kmonotone_estimator(fc, k=ki, n_profile=n_profile,
                                s_max_factor=s_max_factor)
            for ki in range(1, k_max + 1)]
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["BIC"])

    if valid.empty:
        return dict(k_star=np.nan, S_hat_kstar=np.nan,
                    S_hat_cm=np.nan, results_df=df)

    best_row = valid.loc[valid["BIC"].idxmin()]
    cm = cm_estimator(fc, grid_size=25, n_profile=n_profile,
                      s_max_factor=s_max_factor)
    return dict(k_star=int(best_row["k"]),
                S_hat_kstar=best_row["S_hat"],
                S_hat_cm=cm["S_hat"],
                results_df=df)
