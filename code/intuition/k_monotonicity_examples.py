"""
Create a conceptual figure illustrating discrete k-monotonicity.

The curves in this figure are illustrative examples, not fitted empirical data.
They are meant to show that larger k imposes stronger alternating-sign
constraints on higher-order finite differences:

    (-1)^a Delta^a p(j) >= 0, for a = 1, ..., k.

Finite-k examples for k >= 2 use one k-monotone spline Q_l^k on j=0,...,l.
The k=1 example is a hand-built decreasing but visibly irregular PMF. The
k=infinity example is a geometric PMF, which is completely monotone.

The plotted distributions are:
    k=1       hand-built decreasing PMF
    k=2       k-monotone spline Q_15^2
    k=4       k-monotone spline Q_15^4
    k=infty   geometric PMF with alpha=0.24, truncated to j=0,...,15

Run from the repository root:
    python code/intuition/k_monotonicity_examples.py

Output:
    code/intuition/k_monotonicity_examples.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_PATH = Path(__file__).with_name("k_monotonicity_examples.png")
J_MAX = 15
J = np.arange(J_MAX + 1)


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize nonnegative values to sum to one."""
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        raise ValueError("Cannot normalize values with non-positive sum.")
    return values / total


def k_monotone_spline(k: int, ell: int = J_MAX) -> np.ndarray:
    """
    Return the finite k-monotone spline Q_ell^k(j), j=0,...,ell.

    Q_ell^k(j) = C(ell - j + k - 1, k - 1) / C(ell + k, ell),
    for 0 <= j <= ell.
    """
    denominator = math.comb(ell + k, ell)
    values = [
        math.comb(ell - int(j) + k - 1, k - 1) / denominator
        for j in range(ell + 1)
    ]
    return np.array(values, dtype=float)


def geometric_pmf(alpha: float = 0.24) -> np.ndarray:
    """
    Return a truncated geometric PMF on j=0,...,J_MAX.

    A geometric PMF is completely monotone. The truncation is renormalized
    only for plotting on the finite display window.
    """
    return normalize(alpha * (1.0 - alpha) ** J)


def first_order_irregular_pmf() -> np.ndarray:
    """
    Return a decreasing but irregular PMF.

    This is 1-monotone because p(j) >= p(j+1), but it is intentionally not a
    smooth convex sequence, so it visually contrasts with higher-k examples.
    """
    values = np.array(
        [
            0.180,
            0.150,
            0.142,
            0.105,
            0.097,
            0.066,
            0.064,
            0.045,
            0.043,
            0.031,
            0.029,
            0.020,
            0.019,
            0.005,
            0.004,
            0.002,
        ]
    )
    return normalize(values)


def assert_decreasing(values: np.ndarray) -> None:
    """Sanity check for the k=1 illustrative curve."""
    if np.any(np.diff(values) > 1e-12):
        raise AssertionError("The k=1 example must be non-increasing.")


def make_figure() -> None:
    """Create and save the k-monotonicity example figure."""
    curves = [
        (r"$k=1$", first_order_irregular_pmf(), "#4C4C4C"),
        (r"$k=2$", k_monotone_spline(2), "#2F6B9A"),
        (r"$k=4$", k_monotone_spline(4), "#C65A1E"),
        (r"$k=\infty$", geometric_pmf(), "#4F8A5B"),
    ]

    assert_decreasing(curves[0][1])

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for label, values, color in curves:
        ax.plot(
            J,
            values,
            marker="o",
            markersize=4.2,
            linewidth=1.7,
            color=color,
            label=label,
        )

    ax.set_xlabel(r"abundance $j$", fontsize=11)
    ax.set_ylabel(r"probability $p(j)$", fontsize=11)
    ax.set_title(
        r"Illustrative PMFs Under Increasing $k$-Monotonicity",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlim(-0.3, J_MAX + 0.3)
    ax.set_ylim(0, 0.22)
    ax.set_xticks(np.arange(0, J_MAX + 1, 3))
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(
        0.98,
        0.58,
        "Illustrative distributions:\n"
        r"$k=1$: decreasing PMF" "\n"
        r"$k=2$: spline $Q_{15}^{2}$" "\n"
        r"$k=4$: spline $Q_{15}^{4}$" "\n"
        r"$k=\infty$: geometric PMF",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dddddd"},
    )

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_figure()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
