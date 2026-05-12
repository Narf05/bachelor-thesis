"""
Goodness-of-fit and plots for the parametric mixture models.

This script is NOT intended for thesis inclusion.
It imports the fitted models from mixture_models.py and produces:
  - mixture_shakespeare.png / .pdf  (two-panel bar chart + GOF table)

Run after mixture_models.py has been imported (this file does that automatically).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Import fits — this also loads the data and runs the optimisers.
from mixture_models import (
    fitted, freq_of_freq, S_obs, N_tokens,
    negbin_pmf, lnp_pmf, pig_pmf,
)

_HERE = Path(__file__).resolve().parent

# ============================================================
# GOODNESS-OF-FIT  (chi-squared on pooled bins)
# ============================================================
# Pool all bins x >= CHI2_POOL into a single tail bin.
CHI2_POOL = 11
x_gof = np.arange(1, CHI2_POOL + 1)
obs_c = np.array([freq_of_freq.get(x, 0) for x in x_gof])
obs_c[-1] += sum(
    freq_of_freq.get(x, 0)
    for x in range(CHI2_POOL + 1, max(freq_of_freq.keys()) + 1)
)
df_gof = len(x_gof) - 2 - 1   # bins - params - 1

def chi2_stat(trunc_fn, params):
    exp_c     = trunc_fn(x_gof, *params) * S_obs
    exp_c[-1] = S_obs - np.sum(exp_c[:-1])
    return float(np.sum((obs_c - exp_c) ** 2 / (np.abs(exp_c) + 1e-10)))

for name, m in fitted.items():
    m["chi2"] = chi2_stat(m["trunc_fn"], m["params"])

# ============================================================
# COLOURS
# ============================================================
C_NB  = '#1565C0'   # blue
C_LN  = '#E64A19'   # orange-red
C_PIG = '#2E7D32'   # green
C_OBS = '#424242'   # dark grey

model_order  = ["Gamma-Poisson", "LogNormal-Poisson", "Poisson-InvGaussian"]
label_short  = ["Gamma–Poisson", "LogNormal–Poisson", "Poisson–Inv.Gaussian"]
colors       = [C_NB, C_LN, C_PIG]

# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(15, 7.8))
gs  = fig.add_gridspec(
    2, 2,
    height_ratios=[5, 1.7],
    hspace=0.55, wspace=0.32,
    left=0.07, right=0.97, top=0.91, bottom=0.04,
)
ax_left  = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis('off')

# ---- LEFT: zero-truncated fits (observed region x >= 1) ----
xp    = np.arange(1, 11)
obs_p = np.array([freq_of_freq.get(x, 0) for x in xp]) / S_obs
w     = 0.20

ax_left.bar(xp - 1.5 * w, obs_p, w, label='Observed',
            color=C_OBS, alpha=0.85, edgecolor='white', lw=0.5)

for j, (name, lbl, col) in enumerate(zip(model_order, label_short, colors)):
    m      = fitted[name]
    fitted_p = m["trunc_fn"](xp, *m["params"])
    ax_left.bar(
        xp + (j - 0.5) * w, fitted_p, w,
        label=f'{lbl}  ($\\chi^2$={m["chi2"]:.0f})',
        color=col, alpha=0.85, edgecolor='white', lw=0.5,
    )

ax_left.set_xlabel('Word abundance $x$', fontsize=12)
ax_left.set_ylabel('Proportion of word types', fontsize=12)
ax_left.set_title('Observed region ($x \\geq 1$)', fontsize=12,
                  fontweight='bold', pad=10)
ax_left.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
ax_left.set_xticks(xp)

# ---- RIGHT: full PMF including x = 0 ----
xf = np.arange(0, 11)
w2 = 0.25

for j, (name, lbl, col) in enumerate(zip(model_order, label_short, colors)):
    m     = fitted[name]
    pmf_f = m["pmf_fn"](xf, *m["params"])
    ax_right.bar(
        xf + (j - 1) * w2, pmf_f, w2,
        label=f'{lbl}\n$p(0)={m["p0"]:.3f}$, $\\hat{{N}}={m["N_hat"]:,}$',
        color=col, alpha=0.85, edgecolor='white', lw=0.5,
    )

ax_right.axvspan(-0.5, 0.5, alpha=0.08, color='red', zorder=0)
ymax = max(fitted[n]["pmf_fn"](np.array([0]), *fitted[n]["params"])[0]
           for n in model_order)
ax_right.annotate(
    'Unobserved\nregion', xy=(0, ymax * 1.05), fontsize=10,
    ha='center', color='#B71C1C', fontweight='bold',
)
ax_right.set_xlabel('Word abundance $x$', fontsize=12)
ax_right.set_ylabel('$p(x;\\,\\hat{G})$  (full mixture PMF)', fontsize=12)
ax_right.set_title('Full mixture PMF',
                   fontsize=12, fontweight='bold', pad=10)
ax_right.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
ax_right.set_xticks(xf)

fig.suptitle(
    f'Shakespeare complete works — {S_obs:,} observed types '
    f'({N_tokens:,} tokens)',
    fontsize=10.5, style='italic', color='#333', y=0.985,
)

# ---- GOF TABLE ----
col_labels = [
    'Model', '$p(0)$', '$\\hat{N}$', 'NLL', 'AIC',
    f'$\\chi^2$ (df={df_gof})',
]
table_data = [
    [lbl,
     f'{fitted[name]["p0"]:.4f}',
     f'{fitted[name]["N_hat"]:,}',
     f'{fitted[name]["nll"]:.1f}',
     f'{fitted[name]["aic"]:.1f}',
     f'{fitted[name]["chi2"]:.1f}']
    for name, lbl in zip(model_order, label_short)
]

tbl = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.0, 0.0, 1.0, 1.0],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if row == 0:
        cell.set_facecolor('#e8e8e8')
        cell.set_text_props(fontweight='bold')
    else:
        cell.set_facecolor(colors[row - 1] + '18')
    if col == 0:
        cell.set_width(0.22)

ax_table.set_title(
    f'Goodness-of-fit summary  '
    f'($\\chi^2$ bins pooled at $x \\geq {CHI2_POOL}$, df={df_gof};  '
    f'AIC = 2k + 2NLL,  k=2 for all models)',
    fontsize=9.5, pad=4, color='#444',
)

# ============================================================
# SAVE
# ============================================================
out = _HERE / 'mixture_shakespeare'
plt.savefig(out.with_suffix('.png'), dpi=180, bbox_inches='tight')
plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
print(f"\nFigure saved to {out}.png / .pdf")
