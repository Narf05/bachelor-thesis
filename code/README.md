# Code

## Folder structure

```
code/
├── estimators/   # Pure statistical implementations (no I/O, no plotting)
│   ├── coverage.py   — Turing and Chao-Jost sample coverage estimators
│   ├── classical.py  — Good-Turing, Chao1, iChao1, ACE, Jackknife 1 & 2
│   └── breakaway.py  — Frequency-ratio regression estimator
│
├── analysis/     # Scripts that load data and apply estimators
│
└── plots/        # Scripts that generate figures from results
```

## Estimator input convention

All estimators accept a `freq_counts` dict:

```python
freq_counts = {k: f_k}   # f_k = number of species seen exactly k times
```

Example:

```python
from estimators import chao1, ace, jackknife1

counts = {1: 120, 2: 40, 3: 18, 4: 9, 5: 5}
print(chao1(counts))
print(ace(counts, k_cutoff=10))
print(jackknife1(counts))
```
