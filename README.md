# PM-LSH-Style Research Reimplementation

This project is a **research-style Python reimplementation** of the main PM-LSH framework described in:

> Zheng et al., **PM-LSH: A Fast and Accurate LSH Framework for High-Dimensional Approximate NN Search**, PVLDB 2020.

It implements the paper's core ideas:
- Gaussian random projections into an `m`-dimensional projected space
- A **PM-tree** over projected points instead of flat hash buckets
- A **chi-square confidence-interval mapping** from original-space radius to projected-space radius
- A **range-expansion query algorithm** for `(c, k)`-ANN search

## Important note

This is **not** the authors' original C++ codebase and should not be described as an official reproduction. It is best described as:

- **faithful Python reimplementation of the PM-LSH framework**, or
- **PM-LSH-inspired ANN system with PM-tree indexing and confidence-interval search**

## Files

- `pmtree.py` — PM-tree metric index with pivot-ring pruning and Euclidean range queries
- `pm_lsh.py` — PM-LSH index, projection logic, confidence interval parameterization, and `(c, k)`-ANN search
- `datasets.py` — MNIST loader and brute-force exact baseline
- `evaluate.py` — experiment runner and plotting

## Install

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Run

```bash
python evaluate.py --train-size 15000 --test-size 2000 --queries 200 --k 10
```

By default this sweeps the **confidence-net width** (`--search-widths 1.0 1.25 1.5 2.0`)
at a fixed projected dimension `m=15`, and saves a CSV, a JSON of the best row, and
three plots under `pm_lsh_outputs/`:
`recall_vs_time.png`, `ratio_vs_time.png`, `ratio_vs_recall.png`.

## Metrics (matched to the paper)

Both accuracy metrics follow the definitions used in the PM-LSH paper for
`(c, k)`-ANN search, computed against a brute-force exact top-`k` baseline:

- **Overall ratio** `= (1/k) · Σ_i Dist(o_i, q) / Dist(o_i*, q)`, where `o_i` is the
  `i`-th nearest object returned and `o_i*` is the exact `i`-th NN (results paired
  by rank), then averaged over queries. Lower is better; `1.0` is exact.
- **Recall@k** `= |returned ∩ exact_kNN| / k`, averaged over queries.

Efficiency is reported as average query time (ms) and the average number of
candidates / distance computations per query.

## Why you may not reproduce the paper's recall out of the box

This is a `(c, k)`-ANN method, and its accuracy is controlled by the **chi-square
confidence net**, not by the approximation ratio `c`:

- At the paper's default `alpha1 = 1/e` (`search_width = 1.0`), a true neighbor
  sitting at the current search radius is captured by the projected range search
  with probability only `1 - 1/e ≈ 0.63`. The search therefore *systematically*
  misses roughly a third of the true neighbors, so **recall saturates around
  0.6–0.8** even though the overall ratio is already excellent (`≈ 1.02`, well
  inside the `c = 1.5` bound). This is the regime the original run landed in.
- **Recall is recovered by widening the confidence net** (`search_width > 1`,
  equivalently a smaller effective `alpha1` / larger `t`), which pulls more of the
  true `k`-NN into the candidate set before the algorithm's `(c, k)` stopping rule
  fires — at the cost of examining more candidates. In a synthetic sanity check,
  width `1.0 → 1.5` moves recall from `~0.70` to `~1.00` and ratio from `1.012` to
  `1.000`, roughly tripling the candidates examined. That accuracy/efficiency
  tradeoff is exactly what the paper's curves report.
- Note the paper also evaluates on much larger datasets (millions of points, up to
  ~960 dimensions); the random-projection concentration is sharper there than on a
  15k-point MNIST subset, so absolute numbers at a fixed width will differ.

Tune with `--search-widths`, `--c`, `--beta-cap`, and `--m` to trace the curve.


