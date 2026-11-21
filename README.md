# PM-LSH in Python

A from-scratch Python reimplementation of **PM-LSH** (Zheng et al., *PM-LSH: A Fast and Accurate LSH Framework for High-Dimensional Approximate Nearest Neighbor Search*, PVLDB 2020), evaluated on MNIST.

This is **not** the authors' C++ code. It is a clean-room implementation of the paper's core ideas, written to understand the algorithm and measure its accuracy/cost trade-offs.

## What it implements

- **Gaussian random projection** from 784 dimensions down to `m` dimensions.
- **PM-tree** (`pmtree.py`): a metric index over the projected points, with ball-partition nodes and pivot hyper-ring pruning, supporting Euclidean range queries.
- **Chi-square confidence-interval mapping** (`pm_lsh.py`): maps an original-space radius to a projected-space radius, with `alpha2` solved in closed form from the paper's Equation 10 using `scipy.stats.chi2`.
- **(c, k)-ANN query**: range expansion in projected space, then exact reranking of the candidates in the original space.

## Results

MNIST, 15,000 indexed vectors, 200 queries, k = 10, c = 1.5, seed 42. Recall is recall@10 against exact brute-force search. Overall ratio is the mean of approximate / exact neighbor distance (1.0 is perfect).

| m | Recall@10 | Overall ratio | Candidates reranked | Fraction of dataset reranked |
|---|-----------|---------------|---------------------|------------------------------|
| 10 | 0.703 | 1.031 | 693 | 4.6% |
| 15 | 0.731 | 1.028 | 473 | 3.2% |
| 20 | **0.778** | **1.021** | 434 | 2.9% |

The best configuration (m = 20) returns 78% of the true top-10 neighbors, and the distances it returns are within 2.1% of the exact ones on average, while computing exact distances for only 2.9% of the dataset. Full per-run numbers are in [`pm_lsh_outputs/pm_lsh_results.csv`](pm_lsh_outputs/pm_lsh_results.csv).

## Honest limitations

- **It is slower than brute force here.** Average query time is roughly 90-165 ms for PM-LSH versus roughly 35 ms for a vectorized NumPy brute-force scan of 15K vectors. The tree is pure Python, so per-node overhead dominates, and at this scale (15K points) a single matrix operation is hard to beat. The savings in exact distance computations would only turn into wall-clock speedups with a compiled tree and a much larger dataset.
- **Pivot count did not change recall.** Pivots only affect pruning, not which points qualify, so recall is identical for 3, 5 and 7 pivots; only query time differs, and the differences are small and noisy.
- **Small scale.** Only MNIST at 15K points was tested, not million-scale data.
- The "same-label accuracy" column in the CSV (fraction of queries whose top result shares the query's digit label) is a sanity check, not a standard ANN metric.

## Files

| File | Purpose |
|------|---------|
| `pmtree.py` | PM-tree index with pivot-ring pruning and Euclidean range queries |
| `pm_lsh.py` | PM-LSH index: projection, confidence-interval parameters, (c, k)-ANN search |
| `datasets.py` | MNIST loader and brute-force exact baseline |
| `evaluate.py` | Sweeps `m` and pivot count, times PM-LSH and brute force separately, writes CSV/JSON/plots |

## Run

```bash
pip install -r requirements.txt
python evaluate.py --train-size 15000 --test-size 2000 --queries 200 --k 10
```

Outputs go to `pm_lsh_outputs/`.

![Recall vs query time](pm_lsh_outputs/recall_vs_time.png)
![Overall ratio vs query time](pm_lsh_outputs/ratio_vs_time.png)
