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

Outputs are saved under `pm_lsh_outputs/`.


