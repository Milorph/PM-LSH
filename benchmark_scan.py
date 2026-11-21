"""
Recall / latency trade-off of PM-LSH with the vectorized projected-space scan,
against an optimized exact brute-force baseline (BLAS + argpartition).

    python benchmark_scan.py --train-size 60000 --queries 500
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from datasets import load_mnist, brute_force_topk, FastBruteForce
from pm_lsh import PMLSH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=60000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="pm_lsh_outputs_60k/scan_vs_bruteforce.csv")
    args = ap.parse_args()

    X_train, X_test, _, _ = load_mnist(train_size=args.train_size, test_size=2000, random_state=args.seed)
    Q = X_test[: args.queries]
    k = args.k

    ground_truth = [brute_force_topk(X_train, q, k=k) for q in Q]

    fast_bf = FastBruteForce(X_train)
    fast_bf.query(Q[0], k)  # warm-up
    t0 = time.perf_counter()
    for q in Q:
        fast_bf.query(q, k)
    bf_ms = (time.perf_counter() - t0) / len(Q) * 1000.0
    print(f"optimized brute force: {bf_ms:.2f} ms/query")

    rows = []
    for m in (20, 50, 64, 100):
        for c in (1.5, 1.3):
            model = PMLSH(m=m, c=c, random_state=args.seed, search="scan").fit(X_train)
            model.query_ckann(Q[0], k)  # warm-up
            t0 = time.perf_counter()
            results = [model.query_ckann(q, k) for q in Q]
            ms = (time.perf_counter() - t0) / len(Q) * 1000.0

            recall = np.mean([len(set(ids.tolist()) & set(g.tolist())) / k
                              for (ids, _, _), (g, _) in zip(results, ground_truth)])
            ratio = np.mean([np.mean(d / (gd + 1e-12))
                             for (_, d, _), (_, gd) in zip(results, ground_truth)])
            rerank = np.mean([s.candidates_found for _, _, s in results]) / len(X_train)
            row = {
                "m": m, "c": c, "recall_at_k": recall, "overall_ratio": ratio,
                "avg_query_ms": ms, "fast_bruteforce_ms": bf_ms,
                "speedup_vs_fast_bruteforce": bf_ms / ms, "rerank_fraction": rerank,
            }
            rows.append(row)
            print(f"m={m:3d} c={c}: recall={recall:.4f} ratio={ratio:.4f} "
                  f"{ms:.2f} ms ({bf_ms / ms:.1f}x vs brute force), rerank {rerank * 100:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
