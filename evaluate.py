from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import load_mnist, brute_force_topk, FastBruteForce
from pm_lsh import PMLSH


def overall_ratio(approx_dists: np.ndarray, exact_dists: np.ndarray) -> float:
    return float(np.mean(approx_dists / (exact_dists + 1e-12)))


def evaluate_one_config(
    X_train,
    y_train,
    X_test,
    y_test,
    m,
    n_pivots,
    c,
    k,
    n_queries,
    random_state,
    beta_cap,
    rmin_quantile,
):
    model = PMLSH(
        m=m,
        n_pivots=n_pivots,
        c=c,
        random_state=random_state,
        beta_cap=beta_cap,
        rmin_quantile=rmin_quantile,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    build_time = time.perf_counter() - t0

    approx_ids = []
    approx_dists = []
    exact_ids = []
    exact_dists = []
    same_label_hits = 0
    total_candidates = 0
    total_tree_dcs = 0
    total_expansions = 0
    empty_results = 0

    fast_bf = FastBruteForce(X_train)
    model.query_ckann(X_test[0], k=k)  # warm-up
    fast_bf.query(X_test[0], k=k)

    query_time = 0.0
    brute_time = 0.0
    fast_brute_time = 0.0
    for q, yq in zip(X_test[:n_queries], y_test[:n_queries]):
        t0 = time.perf_counter()
        ids, dists, stats = model.query_ckann(q, k=k)
        query_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        bf_ids, bf_dists = brute_force_topk(X_train, q, k=k)
        brute_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        fast_bf.query(q, k=k)
        fast_brute_time += time.perf_counter() - t0

        approx_ids.append(ids)
        approx_dists.append(dists)
        exact_ids.append(bf_ids)
        exact_dists.append(bf_dists)

        total_candidates += stats.candidates_found
        total_tree_dcs += stats.distance_computations
        total_expansions += stats.expansions

        if len(ids) == 0:
            empty_results += 1
        elif y_train[ids[0]] == yq:
            same_label_hits += 1

    recall_vals = []
    ratio_vals = []
    for a_ids, e_ids, a_d, e_d in zip(approx_ids, exact_ids, approx_dists, exact_dists):
        recall_vals.append(len(set(a_ids.tolist()) & set(e_ids.tolist())) / max(len(e_ids), 1))
        if len(a_d) > 0:
            ratio_vals.append(overall_ratio(a_d, e_d))

    return {
        "m": m,
        "n_pivots": n_pivots,
        "c": c,
        "k": k,
        "n_queries": n_queries,
        "build_time_sec": build_time,
        "query_total_time_sec": query_time,
        "avg_query_ms": query_time / n_queries * 1000.0,
        "avg_bruteforce_ms": brute_time / n_queries * 1000.0,
        "avg_fast_bruteforce_ms": fast_brute_time / n_queries * 1000.0,
        "rerank_fraction": (total_candidates / n_queries) / len(X_train),
        "recall": float(np.mean(recall_vals)),
        "overall_ratio": float(np.mean(ratio_vals)) if ratio_vals else float("inf"),
        "same_label_accuracy": same_label_hits / n_queries,
        "avg_candidates": total_candidates / n_queries,
        "avg_tree_distance_computations": total_tree_dcs / n_queries,
        "avg_expansions": total_expansions / n_queries,
        "empty_result_rate": empty_results / n_queries,
        "alpha1": model.alpha1,
        "alpha2": model.alpha2,
        "beta": model.beta,
        "t": model.t,
        "rmin_quantile": model.rmin_quantile,
    }


def plot_tradeoff(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["avg_query_ms"], df["recall"])
    for _, row in df.iterrows():
        plt.annotate(f"m={int(row['m'])},s={int(row['n_pivots'])}", (row["avg_query_ms"], row["recall"]), fontsize=8)
    plt.xlabel("Average Query Time (ms)")
    plt.ylabel("Recall")
    plt.title("PM-LSH Recall vs Query Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recall_vs_time.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["avg_query_ms"], df["overall_ratio"])
    for _, row in df.iterrows():
        plt.annotate(f"m={int(row['m'])},s={int(row['n_pivots'])}", (row["avg_query_ms"], row["overall_ratio"]), fontsize=8)
    plt.xlabel("Average Query Time (ms)")
    plt.ylabel("Overall Ratio")
    plt.title("PM-LSH Overall Ratio vs Query Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratio_vs_time.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=15000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-cap", type=float, default=0.10)
    parser.add_argument("--rmin-quantile", type=float, default=0.35)
    parser.add_argument("--out-dir", type=str, default="pm_lsh_outputs")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_mnist(
        train_size=args.train_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    configs: List[Dict[str, object]] = []
    for m in [10, 15, 20]:
        for s in [3, 5, 7]:
            print(f"Running config: m={m}, pivots={s}")
            row = evaluate_one_config(
                X_train,
                y_train,
                X_test,
                y_test,
                m=m,
                n_pivots=s,
                c=1.5,
                k=args.k,
                n_queries=args.queries,
                random_state=args.seed,
                beta_cap=args.beta_cap,
                rmin_quantile=args.rmin_quantile,
            )
            print(json.dumps(row, indent=2))
            configs.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(configs).sort_values(
        ["recall", "overall_ratio", "avg_candidates", "avg_query_ms"],
        ascending=[False, True, True, True],
    )
    csv_path = os.path.join(args.out_dir, "pm_lsh_results.csv")
    json_path = os.path.join(args.out_dir, "pm_lsh_best.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.iloc[0].to_dict(), f, indent=2)
    plot_tradeoff(df, args.out_dir)

    print("\nTop configurations:")
    print(
        df[
            [
                "m",
                "n_pivots",
                "avg_query_ms",
                "avg_bruteforce_ms",
                "avg_fast_bruteforce_ms",
                "recall",
                "overall_ratio",
                "avg_candidates",
                "rerank_fraction",
                "avg_expansions",
                "beta",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
