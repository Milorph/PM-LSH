from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import load_mnist, brute_force_topk
from pm_lsh import PMLSH


def overall_ratio(approx_dists: np.ndarray, exact_dists: np.ndarray) -> float:
    """Overall ratio as defined in the PM-LSH paper (Zheng et al., PVLDB 2020).

        ratio = (1/k) * sum_i  Dist(o_i, q) / Dist(o_i*, q)

    where o_i is the i-th nearest object returned by the method and o_i* is the
    exact i-th NN. Both distance arrays are assumed sorted ascending so that
    position i pairs the i-th approximate result with the i-th exact result.

    If the method returns fewer than k results, we pair over the results that do
    exist (a missing result cannot be scored as a finite ratio); the shortfall is
    already captured by recall, so the two metrics together describe accuracy.
    """
    m = min(len(approx_dists), len(exact_dists))
    if m == 0:
        return float("nan")
    a = np.asarray(approx_dists[:m], dtype=np.float64)
    e = np.asarray(exact_dists[:m], dtype=np.float64)
    return float(np.mean(a / (e + 1e-12)))


def recall_at_k(approx_ids, exact_ids) -> float:
    """Recall as defined in the PM-LSH paper: the fraction of the exact k NNs
    that the method returns, i.e. |returned ∩ exact_kNN| / k.
    """
    k = len(exact_ids)
    if k == 0:
        return float("nan")
    return len(set(int(i) for i in approx_ids) & set(int(i) for i in exact_ids)) / k


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
    max_expansions,
    search_width,
):
    model = PMLSH(
        m=m,
        n_pivots=n_pivots,
        c=c,
        search_width=search_width,
        random_state=random_state,
        beta_cap=beta_cap,
        rmin_quantile=rmin_quantile,
        max_expansions=max_expansions,
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

    t0 = time.perf_counter()
    for q, yq in zip(X_test[:n_queries], y_test[:n_queries]):
        ids, dists, stats = model.query_ckann(q, k=k)
        bf_ids, bf_dists = brute_force_topk(X_train, q, k=k)

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
    query_time = time.perf_counter() - t0

    recall_vals = []
    ratio_vals = []
    for a_ids, e_ids, a_d, e_d in zip(approx_ids, exact_ids, approx_dists, exact_dists):
        recall_vals.append(recall_at_k(a_ids, e_ids))
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
        "recall": float(np.mean(recall_vals)),
        "overall_ratio": float(np.mean(ratio_vals)) if ratio_vals else float("inf"),
        "same_label_accuracy": same_label_hits / n_queries,
        "avg_candidates": total_candidates / n_queries,
        "avg_tree_distance_computations": total_tree_dcs / n_queries,
        "avg_expansions": total_expansions / n_queries,
        "empty_result_rate": empty_results / n_queries,
        "search_width": model.search_width,
        "alpha1": model.alpha1,
        "effective_alpha1": model.effective_alpha1,
        "alpha2": model.alpha2,
        "beta": model.beta,
        "t": model.t,
        "effective_t": model.effective_t,
        "rmin_quantile": model.rmin_quantile,
    }


def _annotate(df: pd.DataFrame, x: str, y: str) -> None:
    for _, row in df.iterrows():
        plt.annotate(f"w={row['search_width']:g}", (row[x], row[y]), fontsize=8)


def plot_tradeoff(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = df.sort_values("search_width")

    # Recall vs query time (the paper's accuracy/efficiency tradeoff).
    plt.figure(figsize=(7, 5))
    plt.plot(df["avg_query_ms"], df["recall"], marker="o")
    _annotate(df, "avg_query_ms", "recall")
    plt.xlabel("Average Query Time (ms)")
    plt.ylabel("Recall@k")
    plt.title("PM-LSH Recall vs Query Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recall_vs_time.png"), dpi=200)
    plt.close()

    # Overall ratio vs query time.
    plt.figure(figsize=(7, 5))
    plt.plot(df["avg_query_ms"], df["overall_ratio"], marker="o")
    _annotate(df, "avg_query_ms", "overall_ratio")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="ideal (ratio = 1)")
    plt.xlabel("Average Query Time (ms)")
    plt.ylabel("Overall Ratio")
    plt.title("PM-LSH Overall Ratio vs Query Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratio_vs_time.png"), dpi=200)
    plt.close()

    # Ratio vs recall: both accuracy metrics improve together as c -> 1.
    plt.figure(figsize=(7, 5))
    plt.plot(df["recall"], df["overall_ratio"], marker="o")
    _annotate(df, "recall", "overall_ratio")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    plt.xlabel("Recall@k")
    plt.ylabel("Overall Ratio")
    plt.title("PM-LSH Overall Ratio vs Recall (swept over confidence width)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratio_vs_recall.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=15000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=int, default=15, help="projected dimension (paper default ~15)")
    parser.add_argument("--n-pivots", type=int, default=5)
    parser.add_argument("--beta-cap", type=float, default=0.25,
                        help="max candidate-budget fraction beta; larger => higher recall, slower")
    parser.add_argument("--rmin-quantile", type=float, default=0.35)
    parser.add_argument("--max-expansions", type=int, default=25,
                        help="range-expansion steps")
    parser.add_argument("--c", type=float, default=1.5, help="approximation ratio c (ratio bound)")
    parser.add_argument("--search-widths", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0],
                        help="confidence-net width multipliers to sweep; recall rises with width")
    parser.add_argument("--out-dir", type=str, default="pm_lsh_outputs")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_mnist(
        train_size=args.train_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # Sweep the confidence-net width at a fixed projected dimension m. This is the
    # knob that actually traces the paper's accuracy/efficiency tradeoff: at the
    # paper's default alpha1 = 1/e (width = 1.0) the chi-square net captures a true
    # neighbor at the search radius with only ~1 - 1/e ~ 0.63 probability, so the
    # range search misses ~37% of true NNs and recall saturates around 0.6-0.8.
    # Widening the net (width > 1, i.e. a smaller effective alpha1 / larger t)
    # pushes recall toward ~1.0 and ratio toward 1.0 at the cost of examining more
    # candidates. The approximation ratio c only bounds the ratio and does not move
    # recall, so it is held fixed here.
    configs: List[Dict[str, object]] = []
    for width in args.search_widths:
        print(f"Running config: m={args.m}, pivots={args.n_pivots}, c={args.c}, search_width={width}")
        row = evaluate_one_config(
            X_train,
            y_train,
            X_test,
            y_test,
            m=args.m,
            n_pivots=args.n_pivots,
            c=args.c,
            k=args.k,
            n_queries=args.queries,
            random_state=args.seed,
            beta_cap=args.beta_cap,
            rmin_quantile=args.rmin_quantile,
            max_expansions=args.max_expansions,
            search_width=width,
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

    print("\nResults (sorted by recall):")
    print(
        df[
            [
                "m",
                "search_width",
                "effective_alpha1",
                "avg_query_ms",
                "recall",
                "overall_ratio",
                "avg_candidates",
                "avg_expansions",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
