"""
Scalability analysis across graph sizes.

Measures how training time and predictive performance change as the
agricultural graph grows from 200 to 5000 nodes.  We compare GraphSAGE
(the best-performing GNN from the main experiment) against the MLP
baseline to show that the graph-based advantage persists at scale.

Outputs:
    results/tables/scalability_results.csv
    results/figures/scalability_analysis.png
"""

import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator import generate_agricultural_graph
from models import MLPBaseline, GraphSAGEModel, count_parameters
from train import train_model


GRAPH_SIZES = [200, 500, 1000, 2000, 5000]


def run_scalability_analysis() -> pd.DataFrame:
    """
    Train MLP and GraphSAGE at each graph size and record metrics.
    """
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print("=" * 60)
    print("Scalability Analysis: Graph Size vs Performance")
    print("=" * 60)

    rows = []

    for n in GRAPH_SIZES:
        print(f"\n--- Graph size: {n} nodes ---")
        data = generate_agricultural_graph(n_farms=n, seed=42)
        in_dim = data.num_node_features
        hidden, out = 128, 3

        configs = [
            ("MLP", MLPBaseline(in_dim, hidden, out)),
            ("GraphSAGE", GraphSAGEModel(in_dim, hidden, out, num_layers=2)),
        ]

        for model_name, model in configs:
            print(f"  Training {model_name} ...")
            max_epochs = 300 if n <= 2000 else 200
            res = train_model(model, data, epochs=max_epochs, patience=40, verbose=False)

            row = {
                "Graph Size": n,
                "Model": model_name,
                "Test F1 (%)": round(res["test_f1"] * 100, 2),
                "Test Acc (%)": round(res["test_acc"] * 100, 2),
                "Training Time (s)": round(res["elapsed"], 2),
                "Edges": data.num_edges,
                "Avg Degree": round(data.num_edges / data.num_nodes, 1),
            }
            rows.append(row)
            print(f"    F1={row['Test F1 (%)']:.1f}%  Time={row['Training Time (s)']:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/scalability_results.csv", index=False)

    print("\n" + "=" * 60)
    print("SCALABILITY RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    style = {
        "MLP": ("#95a5a6", "s", "MLP (Baseline)"),
        "GraphSAGE": ("#2ecc71", "o", "GraphSAGE"),
    }

    for model_name, (colour, marker, label) in style.items():
        subset = df[df["Model"] == model_name]
        sizes = subset["Graph Size"].values
        f1s = subset["Test F1 (%)"].values
        times = subset["Training Time (s)"].values

        ax1.plot(sizes, f1s, color=colour, marker=marker, linewidth=2,
                 markersize=8, label=label)
        ax2.plot(sizes, times, color=colour, marker=marker, linewidth=2,
                 markersize=8, label=label)

    ax1.set_title("Test Macro-F1 vs Graph Size", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of Farm Nodes")
    ax1.set_ylabel("Macro F1 (%)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Training Time vs Graph Size", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Number of Farm Nodes")
    ax2.set_ylabel("Wall-Clock Time (s)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Scalability: GraphSAGE vs MLP Across Graph Sizes",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig("results/figures/scalability_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure → results/figures/scalability_analysis.png")

    return df


if __name__ == "__main__":
    run_scalability_analysis()
