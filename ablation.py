"""
Ablation study on GCN architecture choices.

Systematically varies three design decisions to measure their individual
contribution to performance:

    (a) Network depth   — 1, 2, 3, 4 graph convolution layers
    (b) Aggregation fn  — GCN (symmetric normalisation) vs SAGE (mean pool)
    (c) Residual links  — with and without skip connections

Each variant is trained from scratch on the same graph using the same
optimiser settings as the main experiment.  Results are saved as a CSV
table and a grouped bar chart.

Outputs:
    results/tables/ablation_results.csv
    results/figures/ablation_study.png
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator import generate_agricultural_graph
from train import train_model


# ── Configurable GCN variant ────────────────────────────────────────────

class GCNVariant(nn.Module):
    """
    Flexible graph convolution network that supports different depths,
    aggregation schemes, and optional residual connections.

    When ``use_residual=True`` a learnable linear projection aligns the
    input dimension to the hidden dimension so that skip connections can
    be applied even when the two differ.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        n_layers: int = 2,
        aggregation: str = "gcn",
        use_residual: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.dropout = dropout

        Conv = GCNConv if aggregation == "gcn" else SAGEConv

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(Conv(in_channels, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))

        for _ in range(max(0, n_layers - 2)):
            self.convs.append(Conv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        if n_layers >= 2:
            self.convs.append(Conv(hidden, out_channels))
        self.head = nn.Linear(hidden, out_channels) if n_layers == 1 else None

        self.res_proj = (
            nn.Linear(in_channels, hidden, bias=False)
            if use_residual and in_channels != hidden
            else None
        )

    def forward(self, x, edge_index):
        # Single-layer variant: one conv + linear head
        if self.head is not None:
            x = F.relu(self.bns[0](self.convs[0](x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return F.log_softmax(self.head(x), dim=1)

        # Multi-layer variant
        for idx, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            h = F.relu(bn(conv(x, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_residual:
                if idx == 0 and self.res_proj is not None:
                    x = h + self.res_proj(x)
                elif h.shape == x.shape:
                    x = h + x
                else:
                    x = h
            else:
                x = h

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


# ── Run the ablation ─────────────────────────────────────────────────────

def run_ablation() -> pd.DataFrame:
    """Train all ablation variants and save results + bar chart."""
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print("=" * 60)
    print("Ablation Study: Depth, Aggregation, Residuals")
    print("=" * 60)

    data = generate_agricultural_graph(n_farms=500)
    in_dim = data.num_node_features
    out_dim = 3
    hidden = 128

    variants = {
        "GCN-2L (base)":     {"n_layers": 2, "aggregation": "gcn",  "use_residual": False},
        "GCN-1L":            {"n_layers": 1, "aggregation": "gcn",  "use_residual": False},
        "GCN-3L":            {"n_layers": 3, "aggregation": "gcn",  "use_residual": False},
        "GCN-4L":            {"n_layers": 4, "aggregation": "gcn",  "use_residual": False},
        "SAGE-2L":           {"n_layers": 2, "aggregation": "sage", "use_residual": False},
        "GCN-2L + Residual": {"n_layers": 2, "aggregation": "gcn",  "use_residual": True},
    }

    rows = []
    for name, cfg in variants.items():
        model = GCNVariant(in_dim, hidden, out_dim, **cfg)
        print(f"\n  Training: {name} ...")
        res = train_model(model, data, epochs=300, patience=40)
        print(f"    F1 = {res['test_f1'] * 100:.2f}%  |  "
              f"Acc = {res['test_acc'] * 100:.2f}%")
        rows.append({
            "Variant": name,
            "Test F1 (%)": f"{res['test_f1'] * 100:.2f}",
            "Test Acc (%)": f"{res['test_acc'] * 100:.2f}",
            "Best Epoch": res["best_epoch"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/ablation_results.csv", index=False)

    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    # ── Bar chart ────────────────────────────────────────────────────
    f1_values = [float(r["Test F1 (%)"]) for r in rows]
    names = [r["Variant"] for r in rows]
    base_f1 = f1_values[0]

    colours = [
        "#e74c3c" if n == "GCN-2L (base)" else "#3498db" for n in names
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, f1_values, color=colours, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, f1_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.axhline(
        y=base_f1, color="#e74c3c", linestyle="--", alpha=0.6,
        label=f"Base config F1 = {base_f1:.1f}%",
    )
    ax.set_title(
        "Ablation: Effect of Depth, Aggregation, and Residuals on Macro F1",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("GNN Variant")
    ax.set_ylabel("Macro F1 (%)")
    ax.set_ylim(max(0, min(f1_values) - 8), min(100, max(f1_values) + 10))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig("results/figures/ablation_study.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → results/figures/ablation_study.png")

    return df


if __name__ == "__main__":
    run_ablation()
