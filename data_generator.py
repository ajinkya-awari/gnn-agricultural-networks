"""
Synthetic agricultural graph dataset generator.

Constructs a spatial graph of farm locations where nodes carry agronomic
features and edges encode geographic proximity.  Disease labels follow a
spatially-correlated spread model influenced by environmental conditions,
reflecting how real pathogen dispersal depends on moisture, rainfall, and
protective factors like pesticide coverage.

Node features (9 per farm):
    crop_type       4-d one-hot  (wheat / rice / corn / sugarcane)
    soil_moisture   continuous   [0, 1]
    temperature     normalised   [0, 1]  — mapped from ~15-40°C
    rainfall        continuous   [0, 1]
    farm_size       categorical  {0.2, 0.5, 1.0}
    pesticide_use   categorical  {0.0, 0.5, 1.0}

Edge construction:
    Euclidean proximity within threshold delta, capped at k=8 nearest
    neighbours per node to bound the degree distribution.

Label generation:
    Multiple outbreak epicentres emit exponential-decay infection probability
    modulated by per-node soil moisture, rainfall, and pesticide factors.
    Percentile-based thresholds produce a realistic three-class distribution
    suitable for multi-class evaluation.
"""

import os

import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Graph construction ───────────────────────────────────────────────────

def generate_agricultural_graph(
    n_farms: int = 500,
    seed: int = 42,
    proximity_threshold: float = 0.12,
    max_neighbours: int = 8,
    disease_spread_radius: float = 0.15,
    n_disease_seeds: int = 25,
) -> Data:
    """
    Build a PyTorch Geometric ``Data`` object representing a regional
    agricultural network.

    Parameters
    ----------
    n_farms : int
        Number of farm nodes in the graph.
    seed : int
        Random seed for full reproducibility.
    proximity_threshold : float
        Maximum Euclidean distance between two farms for an edge to exist.
    max_neighbours : int
        Upper bound on the degree of any single node.
    disease_spread_radius : float
        Length-scale parameter for the exponential infection kernel.
    n_disease_seeds : int
        Number of initial outbreak locations.

    Returns
    -------
    data : torch_geometric.data.Data
        Attributes: x, edge_index, edge_attr, y, pos,
        train_mask, val_mask, test_mask.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # ── Spatial layout ───────────────────────────────────────────────
    locations = rng.uniform(0.0, 1.0, size=(n_farms, 2))

    # ── Node features ────────────────────────────────────────────────
    crop_ids = rng.randint(0, 4, size=n_farms)
    crop_onehot = np.eye(4)[crop_ids]

    soil_moisture = rng.beta(2.0, 5.0, size=n_farms)

    raw_temp = rng.normal(28.0, 5.0, size=n_farms)
    temperature = np.clip((raw_temp - 15.0) / 25.0, 0.0, 1.0)

    rainfall = np.clip(rng.exponential(0.3, size=n_farms), 0.0, 1.0)

    farm_size = rng.choice([0.2, 0.5, 1.0], size=n_farms, p=[0.5, 0.3, 0.2])
    pesticide = rng.choice([0.0, 0.5, 1.0], size=n_farms, p=[0.3, 0.4, 0.3])

    node_features = np.column_stack([
        crop_onehot, soil_moisture, temperature,
        rainfall, farm_size, pesticide,
    ])

    # ── Disease label generation ─────────────────────────────────────
    # The infection model is intentionally spatial: labels depend heavily
    # on proximity to outbreak epicentres.  This means a model that can
    # see neighbour labels/features (i.e. a GNN) has a structural advantage
    # over one that treats each node independently (MLP).
    seed_farms = rng.choice(n_farms, size=n_disease_seeds, replace=False)
    infection_score = np.zeros(n_farms)

    for origin in seed_farms:
        dist = np.linalg.norm(locations - locations[origin], axis=1)
        kernel = np.exp(-dist / disease_spread_radius)
        # Environmental modulation
        kernel *= (1.0 + 1.5 * soil_moisture) * (1.0 + 1.5 * rainfall)
        kernel *= (1.0 - 0.3 * pesticide)
        infection_score += kernel

    # Normalise and discretise via percentile thresholds
    infection_score /= infection_score.max()

    threshold_mild = np.percentile(infection_score, 40)
    threshold_severe = np.percentile(infection_score, 75)

    labels = np.zeros(n_farms, dtype=np.int64)
    labels[infection_score > threshold_mild] = 1
    labels[infection_score > threshold_severe] = 2

    # ── Edge construction ────────────────────────────────────────────
    src_list, dst_list, weight_list = [], [], []

    for i in range(n_farms):
        dists = np.linalg.norm(locations - locations[i], axis=1)
        candidates = np.where((dists > 0) & (dists < proximity_threshold))[0]

        if len(candidates) > max_neighbours:
            closest = np.argsort(dists[candidates])[:max_neighbours]
            candidates = candidates[closest]

        for j in candidates:
            src_list.append(i)
            dst_list.append(j)
            weight_list.append(1.0 / (dists[j] + 1e-6))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(weight_list, dtype=torch.float).unsqueeze(1)

    # ── Train / val / test split (60 / 20 / 20) ─────────────────────
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    pos = torch.tensor(locations, dtype=torch.float)

    perm = torch.randperm(n_farms)
    n_train = int(0.6 * n_farms)
    n_val = int(0.2 * n_farms)

    train_mask = torch.zeros(n_farms, dtype=torch.bool)
    val_mask = torch.zeros(n_farms, dtype=torch.bool)
    test_mask = torch.zeros(n_farms, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, pos=pos,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )

    class_counts = {
        "healthy": int((y == 0).sum()),
        "mild": int((y == 1).sum()),
        "severe": int((y == 2).sum()),
    }
    avg_degree = data.num_edges / data.num_nodes
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"avg degree {avg_degree:.1f}")
    print(f"Features per node: {data.num_node_features}")
    print(f"Class distribution: {class_counts}")

    return data


# ── Visualisation ────────────────────────────────────────────────────────

def visualize_graph(
    data: Data,
    title: str = "Agricultural Disease Spread Graph",
    save_path: str | None = None,
) -> None:
    """
    Plot the farm graph with nodes coloured by disease severity.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    pos = data.pos.numpy()
    labels = data.y.numpy()
    edges = data.edge_index.numpy()

    for k in range(edges.shape[1]):
        i, j = edges[0, k], edges[1, k]
        ax.plot(
            [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
            color="grey", alpha=0.08, linewidth=0.4,
        )

    palette = {"Healthy": "#2ecc71", "Mild": "#f39c12", "Severe": "#e74c3c"}
    for cls, (label_name, colour) in enumerate(palette.items()):
        mask = labels == cls
        ax.scatter(
            pos[mask, 0], pos[mask, 1],
            c=colour, label=label_name, s=22, alpha=0.8, zorder=5,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude (normalised)")
    ax.set_ylabel("Latitude (normalised)")
    ax.legend(fontsize=11, framealpha=0.9, loc="upper right")
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    data = generate_agricultural_graph()
    visualize_graph(data, save_path="results/figures/graph_visualization.png")
    print("Data generation complete.")
