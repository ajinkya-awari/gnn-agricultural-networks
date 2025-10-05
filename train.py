"""
Training pipeline for the GNN comparison study.

Trains all four architectures (MLP, GCN, GraphSAGE, GAT) on the same
agricultural disease graph and produces:

    results/tables/main_results.csv         — per-model accuracy, F1, params
    results/figures/convergence_curves.png   — loss and val-F1 over epochs
    results/figures/confusion_matrices.png   — per-model confusion matrices

Each model is trained with Adam, a ReduceLROnPlateau scheduler, and early
stopping on validation macro-F1.  Class weights are applied to the loss
function to handle the imbalanced distribution of disease severity levels.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

from data_generator import generate_agricultural_graph
from models import (
    MLPBaseline, GCNModel, GraphSAGEModel, GATModel,
    count_parameters, build_model_registry,
)


# ── Class weight computation ─────────────────────────────────────────────

def compute_class_weights(data) -> torch.Tensor:
    """
    Inverse-frequency weighting so the loss pays more attention to
    underrepresented classes (e.g. severe disease).
    """
    train_labels = data.y[data.train_mask]
    counts = torch.bincount(train_labels, minlength=3).float()
    weights = counts.sum() / (len(counts) * counts)
    return weights


# ── Single training step ─────────────────────────────────────────────────

def train_epoch(model, data, optimizer, class_weights):
    """Run one forward + backward pass on the training split."""
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.nll_loss(
        logits[data.train_mask], data.y[data.train_mask],
        weight=class_weights,
    )
    loss.backward()
    optimizer.step()
    return loss.detach().item()


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(model, data, mask):
    """
    Compute accuracy and macro-F1 on the subset of nodes indicated by
    ``mask``.  Returns ``(accuracy, macro_f1, predictions)``.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[mask].argmax(dim=1)
        targets = data.y[mask]

    acc = float((preds == targets).sum()) / int(mask.sum())
    f1 = f1_score(
        targets.numpy(), preds.numpy(), average="macro", zero_division=0,
    )
    return acc, f1, preds.numpy()


# ── Full training loop ───────────────────────────────────────────────────

def train_model(
    model,
    data,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 400,
    patience: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Train a single model with Adam + learning rate scheduling.

    Early stopping monitors validation macro-F1 and restores the best
    checkpoint before evaluating on the held-out test set.
    """
    class_weights = compute_class_weights(data)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=20, factor=0.5,
    )

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    t0 = time.time()

    for epoch in range(epochs):
        loss = train_epoch(model, data, optimizer, class_weights)
        val_acc, val_f1, _ = evaluate(model, data, data.val_mask)
        scheduler.step(val_f1)

        history["train_loss"].append(loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch} "
                          f"(best val F1 at epoch {best_epoch})")
                break

    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    test_acc, test_f1, test_preds = evaluate(model, data, data.test_mask)

    return {
        "history": history,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_preds": test_preds,
        "best_epoch": best_epoch,
        "elapsed": elapsed,
        "params": count_parameters(model),
    }


# ── Plotting utilities ───────────────────────────────────────────────────

COLOUR_MAP = {
    "MLP (Baseline)": "#95a5a6",
    "GCN": "#3498db",
    "GraphSAGE": "#2ecc71",
    "GAT": "#e74c3c",
}

CLASS_NAMES = ["Healthy", "Mild", "Severe"]


def plot_convergence(all_results: dict, save_path: str) -> None:
    """Side-by-side training loss and validation F1 curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in all_results.items():
        colour = COLOUR_MAP.get(name, "#333333")
        h = res["history"]
        axes[0].plot(h["train_loss"], label=name, color=colour, linewidth=1.8)
        axes[1].plot(h["val_f1"], label=name, color=colour, linewidth=1.8)

    axes[0].set_title("Training Loss Convergence", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted NLL Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Macro-F1", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Optimization Convergence: GNN Architectures vs MLP Baseline",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {save_path}")


def plot_confusion_matrices(all_results: dict, data, save_path: str) -> None:
    """Grid of confusion matrices, one per model."""
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))

    if n_models == 1:
        axes = [axes]

    test_true = data.y[data.test_mask].numpy()

    for ax, (name, res) in zip(axes, all_results.items()):
        cm = confusion_matrix(test_true, res["test_preds"], labels=[0, 1, 2])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        )
        ax.set_title(f"{name}\nF1={res['test_f1']*100:.1f}%", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle(
        "Test-Set Confusion Matrices", fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {save_path}")


# ── Main experiment runner ───────────────────────────────────────────────

def run_all_experiments(n_farms: int = 500) -> tuple:
    """
    Train all four models and save results, convergence plots, and
    confusion matrices.
    """
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print("=" * 60)
    print("Generating agricultural graph dataset ...")
    print("=" * 60)
    data = generate_agricultural_graph(n_farms=n_farms)

    in_dim = data.num_node_features
    hidden_dim = 128
    out_dim = 3

    models = build_model_registry(in_dim, hidden_dim, out_dim)

    all_results = {}
    for name, model in models.items():
        n_params = count_parameters(model)
        print(f"\nTraining {name}  ({n_params:,} parameters) ...")
        res = train_model(model, data)
        all_results[name] = res

        print(f"  Test accuracy : {res['test_acc'] * 100:.2f}%")
        print(f"  Test macro-F1 : {res['test_f1'] * 100:.2f}%")
        print(f"  Wall time     : {res['elapsed']:.1f}s")

    plot_convergence(all_results, "results/figures/convergence_curves.png")
    plot_confusion_matrices(all_results, data, "results/figures/confusion_matrices.png")

    rows = []
    for name, res in all_results.items():
        rows.append({
            "Model": name,
            "Test Accuracy (%)": f"{res['test_acc'] * 100:.2f}",
            "Test F1 (Macro %)": f"{res['test_f1'] * 100:.2f}",
            "Parameters": f"{res['params']:,}",
            "Best Epoch": res["best_epoch"],
            "Training Time (s)": f"{res['elapsed']:.1f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/main_results.csv", index=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSaved → results/tables/main_results.csv")

    return all_results, data


if __name__ == "__main__":
    run_all_experiments()
