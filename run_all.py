"""
Master pipeline — runs the full experimental workflow in order.

Steps:
    1.  Generate the synthetic agricultural graph and save a visualisation
    2.  Train all four models (MLP, GCN, GraphSAGE, GAT) and save results
    3.  Run the ablation study on GCN depth and aggregation
    4.  Run the scalability analysis across graph sizes
    5.  Print a summary of all output artefacts

Usage:
    python run_all.py

Expected runtime: ~5-15 minutes on CPU depending on hardware.
"""

import os
import time

BANNER = """
============================================================
  GNN-Based Agricultural Disease Propagation Modelling
  Ajinkya Awari  |  2026
============================================================
"""

print(BANNER)

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

overall_start = time.time()

# ── Step 1: Data generation and graph visualisation ──────────────────
print("[STEP 1/4]  Generating dataset and graph visualisation ...")

from data_generator import generate_agricultural_graph, visualize_graph

data = generate_agricultural_graph()
visualize_graph(
    data,
    title="Agricultural Disease Spread — 500 Farm Graph",
    save_path="results/figures/graph_visualization.png",
)
print("           Done.\n")


# ── Step 2: Model comparison ─────────────────────────────────────────
print("[STEP 2/4]  Training MLP, GCN, GraphSAGE, GAT ...")

from train import run_all_experiments

all_results, _ = run_all_experiments()
print("           Done.\n")


# ── Step 3: Ablation study ───────────────────────────────────────────
print("[STEP 3/4]  Running ablation study ...")

from ablation import run_ablation

run_ablation()
print("           Done.\n")


# ── Step 4: Scalability analysis ─────────────────────────────────────
print("[STEP 4/4]  Running scalability analysis ...")

from scalability import run_scalability_analysis

run_scalability_analysis()
print("           Done.\n")


# ── Summary ──────────────────────────────────────────────────────────
elapsed = time.time() - overall_start
minutes, seconds = divmod(int(elapsed), 60)

print("=" * 60)
print("ALL EXPERIMENTS COMPLETE")
print(f"Total runtime: {minutes}m {seconds}s")
print("=" * 60)
print()
print("Output files:")
print("  results/figures/graph_visualization.png   — farm graph by disease class")
print("  results/figures/convergence_curves.png    — loss and F1 over epochs")
print("  results/figures/confusion_matrices.png    — per-model confusion matrices")
print("  results/figures/ablation_study.png        — ablation bar chart")
print("  results/figures/scalability_analysis.png  — F1 and runtime vs graph size")
print("  results/tables/main_results.csv           — model comparison table")
print("  results/tables/ablation_results.csv       — ablation table")
print("  results/tables/scalability_results.csv    — scalability table")
print()
print("Next step: push to GitHub (see instructions in the guide).")
print("=" * 60)
