"""
Model definitions for the GNN comparison study.

Four architectures are implemented with a shared interface so they can be
trained and evaluated interchangeably:

    MLPBaseline    – graph-unaware feedforward network (lower bound)
    GCNModel       – spectral convolution with symmetric normalisation
    GraphSAGEModel – inductive learning via neighbourhood mean aggregation
    GATModel       – attention-weighted neighbourhood aggregation

All models accept ``(x, edge_index)`` and return log-softmax class scores
of shape ``[n_nodes, n_classes]``.  The MLP ignores ``edge_index`` by
design, providing a baseline that isolates the contribution of graph
structure to predictive performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


# ── MLP Baseline ─────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """
    Standard feedforward network that treats each node independently.

    Because it has no access to edge connectivity, it cannot capture how
    disease risk propagates between neighbouring farms.  Any GNN that
    outperforms this baseline is demonstrably leveraging graph structure.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.fc3(x), dim=1)


# ── GCN ──────────────────────────────────────────────────────────────────

class GCNModel(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, ICLR 2017).

    Layer-wise propagation rule:
        H^{l+1} = sigma( D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H^{l} W^{l} )

    where A_tilde = A + I (adjacency with self-loops) and D_tilde is its
    degree matrix.  The symmetric normalisation prevents feature magnitudes
    from growing with node degree.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = bn(F.relu(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


# ── GraphSAGE ────────────────────────────────────────────────────────────

class GraphSAGEModel(nn.Module):
    """
    GraphSAGE — inductive representation learning (Hamilton et al., NeurIPS 2017).

    Mean aggregation variant:
        h_v^{l+1} = sigma( W * CONCAT(h_v^{l}, MEAN_{u in N(v)} h_u^{l}) )

    Unlike GCN, this architecture can generalise to entirely unseen nodes
    at inference time because it learns an aggregation function rather
    than node-specific embeddings.  This makes it suitable for settings
    where new farms are continuously added to the monitoring network.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = bn(F.relu(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


# ── GAT ──────────────────────────────────────────────────────────────────

class GATModel(nn.Module):
    """
    Graph Attention Network (Velickovic et al., ICLR 2018).

    Multi-head attention mechanism:
        h_v^{l+1} = sigma( CONCAT_{k=1..K} SUM_{j in N(v)} alpha_{ij}^k W^k h_j )

    Attention coefficients:
        e_{ij} = LeakyReLU( a^T [W h_i || W h_j] )
        alpha_{ij} = softmax_j(e_{ij})

    We use K=4 heads in hidden layers and a single averaging head in the
    output layer.  The multi-head setup stabilises training by letting
    different heads attend to different structural patterns.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden_channels // heads,
                    heads=heads, dropout=dropout)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels // heads,
                        heads=heads, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GATConv(hidden_channels, out_channels,
                    heads=1, concat=False, dropout=dropout)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = bn(F.elu(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


# ── Utilities ────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_registry(in_channels: int, hidden: int, out_channels: int) -> dict:
    """
    Convenience factory that returns all four models keyed by name.
    GNN models use 2-layer architectures for this graph size.
    """
    return {
        "MLP (Baseline)": MLPBaseline(in_channels, hidden, out_channels),
        "GCN": GCNModel(in_channels, hidden, out_channels, num_layers=2),
        "GraphSAGE": GraphSAGEModel(in_channels, hidden, out_channels, num_layers=2),
        "GAT": GATModel(in_channels, hidden, out_channels, num_layers=2),
    }


if __name__ == "__main__":
    IN, HIDDEN, OUT = 9, 128, 3
    registry = build_model_registry(IN, HIDDEN, OUT)

    print("Parameter counts:")
    for name, model in registry.items():
        print(f"  {name:<18s}  {count_parameters(model):>8,}")
