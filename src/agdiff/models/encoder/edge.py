from math import pi as PI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, Module, ModuleList, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_sparse import coalesce

from agdiff.utils.chem import BOND_TYPES

from ..common import MeanReadout, MultiLayerPerceptron, SumReadout


class GaussianSmearingEdgeEncoder(Module):

    def __init__(self, num_gaussians=64, cutoff=10.0):
        super().__init__()
        # self.NUM_BOND_TYPES = 22
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.rbf = GaussianSmearing(
            start=0.0, stop=cutoff * 2, num_gaussians=num_gaussians
        )  # Larger `stop` to encode more cases
        self.bond_emb = Embedding(100, embedding_dim=num_gaussians)

    @property
    def out_channels(self):
        return self.num_gaussians * 2

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        edge_attr = torch.cat([self.rbf(edge_length), self.bond_emb(edge_type)], dim=1)
        return edge_attr


class MLPEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim=100):
        super(MLPEdgeEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = nn.Embedding(100, embedding_dim=hidden_dim)

        # Expansion layer for edge features
        self.feature_expansion = nn.Linear(1, hidden_dim)

        # MLP for processing expanded edge features
        self.edge_feature_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim * 2, hidden_dim
            ),  # Concatenated expanded features and bond embeddings
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable combination of MLP output and bond embeddings
        self.combination_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim * 2, hidden_dim
            ),  # Combine edge_feature_mlp output and bond embeddings
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention mechanism for dynamic weighting of edge features
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        # Dynamic feature expansion and embedding
        expanded_features = F.gelu(self.feature_expansion(edge_length))
        bond_features = self.bond_emb(edge_type)

        # Combining expanded edge features with bond embeddings
        combined_features = torch.cat([expanded_features, bond_features], dim=1)

        # Processing combined features through MLP
        edge_features_processed = self.edge_feature_mlp(combined_features)

        # Learnable combination of processed edge features and bond embeddings
        combined_mlp_output = torch.cat([edge_features_processed, bond_features], dim=1)
        edge_attr = self.combination_mlp(combined_mlp_output)

        # Applying attention to edge attributes
        attention_weights = self.attention(edge_attr).expand_as(edge_attr)
        edge_attr_weighted = edge_attr * attention_weights

        return edge_attr_weighted


def get_edge_encoder(cfg):
    if cfg.edge_encoder == "mlp":
        # Only pass the hidden_dim parameter, as MLPEdgeEncoder's updated __init__ method expects
        return MLPEdgeEncoder(cfg.hidden_dim)
    elif cfg.edge_encoder == "gaussian":
        # Ensure GaussianSmearingEdgeEncoder initialization matches its __init__ signature
        return GaussianSmearingEdgeEncoder(
            num_gaussians=cfg.hidden_dim // 2, cutoff=cfg.cutoff
        )
    else:
        raise NotImplementedError(f"Unknown edge encoder: {cfg.edge_encoder}")
