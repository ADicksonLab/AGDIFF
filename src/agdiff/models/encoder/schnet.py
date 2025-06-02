from math import pi as PI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, Module, ModuleList, Sequential
from torch.nn.functional import softmax
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_sparse import coalesce

from agdiff.utils.chem import BOND_TYPES

from ..common import MeanReadout, MultiLayerPerceptron, SumReadout


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AsymmetricSineCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer("freq_k", torch.arange(1, num_basis_k + 1).float())
        self.register_buffer("freq_l", torch.arange(1, num_basis_l + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        # If we don't incorporate `cos`, the embedding of 0-deg and 180-deg will be the
        #  same, which is undesirable.
        s = torch.sin(
            angle.view(-1, 1) * self.freq_k.view(1, -1)
        )  # (num_angles, num_basis_k)
        c = torch.cos(
            angle.view(-1, 1) * self.freq_l.view(1, -1)
        )  # (num_angles, num_basis_l)
        return torch.cat([s, c], dim=-1)


class SymmetricCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer("freq_k", torch.arange(1, num_basis + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        return torch.cos(
            angle.view(-1, 1) * self.freq_k.view(1, -1)
        )  # (num_angles, num_basis)


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  # Learnable parameter
        self.shift = torch.log(torch.tensor(2.0))

    def forward(self, x):
        # Manually apply the beta scaling and softplus
        scaled_x = self.beta * x
        return F.softplus(scaled_x) - self.shift


class DistanceWeightingNetwork(nn.Module):
    def __init__(self, hidden_dim=32):
        super(DistanceWeightingNetwork, self).__init__()
        self.layer1 = Linear(1, hidden_dim)
        self.layer2 = Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, edge_length):
        edge_length = edge_length.unsqueeze(
            -1
        )  # Ensure edge_length is properly shaped for linear layers
        x = self.activation(self.layer1(edge_length))
        weights = torch.sigmoid(
            self.layer2(x)
        )  # Use sigmoid to ensure output weights are between 0 and 1
        return weights.squeeze(
            -1
        )  # Remove the last dimension to match the expected shape


class AttentionModule(nn.Module):
    def __init__(self, feature_size):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_size))

    def forward(self, nn_output, weights):
        context = torch.sigmoid(self.attention_weights)  # Contextual modulation
        return nn_output * context + weights * (1 - context)


class CFConv(MessagePassing):
    def __init__(
        self, in_channels, out_channels, num_filters, nn, cutoff, smooth, hidden_dim=32
    ):
        super(CFConv, self).__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=True)
        self.norm1 = torch.nn.BatchNorm1d(num_filters)
        self.act1 = torch.nn.LeakyReLU(0.2)
        self.lin2 = Linear(num_filters, out_channels)
        self.norm2 = torch.nn.BatchNorm1d(out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth
        self.attention = AttentionModule(num_filters)
        self.distance_weighting = DistanceWeightingNetwork(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        # Compute learnable weights based on edge_length
        learnable_weights = self.distance_weighting(edge_length)

        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * torch.pi / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff)
        else:
            C = torch.exp(-((edge_length - self.cutoff) ** 2) / (2 * self.cutoff**2))
            # C = (edge_length <= self.cutoff).float()
        C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)

        # Combine learnable weights with cutoff-based weights
        combined_weights = learnable_weights * C.view(-1, 1)

        W = self.nn(edge_attr) * combined_weights

        x = self.lin1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        x = self.norm2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        # Define the first pathway MLP
        mlp1 = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        # Define the second pathway MLP (with potentially different processing)
        mlp2 = Sequential(
            Linear(num_gaussians, num_filters // 2),  # Note: Adjusted the size here
            ShiftedSoftplus(),
            Linear(num_filters // 2, num_filters // 2),
        )

        self.conv1 = CFConv(
            hidden_channels, hidden_channels, num_filters, mlp1, cutoff, smooth
        )
        self.conv2 = CFConv(
            hidden_channels, hidden_channels, num_filters // 2, mlp2, cutoff, smooth
        )

        self.act = ShiftedSoftplus()

        self.lin = Linear(
            256, hidden_channels
        )  # Adjusted to match the actual concatenated size

        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_length, edge_attr):
        pathway1 = self.conv1(x, edge_index, edge_length, edge_attr)
        pathway2 = self.conv2(x, edge_index, edge_length, edge_attr)

        # Concatenate features from both pathways
        x_combined = torch.cat([pathway1, pathway2], dim=-1)
        x_combined = self.act(x_combined)
        x_combined = self.lin(x_combined)

        # Compute attention weights
        attention_weights = self.attention(x_combined)

        # Apply attention weights to the interactions
        attended_interactions = x_combined * attention_weights

        return attended_interactions


class AdaptiveScalingModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AdaptiveScalingModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x.view(b, c, -1)).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SchNetEncoder(Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        edge_channels=100,
        cutoff=10.0,
        smooth=False,
    ):
        super(SchNetEncoder, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)
        self.interactions = ModuleList()
        self.scaling_modules = ModuleList()  # Adaptive scaling modules

        for _ in range(num_interactions):
            self.interactions.append(
                InteractionBlock(
                    hidden_channels, edge_channels, num_filters, cutoff, smooth
                )
            )
            self.scaling_modules.append(
                AdaptiveScalingModule(hidden_channels)
            )  # One scaling module per InteractionBlock

    def forward(self, z, edge_index, edge_length, edge_attr, embed_node: bool = True):
        if embed_node:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)
        else:
            h = z

        for interaction, scale in zip(self.interactions, self.scaling_modules):
            interaction_output = interaction(h, edge_index, edge_length, edge_attr)
            scaled_interaction_output = scale(interaction_output.unsqueeze(-1)).squeeze(
                -1
            )  # Assuming h is of shape (batch_size, num_channels)
            h = h + scaled_interaction_output

        return h
