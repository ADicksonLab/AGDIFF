from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_sparse import SparseTensor, matmul

from ..common import MeanReadout, MultiLayerPerceptron, SumReadout


class GINEConv(MessagePassing):
    def __init__(
        self,
        nn: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        activation="softplus",
        **kwargs
    ):
        super(GINEConv, self).__init__(aggr="add", **kwargs)
        self.nn = nn
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.

        # print(f'inside GIN, edge_index.type; {edge_index.dtype}')
        # if isinstance(edge_index, Tensor):
        #     assert edge_attr is not None
        #     assert x[0].size(-1) == edge_attr.size(-1)
        # elif isinstance(edge_index, SparseTensor):
        #     assert x[0].size(-1) == edge_index.size(-1)

        if isinstance(edge_index, Tensor):
            torch._assert(
                edge_attr is not None,
                "edge_attr must not be None when edge_index is a Tensor",
            )
            torch._assert(
                x[0].size(-1) == edge_attr.size(-1),
                "The feature dimensions of x[0] and edge_attr must match",
            )
        elif isinstance(edge_index, SparseTensor):
            torch._assert(
                x[0].size(-1) == edge_index.size(-1),
                "The feature dimensions of x[0] and edge_index must match",
            )

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            return x_j + edge_attr

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class GINEncoder(torch.nn.Module):
    @torch.jit.script_if_tracing
    def __init__(
        self,
        hidden_dim,
        num_convs=3,
        activation="relu",
        short_cut=True,
        concat_hidden=False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.node_emb = nn.Embedding(100, hidden_dim)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  # Add batch normalization layers
        for i in range(self.num_convs):
            self.convs.append(
                GINEConv(
                    MultiLayerPerceptron(
                        hidden_dim, [hidden_dim, hidden_dim], activation=activation
                    ),
                    activation=activation,
                )
            )
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_dim)
            )  # Add batch normalization layer for each conv

    def forward(self, z, edge_index, edge_attr):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """

        node_attr = self.node_emb(z)  # (num_node, hidden)

        hiddens = []
        conv_input = node_attr  # (num_node, hidden)

        for conv_idx, (conv, bn) in enumerate(
            zip(self.convs, self.batch_norms)
        ):  # Use batch normalization
            hidden = conv(conv_input, edge_index, edge_attr)
            hidden = bn(hidden)  # Apply batch normalization

            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)

            # assert hidden.shape == conv_input.shape
            # if self.short_cut and hidden.shape == conv_input.shape:
            #     hidden = hidden + conv_input  # Residual connection

            if self.short_cut:  # trace
                hidden = hidden + conv_input  # Residual connection

            hiddens.append(hidden)
            conv_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return node_feature
