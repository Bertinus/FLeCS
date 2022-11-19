"""Functions for modelling production rates in cells."""
import torch
import torch_scatter
import torch_geometric.nn


def efficient_inplace_message_passing(obj, e_type, e_weights):
    src_type, interaction_type, tgt_type = e_type  # Unpack edge definition.

    parent_indices = obj[e_type].tails
    children_indices = obj[e_type].heads

    edge_messages = e_weights * obj[src_type].state[:, parent_indices]

    torch_scatter.scatter(
        edge_messages, children_indices, dim=1, out=obj[tgt_type].production_rate
    )


class SimpleConv(torch_geometric.nn.MessagePassing):
    def __init__(self, tgt_nodeset_len):
        """

        Args:
            tgt_nodeset_len:
        """
        self.tgt_nodeset_len = tgt_nodeset_len
        super().__init__(aggr="add")

    def forward(self, x, edge_index, edge_weight):
        """

        Args:
            x:
            edge_index:
            edge_weight:

        Returns:

        """
        out = self.propagate(
            edge_index,
            x=x,
            size=(x.shape[1], self.tgt_nodeset_len),
            edge_weight=edge_weight,
        )

        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor):
        """
        Args:
            x_j (torch.Tensor):
            edge_weight (torch.Tensor):
        """
        return edge_weight * x_j
