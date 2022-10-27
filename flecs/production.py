"""Functions for modelling production rates in cells."""
import torch_scatter


def message_passing(obj, e_type):

    src_type, interaction_type, tgt_type = e_type  # Unpack edge definition.

    parent_indices = obj[e_type].tails
    children_indices = obj[e_type].heads

    edge_messages = obj[e_type]["weights"] * obj[src_type].state[:, parent_indices]

    torch_scatter.scatter(
        edge_messages,
        children_indices,
        dim=1,
        out=obj[tgt_type].production_rate,
    )
