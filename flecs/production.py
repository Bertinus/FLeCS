"""Functions for modelling production rates in cells."""
import torch_scatter


def message_passing(obj, e_type, e_weights):
    src_type, interaction_type, tgt_type = e_type  # Unpack edge definition.

    parent_indices = obj[e_type].tails
    children_indices = obj[e_type].heads

    edge_messages = e_weights * obj[src_type].state[:, parent_indices]

    out = torch_scatter.scatter(
        edge_messages,
        children_indices,
        dim=1,
        dim_size=obj[tgt_type].production_rate.shape[1]
    )

    return out


def protein_rna_message_passing(obj, e_type, e_weights):
    src_type, interaction_type, tgt_type = e_type  # Unpack edge definition.

    parent_indices = obj[e_type].tails
    children_indices = obj[e_type].heads

    parent_protein_state = obj[src_type].state[:, parent_indices, :1]
    edge_messages = e_weights * parent_protein_state

    out = torch_scatter.scatter(
        edge_messages,
        children_indices,
        dim=1,
        dim_size=obj[tgt_type].production_rate.shape[1]
    )

    return out
