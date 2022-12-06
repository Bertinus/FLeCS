"""Functions for modelling decay processes in cells."""
from __future__ import annotations
import flecs.cell_population as cp
import torch


def exponential_decay(
    cellpop: cp.CellPopulation, node_type: str, alpha: torch.Tensor = None
):
    """
    Applies an exponential decay with rate *alpha* to the states of a given type of nodes.

    $$
    \operatorname{decay rates}(t) =  alpha \otimes \operatorname{state}(t)
    $$

    Args:
        cellpop: Population of Cells.
        node_type: Type of nodes for which the decay will be applied
        alpha: Decay constant. Shape (1 OR n_cells, 1 OR n_nodes, 1 OR state_dim_per_node)

    Returns:
        decay rates for nodes of type *node_type*.

    """
    return alpha * cellpop[node_type].state
