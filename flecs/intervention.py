import copy
from abc import ABC, abstractmethod
from typing import Dict

import torch

from flecs.cell import Cell

########################################################################################################################
# Intervention Abstract class
########################################################################################################################


class Intervention(ABC):
    """
    Abstract class responsible for intervening on cells, and resetting cells to their default states.
    """

    @abstractmethod
    def reset(self, cell: Cell) -> None:
        """
        Abstract method for resetting cells to their default state.

        Args:
            cell (Cell): Cell object

        """


########################################################################################################################
# Intervention classes
########################################################################################################################


class KnockoutIntervention(Intervention):
    """
    Class to perform Knockout interventions on cells.

    A knockout on node ``k`` is simulated by removing all outgoing edges of node ``k``. This corresponds to a
    complete loss of function of the protein coded by gene ``k``.

    One can perform knockouts on several genes.

    Attributes:
        intervened_nodes (Dict[int, list]): Dictionary whose keys correspond to the nodes that have been knocked out.
            ``intervened_nodes[k]`` contains the list of edges, as well as their attributes, that have been removed from
            the cell because of the knockout of node ``k``.
    """

    def __init__(self):
        self.intervened_nodes: Dict[int, list] = {}

    def intervene(self, cell: Cell, node: int):
        """
        Performs a knockout intervention on the cell by removing all outgoing edges of node ``node``.

        Args:
            cell (Cell): Cell to intervene on.
            node (int): Index of the node on which the knockout is performed.
        """
        if node in self.intervened_nodes:
            raise ValueError("Node {} has already been knocked out".format(node))

        # Make sure the GRN is synchronized
        cell.sync_grn_from_se()

        # Save outgoing edges
        self.intervened_nodes[node] = copy.deepcopy(
            list(cell.grn.out_edges(node, data=True))
        )
        # Truncate the graph
        cell.grn.remove_edges_from(self.intervened_nodes[node])

        # We need to synchronize the structural equation to the intervened graph
        cell.sync_se_from_grn()

    def reset(self, cell: Cell):
        """
        Resets the cell to its default state.

        Args:
            cell (Cell): Cell to reset.
        """

        cell.sync_grn_from_se()

        # Recover original graph if the graph has been intervened
        for node in self.intervened_nodes:
            cell.grn.add_edges_from(self.intervened_nodes[node])

        self.intervened_nodes = {}

        # We need to synchronize the structural equation to the graph
        cell.sync_se_from_grn()


class DrugLinearIntervention(Intervention):
    """
    Class to perform drug (linear) interventions on cells.

    A drug intervention is simulated by shifting the production rates by a fixed vector `drug_direct_effects``,
    representing the direct effects of the drug on the genes:
    $$
    (\operatorname{production rates}) \leftarrow (\operatorname{production rates})
    + (\operatorname{drug direct effects}).
    $$

    One can perform several drug interventions, corresponding to combinations of drugs.

    Attributes:
        sum_of_direct_effects (torch.Tensor, optional): Sum of the direct effects of the drugs that have been applied.
            Shape (n_nodes).
    """

    def __init__(self):
        self.sum_of_direct_effects = None

    def intervene(self, cell: Cell, drug_direct_effects: torch.Tensor):
        """
        Performs a drug intervention on the cell. Production rates are shifted by ``drug_direct_effects``.

        Args:
            cell (Cell): Cell to intervene on.
            drug_direct_effects (torch.Tensor): Direct effects of the drug on the genes. Shape (n_nodes).
        """

        assert len(drug_direct_effects.shape) == 1
        assert drug_direct_effects.shape[0] == cell.n_nodes

        if self.sum_of_direct_effects is None:
            self.sum_of_direct_effects = drug_direct_effects
        else:
            assert cell.n_nodes == self.sum_of_direct_effects.shape[0]
            self.sum_of_direct_effects += drug_direct_effects

        def get_intervened_production_rates(_self, state):
            return (
                _self.structural_equation.get_production_rates(state)
                + self.sum_of_direct_effects[None, :, None]
            )

        # Override the method which generates derivatives
        funcType = type(cell.get_production_rates)
        cell.get_production_rates = funcType(get_intervened_production_rates, cell)

    def reset(self, cell: Cell):
        """
        Resets the cell to its default state.

        Args:
            cell (Cell): Cell to reset.
        """

        def get_default_production_rates(self, state):
            return self.structural_equation.get_production_rates(state)

        # Override the method which generates derivatives
        func_type = type(cell.get_production_rates)
        cell.get_production_rates = func_type(get_default_production_rates, cell)
