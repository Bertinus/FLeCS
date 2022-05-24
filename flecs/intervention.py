from abc import ABC, abstractmethod
from flecs.cell import Cell
import copy
from typing import Dict
import torch


########################################################################################################################
# Intervention Abstract class
########################################################################################################################


class Intervention(ABC):
    """
    Class responsible for intervening on the cell, and resetting to default state
    """

    @abstractmethod
    def reset(self, cell: Cell) -> None:
        """
        Method for resetting the cell to its default state
        """ ""


########################################################################################################################
# Intervention classes
########################################################################################################################


class KnockoutIntervention(Intervention):
    def __init__(self):
        self.intervened_nodes: Dict[int, list] = {}

    def intervene(self, cell: Cell, node: int):
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

        cell.sync_grn_from_se()

        # Recover original graph if the graph has been intervened
        for node in self.intervened_nodes:
            cell.grn.add_edges_from(self.intervened_nodes[node])

        self.intervened_nodes = {}

        # We need to synchronize the structural equation to the graph
        cell.sync_se_from_grn()


class DrugLinearIntervention(Intervention):
    def __init__(self):
        self.sum_of_direct_effects = None

    def intervene(self, cell: Cell, drug_direct_effects: torch.Tensor):

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
        def get_default_production_rates(self, state):
            return self.structural_equation.get_production_rates(state)

        # Override the method which generates derivatives
        func_type = type(cell.get_production_rates)
        cell.get_production_rates = func_type(get_default_production_rates, cell)
