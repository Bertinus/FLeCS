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

    A knockout on gene ``k`` is simulated by removing all outgoing edges of gene ``k``. This corresponds to a
    complete loss of function of the protein coded by gene ``k``.

    One can perform knockouts on several genes.

    Attributes:
        intervened_genes (Dict[int, list]): Dictionary whose keys correspond to the genes that have been knocked out.
            ``intervened_genes[k]`` contains the list of edges, as well as their attributes, that have been removed from
            the cell because of the knockout of gene ``k``.
    """

    def __init__(self):
        self.intervened_genes: Dict[int, list] = {}

    def intervene(self, cell: Cell, gene: int):
        """
        Performs a knockout intervention on the cell by removing all outgoing edges of gene ``gene``.

        Args:
            cell (Cell): Cell to intervene on.
            gene (int): Index of the gene on which the knockout is performed.
        """
        if gene in self.intervened_genes:
            raise ValueError("Gene {} has already been knocked out".format(gene))

        # Make sure the GRN is synchronized
        cell.sync_grn_from_se()

        # Save outgoing edges
        self.intervened_genes[gene] = copy.deepcopy(
            list(cell.grn.out_edges(gene, data=True))
        )
        # Truncate the graph
        cell.grn.remove_edges_from(self.intervened_genes[gene])

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
        for gene in self.intervened_genes:
            cell.grn.add_edges_from(self.intervened_genes[gene])

        self.intervened_genes = {}

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
            Shape (n_genes).
    """

    def __init__(self):
        self.sum_of_direct_effects = None

    def intervene(self, cell: Cell, drug_direct_effects: torch.Tensor):
        """
        Performs a drug intervention on the cell. Production rates are shifted by ``drug_direct_effects``.

        Args:
            cell (Cell): Cell to intervene on.
            drug_direct_effects (torch.Tensor): Direct effects of the drug on the genes. Shape (n_genes).
        """

        assert len(drug_direct_effects.shape) == 1
        assert drug_direct_effects.shape[0] == cell.n_genes

        if self.sum_of_direct_effects is None:
            self.sum_of_direct_effects = drug_direct_effects
        else:
            assert cell.n_genes == self.sum_of_direct_effects.shape[0]
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
