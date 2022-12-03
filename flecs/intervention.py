from abc import ABC, abstractmethod
from typing import Dict, Tuple

from flecs.cell_population import CellPopulation

########################################################################################################################
# Intervention Abstract class
########################################################################################################################


class Intervention(ABC):
    """
    Abstract class responsible for intervening on cells, and resetting cells to their default states.
    """

    @abstractmethod
    def intervene(self, *args, **kwargs) -> None:
        """
        Abstract method for intervening on cells.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Abstract method for resetting cells to their default state.
        """


########################################################################################################################
# Intervention classes
########################################################################################################################


class CrisprIntervention(Intervention):
    def __init__(
        self,
        cells: CellPopulation,
        e_type: Tuple[str, str, str] = ("gene", "activation", "gene"),
    ):
        self.cells = cells
        self.e_type = e_type
        self.intervened_edges: Dict[int, tuple] = {}

    def intervene(self, gene: int) -> None:
        if gene in self.intervened_edges.keys():
            raise ValueError("Gene {} has already been knocked out".format(gene))

        out_edges_indices = self.cells[self.e_type].out_edges(gene)

        # Save the removed edges
        self.intervened_edges[gene] = self.cells[self.e_type].get_edges(
            out_edges_indices
        )

        # Remove edges from the CellPopulation object
        self.cells[self.e_type].remove_edges(out_edges_indices)

    def reset(self) -> None:
        for gene in self.intervened_edges.keys():
            self.cells[self.e_type].add_edges(*self.intervened_edges[gene])
