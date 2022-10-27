from abc import ABC, abstractmethod
from flecs.cell_population import CellPopulation


########################################################################################################################
# Intervention Abstract class
########################################################################################################################


class Intervention(ABC):
    """
    Abstract class responsible for intervening on cells, and resetting cells to their default states.
    """

    @abstractmethod
    def intervene(self, cells: CellPopulation, *args, **kwargs) -> None:
        """
        Abstract method for intervening on cells.

        Args:
            cells (CellPopulation): CellPopulation object

        """

    @abstractmethod
    def reset(self, cells: CellPopulation) -> None:
        """
        Abstract method for resetting cells to their default state.

        Args:
            cells (CellPopulation): CellPopulation object

        """


########################################################################################################################
# Intervention classes
########################################################################################################################
