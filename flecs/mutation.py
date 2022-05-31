from abc import ABC, abstractmethod

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

from flecs.cell_population import CellPopulation

########################################################################################################################
# Mutation Abstract class
########################################################################################################################


class Mutation(ABC):
    """
    Abstract class responsible for applying mutations to cells.
    """

    def duplicate_and_mutate_attribute(self, cells: CellPopulation, attr_name: str):
        """
        Duplicates and mutates the attribute ``attr_name``.

        The attribute is duplicated so that its first dimension matches ``cell.n_cells``. After that, it is mutated to
        induce variations between cells.

        Args:
            cells (CellPopulation): CellPopulation.
            attr_name (str): Name of the attribute.

        """
        n_cells = cells.n_cells
        self._duplicate_attribute(cells, attr_name, n_cells)
        self._mutate_attribute(cells, attr_name)
        cells.sync_grn_from_se()

    @staticmethod
    def _duplicate_attribute(cells: CellPopulation, attr_name: str, n_cells: int):
        """
        Duplicates the attribute ``attr_name``. It assumes that the attribute has not yet been duplicated, i.e.
        ``attr.tensor.shape[0] == 1``.

        Args:
            cells (CellPopulation): CellPopulation.
            attr_name (str): Name of the attribute.
            n_cells: Number of cells.
        """
        tensor = cells.get_attribute(attr_name).tensor

        if tensor.shape[0] != 1:
            raise RuntimeError(
                "Cannot duplicate an attribute which has already been duplicated."
            )

        cells.get_attribute(attr_name).tensor = torch.cat([tensor] * n_cells, dim=0)

    @abstractmethod
    def _mutate_attribute(self, cells: CellPopulation, attr_name: str):
        """
        Mutates the attribute ``attr_name``.

        Args:
            cells (CellPopulation): CellPopulation.
            attr_name (str): Name of the attribute.
        """


########################################################################################################################
# Mutation classes
########################################################################################################################


class GaussianMutation(Mutation):
    """
    Class to apply mutations in the form of Gaussian noise.

    Attributes:
        noise_dist (torch.distributions.normal.Normal): Normal distribution to sample noise.
    """

    def __init__(self, sigma: float):
        """
        Args:
            sigma: standard deviation of the Gaussian noise to be applied.
        """
        self.noise_dist = Normal(0, sigma)

    def _mutate_attribute(self, cells: CellPopulation, attr_name: str):
        """
        Mutates the attribute ``attr_name``.

        Args:
            cells (CellPopulation): CellPopulation.
            attr_name (str): Name of the attribute.
        """
        attr = cells.get_attribute(attr_name)

        attr.tensor += self.noise_dist.sample(attr.tensor.shape)


class BernoulliMutation(Mutation):
    """
    Class to apply mutations wherein each parametric element is set to zero with a probability ``p``.

    Attributes:
        noise_dist (torch.distributions.bernoulli.Bernoulli): Bernoulli distribution to sample noise.
    """

    def __init__(self, p: float):
        """
        Args:
            p (float): probability of being set to zero
        """
        self.noise_dist = Bernoulli(1 - p)

    def _mutate_attribute(self, cells: CellPopulation, attr_name: str):
        """
        Mutates the attribute ``attr_name``.

        Args:
            cells (CellPopulation): CellPopulation.
            attr_name (str): Name of the attribute.
        """
        attr = cells.get_attribute(attr_name)

        attr.tensor *= self.noise_dist.sample(attr.tensor.shape)
