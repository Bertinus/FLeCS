from flecs.cell import Cell
from abc import ABC, abstractmethod
import torch
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

########################################################################################################################
# Mutation Abstract class
########################################################################################################################


class Mutation(ABC):
    """
    Class responsible for applying mutations to Cells
    """

    def duplicate_and_mutate_attribute(self, cell: Cell, attr_name: str):
        n_cells = cell.n_cells
        self._duplicate_attribute(cell, attr_name, n_cells)
        self._mutate_attribute(cell, attr_name)
        cell.sync_grn_from_se()

    @staticmethod
    def _duplicate_attribute(cell: Cell, attr_name: str, n_cells: int):
        tensor = cell.get_parameter(attr_name).tensor

        if tensor.shape[0] != 1:
            raise RuntimeError(
                "Cannot duplicate an attribute which has already been duplicated."
            )

        cell.get_parameter(attr_name).tensor = torch.cat([tensor] * n_cells, dim=0)

    @abstractmethod
    def _mutate_attribute(self, cell: Cell, attr_name: str):
        """
        Method which applies mutations to a given attribute
        """


########################################################################################################################
# Mutation classes
########################################################################################################################


class GaussianMutation(Mutation):
    def __init__(self, sigma: float):
        self.noise_dist = Normal(0, sigma)

    def _mutate_attribute(self, cell: Cell, attr_name: str):
        attr = cell.get_parameter(attr_name)

        attr.tensor += self.noise_dist.sample(attr.tensor.shape)


class BernoulliMutation(Mutation):
    def __init__(self, p: float):
        """

        :param p: probability of being set to zero
        """
        self.noise_dist = Bernoulli(1 - p)

    def _mutate_attribute(self, cell: Cell, attr_name: str):
        attr = cell.get_parameter(attr_name)

        attr.tensor *= self.noise_dist.sample(attr.tensor.shape)
