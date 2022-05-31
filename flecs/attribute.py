from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch.distributions.distribution import Distribution

########################################################################################################################
# Attribute Abstract class
########################################################################################################################


class Attribute(ABC):
    """
    Abstract class representing an attribute of the cell. It can correspond to either a gene or an edge attribute.

    Attributes:
        dim (Tuple[int, ...]): Dimension of the attribute. In the case of a gene attribute,
            each gene will be associated with a torch.Tensor of shape (n_cells, *dim).
        prior_dist (torch.distributions.distribution.Distribution): Prior distribution of the attribute. Can be used
            to initialize or re-sample the attribute.
        tensor (torch.Tensor): Values of the attribute. Shape (n_cells, length, *dim). Length is typically the number
            of genes or edges.
    """

    def __init__(
        self,
        dim: Tuple[int, ...],
        prior_dist: Optional[Distribution] = None,
        tensor: torch.Tensor = None,
    ):
        """
        Args:
            dim (Tuple[int, ...]): Dimension of the attribute. In the case of a gene attribute,
                each gene will be associated with a torch.Tensor of shape (n_cells, *dim).
            prior_dist (torch.distributions.distribution.Distribution, optional): Prior distribution of the attribute.
                Can be used to initialize or re-sample the attribute. Default is None
            tensor (torch.Tensor, optional): Values of the attribute. Shape (n_cells, length, *dim). Length is
                typically the number of genes or edges. Default is None.
        """

        self.dim = dim
        if tensor is None:
            self.tensor = torch.zeros((1, 0, *self.dim))
        else:
            self.tensor = tensor

        self.prior_dist = prior_dist

    @property
    def n_cells(self):
        """
        (``int``) Second dimension of ``self.tensor``. Typically the number of cells.
        """
        return self.tensor.shape[0]

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, t: torch.Tensor):
        if t.shape[2:] != self.dim:
            raise ValueError(
                "Dimension mismatch: New tensor's last dimensions t.shape[2:]= {} must match "
                "attribute.dim= {}".format(t.shape[2:], self.dim)
            )
        self._tensor = t

    def initialize_from_prior_dist(self, length: int):
        """
        Initializes the values of the ``self.tensor`` based on the prior distribution ``self.prior_dist``

        Args:
            length (int): Second dimension for ``self.tensor``. Typically the number of genes or the number of edges.
        """
        if self.prior_dist is None:
            raise RuntimeError(
                "The attribute's prior distribution prior_dist is not defined."
            )
        self.tensor = self.prior_dist.rsample((self.n_cells, length, *self.dim))

    @abstractmethod
    def __repr__(self):
        rep = "Attribute(n_cells={}, dim={}, prior_dist={}, tensor={})".format(
            self.n_cells, self.dim, self.prior_dist, self.tensor
        )
        return rep


########################################################################################################################
# Attribute classes
########################################################################################################################


class GeneAttribute(Attribute):
    """
    Subclass representing a gene attribute.
    """

    def __repr__(self):
        return "Gene" + super().__repr__()


class EdgeAttribute(Attribute):
    """
    Subclass representing an edge attribute.
    """

    def __repr__(self):
        return "Edge" + super().__repr__()
