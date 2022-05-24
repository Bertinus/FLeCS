from abc import ABC, abstractmethod
import torch
from torch.distributions.distribution import Distribution
from typing import Optional

from typing import Tuple

########################################################################################################################
# Parameter Abstract class
########################################################################################################################


class Parameter(ABC):
    """
    Abstract class representing a parameter of the cell. It can correspond to either a node or an edge parameter.

    Attributes:
        dim (Tuple[int, ...]): Dimension of the parameter. In the case of a node parameter,
            each node will be associated with a torch.Tensor of shape (n_cells, *dim).
        prior_dist (torch.distributions.distribution.Distribution): Prior distribution of the parameter. Can be used
            to initialize or re-sample the parameter.
        tensor (torch.Tensor): Values of the parameter. Shape (n_cells, length, *dim). Length is typically the number
            of nodes or edges.
    """

    def __init__(
        self,
        dim: Tuple[int, ...],
        prior_dist: Optional[Distribution] = None,
        tensor: torch.Tensor = None,
    ):
        """
        Args:
            dim (Tuple[int, ...]): Dimension of the parameter. In the case of a node parameter,
                each node will be associated with a torch.Tensor of shape (n_cells, *dim).
            prior_dist (torch.distributions.distribution.Distribution, optional): Prior distribution of the parameter.
                Can be used to initialize or re-sample the parameter. Default is None
            tensor (torch.Tensor, optional): Values of the parameter. Shape (n_cells, length, *dim). Length is
                typically the number of nodes or edges. Default is None.
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
                "parameter.dim= {}".format(t.shape[2:], self.dim)
            )
        self._tensor = t

    def initialize_from_prior_dist(self, length: int):
        """
        Initializes the values of the ``self.tensor`` based on the prior distribution ``self.prior_dist``

        Args:
            length (int): Second dimension for ``self.tensor``. Typically the number of nodes or the number of edges.
        """
        if self.prior_dist is None:
            raise RuntimeError(
                "The parameter's prior distribution prior_dist is not defined."
            )
        self.tensor = self.prior_dist.rsample((self.n_cells, length, *self.dim))

    @abstractmethod
    def __repr__(self):
        rep = "Parameter(n_cells={}, dim={}, prior_dist={}, tensor={})".format(
            self.n_cells, self.dim, self.prior_dist, self.tensor
        )
        return rep


########################################################################################################################
# Parameter classes
########################################################################################################################


class NodeParameter(Parameter):
    """
    Subclass representing a node parameter.
    """

    def __repr__(self):
        return "Node" + super().__repr__()


class EdgeParameter(Parameter):
    """
    Subclass representing an edge parameter.
    """

    def __repr__(self):
        return "Edge" + super().__repr__()
