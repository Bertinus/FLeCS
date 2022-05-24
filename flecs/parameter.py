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
    Class representing a parameter.
    """

    def __init__(
        self,
        dim: Tuple[int, ...],
        prior_dist: Optional[Distribution] = None,
        tensor: torch.Tensor = None,
    ):
        """

        :param dim: Dimension of the parameter.
        :param prior_dist: Prior distribution of the parameter. Can be used to initialize or re-sample the parameter.
        Typically a torch.distribution object.
        :param tensor: torch Tensor containing the values of the parameter. Shape (n_cells, length, *dim).
        Length would typically be the number of nodes or edges in the GRN.
        """

        self.dim = dim
        if tensor is None:
            self.tensor = torch.zeros((1, 0, *self.dim))
        else:
            self.tensor = tensor

        self.prior_dist = prior_dist

    @property
    def n_cells(self):
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

    def initialize_from_prior_dist(self, length):
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
