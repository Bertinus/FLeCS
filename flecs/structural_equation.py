from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch_scatter
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from flecs.parameter import EdgeParameter, GeneParameter, Parameter

########################################################################################################################
# StructuralEquation Abstract class
########################################################################################################################


class StructuralEquation(ABC):
    """
    Abstract Class representing the Structural Equation of the Cell.

    The Structural Equation is responsible for computing the production rates and decay rates of all the genes. It
    represents the Cell as a set Tensors, which can be used for efficient computation and training. Its edges and
    number of genes are based on the structure of a ``GRN`` object.

    Attributes:
        edges (torch.Tensor): Edges in the gene regulatory network of the cell. Shape (n_edges, 2)

    """

    def __init__(self):
        self.edges = torch.zeros((0, 2))

    @abstractmethod
    def get_production_rates(self, state) -> torch.Tensor:
        """
        Abstract method to compute the production rates of all the genes.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Production rates. Shape (n_cells, n_genes, *state_dim)

        """

    @abstractmethod
    def get_decay_rates(self, state) -> torch.Tensor:
        """
        Abstract method to compute the decay rates of all the genes.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Decay rates. Shape (n_cells, n_genes, *gene_state_dim)

        """

    def get_derivatives(self, state: torch.Tensor):
        """
        Computes the time derivative of the state:
        $$
        {d \operatorname{state} \over dt} = (\operatorname{production rates}) - (\operatorname{decay rates}).
        $$

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Time derivative of the state. Shape (n_cells, n_genes, *state_dim).

        """
        all_derivatives = self.get_production_rates(state) - self.get_decay_rates(state)

        return all_derivatives

    @property
    def gene_parameter_dict(self) -> Dict[str, GeneParameter]:
        """
        Returns:
            Dict[str, GeneParameter]: Dictionary containing all gene parameters.
                Keys are the names of the parameters.
        """
        return {
            attr_name: attr
            for attr_name, attr in self.__dict__.items()
            if isinstance(attr, GeneParameter)
        }

    @property
    def edge_parameter_dict(self) -> Dict[str, EdgeParameter]:
        """
        Returns:
            Dict[str, EdgeParameter]: Dictionary containing all edge parameters.
                Keys are the names of the parameters.
        """
        return {
            attr_name: attr
            for attr_name, attr in self.__dict__.items()
            if isinstance(attr, EdgeParameter)
        }

    @property
    def parameter_dict(self) -> Dict[str, Parameter]:
        """
        Returns:
             Dict[str, Parameter]: Dictionary containing all (gene and edge) parameters.
            Keys are the names of the parameters.
        """
        return {**self.gene_parameter_dict, **self.edge_parameter_dict}

    def set_parameter(self, param_name: str, param: Parameter):
        """
        Creates a new attribute named ``param_name`` which point to the ``param`` object.

        Args:
            param_name (str): Name of the parameter.
            param (Parameter): Parameter object.
        """
        self.__setattr__(param_name, param)

    @property
    def n_edges(self):
        """
        (``int``) Number of edges.
        """
        return len(self.edges)

    @property
    def edges(self):
        """
        (``torch.Tensor``) Edges. Shape (n_edges, 2).
        """
        return self._edges

    @edges.setter
    def edges(self, edges):
        assert edges.shape[1] == 2
        self._edges = edges

    @property
    def edge_tails(self):
        """
        (``torch.Tensor``) All parents. Shape (n_edges)
        """
        return self.edges[:, 0]

    @property
    def edge_heads(self):
        """
        (``torch.Tensor``) All children. Shape (n_edges)
        """
        return self.edges[:, 1]

    def initialize_given_structure(self, n_genes: int, edges: torch.Tensor):
        """
        Sets the ``self.edges`` attribute, and initializes, using their prior distributions, all gene and edge
        parameters based on the structure of a graph represented by its number of genes and list of edges.

        Args:
            n_genes: Number of genes.
            edges (torch.Tensor): Edges. Shape (n_edges, 2).
        """
        assert edges.dtype is torch.long
        self.edges = edges

        for param_name in self.gene_parameter_dict:
            self.__getattribute__(param_name).initialize_from_prior_dist(length=n_genes)

        for param_name in self.edge_parameter_dict:
            self.__getattribute__(param_name).initialize_from_prior_dist(
                length=len(self.edges)
            )

    def to(self, device):
        """
        Sends all torch.Tensors of the StructuralEquation object to the device ``device``.

        Args:
            device (torch.cuda.device): Device.
        """
        self.edges = self.edges.to(device)
        for parameter in self.parameter_dict.values():
            parameter.tensor = parameter.tensor.to(device)

    def __repr__(self):
        se_repr = "StructuralEquation containing:\n"
        se_repr += "Gene parameters:\n{}\n".format(self.gene_parameter_dict)
        se_repr += "Edge parameters:\n{}".format(self.edge_parameter_dict)

        return se_repr


########################################################################################################################
# StructuralEquation classes
########################################################################################################################


class SigmoidLinearSE(StructuralEquation):
    """
    Structural Equation implementing production rates:

    $$
    (\operatorname{production rates})_i = \operatorname{sigmoid}(\sum_{j \in PA_i} \omega_{ji} \cdot
    \operatorname{state}_j)
    $$

    and decay rates:
    $$
    (\operatorname{decay rates}) = (\operatorname{gene decays}) \cdot \operatorname{state}.
    $$

    Attributes:
        edges (torch.Tensor): Edges in the gene regulatory network of the cell. Shape (n_edges, 2)
        gene_decay (GeneParameter): Rate of exponential decay of the genes.
        weights (EdgeParameter): linear strength of regulation between genes.
    """

    def __init__(self, gene_decay: GeneParameter = None, weights: EdgeParameter = None):
        super().__init__()

        self.gene_decay = (
            gene_decay
            if gene_decay is not None
            else GeneParameter(dim=(1,), prior_dist=Gamma(concentration=10, rate=10))
        )
        self.weights = (
            weights
            if weights is not None
            else EdgeParameter(dim=(1,), prior_dist=Normal(0, 1))
        )

    def get_production_rates(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the production rates of all the genes.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Production rates. Shape (n_cells, n_genes, *state_dim)

        """
        # Compute messages
        parent_inputs = state[:, self.edge_tails]
        edge_messages = parent_inputs * self.weights.tensor

        # Send messages to edge heads
        lin_fct = torch_scatter.scatter(edge_messages, self.edge_heads, dim=1)

        return torch.sigmoid(lin_fct)

    def get_decay_rates(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the decay rates of all the genes.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Decay rates. Shape (n_cells, n_genes, *gene_state_dim)

        """
        return self.gene_decay.tensor * state
