from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch_scatter
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from flecs.attribute import EdgeAttribute, GeneAttribute, Attribute

########################################################################################################################
# StructuralEquation Abstract class
########################################################################################################################


class StructuralEquation(ABC):
    """
    Abstract Class representing the Structural Equation of the CellPopulation.

    The Structural Equation is responsible for computing the production rates and decay rates of all the genes. It
    represents the CellPopulation as a set Tensors, which can be used for efficient computation and training. Its edges and
    number of genes are based on the structure of a ``GRN`` object.

    Attributes:
        edges (torch.Tensor): Edges in the gene regulatory network of the cells. Shape (n_edges, 2)

    """

    def __init__(self):
        self.edges = torch.zeros((0, 2))

    @abstractmethod
    def get_production_rates(self, state) -> torch.Tensor:
        """
        Abstract method to compute the production rates of all the genes.

        Args:
            state (torch.Tensor): State of the cells. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Production rates. Shape (n_cells, n_genes, *state_dim)

        """

    @abstractmethod
    def get_decay_rates(self, state) -> torch.Tensor:
        """
        Abstract method to compute the decay rates of all the genes.

        Args:
            state (torch.Tensor): State of the cells. Shape (n_cells, n_genes, *state_dim)

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
            state (torch.Tensor): State of the cells. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Time derivative of the state. Shape (n_cells, n_genes, *state_dim).

        """
        all_derivatives = self.get_production_rates(state) - self.get_decay_rates(state)

        return all_derivatives

    @property
    def gene_attribute_dict(self) -> Dict[str, GeneAttribute]:
        """
        Returns:
            Dict[str, GeneAttribute]: Dictionary containing all gene attributes.
                Keys are the names of the attributes.
        """
        return {
            attr_name: attr
            for attr_name, attr in self.__dict__.items()
            if isinstance(attr, GeneAttribute)
        }

    @property
    def edge_attribute_dict(self) -> Dict[str, EdgeAttribute]:
        """
        Returns:
            Dict[str, EdgeAttribute]: Dictionary containing all edge attributes.
                Keys are the names of the attributes.
        """
        return {
            attr_name: attr
            for attr_name, attr in self.__dict__.items()
            if isinstance(attr, EdgeAttribute)
        }

    @property
    def attribute_dict(self) -> Dict[str, Attribute]:
        """
        Returns:
             Dict[str, Attribute]: Dictionary containing all (gene and edge) attributes.
            Keys are the names of the attributes.
        """
        return {**self.gene_attribute_dict, **self.edge_attribute_dict}

    def set_attribute(self, attr_name: str, attribute: Attribute):
        """
        Creates a new attribute named ``attr_name`` which point to the ``attribute`` object.

        Args:
            attr_name (str): Name of the attribute.
            attribute (Attribute): Attribute object.
        """
        self.__setattr__(attr_name, attribute)

    @property
    def n_genes(self):
        """
        (``int``) Number of genes.
        """
        return int(self.edges.max()) + 1

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
        attributes based on the structure of a graph represented by its number of genes and list of edges.

        Args:
            n_genes: Number of genes.
            edges (torch.Tensor): Edges. Shape (n_edges, 2).
        """
        assert edges.dtype is torch.long
        self.edges = edges

        for attr_name in self.gene_attribute_dict:
            self.__getattribute__(attr_name).initialize_from_prior_dist(length=n_genes)

        for attr_name in self.edge_attribute_dict:
            self.__getattribute__(attr_name).initialize_from_prior_dist(
                length=len(self.edges)
            )

    def to(self, device):
        """
        Sends all torch.Tensors of the StructuralEquation object to the device ``device``.

        Args:
            device (torch.cuda.device): Device.
        """
        self.edges = self.edges.to(device)
        for attribute in self.attribute_dict.values():
            attribute.tensor = attribute.tensor.to(device)

    def __repr__(self):
        se_repr = "StructuralEquation containing:\n"
        se_repr += "Gene attributes:\n{}\n".format(self.gene_attribute_dict)
        se_repr += "Edge attributes:\n{}".format(self.edge_attribute_dict)

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
        edges (torch.Tensor): Edges in the gene regulatory network of the cells. Shape (n_edges, 2)
        gene_decay (GeneAttribute): Rate of exponential decay of the genes.
        weights (EdgeAttribute): linear strength of regulation between genes.
    """

    def __init__(self, gene_decay: GeneAttribute = None, weights: EdgeAttribute = None):
        super().__init__()

        self.gene_decay = (
            gene_decay
            if gene_decay is not None
            else GeneAttribute(dim=(1,), prior_dist=Gamma(concentration=10, rate=10))
        )
        self.weights = (
            weights
            if weights is not None
            else EdgeAttribute(dim=(1,), prior_dist=Normal(0, 1))
        )

    def get_production_rates(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the production rates of all the genes.

        Args:
            state (torch.Tensor): State of the cells. Shape (n_cells, n_genes, *state_dim)

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
            state (torch.Tensor): State of the cells. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Decay rates. Shape (n_cells, n_genes, *gene_state_dim)

        """
        return self.gene_decay.tensor * state
