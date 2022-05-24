from abc import ABC, abstractmethod
import torch
from typing import Dict
from flecs.parameter import NodeParameter, EdgeParameter, Parameter
import torch_scatter
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal


########################################################################################################################
# StructuralEquation Abstract class
########################################################################################################################


class StructuralEquation(ABC):
    """
    Class responsible for computing production rates and decay rates.

    """

    def __init__(self):
        self.edges = torch.zeros((0, 2))

    @abstractmethod
    def get_production_rates(self, state) -> torch.Tensor:
        """
        :param state: Tensor of shape (n_cells, n_nodes, *node_state_dim)
        :return: Production rates with shape (n_cells, n_nodes, *node_state_dim)
        """

    @abstractmethod
    def get_decay_rates(self, state) -> torch.Tensor:
        """
        :param state: Tensor of shape (n_cells, n_nodes, *node_state_dim)
        :return: Decay rates with shape (n_cells, n_nodes, *node_state_dim)
        """

    def get_derivatives(self, state: torch.Tensor):
        all_derivatives = self.get_production_rates(state) - self.get_decay_rates(state)

        return all_derivatives

    @property
    def node_parameter_dict(self) -> Dict[str, NodeParameter]:
        """
        :return: Dictionary containing all node parameters. Keys are the names of the parameters
        """
        return {
            attr_name: attr
            for attr_name, attr in self.__dict__.items()
            if isinstance(attr, NodeParameter)
        }

    @property
    def edge_parameter_dict(self) -> Dict[str, NodeParameter]:
        """
        :return: Dictionary containing all edge parameters. Keys are the names of the parameters
        """
        return {
            attr_name: attr
            for attr_name, attr in self.__dict__.items()
            if isinstance(attr, EdgeParameter)
        }

    @property
    def parameter_dict(self) -> Dict[str, Parameter]:
        """
        :return: Dictionary containing both node and edge parameters. Keys are the names of the parameters
        """
        return {**self.node_parameter_dict, **self.edge_parameter_dict}

    def set_parameter(self, param_name: str, param: Parameter):
        self.__setattr__(param_name, param)

    @property
    def n_nodes(self):
        return int(self.edges.max()) + 1

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, edges):
        assert edges.shape[1] == 2
        self._edges = edges

    @property
    def edge_tails(self):
        return self.edges[:, 0]

    @property
    def edge_heads(self):
        return self.edges[:, 1]

    def initialize_given_structure(self, n_nodes: int, edges: torch.Tensor) -> None:
        """
        Initializes the attributes of the StructuralEquation object based on the structure of a GRN.
        :param n_nodes: Number of nodes
        :param edges: torch Tensor of shape (n_edges, 2) containing all the edges
        """
        assert edges.dtype is torch.long
        self.edges = edges

        for param_name in self.node_parameter_dict:
            self.__getattribute__(param_name).initialize_from_prior_dist(length=n_nodes)

        for param_name in self.edge_parameter_dict:
            self.__getattribute__(param_name).initialize_from_prior_dist(
                length=len(self.edges)
            )

    def to(self, device):
        self.edges = self.edges.to(device)
        for parameter in self.parameter_dict.values():
            parameter.tensor = parameter.tensor.to(device)

    def __repr__(self):
        se_repr = "StructuralEquation containing:\n"
        se_repr += "Node parameters:\n{}\n".format(self.node_parameter_dict)
        se_repr += "Edge parameters:\n{}".format(self.edge_parameter_dict)

        return se_repr


########################################################################################################################
# StructuralEquation classes
########################################################################################################################


class SigmoidLinearSE(StructuralEquation):
    """
    Class which represents structural equations of the form:
        dstate/dt = sigmoid(weights_of_parents.dot(states_of_parents)) - gene_decay * state
    """

    def __init__(
        self,
        gene_decay: NodeParameter = NodeParameter(
            dim=(1,), prior_dist=Gamma(concentration=10, rate=10)
        ),
        weights: EdgeParameter = EdgeParameter(dim=(1,), prior_dist=Normal(0, 1)),
    ):
        super().__init__()

        self.gene_decay = gene_decay
        self.weights = weights

    def get_production_rates(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes sigmoid(weight_of_parents.dot(state_of_parents))

        :param state: torch tensor of shape (n_cells, n_nodes, *node_state_dim)
        :return: Production rates with shape (n_cells, n_nodes, *node_state_dim)
        """
        # Compute messages
        parent_inputs = state[:, self.edge_tails]
        edge_messages = parent_inputs * self.weights.tensor

        # Send messages to edge heads
        lin_fct = torch_scatter.scatter(edge_messages, self.edge_heads, dim=1)

        return torch.sigmoid(lin_fct)

    def get_decay_rates(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes gene_decay * state
        :param state: torch tensor of shape (n_cells, n_nodes, *node_state_dim)
        :return: Decay rates with shape (n_cells, n_nodes, *node_state_dim)
        """
        return self.gene_decay.tensor * state
