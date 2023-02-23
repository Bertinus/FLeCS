from __future__ import annotations
from abc import ABC, abstractmethod
import flecs.sets as sets
from typing import Tuple, Dict, Union, List
from flecs.data.utils import load_interaction_data
from flecs.data.interaction_data import InteractionData, SetData, EdgeType
from torch.distributions.normal import Normal
import flecs.decay as dc
import networkx as nx
from flecs.production import SimpleConv
import torch


########################################################################################################################
# Cell Population abstract class
########################################################################################################################


class CellPopulation(ABC, torch.nn.Module):
    def __init__(
        self,
        interaction_graph: Union[InteractionData, nx.DiGraph],
        n_cells: int = 1,
        per_node_state_dim: int = 1,
        scale_factor_state_prior: float = 10.0,
    ):
        """
        A population of cells. The mechanisms of cells are based on a graph with different types of nodes and edges.
        Cell dynamics can be computed based on these mechanisms.

        * Examples of node types include "proteins", "small molecules", "gene/RNA".
        * Examples of edge types include ("gene", "activates", "gene"), ("protein", "catalyses", "small molecule").

        All nodes/edges of a given type are grouped in a `NodeSet`/`EdgeSet` object.

        A specific node usually corresponds to a specific molecule (e.g. RNA from gene X) whose concentration
        (and potentially other properties) is tracked. Together, the tracked properties of all the nodes define the
        state of the cell.

        Production rates and Decay rates (for all the tracked properties of all the nodes) can be computed and depend
        on the state of the cell, as well as some node parameters and edge parameters.

        To define your own `CellPopulation` class inheriting from this class, you need to implement the two methods
        `compute_production_rates` and `compute_decay_rates`. You may also want to override
        `sample_from_state_prior_dist` in order to choose your own prior distribution over the state of cells.

        Args:
            interaction_graph: Graph on which the mechanisms of cells will be based.
            n_cells: Number of cells in the population.
            per_node_state_dim: Dimension of the state associated with each node.
            scale_factor_state_prior: Value at which the state will be initialized by default
        """
        super().__init__()

        if not isinstance(interaction_graph, InteractionData):
            assert isinstance(interaction_graph, nx.DiGraph)
            interaction_graph = InteractionData(interaction_graph)

        # str type of node (e.g., gene, protein).
        self._node_set_dict: Dict[str, sets.NodeSet] = {}
        # str types of interactions (src, interaction_type, dest).
        self._edge_set_dict: Dict[Tuple[str, str, str], sets.EdgeSet] = {}

        self.initialize_from_interaction_graph(interaction_graph)

        # Create state and production/decay rates as empty tensors
        self._state = torch.empty((n_cells, self.n_nodes, per_node_state_dim))
        self.decay_rates = torch.empty((n_cells, self.n_nodes, per_node_state_dim))
        self.production_rates = torch.empty((n_cells, self.n_nodes, per_node_state_dim))

        self.scale_factor_state_prior = scale_factor_state_prior

        # Initialize
        self.reset_state()

    @property
    def state(self):
        return self._state

    def sample_from_state_prior_dist(self, shape=None) -> torch.Tensor:
        """
        Method which samples from the prior distribution of the state of the cell population.

        Args:
            shape: Shape of the output sample. By default, it will return a sample of the same shape as the current
                state.

        Returns:
            Tensor with the same shape as `self.state`
        """
        if shape is None:
            shape = self.state.shape

        return self.scale_factor_state_prior * torch.ones(shape)

    def reset_state(self):
        """
        Resets the state, production_rates and decay_rates attributes of the cell population.
        """
        self._state = self.sample_from_state_prior_dist()
        self.production_rates = torch.empty(self.state.shape)
        self.decay_rates = torch.empty(self.state.shape)

    def __getitem__(
        self, key: Union[str, Tuple[str, str, str]]
    ) -> Union[sets.NodeSet, sets.EdgeSet]:
        if type(key) is tuple:
            return self._edge_set_dict[key]
        else:
            return self._node_set_dict[key]

    def __setitem__(
        self,
        key: Union[str, Tuple[str, str, str]],
        value: Union[sets.NodeSet, sets.EdgeSet],
    ):
        if type(key) is tuple:
            assert isinstance(value, sets.EdgeSet)
            assert key not in self._edge_set_dict
            self._edge_set_dict[key] = value
        else:
            assert isinstance(value, sets.NodeSet)
            assert key not in self._node_set_dict
            self._node_set_dict[key] = value

    def __delitem__(self, key: Union[str, Tuple[str, str, str]]):
        if type(key) is tuple:
            del self._edge_set_dict[key]
        else:
            self._delete_node_set(key)

    @property
    def n_cells(self) -> int:
        """(`int`): Number of cells in the population"""
        return self.state.shape[0]

    @property
    def n_nodes(self) -> int:
        """(`int`): Number of nodes in the underlying cell mechanisms."""
        return sum([len(node_set) for node_set in self._node_set_dict.values()])

    @property
    def node_types(self) -> List[str]:
        """(`List[str]`): List the different types of nodes. Each node type is associated with a NodeSet object."""
        return list(self._node_set_dict.keys())

    @property
    def edge_types(self) -> List[Tuple[str, str, str]]:
        """(`List[str]`): List the different types of edges. Each edge type is associated with an EdgeSet object."""
        return list(self._edge_set_dict.keys())

    @abstractmethod
    def compute_production_rates(self) -> None:
        """Abstract method. Should update `self.production_rates`"""
        pass

    @abstractmethod
    def compute_decay_rates(self) -> None:
        """Abstract method. Should update `self.decay_rates`"""
        pass

    def get_production_rates(self) -> torch.Tensor:
        """Computes and returns the production rates of the system."""
        self.compute_production_rates()
        return self.production_rates

    def get_decay_rates(self) -> torch.Tensor:
        """Computes and returns the decay rates of the system."""
        self.compute_decay_rates()
        return self.decay_rates

    def get_derivatives(self, state: torch.Tensor) -> torch.Tensor:
        """Computes and returns the time derivatives of the system for a given state.

        Args:
            state: State of the Cell Population for which derivatives should be computed.

        Returns:
            time derivatives of all the tracked properties of the Cell Population.
        """
        self._state = state
        return self.get_production_rates() - self.get_decay_rates()

    def _get_node_set(self, n_type_data: SetData) -> sets.NodeSet:
        """
        Given node type data Dict[AttributeName, AttributeList], returns a `NodeSet` with the associated attributes.
        """
        idx_low = int(min(n_type_data["idx"]))
        idx_high = int(max(n_type_data["idx"])) + 1
        n_type_data.pop("idx", None)

        return sets.NodeSet(self, idx_low, idx_high, attribute_dict=n_type_data)

    def _get_edge_set(self, e_type: EdgeType, e_type_data: SetData) -> sets.EdgeSet:
        """
        Given edge type data Dict[AttributeName, AttributeList], returns an `EdgeSet` with the associated attributes.
        """
        edges = e_type_data["idx"]
        # We shift the edge tail/head indices by idx_low for the source/target node type
        edges[:, 0] -= self[e_type[0]].idx_low  # e_type[0] = Source
        edges[:, 1] -= self[e_type[2]].idx_low  # e_type[2] = Target
        e_type_data.pop("idx", None)

        return sets.EdgeSet(edges, attribute_dict=e_type_data)

    def initialize_from_interaction_graph(
        self, interaction_graph: InteractionData
    ) -> None:
        """Initalizaes NodeSet and EdgeSets from an InteractionData object.

        Args:
            interaction_graph: Interaction graph from which `NodeSet` and `EdgeSet` objects should be initialized.
        """
        node_data_dict = interaction_graph.get_formatted_node_data()
        edge_data_dict = interaction_graph.get_formatted_edge_data()

        for n_type, n_type_data in node_data_dict.items():
            self[n_type] = self._get_node_set(n_type_data)

        for e_type, e_type_data in edge_data_dict.items():
            self[e_type] = self._get_edge_set(e_type, e_type_data)

    def set_production_rates_to_zero(self) -> None:
        """Sets all production rates for all nodes to zero."""
        for n_type in self.node_types:
            self[n_type].production_rates = torch.zeros(
                self[n_type].production_rates.shape
            )

    def _extend_state(self, n_added_nodes: int) -> None:
        """Appends a number of nodes initialized to the state of the cell population.

        The shapes of the production rates and decay rates get updated accordingly.

        Args:
            n_added_nodes: Number of nodes to be added
        """
        added_node_shape = (self.n_cells, n_added_nodes, self.state.shape[2])
        added_state = self.sample_from_state_prior_dist(added_node_shape)

        self._state = torch.cat([self._state, added_state], dim=1)

        # Production and decay rates also get extended
        self.production_rates = torch.cat(
            [self.production_rates, torch.empty(added_node_shape)], dim=1
        )
        self.decay_rates = torch.cat(
            [self.decay_rates, torch.empty(added_node_shape)], dim=1
        )

    def append_node_set(
        self,
        n_type: str,
        n_added_nodes: int,
        attribute_dict: Dict[str, torch.Tensor] = None,
    ):
        """Adds a node set object to the cell population.

        Args:
            n_type: Name of the node type to be added.
            n_added_nodes: Number of nodes in the set to be added.
            attribute_dict: Dict of node attributes.
        """
        assert isinstance(n_type, str)
        self._extend_state(n_added_nodes)
        self[n_type] = sets.NodeSet(
            self,
            self.n_nodes,
            self.n_nodes + n_added_nodes,
            attribute_dict=attribute_dict,
        )

    def _delete_node_set(self, n_type: str) -> None:
        """Removes a node set from the cell population object.

        The state / production rates / decay rates get truncated accordingly.

        Args:
            n_type: Node type to be removed.
        """
        assert isinstance(n_type, str)

        node_set_to_be_del = self[n_type]

        # Remove the corresponding state / production rates / decay rates
        self._state = torch.cat(
            [
                self.state[:, : node_set_to_be_del.idx_low],
                self.state[:, node_set_to_be_del.idx_high :],
            ],
            dim=1,
        )

        self.production_rates = torch.cat(
            [
                self.production_rates[:, : node_set_to_be_del.idx_low],
                self.production_rates[:, node_set_to_be_del.idx_high :],
            ],
            dim=1,
        )

        self.decay_rates = torch.cat(
            [
                self.decay_rates[:, : node_set_to_be_del.idx_low],
                self.decay_rates[:, node_set_to_be_del.idx_high :],
            ],
            dim=1,
        )

        # Removing the node set
        del self._node_set_dict[n_type]

        # Adapt the indices of other node sets
        for other_n_type in self.node_types:
            if self[other_n_type].idx_low >= node_set_to_be_del.idx_high:
                self[other_n_type].idx_low -= len(node_set_to_be_del)
                self[other_n_type].idx_high -= len(node_set_to_be_del)

    def parameters(self, recurse: bool = True):
        """Yields all cell population parameters."""
        for k, n_set in self._node_set_dict.items():
            yield from n_set.parameters(recurse=recurse)
        for k, e_set in self._edge_set_dict.items():
            yield from e_set.parameters(recurse=recurse)
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def get_interaction_data(self):
        g = nx.DiGraph()

        for n_type in self.node_types:
            n_set = self[n_type]
            n_set_attr_dict = self[n_type].element_level_attr_dict

            n_set_list_attr_dict = {k: v for k, v in n_set_attr_dict.items() if isinstance(v, list)}
            n_set_tensor_attr_dict = {k: v for k, v in n_set_attr_dict.items() if not isinstance(v, list)}

            for idx in range(n_set.idx_low, n_set.idx_high):
                node_list_attr = {k: v[idx - n_set.idx_low] for k, v in n_set_list_attr_dict.items()}
                node_tensor_attr = {k: v[:, idx - n_set.idx_low] for k, v in n_set_tensor_attr_dict.items()}
                g.add_node(idx, type=n_type, **node_list_attr, **node_tensor_attr)

        for e_type in self.edge_types:
            e_set = self[e_type]
            e_set_attr_dict = self[e_type].element_level_attr_dict

            e_set_list_attr_dict = {k: v for k, v in e_set_attr_dict.items() if isinstance(v, list)}
            e_set_tensor_attr_dict = {k: v for k, v in e_set_attr_dict.items() if not isinstance(v, list)}

            for e_idx in range(len(e_set)):
                edge_list_attr = {k: v[e_idx] for k, v in e_set_list_attr_dict.items()}
                edge_tensor_attr = {k: v[:, e_idx] for k, v in e_set_tensor_attr_dict.items()}
                src_idx = int(e_set.edges[e_idx, 0] + self[e_type[0]].idx_low)
                tgt_idx = int(e_set.edges[e_idx, 1] + self[e_type[2]].idx_low)
                g.add_edge(src_idx, tgt_idx, type=e_type[1], **edge_list_attr, **edge_tensor_attr)

        return InteractionData(g)

    def draw(self):
        self.get_interaction_data().draw()

    def __repr__(self):
        return "CellPopulation. {} nodes and {} cells.\n".format(
            self.n_nodes, self.n_cells
        )

    def __str__(self):
        s = self.__repr__()

        s += "\tNodeSets:\n"
        for k, v in self._node_set_dict.items():
            s += "\t\t{}: {}\n".format(k, v)

        s += "\tEdgeSets:\n"
        for k, v in self._edge_set_dict.items():
            s += "\t\t{}: {}\n".format(k, v)

        return s


########################################################################################################################
# Cell Population classes
########################################################################################################################


class TestCellPop(CellPopulation):
    def __init__(self, n_cells: int = 1):
        """
        Basic Test Cell Population.

        Mechanisms are based on the calcium signaling pathway from KEGG:

        * 60 nodes and 57 edges.
        * 2 different types of nodes: ['compound', 'gene'].
        * 5 different types of interactions: ['', 'activation', 'binding/association', 'compound', 'inhibition'].

        Each edge type is associated with a graph convolution operation. Together these graph convolutions are used to
        compute the production rates:

        ```
        self[tgt_n_type].production_rates += self[e_type].simple_conv(
            x=self[src_n_type].state,
            edge_index=self[e_type].edges.T,
            edge_weight=self[e_type].weights,
        )
        ```

        Decay rates are exponential decays:

        ```
        self[n_type].decay_rates = exponential_decay(self, n_type, alpha=self[n_type].alpha)
        ```

        Args:
            n_cells: Number of cells in the population
        """
        interaction_graph = load_interaction_data("test")
        super().__init__(interaction_graph, n_cells=n_cells)

        # Initialize additional node attributes.
        self["gene"].init_param(name="alpha", dist=Normal(5, 1))
        self["compound"].init_param(name="alpha", dist=Normal(5, 1))

        # Initialize additional edge attributes.
        for e_type in self.edge_types:
            self[e_type].init_param(name="weights", dist=Normal(0, 1))
            self[e_type].simple_conv = SimpleConv(tgt_nodeset_len=len(self[e_type[2]]))

    def compute_production_rates(self):
        self.set_production_rates_to_zero()
        for e_type in self.edge_types:
            src_n_type, interaction_type, tgt_n_type = e_type
            self[tgt_n_type].production_rates += self[e_type].simple_conv(
                x=self[src_n_type].state,
                edge_index=self[e_type].edges.T,
                edge_weight=self[e_type].weights,
            )

        self.production_rates = torch.sigmoid(self.production_rates)

    def compute_decay_rates(self):
        for n_type in self.node_types:
            self[n_type].decay_rates = dc.exponential_decay(
                self, n_type, alpha=self[n_type].alpha
            )


class ProteinRNACellPop(CellPopulation):
    def __init__(self, n_cells: int = 1):
        """
        Cell Population which tracks the concentration of RNA and the concentration of protein for each gene.

        Mechanisms are based on the calcium signaling pathway from KEGG:

        * 60 nodes and 57 edges.
        * 2 different types of nodes: ['compound', 'gene'].
        * 5 different types of interactions: ['', 'activation', 'binding/association', 'compound', 'inhibition'].

        Each edge type is associated with a graph convolution operation. Together these graph convolutions are used to
        compute the production rates:

        For edges between genes, of type ("gene", *, "gene"),  messages are passed from the source proteins to the
        target RNA. This aims at modeling transcriptional regulation by Transcription Factor proteins:

        ```
        rna_prod_rates += self[e_type].simple_conv(
            x=protein_state,
            edge_index=self[e_type].edges.T,
            edge_weight=self[e_type].weights,
        )
        ```

        For the other types of edges, default graph convolutions are used:

        ```
        self[tgt_n_type].production_rates += self[e_type].simple_conv(
            x=self[src_n_type].state,
            edge_index=self[e_type].edges.T,
            edge_weight=self[e_type].weights,
        )
        ```

        Decay rates are exponential decays:

        ```
        self[n_type].decay_rates = exponential_decay(self, n_type, alpha=self[n_type].alpha)
        ```

        Args:
            n_cells: Number of cells in the population
        """
        interaction_graph = load_interaction_data("test")
        super().__init__(interaction_graph, n_cells=n_cells, per_node_state_dim=2)

        # Initialize additional node attributes.
        self["gene"].init_param(
            name="alpha", dist=Normal(5, 1), shape=(1, len(self["gene"]), 2)
        )
        self["gene"].init_param(
            name="translation_rates", dist=Normal(5, 1), shape=(1, len(self["gene"]), 1)
        )

        self["compound"].init_param(
            name="alpha", dist=Normal(5, 0.01), shape=(1, len(self["compound"]), 2)
        )

        # Initialize additional edge attributes.
        for e_type in self.edge_types:
            self[e_type].init_param(name="weights", dist=Normal(0, 1))
            self[e_type].simple_conv = SimpleConv(tgt_nodeset_len=len(self[e_type[2]]))

    def compute_production_rates(self):
        self.set_production_rates_to_zero()

        for e_type in self.edge_types:
            src_n_type, interaction_type, tgt_n_type = e_type

            if e_type[0] == e_type[2] == "gene":  # Edges between genes
                # RNA production depends on the concentration of parent proteins
                rna_prod_rates = self["gene"].production_rates[:, :, 0:1]
                protein_state = self["gene"].state[:, :, 1:2]

                rna_prod_rates += self[e_type].simple_conv(
                    x=protein_state,
                    edge_index=self[e_type].edges.T,
                    edge_weight=self[e_type].weights,
                )
            else:
                # Regular message passing
                self[tgt_n_type].production_rates += self[e_type].simple_conv(
                    x=self[src_n_type].state,
                    edge_index=self[e_type].edges.T,
                    edge_weight=self[e_type].weights,
                )

        # Protein production depends on the concentration of the RNA coding for that protein
        protein_prod_rates = self["gene"].production_rates[:, :, 1:2]
        protein_prod_rates += (
            self["gene"].translation_rates * self["gene"].state[:, :, 0:1]
        )

    def compute_decay_rates(self):
        for n_type in self.node_types:
            self[n_type].decay_rates = dc.exponential_decay(
                self, n_type, alpha=self[n_type].alpha
            )


class Fantom5CovidCellPop(CellPopulation):
    def __init__(self, n_cells: int = 1):
        interaction_graph = load_interaction_data(
            "fantom5_covid_related_subgraph",
            realnet_tissue_type_file="15_myeloid_leukemia.txt.gz",
        )

        super().__init__(interaction_graph, n_cells=n_cells, per_node_state_dim=1)

        # Initialize additional node attributes.
        for n_type in ["TF_gene", "gene"]:
            self[n_type].init_param(name="alpha", dist=Normal(5, 1))
            self[n_type].init_param(name="translation_rates", dist=Normal(5, 1))
        # Initialize additional edge attributes.
        for e_type in self.edge_types:
            self[e_type].init_param(name="weights", dist=Normal(0, 1))
            self[e_type].simple_conv = SimpleConv(tgt_nodeset_len=len(self[e_type[2]]))

    def compute_production_rates(self):
        self.set_production_rates_to_zero()

        for e_type in self.edge_types:
            src_n_type, interaction_type, tgt_n_type = e_type
            # Regular message passing
            self[tgt_n_type].production_rates += self[e_type].simple_conv(
                x=self[src_n_type].state,
                edge_index=self[e_type].edges.T,
                edge_weight=self[e_type].weights,
            )

    def compute_decay_rates(self):
        for n_type in self.node_types:
            self[n_type].decay_rates = dc.exponential_decay(
                self, n_type, alpha=self[n_type].alpha
            )


if __name__ == "__main__":
    from flecs.trajectory import simulate_deterministic_trajectory
    from flecs.utils import plot_trajectory, set_seed
    import matplotlib.pyplot as plt

    set_seed(0)

    # Simulate trajectories.
    cell_pop = ProteinRNACellPop()

    cell_pop.draw()
    plt.show()

    cell_traj = simulate_deterministic_trajectory(cell_pop, torch.linspace(0, 1, 100))

    plot_trajectory(cell_traj, legend=False)
    plt.show()