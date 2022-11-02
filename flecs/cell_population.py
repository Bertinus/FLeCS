from abc import ABC, abstractmethod
from flecs.edge_set import EdgeSet
from flecs.node_set import NodeSet
from typing import Tuple, Dict, Union
import torch


class CellPopulation(ABC):
    def __init__(self, interaction_graph, n_cells=1):
        """A population of independnet cells (no cell-cell interactions).

        Args:
            interaction_graph ():
            n_cells (int): Number of independent cells in the population.
        """
        # str type of node (e.g., gene, protein).
        self._node_set_dict: Dict[str, NodeSet] = {}
        # str types of interactions (src, interaction_type, dest).
        self._edge_set_dict: Dict[Tuple[str, str, str], EdgeSet] = {}

        self.initialize_from_interaction_graph(interaction_graph)

        self.state = 10 * torch.ones((n_cells, self.n_nodes))
        self.decay_rates = torch.empty((n_cells, self.n_nodes))
        self.production_rates = torch.empty((n_cells, self.n_nodes))

    def __getitem__(
        self, key: Union[str, Tuple[str, str, str]]
    ) -> Union[NodeSet, EdgeSet]:
        if type(key) is tuple:
            return self._edge_set_dict[key]
        else:
            return self._node_set_dict[key]

    def __setitem__(
        self, key: Union[str, Tuple[str, str, str]], value: Union[NodeSet, EdgeSet]
    ):
        if type(key) is tuple:
            assert isinstance(value, EdgeSet)
            assert key not in self._edge_set_dict
            self._edge_set_dict[key] = value
        else:
            assert isinstance(value, NodeSet)
            assert key not in self._node_set_dict
            self._node_set_dict[key] = value

    @property
    def n_cells(self) -> int:
        return self.state.shape[0]

    @property
    def n_nodes(self) -> int:
        return sum([len(node_set) for node_set in self._node_set_dict.values()])

    @property
    def node_types(self):
        return list(self._node_set_dict.keys())

    @property
    def edge_types(self):
        return list(self._edge_set_dict.keys())

    @abstractmethod
    def get_production_rates(self):
        pass

    @abstractmethod
    def get_decay_rates(self):
        pass

    def get_derivatives(self, state):
        """Estimates derivative of system using first differences."""
        self.state = state
        return self.get_production_rates() - self.get_decay_rates()

    def get_node_set(self, n_type_data):
        """Given node type data, return a node set with the associated attributes."""
        idx_low = int(min(n_type_data["idx"]))
        idx_high = int(max(n_type_data["idx"]))
        n_type_data.pop("idx", None)

        attr_dict = {
            k: v for k, v in n_type_data.items() if isinstance(v, torch.Tensor)
        }

        return NodeSet(self, idx_low, idx_high, attribute_dict=attr_dict)

    def get_edge_set(self, e_type, e_type_data):
        """ "Given an edge type, and edge data, return an edge set."""
        edges = e_type_data["idx"]
        edges[:, 0] -= self[e_type[0]].idx_low  # e_type[0] = Source
        edges[:, 1] -= self[e_type[2]].idx_low  # e_type[2] = Target
        e_type_data.pop("idx", None)

        attr_dict = {
            k: v for k, v in e_type_data.items() if isinstance(v, torch.Tensor)
        }

        return EdgeSet(edges, attribute_dict=attr_dict)

    def initialize_from_interaction_graph(self, interaction_graph):
        """Initializes a graph from an `interaction_graph` object."""
        node_data_dict = interaction_graph.get_formatted_node_data()
        edge_data_dict = interaction_graph.get_formatted_edge_data()

        for n_type, n_type_data in node_data_dict.items():
            self[n_type] = self.get_node_set(n_type_data)

        for e_type, e_type_data in edge_data_dict.items():
            self[e_type] = self.get_edge_set(e_type, e_type_data)

    def set_production_rates_to_zero(self):
        for n_type in self.node_types:
            self[n_type].production_rate = torch.zeros(
                self[n_type].production_rate.shape
            )

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
            s += "\t\t{}: {}".format(k, v)

        return s


if __name__ == "__main__":
    from flecs.data.interaction_data import load_interaction_data
    from flecs.decay import alpha_decay
    from flecs.initializers import init_normal
    from flecs.production import message_passing
    from flecs.trajectory import simulate_deterministic_trajectory
    from flecs.utils import plot_trajectory
    import matplotlib.pyplot as plt

    # Define your own Cell population object
    class TestCellPop(CellPopulation):
        def __init__(self):
            """
            Information about the test interaction data:
                60 nodes and 57 edges.
                2 different types of nodes: ['compound', 'gene'].
                5 different types of interactions: ['', 'activation',
                    'binding/association', 'compound', 'inhibition'].
            """
            interaction_graph = load_interaction_data("test")
            super().__init__(interaction_graph)

            # Initialize additional node attributes.
            init_normal(self, "gene", "alpha", 5, 0.01)
            init_normal(self, "compound", "alpha", 5, 0.01)

            # Initialize additional edge attributes.
            for e_type in self.edge_types:
                init_normal(self, e_type, "weights", 0, 1)

            self.decay_rates_fn = alpha_decay  # Define your own!
            self.production_rate_fn = message_passing  # Define your own!

        def get_production_rates(self):
            """Applies a generic production rate fn to each edge type individually."""
            self.set_production_rates_to_zero()
            for e_type in self.edge_types:
                self.production_rate_fn(self, e_type)

            return self.production_rates

        def get_decay_rates(self):
            """Applies a generic decay fn to each node type individually."""
            for n_type in self.node_types:
                self[n_type].decay_rate = self.decay_rates_fn(self, n_type)

            return self.decay_rates

    # Simulate trajectories.
    cell_pop = TestCellPop()
    cell_traj = simulate_deterministic_trajectory(cell_pop, torch.linspace(0, 1, 100))

    plot_trajectory(cell_traj, legend=False)
    plt.show()
