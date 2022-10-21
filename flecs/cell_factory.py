import torch
from flecs.node_set import NodeSet
from flecs.edge_set import EdgeSet
from torch.distributions.normal import Normal
from flecs.cell_population import CellPopulation
from flecs.data.interaction_data import load_interaction_data, InteractionGraph
import torch_scatter


class CellFactory:
    """
    Class that initializes a cell population.
    """
    def __init__(self, interaction_graph: InteractionGraph):

        self._node_data_dict = interaction_graph.get_formatted_node_data()
        self._edge_data_dict = interaction_graph.get_formatted_edge_data()

    def initialize_cell_pop(self):

        class TestCellPop(CellPopulation):
            def __init__(self):
                super().__init__()

            def get_production_rates(self):
                for n_type in self.node_types:
                    self[n_type].production_rate = torch.zeros(self[n_type].production_rate.shape)

                # Pass messages
                for e_type in self.edge_types:
                    src_n_type = e_type[0]
                    tgt_n_type = e_type[2]

                    parent_indices = self[e_type].tails
                    children_indices = self[e_type].heads

                    state_of_parent_indices = self[src_n_type].state[:, parent_indices]
                    edge_messages = self[e_type]["weights"] * state_of_parent_indices

                    torch_scatter.scatter(edge_messages, children_indices, dim=1, out=self[tgt_n_type].production_rate)

                return self.production_rates

            def get_decay_rates(self):
                for n_type in self.node_types:
                    self[n_type].decay_rate = self[n_type]["alpha"] * self[n_type].state

                return self.decay_rates

        return TestCellPop()

    def generate_cell_pop(self) -> CellPopulation:

        cell_pop = self.initialize_cell_pop()

        for n_type, n_type_data in self._node_data_dict.items():
            cell_pop[n_type] = self.get_node_set(cell_pop, n_type_data)
            cell_pop[n_type]['alpha'] = Normal(5, 0.01).sample((len(cell_pop[n_type]),))

        for e_type, e_type_data in self._edge_data_dict.items():
            cell_pop[e_type] = self.get_edge_set(cell_pop, e_type, e_type_data)
            cell_pop[e_type]['weights'] = Normal(0, 1).sample((len(cell_pop[e_type]),))

        cell_pop.state = 10 * torch.ones((1, cell_pop.n_nodes))
        cell_pop.decay_rates = torch.ones((1, cell_pop.n_nodes))
        cell_pop.production_rates = torch.ones((1, cell_pop.n_nodes))

        return cell_pop

    @staticmethod
    def get_node_set(cell_pop, n_type_data):

        idx_low = int(min(n_type_data['idx']))
        idx_high = int(max(n_type_data['idx']))

        n_type_data.pop('idx', None)
        attr_dict = {k: v for k, v in n_type_data.items() if isinstance(v, torch.Tensor)}

        return NodeSet(cell_pop, idx_low, idx_high, attribute_dict=attr_dict)

    @staticmethod
    def get_edge_set(cell_pop, e_type, e_type_data):

        edges = e_type_data['idx']

        edges[:, 0] -= cell_pop[e_type[0]]._idx_low
        edges[:, 1] -= cell_pop[e_type[2]]._idx_low

        e_type_data.pop('idx', None)
        attr_dict = {k: v for k, v in e_type_data.items() if isinstance(v, torch.Tensor)}

        return EdgeSet(edges, attribute_dict=attr_dict)


def main():
    """Explore this :D"""

    interaction_graph = load_interaction_data("test")
    factory = CellFactory(interaction_graph)

    cell_pop = factory.generate_cell_pop()

    import copy
    import torch

    def simulate_deterministic_trajectory_euler_steps(cells: CellPopulation, time_range: torch.Tensor) -> torch.Tensor:
        # Store cell state at each time step
        trajectory = [copy.deepcopy(cells.state[None, :, :])]

        with torch.no_grad():
            for i in range(1, len(time_range)):
                tau = time_range[i] - time_range[i - 1]
                cells.state += tau * cells.get_derivatives(cells.state)
                trajectory.append(copy.deepcopy(cells.state[None, :, :]))

        return torch.cat(trajectory)

    cell_traj = simulate_deterministic_trajectory_euler_steps(cell_pop, torch.linspace(0, 1, 100))

    from flecs.utils import plot_trajectory
    import matplotlib.pyplot as plt

    plot_trajectory(cell_traj)
    plt.show()


if __name__ == "__main__":
    main()
