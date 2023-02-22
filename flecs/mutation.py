from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch

from flecs.cell_population import CellPopulation
from flecs.sets import NodeSet
from flecs.sets import EdgeSet

from typing import Union

########################################################################################################################
# Mutation functions
########################################################################################################################


def apply_bernoulli_mutation(
    obj: Union[CellPopulation, NodeSet, EdgeSet], attr_name: str, p: float, n_cells: int
) -> None:
    """
    Function which mutates a specific attribute of a CellPopulation/NodeSet/EdgeSet with probability *p*.

    The attribute will first be duplicated so that each cell in the population gets its own copy of the attribute.
    Mutations then happen with probability *p*, and independently for each cell.

    A mutation corresponds to setting the cell-specific attribute to zero.

    Args:
        obj: CellPopulation/NodeSet/EdgeSet whose attribute we want to mutate.
        attr_name: Name of the attribute we want to mutate.
        p: Probability of mutation.
        n_cells: Number of cells in the population.
    """

    duplicate_attribute(obj, attr_name, n_cells)
    noise_dist = Bernoulli(1 - p)

    attr_value = obj.element_level_attr_dict[attr_name]

    obj.__setattr__(attr_name, attr_value * noise_dist.sample(attr_value.shape))


def apply_gaussian_mutation(
    obj: Union[CellPopulation, NodeSet, EdgeSet],
    attr_name: str,
    sigma: float,
    n_cells: int,
):
    """
    Function which mutates a specific attribute of a CellPopulation/NodeSet/EdgeSet by adding a gaussian noise.

    The attribute will first be duplicated so that each cell in the population gets its own copy of the attribute.
    Independent gaussian noises are then added to the cell-specific attributes.

    Args:
        obj: CellPopulation/NodeSet/EdgeSet whose attribute we want to mutate.
        attr_name: Name of the attribute we want to mutate.
        sigma: Standard deviation of the gaussian noise applied.
        n_cells: Number of cells in the population.
    """

    duplicate_attribute(obj, attr_name, n_cells)
    noise_dist = Normal(0, sigma)

    attr_value = obj.element_level_attr_dict[attr_name]

    obj.__setattr__(attr_name, attr_value + noise_dist.sample(attr_value.shape))


########################################################################################################################
# Auxiliary functions
########################################################################################################################


def duplicate_attribute(
    obj: Union[CellPopulation, NodeSet, EdgeSet], attr_name: str, n_cells: int
):
    assert n_cells > 1

    attr_value = obj.element_level_attr_dict[attr_name]

    if attr_value.shape[0] != 1:
        raise RuntimeError(
            "Cannot duplicate an attribute which has already been duplicated."
        )

    duplicated_attr_value = torch.cat([attr_value] * n_cells, dim=0)

    obj.__setattr__(attr_name, duplicated_attr_value)


if __name__ == "__main__":
    from flecs.trajectory import simulate_deterministic_trajectory
    from flecs.utils import plot_trajectory, set_seed
    import matplotlib.pyplot as plt
    import torch
    from flecs.cell_population import TestCellPop

    set_seed(0)

    cell_pop = TestCellPop(n_cells=3)

    # apply_bernoulli_mutation(cell_pop["gene"], "alpha", p=0.5, n_cells=3)

    # apply_gaussian_mutation(cell_pop['gene', 'activation', 'gene'], "weights", sigma=1., n_cells=3)

    # Simulate trajectory
    cell_traj = simulate_deterministic_trajectory(cell_pop, torch.linspace(0, 1, 100))

    plot_trajectory(cell_traj[:, 0], legend=False)
    plt.show()

    plot_trajectory(cell_traj[:, 1], legend=False)
    plt.show()

    plot_trajectory(cell_traj[:, 2], legend=False)
    plt.show()
