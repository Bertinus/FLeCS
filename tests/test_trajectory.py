import pytest
import torch
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from flecs.cell_population import CellPopulation
from flecs.grn import RandomGRN
from flecs.attribute import EdgeAttribute, GeneAttribute
from flecs.structural_equation import SigmoidLinearSE
from flecs.trajectory import (
    simulate_deterministic_trajectory,
    simulate_deterministic_trajectory_euler_steps,
    simulate_stochastic_trajectory,
)
from flecs.utils import plot_trajectory


@pytest.fixture
def my_cells():
    grn = RandomGRN(10, 3)
    linear_se = SigmoidLinearSE(
        gene_decay=GeneAttribute(dim=(1,), prior_dist=Gamma(10, 10)),
        weights=EdgeAttribute(dim=(1,), prior_dist=Normal(1, 1)),
    )

    cells = CellPopulation(grn=grn, structural_equation=linear_se)

    cells.state = 10 * torch.ones((1, 10, 1))

    return cells


def test_simulate_deterministic_trajectory(my_cells):
    time_range = torch.linspace(0, 1, 100)

    cell_traj = simulate_deterministic_trajectory(my_cells, time_range)

    assert cell_traj.shape == (100, 1, my_cells._grn.n_genes, 1)

    assert not torch.equal(cell_traj[0], cell_traj[-1])


def test_simulate_deterministic_trajectory_euler_steps(my_cells):
    time_range = torch.linspace(0, 1, 100)

    cell_traj = simulate_deterministic_trajectory_euler_steps(my_cells, time_range)

    assert cell_traj.shape == (100, 1, my_cells._grn.n_genes, 1)

    assert not torch.equal(cell_traj[0], cell_traj[-1])


def test_simulate_stochastic_trajectory(my_cells):
    time_range = torch.linspace(0, 1, 100)

    cell_traj = simulate_stochastic_trajectory(my_cells, time_range)

    assert cell_traj.shape == (100, 1, my_cells._grn.n_genes, 1)

    assert not torch.equal(cell_traj[0], cell_traj[-1])


def test_solver_consistent_with_euler_steps(my_cells):
    time_range = torch.linspace(0, 1, 1000)
    euler_steps_cell_traj = simulate_deterministic_trajectory_euler_steps(
        my_cells, time_range
    )

    my_cells.state = 10 * torch.ones((1, 10, 1))
    solver_cell_traj = simulate_deterministic_trajectory(my_cells, time_range)

    assert torch.isclose(
        solver_cell_traj.reshape(-1),
        euler_steps_cell_traj.reshape(-1),
        rtol=1e-2,
        atol=1e-2,
    ).all()


def test_plot_trajectory(my_cells):
    time_range = torch.linspace(0, 1, 100)
    cell_traj = simulate_stochastic_trajectory(my_cells, time_range)

    # With timepoints
    plot_trajectory(cell_traj, time_points=time_range)

    # Without timepoints
    plot_trajectory(cell_traj)

    my_cells.state = 10 * torch.ones((3, 10, 1))
    cell_traj = simulate_stochastic_trajectory(my_cells, time_range)
    with pytest.raises(RuntimeWarning):
        plot_trajectory(cell_traj, time_points=time_range)
