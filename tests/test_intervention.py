import pytest
import torch

from flecs.cell_population import TestCellPop
from flecs.intervention import CrisprIntervention
from flecs.trajectory import simulate_deterministic_trajectory
from flecs.utils import set_seed


@pytest.fixture
def cell_pop():
    return TestCellPop()


def test_inverventions(cell_pop):
    set_seed(0)
    intervention = CrisprIntervention(cell_pop)

    # Simulate trajectory for CRISPR.
    cell_traj_0 = simulate_deterministic_trajectory(cell_pop, torch.linspace(0, 1, 100))

    intervention.intervene(gene=0)  # Intervene on gene 0
    cell_pop.state = 10 * torch.ones(cell_pop.state.shape)  # Set initial state

    # Simulate trajectory twice, with intervention reset between.
    cell_traj_1 = simulate_deterministic_trajectory(cell_pop, torch.linspace(0, 1, 100))

    intervention.reset()  # Reset intervention
    cell_pop.state = 10 * torch.ones(cell_pop.state.shape)  # Set initial state

    cell_traj_2 = simulate_deterministic_trajectory(cell_pop, torch.linspace(0, 1, 100))

    assert torch.allclose(cell_traj_2[-1, ...], cell_traj_0[-1, ...])
    assert not torch.allclose(cell_traj_2[-1, ...], cell_traj_1[-1, ...])
    assert not torch.allclose(cell_traj_1[-1, ...], cell_traj_0[-1, ...])
