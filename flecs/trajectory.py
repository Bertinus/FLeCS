import copy

import torch
import torch.nn.functional as F

from flecs.cell_population import CellPopulation


def simulate_deterministic_trajectory_euler_steps(
    cells: CellPopulation, time_range: torch.Tensor
) -> torch.Tensor:
    """
    Simulates the deterministic trajectory of the cells using Euler's method:
    $$
    \operatorname{state}(t + \Delta t) = \operatorname{state}(t) + {d \operatorname{state} \over dt} \Delta t.
    $$

    Args:
        cells (CellPopulation): CellPopulation.
        time_range (1D torch.Tensor): Time points at which the cells state should be evaluated.

    Returns:
        torch.Tensor: Trajectory of shape (n_time_points, n_cells, n_genes, *state_dim)
    """

    # Store cell state at each time step
    trajectory = [copy.deepcopy(cells.state[None, :, :])]

    with torch.no_grad():
        for i in range(1, len(time_range)):
            tau = time_range[i] - time_range[i - 1]
            cells.state += tau * cells.get_derivatives(cells.state)
            trajectory.append(copy.deepcopy(cells.state[None, :, :]))

    return torch.cat(trajectory)


def simulate_deterministic_trajectory(
    cells: CellPopulation, time_range: torch.Tensor, method="dopri5"
) -> torch.Tensor:
    """
    Simulates the deterministic trajectory of the cells using the ``torchdiffeq`` solver.

    Args:
        cells (CellPopulation): CellPopulation.
        time_range (1D torch.Tensor): Time points at which the cells state should be evaluated.
        method (str): argument for the solver.

    Returns:
        torch.Tensor: Trajectory of shape (n_time_points, n_cells, n_genes, *state_dim)
    """
    from torchdiffeq import odeint

    def derivatives_for_solver(_, state):
        """
        Utility method for compatibility with the solver.
        """
        # Get derivatives
        return cells.get_derivatives(state)

    # Simulate trajectory
    trajectory = odeint(
        derivatives_for_solver, cells.state, time_range, method=method
    )

    return trajectory


def simulate_stochastic_trajectory(cells: CellPopulation, time_range: torch.Tensor):
    """
    Simulates stochastic trajectories of the cell using the tau-leaping method, which is a variation of
    the Gillespie algorithm:
    $$
    \operatorname{state}(t + \Delta t) = \operatorname{state}(t)
    + \operatorname{Pois}[\Delta t \cdot (\operatorname{production rates})]
    - \operatorname{Pois}[\Delta t \cdot (\operatorname{decay rates})].
    $$

    Args:
        cells (CellPopulation): CellPopulation.
        time_range (1D torch.Tensor): Time points at which the cells state should be evaluated.

    Returns:
        torch.Tensor: Trajectory of shape (n_time_points, n_cells, n_genes, *state_dim)
    """
    # Store cell state at each time step
    trajectory = [copy.deepcopy(cells.state[None, :, :])]

    with torch.no_grad():
        for i in range(1, len(time_range)):
            tau = time_range[i] - time_range[i - 1]
            production_rates = cells.get_production_rates(cells.state)
            decay_rates = cells.get_decay_rates(cells.state)

            cells.state += torch.poisson(tau * production_rates) - torch.poisson(
                tau * decay_rates
            )

            # Make sure state is positive
            cells.state = F.relu(cells.state)

            trajectory.append(copy.deepcopy(cells.state[None, :, :]))

    return torch.cat(trajectory)
