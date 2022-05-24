from flecs.cell import Cell
import torch
import copy
import torch.nn.functional as F


def simulate_deterministic_trajectory_euler_steps(
    cell: Cell, time_range: torch.Tensor
) -> torch.Tensor:
    """
    Simulates the deterministic trajectory of the cell using Euler's method:
    $$
    \operatorname{state}(t + \Delta t) = \operatorname{state}(t) + {d \operatorname{state} \over dt} \Delta t.
    $$

    Args:
        cell (Cell): Cell.
        time_range (1D torch.Tensor): Time points at which the cell state should be evaluated.

    Returns:
        torch.Tensor: Trajectory of shape (n_time_points, n_cells, n_nodes, *state_dim)
    """

    # Store cell state at each time step
    trajectory = [copy.deepcopy(cell.state[None, :, :])]

    with torch.no_grad():
        for i in range(1, len(time_range)):
            tau = time_range[i] - time_range[i - 1]
            cell.state += tau * cell.get_derivatives(cell.state)
            trajectory.append(copy.deepcopy(cell.state[None, :, :]))

    return torch.cat(trajectory)


def simulate_deterministic_trajectory(
    cell: Cell, time_range: torch.Tensor, method="dopri5"
) -> torch.Tensor:
    """
    Simulates the deterministic trajectory of the cell using the ``torchdiffeq`` solver.

    Args:
        cell (Cell): Cell.
        time_range (1D torch.Tensor): Time points at which the cell state should be evaluated.
        method (str): argument for the solver.

    Returns:
        torch.Tensor: Trajectory of shape (n_time_points, n_cells, n_nodes, *state_dim)
    """
    from torchdiffeq import odeint

    def derivatives_for_solver(_, state):
        """
        Utility method for compatibility with the solver.
        """
        # Get derivatives
        return cell.get_derivatives(state)

    # Simulate trajectory
    with torch.no_grad():
        trajectory = odeint(
            derivatives_for_solver, cell.state, time_range, method=method
        )

    return trajectory


def simulate_stochastic_trajectory(cell: Cell, time_range: torch.Tensor):
    """
    Simulates stochastic trajectories of the cell using the tau-leaping method, which is a variation of
    the Gillespie algorithm:
    $$
    \operatorname{state}(t + \Delta t) = \operatorname{state}(t)
    + \operatorname{Pois}[\Delta t \cdot (\operatorname{production rates})]
    - \operatorname{Pois}[\Delta t \cdot (\operatorname{decay rates})].
    $$

    Args:
        cell (Cell): Cell.
        time_range (1D torch.Tensor): Time points at which the cell state should be evaluated.

    Returns:
        torch.Tensor: Trajectory of shape (n_time_points, n_cells, n_nodes, *state_dim)
    """
    # Store cell state at each time step
    trajectory = [copy.deepcopy(cell.state[None, :, :])]

    with torch.no_grad():
        for i in range(1, len(time_range)):
            tau = time_range[i] - time_range[i - 1]
            production_rates = cell.get_production_rates(cell.state)
            decay_rates = cell.get_decay_rates(cell.state)

            cell.state += torch.poisson(tau * production_rates) - torch.poisson(
                tau * decay_rates
            )

            # Make sure state is positive
            cell.state = F.relu(cell.state)

            trajectory.append(copy.deepcopy(cell.state[None, :, :]))

    return torch.cat(trajectory)
