from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_project_root() -> Path:
    """
    Returns:
        str: Path to the root of the project.

    """
    return Path(__file__).parent.parent


def set_seed(k):
    """
    Sets the numpy and torch seeds.

    Args:
        k: seed.
    """
    np.random.seed(k)
    torch.random.manual_seed(k)


########################################################################################################################
# Plotting
########################################################################################################################


def plot_trajectory(trajectory, time_points=None, legend=True):
    """
    Function to plot the time evolution of the state of a cell.

    Args:
        trajectory (torch.Tensor): States of the cell observed at the different time points.
            Shape (n_time_points, 1, n_genes).
        time_points (list or 1D numpy.array): Times of the observations.
        legend (bool): Whether to add a legend to the plot.

    Raises:
        RuntimeWarning: If the trajectory contains more than one cell. Only the trajectory of the first cell is plotted.

    """
    if len(trajectory.shape) == 3:
        trajectory = trajectory[:, None, :, :]
    assert len(trajectory.shape) == 4

    if trajectory.shape[1] != 1:
        raise RuntimeWarning(
            "Trajectory contains {} cells. Plotting the trajectory of the first cell only.".format(
                trajectory.shape[1]
            )
        )

    for gene in range(trajectory.shape[2]):
        if time_points is not None:
            plt.plot(time_points, trajectory[:, 0, gene], label="gene " + str(gene))
        else:
            plt.plot(trajectory[:, 0, gene], label="gene " + str(gene))
    plt.xlabel("time")
    plt.ylabel("gene expressions")
    if legend:
        plt.legend()
    plt.grid()
