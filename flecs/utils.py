import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch


def get_project_root():
    return Path(__file__).parent.parent


def set_seed(k):
    np.random.seed(k)
    torch.random.manual_seed(k)


########################################################################################################################
# Plotting
########################################################################################################################


def plot_trajectory(trajectory, timepoints=None, legend=True):
    """
    Creates a plot showing the time evolution of genes

    :param trajectory: torch tensor or numpy array of shape (n_time_points, n_genes, n_cells)
    :param timepoints: list or 1D numpy array containing the times of the observations
    :param legend: Boolean. Whether to add a legend to the plot
    """
    if trajectory.shape[1] != 1:
        raise RuntimeWarning(
            "Trajectory contains {} cells. Plotting the trajectory of the first cell only.".format(
                trajectory.shape[1]
            )
        )

    for gene in range(trajectory.shape[2]):
        if timepoints is not None:
            plt.plot(timepoints, trajectory[:, 0, gene], label="gene " + str(gene))
        else:
            plt.plot(trajectory[:, 0, gene], label="gene " + str(gene))
    plt.xlabel("time")
    plt.ylabel("gene expressions")
    if legend:
        plt.legend()
    plt.grid()
