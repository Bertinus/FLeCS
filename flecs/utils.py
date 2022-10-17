from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_project_root() -> str:
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


########################################################################################################################
# Low Rank Attention
########################################################################################################################

class LowRankAttention(torch.nn.Module):
    """
    Low Rank Global Attention. see https://arxiv.org/pdf/2006.07846.pdf
    """

    def __init__(self, k, d, dropout):
        """
        :param k: rank of the attention matrix
        :param d: dimension of the embeddings on which attention is performed
        :param dropout: probability of dropout
        """
        super().__init__()
        self.w = torch.nn.Sequential(torch.nn.Linear(d, 4 * k), torch.nn.ReLU())
        self.activation = torch.nn.ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X):
        tmp = self.w(X)
        u = tmp[:, : self.k]
        v = tmp[:, self.k: 2 * self.k]
        z = tmp[:, 2 * self.k: 3 * self.k]
        t = tmp[:, 3 * self.k:]
        v_t = torch.t(v)
        # normalization
        d = joint_normalize2(u, v_t)
        res = torch.mm(u, torch.mm(v_t, z))
        res = torch.cat((res * d, t), dim=1)
        return self.dropout(res)


def joint_normalize2(U, V_T):
    # U and V_T are in block diagonal form
    if torch.cuda.is_available():
        tmp_ones = torch.ones((V_T.shape[1], 1)).to("cuda")
    else:
        tmp_ones = torch.ones((V_T.shape[1], 1))
    norm_factor = torch.mm(U, torch.mm(V_T, tmp_ones))
    norm_factor = (torch.sum(norm_factor) / U.shape[0]) + 1e-6
    return 1 / norm_factor


def weight_init(layer):
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)


class MyAttention(torch.nn.Module):

    def __init__(self, k, d):
        """
        :param k: rank of the attention matrix
        :param d: dimension of the embeddings on which attention is performed
        """
        super().__init__()
        self.w = torch.nn.Sequential(torch.nn.Linear(d, 2 * k), torch.nn.ReLU())
        self.apply(weight_init)
        self.k = k

    def forward(self, X, z):
        tmp = self.w(X)
        u = tmp[:, : self.k]
        v = tmp[:, self.k: 2 * self.k]
        v_t = torch.t(v)
        res = torch.mm(u, torch.mm(v_t, z))
        # normalization
        d = joint_normalize2(u, v_t)
        res *= d
        return res


if __name__ == '__main__':

    import torch
    X = torch.randn((200, 50))
    my_att = MyAttention(k=100, d=50)
