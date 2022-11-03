import flecs.cell_population
from flecs.cell_population import TestCellPop
from flecs.utils import set_seed
from flecs.trajectory import simulate_deterministic_trajectory
import torch


class TrainableTestCellPop(TestCellPop, torch.nn.Module):
    def __init__(self):
        flecs.cell_population.TestCellPop.__init__(self)
        torch.nn.Module.__init__(self)

        self["gene"]["alpha"] = torch.nn.Parameter(self["gene"]["alpha"])

        self.parameter_list = torch.nn.ParameterList([self["gene"]["alpha"]])


set_seed(0)

ground_truth_cellpop = TestCellPop()
trainable_cellpop = TrainableTestCellPop()

optimizer = torch.optim.Adam(trainable_cellpop.parameters(), lr=0.01)
loss = torch.nn.MSELoss()

for _ in range(10):
    optimizer.zero_grad()

    # Re-initialize states
    ground_truth_cellpop.reset_state()
    trainable_cellpop.reset_state()

    # Compute trajectories
    gt_cell_traj = simulate_deterministic_trajectory(ground_truth_cellpop, torch.linspace(0, 1, 100))
    tr_cell_traj = simulate_deterministic_trajectory(trainable_cellpop, torch.linspace(0, 1, 100))

    epoch_loss = loss(gt_cell_traj, tr_cell_traj)
    epoch_loss.backward()
    optimizer.step()

    print(epoch_loss.item())
