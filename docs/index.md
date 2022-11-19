<img src="flecs_logo.png" alt="flecs_logo" width="600"/>

# Flexible and Learnable Cell Simulations


## Overview

TODO

## Installation

TODO

## Quick usage

Minimal example to train a `CellPopulation` object:

```
from flecs.cell_population import TestCellPop
from flecs.utils import set_seed
from flecs.trajectory import simulate_deterministic_trajectory
import torch

set_seed(0)
n_epochs = 10
incubation_time_range = torch.linspace(0, 1, 100)

# Initialize a ground truth CellPopulation and a trainable CellPopulation
ground_truth_cellpop = TestCellPop()
trainable_cellpop = TestCellPop()

# Cast some tensors of the trainable CellPopulation object as a torch Parameters. That's it!
trainable_cellpop["gene"].alpha = torch.nn.Parameter(trainable_cellpop["gene"].alpha)
trainable_cellpop[("gene", "activation", "gene")].weights = torch.nn.Parameter(
    trainable_cellpop[("gene", "activation", "gene")].weights
)

optimizer = torch.optim.Adam(trainable_cellpop.parameters(), lr=0.01)
loss = torch.nn.MSELoss()

# Training loop
for _ in range(n_epochs):
    optimizer.zero_grad()

    # Re-initialize states
    ground_truth_cellpop.reset_state()
    trainable_cellpop.reset_state()

    # Compute trajectories
    gt_cell_traj = simulate_deterministic_trajectory(
        ground_truth_cellpop, incubation_time_range
    )
    tr_cell_traj = simulate_deterministic_trajectory(
        trainable_cellpop, incubation_time_range
    )

    epoch_loss = loss(gt_cell_traj, tr_cell_traj)
    epoch_loss.backward()
    optimizer.step()

    print(epoch_loss.item())
```

