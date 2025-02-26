import os
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
import torch
from flecs.utils import set_seed, get_project_root
from flecs.sc.dataset import Paul15Dataset
from flecs.sc.utils import train_epoch
from flecs.sc.model import ProteinLevelPop, GRNCellPop
import numpy as np

########################################################################################################################
# Params
########################################################################################################################

set_seed(0)
train_len = 1000  # 6000  # 2048
valid_len = 1
learning_rate = 0.0005  # 0.005
batch_size = 8
sparsity_rand_edges = 0.1

########################################################################################################################
# Build dataloaders
########################################################################################################################

adata = sc.read_h5ad(os.path.join(get_project_root(),
                                    "datasets", "PerturbSeq", "processed",
                                  "adata_processed_with_obs_pseudotime_paths.h5ad")
                     )

# Remove last time point in S -> G2 paths
s_g2_obs_paths = adata.uns["s_g2_obs_shortest_paths"]
s_g2_obs_paths = {k: v[:-2] for k, v in s_g2_obs_paths.items()}

unsorted_shortest_paths = {**adata.uns["g1_s_obs_shortest_paths"], **s_g2_obs_paths,
                           **adata.uns["g2_g1_obs_shortest_paths"]}

unsorted_dataset_late_1 = Paul15Dataset(adata, unsorted_shortest_paths, path_length=6, option="late")
unsorted_dataset_late_2 = Paul15Dataset(adata, unsorted_shortest_paths, path_length=4, option="late")
unsorted_dataset_late_3 = Paul15Dataset(adata, unsorted_shortest_paths, path_length=2, option="late")
unsorted_dataset_early = Paul15Dataset(adata, unsorted_shortest_paths, path_length=3, option="early")

print("dataset length", len(unsorted_dataset_late_1), len(unsorted_dataset_early))

test_len_late_1 = len(unsorted_dataset_late_1) - train_len - valid_len
test_len_late_2 = len(unsorted_dataset_late_2) - train_len - valid_len
test_len_late_3 = len(unsorted_dataset_late_3) - train_len - valid_len
test_len_early = len(unsorted_dataset_early) - train_len - valid_len

train_dataset_early, valid_dataset_early, test_dataset_early = torch.utils.data.random_split(
    unsorted_dataset_early,
    [train_len, valid_len, test_len_early],
    generator=torch.Generator().manual_seed(0)
)

train_dataset_late_1, valid_dataset_late_1, test_dataset_late_1 = torch.utils.data.random_split(
    unsorted_dataset_late_1,
    [train_len, valid_len, test_len_late_1],
    generator=torch.Generator().manual_seed(0)
)

train_dataset_late_2, valid_dataset_late_2, test_dataset_late_2 = torch.utils.data.random_split(
    unsorted_dataset_late_2,
    [train_len, valid_len, test_len_late_2],
    generator=torch.Generator().manual_seed(0)
)

train_dataset_late_3, valid_dataset_late_3, test_dataset_late_3 = torch.utils.data.random_split(
    unsorted_dataset_late_3,
    [train_len, valid_len, test_len_late_3],
    generator=torch.Generator().manual_seed(0)
)

train_dataloader_early = DataLoader(train_dataset_early, batch_size=batch_size, shuffle=True, drop_last=True)
train_dataloader_late_1 = DataLoader(train_dataset_late_1, batch_size=batch_size, shuffle=True, drop_last=True)
train_dataloader_late_2 = DataLoader(train_dataset_late_2, batch_size=batch_size, shuffle=True, drop_last=True)
train_dataloader_late_3 = DataLoader(train_dataset_late_3, batch_size=batch_size, shuffle=True, drop_last=True)

########################################################################################################################
# Model
########################################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mycellpop = GRNCellPop(adata=adata,
                            batch_size=batch_size,
                            n_latent_var=0,
                            use_2nd_order_interactions=False
                            ).to(device)

optimizer = torch.optim.Adam(mycellpop.parameters(), lr=learning_rate)
loss = torch.nn.MSELoss()

########################################################################################################################
# Train
########################################################################################################################

print("Start training")

for epoch in range(1200):
    print("New epoch", epoch)

    if epoch == 600:
        optimizer = torch.optim.Adam(mycellpop.parameters(), lr=learning_rate / 10)

    train_epoch(mycellpop, train_dataloader_early, optimizer, path_length=3, loss=loss, max_n_batch=1)

    train_epoch(mycellpop, train_dataloader_late_1, optimizer, path_length=6, loss=loss, max_n_batch=1)
    train_epoch(mycellpop, train_dataloader_late_2, optimizer, path_length=4, loss=loss, max_n_batch=1)
    train_epoch(mycellpop, train_dataloader_late_3, optimizer, path_length=2, loss=loss, max_n_batch=2)

torch.save({
    **mycellpop["gene"].state_dict(),
    **mycellpop["gene", "regulates", "gene"].state_dict()
},
    os.path.join(get_project_root(), "logs", "trained_mycellpop_jan_02_3.pt")
)
