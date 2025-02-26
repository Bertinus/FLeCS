import os
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
import torch
from flecs.utils import set_seed, get_project_root
from flecs.sc.dataset import Paul15Dataset
from flecs.sc.utils import train_epoch, subsample_to_length
from flecs.sc.model import GRNCellPop

########################################################################################################################
# Params
########################################################################################################################

set_seed(0)
train_len = 2048
valid_len = 256
learning_rate = 0.005
batch_size = 8
path_length = 3

########################################################################################################################
# Build dataloaders
########################################################################################################################

adata = sc.read_h5ad(os.path.join(get_project_root(),
                                  "figure2", "adata_processed_with_paths_magic.h5ad")
                     )

unsorted_shortest_paths = adata.uns["unsorted_shortest_paths"]
# We subsample the paths to make sure they are of length n_bins
unsorted_shortest_paths = {k: v for k, v in unsorted_shortest_paths.items() if len(v) >= path_length}
unsorted_shortest_paths = {k: subsample_to_length(v, path_length) for k, v in unsorted_shortest_paths.items()}
# Initialize dataset
unsorted_dataset = Paul15Dataset(adata, unsorted_shortest_paths, path_length=path_length, option="late")

test_len = len(unsorted_dataset) - train_len - valid_len

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    unsorted_dataset, [train_len, valid_len, test_len], generator=torch.Generator().manual_seed(0)
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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

for epoch in range(10):
    print("New epoch", epoch)

    if epoch == 5:
        optimizer = torch.optim.Adam(mycellpop.parameters(), lr=learning_rate / 10)

    train_epoch(mycellpop, train_dataloader, optimizer, path_length=path_length, loss=loss)

torch.save({
    **mycellpop["gene"].state_dict(),
    **mycellpop["gene", "regulates", "gene"].state_dict()
},
    "trained_mycellpop_24_jul_19_2.pt"
)
