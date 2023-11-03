import os
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
import torch
from flecs.utils import set_seed, get_project_root
from flecs.sc.dataset import Paul15Dataset
from flecs.sc.model import GRNCellPop
from flecs.sc.utils import train_epoch

########################################################################################################################
# Params
########################################################################################################################

set_seed(0)
train_len = 2048
valid_len = 64
learning_rate = 0.0005  # 0.0005
batch_size = 8

########################################################################################################################
# Build dataloaders
########################################################################################################################

adata = sc.read_h5ad(os.path.join(get_project_root(),
                                  "datasets", "Paul15", "processed", "adata_processed_with_paths_magic.h5ad")
                     )

cebpa_ko_shortest_paths = adata.uns["cebpa_ko_shortest_paths"]
cebpa_ko_dataset_late = Paul15Dataset(adata, cebpa_ko_shortest_paths, path_length=10, option="late")
cebpa_ko_dataset_early = Paul15Dataset(adata, cebpa_ko_shortest_paths, path_length=3, option="early")

print(len(cebpa_ko_dataset_early), len(cebpa_ko_dataset_late))

test_len_late = len(cebpa_ko_dataset_late) - train_len - valid_len
test_len_early = len(cebpa_ko_dataset_early) - train_len - valid_len

train_dataset_early, valid_dataset_early, test_dataset_early = torch.utils.data.random_split(
    cebpa_ko_dataset_early,
    [train_len, valid_len, test_len_early],
    generator=torch.Generator().manual_seed(0)
)

train_dataset_late, valid_dataset_late, test_dataset_late = torch.utils.data.random_split(
    cebpa_ko_dataset_late,
    [train_len, valid_len, test_len_late],
    generator=torch.Generator().manual_seed(0)
)

train_dataloader_early = DataLoader(train_dataset_early, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataloader_early = DataLoader(valid_dataset_early, batch_size=batch_size, shuffle=True, drop_last=True)

train_dataloader_late = DataLoader(train_dataset_late, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataloader_late = DataLoader(valid_dataset_late, batch_size=batch_size, shuffle=True, drop_last=True)

########################################################################################################################
# Model
########################################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mycellpop = GRNCellPop(adata=adata,
                       batch_size=batch_size,
                       n_latent_var=50,
                       use_2nd_order_interactions=False
                       ).to(device)
mycellpop.t = 0.  # Trick for training

# Load trained weights
trained_state_dict = torch.load("trained_mycellpop_nov_2_6.pt")
mycellpop["gene"].alpha.data = trained_state_dict["alpha"]
mycellpop["gene"].bias.data = trained_state_dict["bias"]
mycellpop["gene", "regulates", "gene"].simple_conv_weights.data = trained_state_dict["simple_conv_weights"]

mycellpop["gene"].alpha.requires_grad = False
mycellpop["gene"].bias.requires_grad = False
mycellpop["gene", "regulates", "gene"].simple_conv_weights.requires_grad = False

optimizer = torch.optim.Adam(mycellpop.parameters(), lr=learning_rate)
loss = torch.nn.MSELoss()

mycellpop.intervene("cebpa")

########################################################################################################################
# Train
########################################################################################################################

for epoch in range(100):
    print("New epoch", epoch)
    if epoch > 10:
        train_epoch(mycellpop, train_dataloader_late, optimizer, path_length=10, loss=loss, max_n_batch=1)
        train_epoch(mycellpop, train_dataloader_early, optimizer, path_length=3, loss=loss, max_n_batch=3)
    else:
        train_epoch(mycellpop, train_dataloader_early, optimizer, path_length=3, loss=loss, max_n_batch=3)

    if epoch == 11:
        # Reinitialize after 10 epochs to avoid momentum from first epochs
        optimizer = torch.optim.Adam(mycellpop.parameters(), lr=learning_rate)

    if epoch >= 20 and mycellpop.t < 1.:
        mycellpop.t += 0.02

    print("t", mycellpop.t)

torch.save({
    **mycellpop["gene"].state_dict(),
    **mycellpop["gene", "regulates", "gene"].state_dict(),
    **mycellpop.interventional_model.state_dict()
},
    "trained_mycellpop_nov_3_7.pt"
)
