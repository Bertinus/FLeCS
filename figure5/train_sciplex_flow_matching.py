import os
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
import torch
from flecs.utils import set_seed, get_project_root
from flecs.sc.dataset import Paul15Dataset
from flecs.sc.utils import train_epoch, train_epoch_with_double_flow_matching
from flecs.sc.model import ProteinLevelPop, GRNCellPop
import numpy as np

########################################################################################################################
# Params
########################################################################################################################

set_seed(0)
train_len = 1024  # 6000  # 2048
valid_len = 1
learning_rate = 0.0005  # 0.005
batch_size = 8
sparsity_rand_edges = 0.1
noise_scaling_factor = 0.000001

########################################################################################################################
# Build dataloaders
########################################################################################################################

adata = sc.read_h5ad(os.path.join(get_project_root(),
                                  "datasets/Sciplex3/processed/SrivatsanTrapnell2020_sciplex3_with_all_paths.h5ad")
                     )

gene_std = adata.X.std(axis=0)

# Add random edges to counter the extreme sparsity of the original adjacency matrix
adj_mat = adata.varp["grn_adj_mat"]
adj_mat = adj_mat + np.random.choice([0, 1], size=adj_mat.shape, p=[1.-sparsity_rand_edges, sparsity_rand_edges])
print("edge density", adj_mat.sum() / (4000 * 4000))
adata.varp["grn_adj_mat"] = adj_mat

obs_shortest_paths = adata.uns["obs_shortest_paths"]
obs_dataset = Paul15Dataset(adata, obs_shortest_paths, path_length=2, option="late")
test_len = len(obs_dataset) - train_len - valid_len

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    obs_dataset, [train_len, valid_len, test_len], generator=torch.Generator().manual_seed(0)
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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

for epoch in range(50):
    train_epoch_with_double_flow_matching(mycellpop, train_dataloader, optimizer, path_length=2, loss=loss,
                                          max_n_batch=None, noise_level=noise_scaling_factor*gene_std)

torch.save({
    **mycellpop["gene"].state_dict(),
    **mycellpop["gene", "regulates", "gene"].state_dict()
},
    os.path.join(get_project_root(), "logs", "trained_mycellpop_mar_12_1.pt")
)
