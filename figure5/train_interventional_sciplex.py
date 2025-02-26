import os
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
import torch
from flecs.utils import set_seed, get_project_root
from flecs.sc.dataset import SciplexDataset
from flecs.sc.utils import train_epoch_with_double_flow_matching_and_interventions
from flecs.sc.model import SciplexGRNCellPop
import numpy as np

########################################################################################################################
# Params
########################################################################################################################

set_seed(0)
train_len = 30  # Number of environments
valid_len = 14
learning_rate = 0.0005
batch_size = 8
sparsity_rand_edges = 0.1
noise_scaling_factor = 0.000001

########################################################################################################################
# Preprocess data
########################################################################################################################

adata = sc.read_h5ad(os.path.join(get_project_root(),
                                  "datasets/Sciplex3/processed/SrivatsanTrapnell2020_sciplex3_with_all_paths.h5ad")
                     )

gene_std = adata.X.std(axis=0)

# Add random edges to counter the extreme sparsity of the original adjacency matrix
adj_mat = adata.varp["grn_adj_mat"]
adj_mat = adj_mat + np.random.choice([0, 1], size=adj_mat.shape, p=[1. - sparsity_rand_edges, sparsity_rand_edges])
print("edge density", adj_mat.sum() / (4000 * 4000))
adata.varp["grn_adj_mat"] = adj_mat

# Retrieve fingerprints
ChEMBL_to_morgan_fp_dict = adata.uns["ChEMBL_to_morgan_fp_dict"]

# Build dictionary mapping names to chembl-IDs
indices_with_id = ~adata.obs["chembl-ID"].isna() & adata.obs["chembl-ID"].apply(lambda x: x in ChEMBL_to_morgan_fp_dict)
perts_and_chembl = adata[indices_with_id].obs[['perturbation', 'chembl-ID']].copy()
perts_and_chembl.drop_duplicates(inplace=True)
pert_to_chembl_dict = perts_and_chembl.set_index("perturbation").to_dict()['chembl-ID']

# Retrieve perts for which paths are available
avail_paths = [k for k in list(adata.uns.keys()) if k.endswith("shortest_paths")]
avail_pert_with_path = [p[:-15] for p in avail_paths]

# Build dictionary mapping names to fingerprints
usable_perts = list(set(avail_pert_with_path).intersection(list(pert_to_chembl_dict.keys())))
usable_perts.sort()
usable_perts_to_fp_dict = {p: ChEMBL_to_morgan_fp_dict[pert_to_chembl_dict[p]] for p in usable_perts}

########################################################################################################################
# Build dataloaders
########################################################################################################################

train_perts, valid_perts = torch.utils.data.random_split(
    usable_perts, [train_len, valid_len], generator=torch.Generator().manual_seed(0)
)
train_perts, valid_perts = list(train_perts), list(valid_perts)

print("train_perts", train_perts)

train_perts_with_fp_dict = {k: v for k, v in usable_perts_to_fp_dict.items() if k in train_perts}
valid_perts_with_fp_dict = {k: v for k, v in usable_perts_to_fp_dict.items() if k in valid_perts}

train_dataset = SciplexDataset(adata, train_perts_with_fp_dict)
valid_dataset = SciplexDataset(adata, valid_perts_with_fp_dict)

train_dataset_in_samp, train_dataset_out_samp = \
    torch.utils.data.random_split(train_dataset, 
                                  [int(0.8*len(train_dataset)), len(train_dataset) - int(0.8*len(train_dataset))], 
                                  generator=torch.Generator().manual_seed(0)
                                  )

# Save datasets indices for later validation
np.save(os.path.join(get_project_root(), "figure5", "train_dataset_pert_paths.npy"), np.array(train_dataset.pert_paths, dtype=object))
np.save(os.path.join(get_project_root(), "figure5", "sciplex_in_sample_indices.npy"), train_dataset_in_samp.indices)
np.save(os.path.join(get_project_root(), "figure5", "sciplex_out_sample_indices.npy"), train_dataset_out_samp.indices)

train_dataloader = DataLoader(train_dataset_in_samp, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

########################################################################################################################
# Model
########################################################################################################################

print("Initialize model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mycellpop = SciplexGRNCellPop(adata=adata,
                              batch_size=batch_size,
                              n_latent_var=0,
                              use_2nd_order_interactions=False
                              ).to(device)

mycellpop.t = 0.  # Trick for training

# Load trained weights
trained_state_dict = torch.load(os.path.join(get_project_root(), "logs/trained_mycellpop_mar_12_1.pt"))
mycellpop["gene"].alpha.data = trained_state_dict["alpha"]
mycellpop["gene"].bias.data = trained_state_dict["bias"]
mycellpop["gene", "regulates", "gene"].simple_conv_weights.data = trained_state_dict["simple_conv_weights"]

mycellpop["gene"].alpha.requires_grad = False
mycellpop["gene"].bias.requires_grad = False
mycellpop["gene", "regulates", "gene"].simple_conv_weights.requires_grad = False

optimizer = torch.optim.Adam(mycellpop.parameters(), lr=learning_rate)
loss = torch.nn.MSELoss()

########################################################################################################################
# Train
########################################################################################################################

print("Start training")

for epoch in range(100):
    train_epoch_with_double_flow_matching_and_interventions(mycellpop, train_dataloader, optimizer, path_length=2,
                                                            loss=loss, max_n_batch=40,
                                                            noise_level=noise_scaling_factor * gene_std)

    if epoch >= 10 and mycellpop.t < 1.:
        mycellpop.t += 0.02

    print("t", mycellpop.t)

torch.save({
    **mycellpop["gene"].state_dict(),
    **mycellpop["gene", "regulates", "gene"].state_dict(),
    'interv_decay': mycellpop.interventional_model_dec.state_dict(),
    'interv_prod': mycellpop.interventional_model_prod.state_dict()
},
    os.path.join(get_project_root(), "logs", "trained_mycellpop_mar_19_1_new_4.pt")
)
