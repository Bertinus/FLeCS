import os
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
import torch
from flecs.utils import set_seed, get_project_root
from flecs.sc.dataset import Paul15Dataset
from flecs.sc.utils import train_epoch
from flecs.sc.model import GRNCellPop

########################################################################################################################
# Params
########################################################################################################################

for seed in range(10):
    set_seed(seed)

    train_len = 512
    valid_len = 8
    learning_rate = 0.005
    batch_size = 8

    ####################################################################################################################
    # Build dataloaders
    ####################################################################################################################

    adata = sc.read_h5ad(os.path.join(get_project_root(), "figure2", "processed",
                                      "adata_processed_with_paths_magic.h5ad")
                         )

    cebpe_ko_shortest_paths = adata.uns["cebpe_ko_shortest_paths"]
    cebpe_ko_dataset_late = Paul15Dataset(adata, cebpe_ko_shortest_paths, path_length=10, option="late")
    cebpe_ko_dataset_early = Paul15Dataset(adata, cebpe_ko_shortest_paths, path_length=3, option="early")

    test_len_late = len(cebpe_ko_dataset_late) - train_len - valid_len
    test_len_early = len(cebpe_ko_dataset_early) - train_len - valid_len

    train_dataset_early, valid_dataset_early, test_dataset_early = torch.utils.data.random_split(
        cebpe_ko_dataset_early,
        [train_len, valid_len, test_len_early],
        generator=torch.Generator().manual_seed(0)
    )

    train_dataset_late, valid_dataset_late, test_dataset_late = torch.utils.data.random_split(
        cebpe_ko_dataset_late,
        [train_len, valid_len, test_len_late],
        generator=torch.Generator().manual_seed(0)
    )

    train_dataloader_early = DataLoader(train_dataset_early, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader_early = DataLoader(valid_dataset_early, batch_size=batch_size, shuffle=True, drop_last=True)

    train_dataloader_late = DataLoader(train_dataset_late, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader_late = DataLoader(valid_dataset_late, batch_size=batch_size, shuffle=True, drop_last=True)

    ####################################################################################################################
    # Model
    ####################################################################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mycellpop = GRNCellPop(adata=adata,
                           batch_size=batch_size,
                           n_latent_var=0,
                           use_2nd_order_interactions=False
                           ).to(device)

    # Remove outgoing edges of Cebpe
    cebpe_idx = mycellpop["gene"].name.index('Cebpe')
    outgoing_cebpe_edges = (mycellpop['gene', 'regulates', 'gene'].edges[:, 0] == cebpe_idx)
    mycellpop['gene', 'regulates', 'gene'].remove_edges(outgoing_cebpe_edges)

    optimizer = torch.optim.Adam(mycellpop.parameters(), lr=learning_rate)
    loss = torch.nn.MSELoss()

    ####################################################################################################################
    # Train
    ####################################################################################################################

    for epoch in range(200):
        print("New epoch", epoch)

        if epoch == 100:
            optimizer = torch.optim.Adam(mycellpop.parameters(), lr=learning_rate / 10)

        train_epoch(mycellpop, train_dataloader_late, optimizer, path_length=10, loss=loss, max_n_batch=1)
        train_epoch(mycellpop, train_dataloader_early, optimizer, path_length=3, loss=loss, max_n_batch=10)

    torch.save({
        **mycellpop["gene"].state_dict(),
        **mycellpop["gene", "regulates", "gene"].state_dict()
    },
        "trained_mycellpop_feb_1_3_seed_" + str(seed) + ".pt"
    )
