from tqdm import tqdm
import networkx as nx
import seaborn as sns
import sklearn
import scipy
import ot
import matplotlib.pyplot as plt
import numpy as np
import torch
from flecs.trajectory import simulate_deterministic_trajectory


def train_epoch(model, train_dataloader, optimizer, path_length, loss, max_n_batch=None):
    all_train_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpt = 0

    for expr_traj in train_dataloader:

        expr_traj = expr_traj.to(device)
        optimizer.zero_grad()

        # Set cell state to the first cell in the trajectory
        model.set_visible_state(expr_traj[:, 0])

        # Compute dynamics
        traj = simulate_deterministic_trajectory(model, torch.linspace(0, path_length-1, path_length).to(device))
        traj = traj[:, :, :model.n_genes, 0]

        batch_loss = loss(traj, expr_traj)
        batch_loss.backward()
        all_train_losses.append(batch_loss.item())

        optimizer.step()

        cpt += 1
        if max_n_batch is not None and cpt >= max_n_batch:
            break

    print("train loss", np.mean(all_train_losses))


def eval_epoch(model, eval_dataloader, path_length, loss, max_n_batch=None):
    all_eval_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpt = 0

    with torch.no_grad():
        for expr_traj in eval_dataloader:

            expr_traj = expr_traj.to(device)

            # Set cell state to the first cell in the trajectory
            model.set_visible_state(expr_traj[:, 0])

            # Compute dynamics
            traj = simulate_deterministic_trajectory(model, torch.linspace(0, path_length-1, path_length).to(device))
            traj = traj[:, :, :model.n_genes, 0]

            batch_loss = loss(traj, expr_traj)
            all_eval_losses.append(batch_loss.item())

            cpt += 1
            if max_n_batch is not None and cpt >= max_n_batch:
                break

    print("eval loss", np.mean(all_eval_losses))


def get_cell_indices(adata):
    # Get the indices of cells for the different populations
    cmp_cells = list(adata.obs[adata.obs["Batch_desc"] == 'CMP Flt3+ Csf1r+'].index.astype("int"))
    unsorted_cells = list(adata.obs[adata.obs["Batch_desc"] == 'Unsorted myeloid'].index.astype("int"))
    cebpa_cells = list(adata.obs[adata.obs["Batch_desc"] == 'Cebpa KO'].index.astype("int"))
    cebpe_cells = list(adata.obs[adata.obs["Batch_desc"] == 'Cebpe KO'].index.astype("int"))

    return cmp_cells, unsorted_cells, cebpa_cells, cebpe_cells


def adapt_source_population_size(source_population, target_population):
    if len(target_population) > len(source_population):
        # Upsample the source population
        q = len(target_population) // len(source_population)
        r = len(target_population) % len(source_population)

        return source_population * q + source_population[:r]

    if len(target_population) < len(source_population):
        # Subsample the source population
        return source_population[:len(target_population)]

    return source_population


def compute_optimal_transport(adata, source_population, target_population):
    assert len(source_population) == len(target_population)

    source_coord = adata.obsm["X_umap"][source_population]
    target_coord = adata.obsm["X_umap"][target_population]

    pop_size = len(source_coord)
    dist_mat = scipy.spatial.distance_matrix(source_coord, target_coord, p=2)

    T = ot.emd([1.] * pop_size, [1.] * pop_size, dist_mat)
    ot_mapping = np.where(T)

    ot_mapping = np.array([[source_population[idx] for idx in ot_mapping[0]],
                           [target_population[idx] for idx in ot_mapping[1]]])

    return ot_mapping


def compute_knn_weighted_adj_mat(coord):
    knn_adj_mat = sklearn.neighbors.kneighbors_graph(coord, n_neighbors=30).toarray()
    # Compute distances
    dist_mat = scipy.spatial.distance_matrix(coord, coord, p=2)
    # Weighted adjacency matrix
    knn_adj_mat *= dist_mat

    return knn_adj_mat


def compute_env_shortest_paths_with_ot(adata, env):
    assert env in ["Cebpa KO", "Cebpe KO", "Unsorted myeloid"]

    cmp_cells, unsorted_cells, cebpa_cells, cebpe_cells = get_cell_indices(adata)

    cells_from_env = {"Cebpa KO": cebpa_cells,
                      "Cebpe KO": cebpe_cells,
                      "Unsorted myeloid": unsorted_cells}

    all_coord = adata.obsm['X_umap']

    # Compute knn adj mat among cells from the environment
    all_coord_for_env = all_coord[cells_from_env[env] + cmp_cells]
    env_knn_adj_mat = compute_knn_weighted_adj_mat(all_coord_for_env)

    # Expend the adjacency matrix to include rows and columnns corresponding to cells from other envs
    knn_adj_mat = np.zeros((len(all_coord), len(all_coord)))
    for idx, env_cell in enumerate(cells_from_env[env] + cmp_cells):
        knn_adj_mat[env_cell, cells_from_env[env] + cmp_cells] = env_knn_adj_mat[idx]

    # Build knn graph
    knn_graph = nx.from_numpy_matrix(knn_adj_mat)

    # Compute OT matching
    resampled_cmp_cells = adapt_source_population_size(cmp_cells, cells_from_env[env])
    ot_mapping = compute_optimal_transport(adata, resampled_cmp_cells, cells_from_env[env])

    env_shortest_paths = {}
    # Compute shortest paths
    for match in tqdm(ot_mapping.T):
        env_shortest_paths[match[1]] = nx.shortest_path(knn_graph,
                                                        source=match[0],
                                                        target=match[1],
                                                        weight="weight")
    return env_shortest_paths


def plot_cell_type_distribution(adata, cell_type, root_cell=None):
    coords = adata[adata.obs["Batch_desc"] == cell_type].obsm['X_umap']

    plt.scatter(coords[:, 0], coords[:, 1])
    sns.kdeplot(coords[:, 0], coords[:, 1], cmap="Reds")

    if root_cell is not None:
        root_cell_coord = adata.obsm["X_umap"][root_cell]
        plt.scatter(root_cell_coord[0], root_cell_coord[1], marker='*', s=300, zorder=2, c="r")

