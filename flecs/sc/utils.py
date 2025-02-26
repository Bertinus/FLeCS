import random

from tqdm import tqdm
import networkx as nx
import sklearn
import scipy
import ot
import numpy as np
import torch
from flecs.trajectory import simulate_deterministic_trajectory
import scanpy as sc

########################################################################################################################
# Methods to train and evaluate models
########################################################################################################################


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


def train_epoch_with_double_flow_matching(model, train_dataloader, optimizer, path_length, loss, max_n_batch=None,
                                   noise_level=1):
    all_train_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpt = 0

    assert path_length == 2

    for expr_traj in train_dataloader:

        expr_traj = expr_traj.to(device)
        optimizer.zero_grad()

        # Sample two times in [0, 1]
        t = torch.sort(torch.rand(2).to(device))[0]

        # Linear interpol. between initial and final expr.
        init_expr = (1-t[0])*expr_traj[:, 0] + t[0]*expr_traj[:, 1]
        final_expr = (1 - t[1]) * expr_traj[:, 0] + t[1] * expr_traj[:, 1]

        # Add noise
        t_interval = t[1] - t[0]
        gaussian_noise = torch.Tensor(noise_level[None, :]).to(device) * \
                         torch.sqrt(t_interval*(1-t_interval))*torch.randn(init_expr.shape[0],
                                                                           init_expr.shape[1]).to(device)

        # Set cell state to the first cell in the trajectory
        model.set_visible_state(init_expr + gaussian_noise)

        # Compute dynamics
        traj = simulate_deterministic_trajectory(model, torch.linspace(t[0].cpu().item(), t[1].cpu().item(),
                                                                       path_length).to(device))
        traj = traj[:, :, :model.n_genes, 0]

        batch_loss = loss(traj[:, 1], final_expr)
        batch_loss.backward()
        all_train_losses.append(batch_loss.item())

        optimizer.step()

        cpt += 1
        if max_n_batch is not None and cpt >= max_n_batch:
            break

    print("train loss", np.mean(all_train_losses))


def train_epoch_with_double_flow_matching_and_interventions(model, train_dataloader, optimizer, path_length, loss,
                                                            max_n_batch=None, noise_level=1):
    all_train_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpt = 0

    assert path_length == 2

    for fp, expr_traj in train_dataloader:

        expr_traj = expr_traj.to(device)
        fp = fp.to(device)
        optimizer.zero_grad()

        # Sample two times in [0, 1]
        t = torch.sort(torch.rand(2).to(device))[0]

        # Linear interpol. between initial and final expr.
        init_expr = (1-t[0])*expr_traj[:, 0] + t[0]*expr_traj[:, 1]
        final_expr = (1 - t[1]) * expr_traj[:, 0] + t[1] * expr_traj[:, 1]

        # Add noise
        t_interval = t[1] - t[0]
        gaussian_noise = torch.Tensor(noise_level[None, :]).to(device) * \
                         torch.sqrt(t_interval*(1-t_interval))*torch.randn(init_expr.shape[0],
                                                                           init_expr.shape[1]).to(device)

        # Set cell state to the first cell in the trajectory
        model.set_visible_state(init_expr + gaussian_noise)
        model.intervene(fp)

        # Compute dynamics
        traj = simulate_deterministic_trajectory(model, torch.linspace(t[0].cpu().item(), t[1].cpu().item(),
                                                                       path_length).to(device))
        traj = traj[:, :, :model.n_genes, 0]

        batch_loss = loss(traj[:, 1], final_expr)
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

########################################################################################################################
# Methods to extract trajectories of cells
########################################################################################################################


def adapt_source_population_size(source_population, target_population, shuffle=False):
    if shuffle:
        random.shuffle(source_population)
    if len(target_population) > len(source_population):
        # Upsample the source population
        q = len(target_population) // len(source_population)
        r = len(target_population) % len(source_population)

        return source_population * q + source_population[:r]

    if len(target_population) < len(source_population):
        # Subsample the source population
        return source_population[:len(target_population)]

    return source_population


def compute_optimal_transport(adata, source_population, target_population, option="PCA", n_pca=20):
    assert option in ["PCA", "Umap"]
    assert len(source_population) == len(target_population)

    if option == "PCA":
        # We only take n_pca components into account
        source_coord = adata.obsm["X_pca"][source_population][:, :n_pca]
        target_coord = adata.obsm["X_pca"][target_population][:, :n_pca]
    else:
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


def compute_cell_knn_paths(_adata,  source_cells, target_cells,  n_bins=10, option="Umap", n_pca=20):
    assert option in ["PCA", "Umap"]

    source_target_adata = _adata[source_cells + target_cells].copy()

    if option == "PCA":
        # We only take n_pca components into account
        all_coord = source_target_adata.obsm["X_pca"][:, :n_pca]
    else:
        all_coord = source_target_adata.obsm["X_umap"]

    # Compute knn adj mat among cells from the environment
    env_knn_adj_mat = compute_knn_weighted_adj_mat(all_coord)

    # Expend the adjacency matrix to include rows and columnns corresponding to cells from other envs
    knn_adj_mat = np.zeros((len(_adata), len(_adata)))
    for idx, env_cell in enumerate(source_cells + target_cells):
        knn_adj_mat[env_cell, source_cells + target_cells] = env_knn_adj_mat[idx]

    # Build knn graph
    knn_graph = nx.from_numpy_matrix(knn_adj_mat)

    # Compute OT matching
    resampled_source_cells = adapt_source_population_size(source_cells, target_cells)
    ot_mapping = compute_optimal_transport(_adata, resampled_source_cells, target_cells, option=option, n_pca=n_pca)

    env_shortest_paths = {}
    # Compute shortest paths
    for match in tqdm(ot_mapping.T):
        env_shortest_paths[match[1]] = nx.shortest_path(knn_graph,
                                                        source=match[0],
                                                        target=match[1],
                                                        weight="weight")

    if n_bins is not None:
        # We subsample the paths to make sure they are of length n_bins
        env_shortest_paths = {k: v for k, v in env_shortest_paths.items() if len(v) >= n_bins}
        env_shortest_paths = {k: subsample_to_length(v, n_bins) for k, v in env_shortest_paths.items()}

    # Convert keys to strings
    env_shortest_paths = {str(k): v for k, v in env_shortest_paths.items()}

    return env_shortest_paths


def subsample_to_length(path_list, target_length):
    assert len(path_list) >= target_length

    _step = (len(path_list)-1) // (target_length-1)
    rest = (len(path_list)-1) % (target_length-1)

    idx = 0
    subsampled_path_list = [path_list[idx]]
    for i in range(target_length - 1 - rest):
        idx += _step
        subsampled_path_list.append(path_list[idx])
    for i in range(rest):
        idx += _step + 1
        subsampled_path_list.append(path_list[idx])

    return subsampled_path_list


def compute_pseudotime_quantiles(_adata, n_bins=10):
    _adata.obs["pseudotime_quantile"] = 0

    for q in range(n_bins):
        quantile = np.quantile(_adata.obs["dpt_pseudotime"], q/n_bins)

        _adata.obs["pseudotime_quantile"] += (_adata.obs["dpt_pseudotime"] >= quantile).astype(int)


def compute_dpt_quantile_paths(_adata, source_cells, target_cells, root_index, invert_dpt=False, n_bins=10,
                                   option="Umap", n_pca=None):
    assert option in ["PCA", "Umap"]
    source_target_adata = _adata[source_cells + target_cells].copy()

    source_target_adata.uns['iroot'] = root_index

    # Compute pseudotime
    sc.tl.diffmap(source_target_adata)
    sc.tl.dpt(source_target_adata)
    if invert_dpt:
        source_target_adata.obs["dpt_pseudotime"] = 1 - source_target_adata.obs["dpt_pseudotime"]

    # Compute quantiles
    compute_pseudotime_quantiles(source_target_adata, n_bins=n_bins)

    # Compute OT matchings between consecutive bins
    env_shortest_paths = {}

    for i in range(1, n_bins):
        i_cells = source_target_adata.obs[source_target_adata.obs["pseudotime_quantile"] == i].index.tolist()
        i_p_1_cells = source_target_adata.obs[source_target_adata.obs["pseudotime_quantile"] == i + 1].index.tolist()

        i_cells = [int(c) for c in i_cells]
        i_p_1_cells = [int(c) for c in i_p_1_cells]

        resampled_i_cells = adapt_source_population_size(i_cells, i_p_1_cells)
        ot_mapping = compute_optimal_transport(_adata, resampled_i_cells, i_p_1_cells, option=option, n_pca=n_pca)

        for match in ot_mapping.T:
            if match[0] in env_shortest_paths.keys():
                traj = env_shortest_paths[match[0]]
                traj.append(match[1])
                del env_shortest_paths[match[0]]
            else:
                traj = [match[0], match[1]]

            env_shortest_paths[match[1]] = traj

    # Convert keys to strings
    env_shortest_paths = {str(k): v for k, v in env_shortest_paths.items()}

    return env_shortest_paths
