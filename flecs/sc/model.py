import numpy as np
import torch
from flecs.cell_population import CellPopulation
from flecs.utils import get_project_root
import scanpy as sc
import os
import networkx as nx


class GRNCellPop(CellPopulation):
    def __init__(self, adata, batch_size, n_latent_var, use_2nd_order_interactions=False):
        adj_mat = adata.varp["grn_adj_mat"]
        self.n_genes = adj_mat.shape[0]
        self.n_latent_var = n_latent_var

        self.t = 1.  # Trick for training. Default behaviour when t = 1.
        self.ko_gene_embedding = None

        if use_2nd_order_interactions:
            adj_mat = adj_mat.dot(adj_mat).astype(bool).astype(int)

        # Add latent variables
        adj_mat = self._add_latent_variables_to_adj_mat(adj_mat)

        # Create graph
        g = self._init_graph(adj_mat, var_names=list(adata.var.index))

        super().__init__(g, n_cells=batch_size, per_node_state_dim=1, scale_factor_state_prior=1.)

        self["gene", "regulates", "gene"].init_edge_conv(name="simple_conv", out_channels=1)
        self["gene"].init_param(name="bias", dist=torch.distributions.Normal(0, 0.01))
        self["gene"].init_param(name="alpha", dist=torch.distributions.Gamma(2, 10))

        self.relu = torch.nn.ReLU()
        self.activation = torch.nn.Sigmoid()

        self.interventional_model, self.perturbseq = self.initialize_interventional_model()

    def intervene(self, gene_name=None):
        if gene_name is None:
            self.ko_gene_embedding = None
            return

        pert_indices = [x for x in list(self.perturbseq.obs.index) if gene_name.upper() in x]
        assert len(pert_indices) == 1

        pert_index = pert_indices[0]
        self.ko_gene_embedding = torch.tensor(self.perturbseq[pert_index].obsm["X_pca"].copy()[0]).to(self.state.device)

    def initialize_interventional_model(self):
        # Initialize interventional model
        perturbseq = sc.read_h5ad(
            os.path.join(get_project_root(), "datasets", "PerturbSeq", "K562_gwps_raw_bulk_01.h5ad"),
            backed='r'
        )
        sc.tl.pca(perturbseq, svd_solver="arpack", n_comps=128)

        interventional_model = torch.nn.Sequential(torch.nn.Linear(128 + self.n_nodes, self.n_nodes))

        return interventional_model, perturbseq

    def compute_production_rates(self):
        gene_regulation_embeddings = self["gene", "regulates", "gene"].conv(x=self.state, name="simple_conv")

        if self.ko_gene_embedding is not None:
            interv_input = torch.cat((gene_regulation_embeddings, self.ko_gene_embedding[None, :, None]\
                                      * torch.ones((self.state.shape[0], 1, 1)).to(self.state.device)), dim=1)
            interv_effect = self.interventional_model(interv_input[:, :, 0])[:, :, None]

            gene_regulation_embeddings = self.t * gene_regulation_embeddings + interv_effect

        self.production_rates = self.activation(gene_regulation_embeddings + self["gene"].bias)

    def compute_decay_rates(self):
        self.decay_rates = self["gene"].alpha * self.state

    def _add_latent_variables_to_adj_mat(self, adj_mat):
        full_adj_mat = np.ones((self.n_genes + self.n_latent_var, self.n_genes + self.n_latent_var))
        full_adj_mat[:self.n_genes, :self.n_genes] = adj_mat

        return full_adj_mat

    def _init_graph(self, adj_mat, var_names):
        assert len(var_names) == self.n_genes

        g = nx.DiGraph(nx.from_numpy_matrix(adj_mat))
        nx.set_node_attributes(g,
                               {
                                   **{k: v for k, v in enumerate(var_names)},
                                   **{self.n_genes + i: "latent_" + str(i) for i in range(self.n_latent_var)}
                               },
                               name="name")

        return g

    def set_visible_state(self, visible_state):
        """

        Args:
            visible_state:

        Returns:

        """
        if len(visible_state.shape) == 2:
            # Add a third dimension if necessary
            visible_state = visible_state[:, :, None]

        latent_state = torch.zeros((self.state.shape[0], self.n_latent_var, self.state.shape[2])).to(self.state.device)
        self.state = torch.cat((visible_state, latent_state), dim=1)
