import numpy as np
import torch
from flecs.cell_population import CellPopulation
import networkx as nx


class GRNCellPop(CellPopulation):
    def __init__(self, adata, batch_size, n_latent_var, use_2nd_order_interactions=False):
        adj_mat = adata.varp["grn_adj_mat"]
        self.n_genes = adj_mat.shape[0]
        self.n_latent_var = n_latent_var
        self.ko_gene = None

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

    def compute_production_rates(self):
        if self.ko_gene is not None:
            regulators_state = self.state.clone()
            regulators_state[:, self.ko_gene] = 0.
        else:
            regulators_state = self.state

        gene_regulation_embeddings = self["gene", "regulates", "gene"].conv(x=regulators_state, name="simple_conv")
        self.production_rates = self.activation(gene_regulation_embeddings + self["gene"].bias)

    def compute_decay_rates(self):
        self.decay_rates = self["gene"].alpha * self.state
        self.decay_rates = self.decay_rates

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
