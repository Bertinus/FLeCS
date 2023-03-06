import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import pandas as pd
from torch.distributions import Normal

from flecs.cell_population import CellPopulation
from flecs.production import SimpleConv
from flecs.sets import EdgeSet, NodeSet
from flecs.utils import get_project_root

########################################################################################################################
# Intervention Abstract class
########################################################################################################################


class Intervention(ABC):
    """
    Abstract class responsible for intervening on cells, and resetting cells to their default states.
    """

    @abstractmethod
    def intervene(self, *args, **kwargs) -> None:
        """
        Abstract method for intervening on cells.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Abstract method for resetting cells to their default state.
        """


########################################################################################################################
# Intervention classes
########################################################################################################################


# TODO: Generalize Add Virus.
class SARSCov2Intervention(Intervention):
    def __init__(
        self,
        cells: CellPopulation,
    ):
        self.cells = cells

        # Load covid human interactions
        covid_human_interactions = pd.read_csv(
            os.path.join(
                get_project_root(), "datasets", "SarsCov2", "covid_krogan_ppi.csv"
            )
        )

        self.covid_proteins = (
            covid_human_interactions["covid_protein"].unique().tolist()
        )

        self.edges_to_genes, self.edges_to_TF = self.get_edge_indices(
            covid_human_interactions
        )

    @staticmethod
    def get_index(cells, n):
        if n in cells["TF_gene"].name:
            return cells["TF_gene"].name.index(n)
        elif n in cells["gene"].name:
            return cells["gene"].name.index(n)
        else:
            return None

    def intervene(self, sars_cov_2_concentration: float = 10.0) -> None:
        # Add a node set
        self.cells.append_node_set(
            "sars_cov_2_protein",
            n_added_nodes=len(self.covid_proteins),
            attribute_dict={"name": self.covid_proteins},
        )

        # Set the concentration of covid protein nodes
        self.cells["sars_cov_2_protein"].state = sars_cov_2_concentration

        # Define nodeset parameters needed to compute production / decay rates
        self.cells["sars_cov_2_protein"].init_param(name="alpha", dist=Normal(5, 0.01))

        # Add edge sets
        self.cells["sars_cov_2_protein", "disrupts", "gene"] = EdgeSet(
            edges=self.edges_to_genes
        )
        self.cells["sars_cov_2_protein", "disrupts", "TF_gene"] = EdgeSet(
            edges=self.edges_to_TF
        )

        # Define nodeset parameters needed to compute production / decay rates
        for e_type in [
            ("sars_cov_2_protein", "disrupts", "gene"),
            ("sars_cov_2_protein", "disrupts", "TF_gene"),
        ]:
            self.cells[e_type].init_param(name="weights", dist=Normal(-10, 1))
            self.cells[e_type].simple_conv = SimpleConv(
                tgt_nodeset_len=len(self.cells[e_type[2]])
            )

    def reset(self) -> None:
        del self.cells["sars_cov_2_protein"]
        del self.cells["sars_cov_2_protein", "disrupts", "gene"]
        del self.cells["sars_cov_2_protein", "disrupts", "TF_gene"]

    def get_edge_indices(self, covid_human_interactions):
        # Compute the indices of source covid nodes
        covid_human_interactions["covid_protein_idx"] = covid_human_interactions[
            "covid_protein"
        ].apply(lambda n: self.covid_proteins.index(n))

        # Check whether the target is a regular gene or a transcription factor
        covid_human_interactions["targets_gene"] = covid_human_interactions[
            "human_gene_hgnc_id"
        ].apply(lambda n: n in self.cells["gene"].name)
        covid_human_interactions["targets_TF_gene"] = covid_human_interactions[
            "human_gene_hgnc_id"
        ].apply(lambda n: n in self.cells["TF_gene"].name)

        # Compute the indices of target human nodes
        covid_human_interactions["human_gene_idx"] = covid_human_interactions[
            "human_gene_hgnc_id"
        ].apply(lambda n: self.get_index(self.cells, n))

        # Get edge index tensors
        edges_to_genes = torch.Tensor(
            covid_human_interactions[covid_human_interactions["targets_gene"]][
                ["covid_protein_idx", "human_gene_idx"]
            ].to_numpy()
        ).long()

        edges_to_TF = torch.Tensor(
            covid_human_interactions[covid_human_interactions["targets_TF_gene"]][
                ["covid_protein_idx", "human_gene_idx"]
            ].to_numpy()
        ).long()

        return edges_to_genes, edges_to_TF


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    from flecs.cell_population import Fantom5CovidCellPop
    from flecs.trajectory import simulate_deterministic_trajectory
    from flecs.utils import plot_trajectory, set_seed

    set_seed(0)

    cell_pop = Fantom5CovidCellPop()
    intervention = SARSCov2Intervention(cell_pop)

    # Simulate trajectory
    cell_traj = simulate_deterministic_trajectory(cell_pop, torch.linspace(0, 0.1, 100))
    plot_trajectory(cell_traj[:, :, :900], legend=False)
    plt.show()

    # Intervene on gene 0
    intervention.intervene(sars_cov_2_concentration=10)
    # Set initial state
    cell_pop.reset_state()

    # Simulate trajectory
    cell_traj = simulate_deterministic_trajectory(cell_pop, torch.linspace(0, 0.1, 100))
    plot_trajectory(cell_traj[:, :, :900], legend=False)
    plt.show()

    # Reset intervention
    intervention.reset()
    # Set initial state
    cell_pop.reset_state()

    # Simulate trajectory
    cell_traj = simulate_deterministic_trajectory(cell_pop, torch.linspace(0, 0.1, 100))
    plot_trajectory(cell_traj[:, :, :900], legend=False)
    plt.show()
