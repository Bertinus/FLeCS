import os
from flecs.utils import get_project_root
import torch
import numpy as np
import networkx as nx
from flecs.data.interaction_data import InteractionData
from flecs.data.random_graphs import get_random_adjacency_mat, get_graph_from_adj_mat
from flecs.data.gene_regulatory_networks import get_regulondb_graph, get_realnet_graph
from flecs.data.composite_graphs import get_grn_string_composite_graph
from flecs.data.protein_interactions import get_string_graph
from flecs.data.pathways import get_calcium_signaling_pathway
from typing import List

########################################################################################################################
# Utils to load databases
########################################################################################################################


def available_fantom5_tissue_type_files() -> List[str]:
    """
    List all tissue type files available in the Fantom5 database.

    Returns:
        all high-level Fantom5 networks available."""
    return [
        f
        for f in os.listdir(
            os.path.join(
                get_project_root(),
                "datasets",
                "RealNet",
                "Network_compendium",
                "Tissue-specific_regulatory_networks_FANTOM5-v1",
                "32_high-level_networks",
            )
        )
        if f.strip(".gz").endswith(".txt")  # Matches .txt.gz and .txt
    ]


def load_interaction_data(
    interaction_type: str,
    realnet_tissue_type_file: str = None,
    tf_only: bool = False,
    subsample_edge_prop: float = 1.0,
    n_nodes: int = None,
    avg_num_parents: int = None,
) -> InteractionData:
    """
    Utility function which loads interaction data from one of the databases, and returns it as a `InteractionData`
    object.

    Args:
        interaction_type: available options: ["test", "calcium_pathway", "regulon_db", "encode", "fantom5", "string",
            "composite", "random"].
        realnet_tissue_type_file: only used when `interaction_type` is "fantom5" or "composite".
        tf_only: whether to restrict to transcription factor nodes only.
        subsample_edge_prop:
        n_nodes: only used when `interaction_type` is "random".
        avg_num_parents: only used when `interaction_type` is "random".

    Returns:
        InteractionData
    """
    assert interaction_type in [
        "test",
        "calcium_pathway",
        "regulon_db",
        "encode",
        "fantom5",
        "string",
        "composite",
        "random",
    ]

    if interaction_type == "test":
        nx_graph = get_calcium_signaling_pathway()
        # add a dummy numerical node attribute
        node_attr = {}
        for node in nx_graph.nodes(data=True):
            if node[1]["type"] == "gene":
                node_attr[node[0]] = {
                    "basal_expression": torch.tensor(np.random.exponential())[None]
                }

        nx.set_node_attributes(nx_graph, node_attr)

        # add a dummy numerical edge attribute
        edge_attr = {}
        for edge in nx_graph.edges(data=True):
            if edge[2]["type"] == "activation":
                edge_attr[(edge[0], edge[1])] = {
                    "strength": torch.tensor(np.random.randn())[None]
                }

        nx.set_edge_attributes(nx_graph, edge_attr)
        return InteractionData(nx_graph)

    elif interaction_type == "calcium_pathway":
        return InteractionData(get_calcium_signaling_pathway())

    elif interaction_type == "regulon_db":
        return InteractionData(get_regulondb_graph(tf_only=tf_only))

    elif interaction_type == "encode":
        encode_graph = get_realnet_graph(
            path_to_file=os.path.join(
                "RealNet",
                "Network_compendium",
                "Other_networks",
                "Global_regulatory_ENCODE",
                "ENCODE-nets.proximal_raw.distal.txt.gz",
            ),
            tf_only=tf_only,
            subsample_edge_prop=subsample_edge_prop,
        )
        return InteractionData(encode_graph)

    elif interaction_type == "fantom5":
        if realnet_tissue_type_file not in available_fantom5_tissue_type_files():
            raise ValueError(
                "When loading GRNs from fantom5, the 'realnet_tissue_type_file'"
                "argument needs to be specified. To get available files, run:"
                "'flecs.data.interaction_data.available_tissue_type_files()'"
            )
        fantom5_graph = get_realnet_graph(
            path_to_file=os.path.join(
                "RealNet",
                "Network_compendium",
                "Tissue-specific_regulatory_networks_FANTOM5-v1",
                "32_high-level_networks",
                realnet_tissue_type_file,
            ),
            tf_only=tf_only,
            subsample_edge_prop=subsample_edge_prop,
        )
        return InteractionData(fantom5_graph)

    elif interaction_type == "string":
        string_graph = get_string_graph(
            path_to_file=os.path.join(
                "STRING", "9606.protein.physical.links.detailed.v11.5.txt.gz"
            ),
            experimental_only=True,
            subsample_edge_prop=subsample_edge_prop,
        )
        return InteractionData(string_graph)

    elif interaction_type == "composite":
        string_graph = get_string_graph(
            path_to_file=os.path.join(
                "STRING", "9606.protein.physical.links.detailed.v11.5.txt.gz"
            ),
            experimental_only=True,
            subsample_edge_prop=subsample_edge_prop,
        )

        if realnet_tissue_type_file not in available_fantom5_tissue_type_files():
            raise ValueError(
                "When loading GRNs from fantom5, the 'realnet_tissue_type_file'"
                "argument needs to be specified. To get available files, run:"
                "'flecs.data.interaction_data.available_tissue_type_files()'"
            )

        fantom5_graph = get_realnet_graph(
            path_to_file=os.path.join(
                "RealNet",
                "Network_compendium",
                "Tissue-specific_regulatory_networks_FANTOM5-v1",
                "32_high-level_networks",
                realnet_tissue_type_file,
            ),
            tf_only=tf_only,
            subsample_edge_prop=subsample_edge_prop,
        )

        composite_graph = get_grn_string_composite_graph(fantom5_graph, string_graph)
        return InteractionData(composite_graph)

    elif interaction_type == "random":
        assert n_nodes is not None, "Please specify 'n_nodes'."
        assert avg_num_parents is not None, "Please specify 'avg_num_parents'."
        random_graph = get_graph_from_adj_mat(
            get_random_adjacency_mat(n_nodes=n_nodes, avg_num_parents=avg_num_parents)
        )
        return InteractionData(random_graph)


def main():
    print(
        "calcium_pathway: {}".format(
            load_interaction_data("calcium_pathway").__repr__()
        )
    )
    print("regulon_db: {}".format(load_interaction_data("regulon_db").__repr__()))
    print(
        "regulon_db: {}".format(
            load_interaction_data("regulon_db", tf_only=True).__repr__()
        )
    )
    print("encode: {}".format(load_interaction_data("encode").__repr__()))
    print("encode: {}".format(load_interaction_data("encode", tf_only=True).__repr__()))
    print(
        "encode: {}".format(
            load_interaction_data(
                "encode", tf_only=True, subsample_edge_prop=0.5
            ).__repr__()
        )
    )
    print(
        "fantom5: {}".format(
            load_interaction_data(
                "fantom5", realnet_tissue_type_file="01_neurons_fetal_brain.txt.gz"
            ).__repr__()
        )
    )
    print(
        "fantom5: {}".format(
            load_interaction_data(
                "fantom5",
                realnet_tissue_type_file="01_neurons_fetal_brain.txt.gz",
                tf_only=True,
            ).__repr__()
        )
    )

    print(
        "fantom5: {}".format(
            load_interaction_data(
                "fantom5",
                realnet_tissue_type_file="01_neurons_fetal_brain.txt.gz",
                tf_only=True,
                subsample_edge_prop=0.5,
            ).__repr__()
        )
    )

    print(
        "string: {}".format(
            load_interaction_data(
                "string",
                subsample_edge_prop=0.5,
            ).__repr__()
        )
    )

    print(
        "composite: {}".format(
            load_interaction_data(
                "composite",
                realnet_tissue_type_file="01_neurons_fetal_brain.txt.gz",
            ).__repr__()
        )
    )

    print(
        "random: {}".format(
            load_interaction_data("random", n_nodes=10, avg_num_parents=3).__repr__()
        )
    )


if __name__ == "__main__":
    main()
