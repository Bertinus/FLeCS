import networkx as nx
import numpy as np
import pandas as pd
import os
import torch
import random
from flecs.utils import get_project_root

########################################################################################################################
# RegulonDB
########################################################################################################################


def get_regulondb_graph(tf_only=False):
    """
    Initializes a networkx directed graph based on the RegulonDB database. RegulonDB provides the Escherichia
    coli K-12 Transcriptional Regulatory Network. Available at http://regulondb.ccg.unam.mx/.

    :param tf_only: If True, genes that are not transcription factors are not part of the graph
    :return: Directed networkx graph with 'activator' edge attributes (value in [0, 1]) and 'is_TF' node attributes
        (value in [0, 1])
    """
    gene_to_idx_dict, all_edges, all_edges_is_activator = get_regulondb_edges(tf_only)

    regulon_graph = nx.DiGraph()

    # Initialize all nodes
    for i in range(len(gene_to_idx_dict)):
        regulon_graph.add_node(i)

    # Initialize all edges
    regulon_graph.add_edges_from(all_edges)

    # Add 'activator' edge attributes
    attrs = {
        all_edges[i]: {
            "activator": torch.tensor([all_edges_is_activator[i]]),
            "type": "",
        }
        for i in range(len(all_edges))
    }
    nx.set_edge_attributes(regulon_graph, attrs)

    # Add a node attribute to indicate whether a gene is a transcription factor or not
    if not tf_only:
        all_tf_indices = np.unique([edge[0] for edge in all_edges])
        nx.set_node_attributes(
            regulon_graph,
            {
                i: {"type": "TF" if i in all_tf_indices else ""}
                for i in range(len(gene_to_idx_dict))
            },
        )
    else:
        nx.set_node_attributes(
            regulon_graph, {i: {"type": "TF"} for i in range(len(gene_to_idx_dict))}
        )

    return regulon_graph


def get_regulondb_edges(tf_only=False):
    """
    Get the list of edges and "is_act" edge attributes from the RegulonDB database

    :param tf_only: If True, genes that are not transcription factors are not part of
        the graph.
    :return: gene_to_idx_dict: dictionary mapping gene names to their indices.
            all_edges: list of tuples representing edges in the graph.
            all_edges_is_activator: list with same length as all_edges. Values in
                [0, 1], to indicate whether this edge.
            corresponds to an activator or repressor interaction.
    """
    network_df = load_tf_tf_network() if tf_only else load_tf_gene_network()

    # All names to lowercase
    network_df["TF_name"] = network_df["TF_name"].apply(lambda s: s.lower())
    network_df["regulated_name"] = network_df["regulated_name"].apply(
        lambda s: s.lower()
    )

    # Get list of all genes
    all_genes = list(
        set(network_df["TF_name"].unique()).union(network_df["regulated_name"].unique())
    )
    all_genes.sort()

    # List of TFs which do not have any regulator
    tf_with_no_regulator = list(
        set(network_df["TF_name"].unique()) - set(network_df["regulated_name"].unique())
    )

    # Arbitrarily add negative self loop to all TFs that do not have any regulator
    for tf_name in tf_with_no_regulator:
        network_df = pd.concat(
            [
                network_df,
                pd.DataFrame([[tf_name, tf_name, "-"]], columns=network_df.columns),
            ]
        )

    # Dictionary mapping genes to their node index
    gene_to_idx_dict = {all_genes[i]: i for i in range(len(all_genes))}

    # Add node index information to the dataframe
    network_df["TF_index"] = network_df["TF_name"].apply(
        lambda name: gene_to_idx_dict[name]
    )
    network_df["regulated_index"] = network_df["regulated_name"].apply(
        lambda name: gene_to_idx_dict[name]
    )

    # Create list of edges
    all_edges = list(network_df[["TF_index", "regulated_index"]].to_numpy())
    all_edges = [tuple(edge) for edge in all_edges]

    # For regulatory effects, arbitrarily replace "?" by "-"
    network_df["regulatory_effect"] = network_df["regulatory_effect"].apply(
        lambda e: "-" if e == "?" else e
    )
    network_df["activator"] = network_df["regulatory_effect"].apply(
        lambda e: 1 if e == "+" else 0
    )

    # Get edge attribute
    all_edges_is_activator = list(network_df["activator"].to_numpy())

    return gene_to_idx_dict, all_edges, all_edges_is_activator


def load_and_crop_regulon_db_file(filename, n_header_lines):
    """Loads a .csv file and removes the header."""
    f = open(filename, "r", encoding="ISO-8859-1")
    lines = f.readlines()[n_header_lines + 1 :]  # TODO: how general is this?
    df = pd.DataFrame([line.strip().split("\t") for line in lines])

    return df


def load_tf_gene_network(n_header_lines=38):
    """Loads a transcription factor gene network from RegulonDB.

    Args:
        n_header_lines (int): Number of lines to skip in input file when constructing
            the dataframe.
    Returns: A dataframe containing the transcription factor name, target name, and
        effect valence.
    """
    tf_gene_df = load_and_crop_regulon_db_file(
        os.path.join(
            get_project_root(), "datasets", "RegulonDB", "network_tf_gene.txt"
        ),
        n_header_lines,
    )

    # Name columns.
    tf_gene_df.columns = [
        "TF_ID",
        "TF_name",
        "regulated_ID",
        "regulated_name",
        "regulatory_effect",
        "evidence",
        "evidence_type",
    ]

    # Removes unused columns.
    return tf_gene_df.loc[:, ["TF_name", "regulated_name", "regulatory_effect"]]


def load_tf_tf_network(n_header_lines=35):
    """Loads a transcription factor transcription factor network from RegulonDB.

    Args:
        n_header_lines (int): Number of lines to skip in input file when constructing
            the dataframe.
    Returns: A dataframe containing the transcription factor name, target name, and
        effect valence.
    """
    tf_tf_df = load_and_crop_regulon_db_file(
        os.path.join(get_project_root(), "datasets", "RegulonDB", "network_tf_tf.txt"),
        n_header_lines,
    )

    # Name columns.
    tf_tf_df.columns = [
        "TF_name",
        "regulated_name",
        "regulatory_effect",
        "evidence",
        "evidence_type",
    ]

    # Removes unused columns.
    return tf_tf_df.loc[:, ["TF_name", "regulated_name", "regulatory_effect"]]


def get_realnet_graph(
    path_to_file,
    tf_only=False,
    subsample_edge_prop=1,
):
    """
    Initializes a networkx directed graph based on the RealNet database.
    :return: Directed networkx graph with 'is_TF' node attributes (value in [0, 1])
    """
    network_df = pd.read_csv(
        os.path.join(get_project_root(), "datasets", path_to_file),
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["TF gene", "target gene"],
        compression="gzip",
    )
    network_df = network_df.dropna()

    # Get list of all genes
    list_of_all_genes = list(
        np.unique(network_df["TF gene"].tolist() + network_df["target gene"].tolist())
    )

    if tf_only:
        # Get list of all TFs
        list_of_all_TFs = list(np.unique(network_df["TF gene"].tolist()))

        # Remove edges which point to a gene which is not a TF
        network_df = network_df[
            network_df["target gene"].apply(lambda name: name in list_of_all_TFs)
        ]

        # Update list of all genes
        list_of_all_genes = list(
            np.unique(
                network_df["TF gene"].tolist() + network_df["target gene"].tolist()
            )
        )

    # Create dictionary mapping gene names to their index
    gene_to_idx_dict = {list_of_all_genes[i]: i for i in range(len(list_of_all_genes))}

    # List of TFs which do not have any regulator
    tf_with_no_regulator = list(
        set(network_df["TF gene"].unique()) - set(network_df["target gene"].unique())
    )

    # Arbitrarily add self loop to all TFs that do not have any regulator
    for tf_name in tf_with_no_regulator:
        network_df = pd.concat(
            [
                network_df,
                pd.DataFrame([[tf_name, tf_name]], columns=network_df.columns),
            ]
        )

    # Add node index information to the dataframe
    network_df["TF_index"] = network_df["TF gene"].apply(
        lambda name: gene_to_idx_dict[name]
    )
    network_df["regulated_index"] = network_df["target gene"].apply(
        lambda name: gene_to_idx_dict[name]
    )

    # Create list of edges
    all_edges = list(network_df[["TF_index", "regulated_index"]].to_numpy())
    all_edges = [tuple(edge) for edge in all_edges]

    # If we subsample edges
    if subsample_edge_prop != 1:
        assert 0 < subsample_edge_prop < 1
        random.shuffle(all_edges)
        all_edges = all_edges[: int(subsample_edge_prop * len(all_edges))]

    # Initialize networkx graph
    realnet_graph = nx.DiGraph()

    # Initialize all nodes
    for name, idx in gene_to_idx_dict.items():
        realnet_graph.add_node(idx, name=name)

    # Initialize all edges
    realnet_graph.add_edges_from(all_edges, type="")

    # Add a node attribute to indicate whether a gene is a transcription factor or not
    all_tf_indices = (
        list(gene_to_idx_dict.values())
        if tf_only
        else list(network_df["TF_index"].unique())
    )
    nx.set_node_attributes(
        realnet_graph,
        {
            i: {"type": "TF" if i in all_tf_indices else ""}
            for i in gene_to_idx_dict.values()
        },
    )

    return realnet_graph
