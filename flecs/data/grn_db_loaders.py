import networkx as nx
import numpy as np
import pandas as pd
import os
import torch
import random
from flecs.utils import get_project_root
import mygene
import hashlib


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
                i: {"type": "TF_gene" if i in all_tf_indices else "gene"}
                for i in range(len(gene_to_idx_dict))
            },
        )
    else:
        nx.set_node_attributes(
            regulon_graph,
            {i: {"type": "TF_gene"} for i in range(len(gene_to_idx_dict))},
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


########################################################################################################################
# RealNet (ENCODE and Fantom5)
########################################################################################################################


def get_realnet_graph(
    path_to_file,
    tf_only=False,
    subsample_edge_prop=1.0,
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
            i: {"type": "TF_gene" if i in all_tf_indices else "gene"}
            for i in gene_to_idx_dict.values()
        },
    )

    return realnet_graph


########################################################################################################################
# String Database
########################################################################################################################


def get_string_graph(
    path_to_file,
    experimental_only=True,
    subsample_edge_prop=1.0,
):
    """
    Initializes a networkx directed graph based on the STRING database.
    :return: Directed networkx graph with 'is_TF' node attributes (value in [0, 1])
    """
    df = pd.read_csv(
        os.path.join(get_project_root(), "datasets", path_to_file),
        sep=" ",
        compression="gzip",
    )

    if experimental_only:
        df = df.loc[df.experimental > 0]

    all_proteins = list(np.unique(df.protein1.tolist() + df.protein2.tolist()))
    protein_to_idx = {all_proteins[i]: i for i in range(len(all_proteins))}

    # Add node index information to the dataframe.
    df["src_protein_index"] = df.protein1.apply(lambda name: protein_to_idx[name])
    df["tgt_protein_index"] = df.protein2.apply(lambda name: protein_to_idx[name])

    # Create list of edges.
    all_edges = list(df[["src_protein_index", "tgt_protein_index"]].to_numpy())
    all_edges = [tuple(edge) for edge in all_edges]

    # If we subsample edges.
    if subsample_edge_prop != 1:
        assert 0 < subsample_edge_prop < 1
        random.shuffle(all_edges)
        all_edges = all_edges[: int(subsample_edge_prop * len(all_edges))]

    # Initialize networkx graph.
    graph = nx.DiGraph()

    # Initialize all nodes.
    for name, idx in protein_to_idx.items():
        ensembl_id = name[5:]  # Remove "9606." at the beginning of the name
        graph.add_node(idx, name=ensembl_id)

    # Initialize all edges.
    graph.add_edges_from(all_edges, type="physical")

    # Add a dummy node attribute to all nodes.
    nx.set_node_attributes(
        graph,
        {i: {"type": "protein"} for i in protein_to_idx.values()},
    )

    return graph


########################################################################################################################
# Composite graph
########################################################################################################################


def get_composite_graph(realnet_graph, string_graph):
    prot_name_to_gene_name_dict = get_protein_gene_mapping(realnet_graph, string_graph)

    # Get subgraph of realnet graph that is covered by the mapping
    realnet_nodes_covered_in_mapping = [
        k
        for k, v in dict(realnet_graph.nodes(data=True)).items()
        if v["name"] in prot_name_to_gene_name_dict.values()
    ]
    realnet_graph = realnet_graph.subgraph(realnet_nodes_covered_in_mapping)

    # Get subgraph of string graph that is covered by the mapping
    string_nodes_covered_in_mapping = [
        k
        for k, v in dict(string_graph.nodes(data=True)).items()
        if v["name"] in prot_name_to_gene_name_dict.keys()
    ]
    string_graph = string_graph.subgraph(string_nodes_covered_in_mapping)

    # Relabel nodes
    realnet_node_list = list(realnet_graph.nodes)
    string_node_list = list(string_graph.nodes)

    new_node_names_realnet = {
        realnet_node_list[i]: i for i in range(len(realnet_node_list))
    }
    new_node_names_string = {
        string_node_list[i]: i + len(realnet_node_list)
        for i in range(len(string_node_list))
    }

    realnet_graph = nx.relabel_nodes(realnet_graph, new_node_names_realnet)
    string_graph = nx.relabel_nodes(string_graph, new_node_names_string)

    # Create composite graph
    composite_graph = nx.DiGraph()

    # Add nodes
    composite_graph.add_nodes_from(realnet_graph.nodes(data=True))
    composite_graph.add_nodes_from(string_graph.nodes(data=True))

    # Add protein interaction edges
    composite_graph.add_edges_from(string_graph.edges(data=True))

    # Add 'gene codes for protein' edges
    name_to_idx_dict = {
        v: k for k, v in nx.get_node_attributes(composite_graph, "name").items()
    }
    gene_codes_for_protein_edges = [
        (name_to_idx_dict[v], name_to_idx_dict[k], {"type": "codes_for"})
        for k, v in prot_name_to_gene_name_dict.items()
    ]
    composite_graph.add_edges_from(gene_codes_for_protein_edges)

    # Add regulation edges
    regulation_edges = []
    for e in realnet_graph.edges(data=True):
        assert e[2] == {"type": ""}
        children_nodes = list(composite_graph[e[0]])
        if len(children_nodes) >= 1:
            # If more than one child, we arbitrarily choose the first one
            src_protein = children_nodes[0]
            regulation_edges.append((src_protein, e[1], {"type": "regulates"}))
        else:
            raise RuntimeError("Gene node {} has no protein child node.".format(e[0]))

    composite_graph.add_edges_from(regulation_edges)

    return composite_graph


def get_protein_gene_mapping(realnet_graph, string_graph):
    mg = mygene.MyGeneInfo()

    # Build ID mapping dictionary
    hashcode = hashlib.sha256(
        (
            str(realnet_graph.nodes(data=True)) + str(string_graph.nodes(data=True))
        ).encode("utf-8")
    ).hexdigest()

    path_to_mapping_dict = os.path.join(
        get_project_root(),
        "datasets",
        "STRING",
        "prot_name_to_gene_name_dict_" + hashcode + ".npy",
    )

    if os.path.isfile(path_to_mapping_dict):
        prot_name_to_gene_name_dict = np.load(
            path_to_mapping_dict, allow_pickle=True
        ).item()
    else:
        print("Building the mapping dictionary. Only happens the first time.")
        realnet_gene_name_set = set(
            nx.get_node_attributes(realnet_graph, name="name").values()
        )
        string_prot_name_set = set(
            nx.get_node_attributes(string_graph, name="name").values()
        )

        query_results = mg.querymany(string_prot_name_set, scopes="ensembl.protein")

        prot_name_to_gene_name_dict = {}
        for q in query_results:
            if "symbol" not in q.keys():
                print("No symbol", q)
            elif q["query"] in q.keys():
                print("Key already in dict")
            elif q["symbol"] in realnet_gene_name_set:
                # Only add mapping if the gene name is covered in the realnet database
                prot_name_to_gene_name_dict[q["query"]] = q["symbol"]

        np.save(path_to_mapping_dict, prot_name_to_gene_name_dict)

    return prot_name_to_gene_name_dict


if __name__ == "__main__":
    from flecs.data.interaction_data import load_interaction_data

    load_interaction_data("string")
