from flecs.utils import get_project_root
import numpy as np
import pandas as pd
import os
import networkx as nx
import random


########################################################################################################################
# String Database
########################################################################################################################


def get_string_graph(
    path_to_file: str,
    experimental_only: bool = True,
    subsample_edge_prop: float = 1.0,
) -> nx.DiGraph:
    """
    Initializes a networkx DiGraph based on the String database.

    Args:
        path_to_file: path to the file to load.
        experimental_only: restrict to experimentally validated interactions.
        subsample_edge_prop: Between 0. and 1. If strictly smaller than 1. edges will be subsampled randomly.

    Returns:
        Directed networkx graph with a "type" node attributes (value in ["protein"])
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
