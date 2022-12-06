import os
import mygene
import networkx as nx
import hashlib
from typing import Dict
from flecs.utils import get_project_root
from flecs.data.gene_regulatory_networks import get_realnet_graph
import pandas as pd
import numpy as np

########################################################################################################################
# Composite graph
########################################################################################################################


def get_grn_string_composite_graph(
    realnet_graph: nx.DiGraph, string_graph: nx.DiGraph
) -> nx.DiGraph:
    """
    Initializes a composite graph from a RealNet graph and a String graph. This composite graph contains both "gene"
    nodes and "protein" nodes.

    ("protein" to "protein") interactions are derived from the String graph. Each gene is
    associated with the protein it codes for via a ("gene" to "protein") interaction. If a gene A regulates another
    gene B, according to the RealNet graph, the composite graph contains an edge "protein A" -> "gene B" where
    "protein A" is the protein coded by "gene A"

    Args:
        realnet_graph:
        string_graph:

    Returns:
        Composite graph.

    """
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


def get_covid_related_realnet_subgraph(
    path_to_file: str, subsample_edge_prop: float = 1.0
):
    """
    Loads a realnet graph and keeps the subgraph on all TFs + all genes targeted by Sars-Cov-2.

    Args:
        path_to_file: path to the file to load.
        subsample_edge_prop: Proportion of edges to keep.

    Returns:
        Subgraph as a networkx DiGraph object.

    """
    # Load Fantom5 myeloid leukemia
    g = get_realnet_graph(
        path_to_file=path_to_file,
        tf_only=False,
        subsample_edge_prop=subsample_edge_prop,
    )

    # Let us restrict ourselves to Transcription factors and genes targets by SarsCov2
    covid_human_interactions = pd.read_csv(
        os.path.join(get_project_root(), "datasets", "SarsCov2", "covid_krogan_ppi.csv")
    )

    genes_targeted_by_sarscov2 = covid_human_interactions["human_gene_hgnc_id"].unique()

    kept_genes = [
        k
        for k, v in dict(g.nodes(data=True)).items()
        if (v["name"] in genes_targeted_by_sarscov2 or v["type"] == "TF_gene")
    ]

    return g.subgraph(kept_genes)


########################################################################################################################
# Auxiliary methods
########################################################################################################################


def get_protein_gene_mapping(
    realnet_graph: nx.DiGraph, string_graph: nx.DiGraph
) -> Dict[str, str]:
    """

    Finds the mapping between genes in *realnet_graph* and proteins in *string_graph*. Each protein is associated with
    the gene that codes for that protein.

    Args:
        realnet_graph:
        string_graph:

    Returns:
        prot_name_to_gene_name_dict: Dictionary mapping protein Ensembl names to gene names.

    """
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
