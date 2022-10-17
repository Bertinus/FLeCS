from Bio.KEGG.KGML.KGML_parser import read
import networkx as nx
import numpy as np
import os
from flecs.utils import get_project_root


def load_calcium_signaling_pathway() -> nx.DiGraph:
    pathway = read(open(os.path.join(get_project_root(), 'datasets', 'KEGG', 'hsa04020.xml'), 'r'))

    pathway.remove_entry(pathway.orthologs[0])
    for i in range(6):
        pathway.remove_entry(pathway.maps[0])

    G = nx.DiGraph()

    # Add gene nodes
    for i in range(len(pathway.genes)):
        node_id = pathway.genes[i].id
        node_type = pathway.genes[i].type
        node_name = pathway.genes[i].name

        G.add_node(node_id)
        G.nodes[node_id]["type"] = node_type
        G.nodes[node_id]["name"] = node_name.split(' ')[0]

    # Add compound nodes
    for i in range(len(pathway.compounds)):
        node_id = pathway.compounds[i].id
        node_type = pathway.compounds[i].type
        node_name = pathway.compounds[i].name

        G.add_node(node_id)
        G.nodes[node_id]["type"] = node_type
        G.nodes[node_id]["name"] = node_name

    # Add edges
    for i in range(len(pathway.relations)):
        u, v = pathway.relations[i].entry1.id, pathway.relations[i].entry2.id
        if u in G and v in G:  # Make sure that both nodes exist (they are not "groups")
            G.add_edge(u, v)
            if pathway.relations[i].subtypes:
                G.edges[u, v]["type"] = pathway.relations[i].subtypes[0][0]
            else:
                G.edges[u, v]["type"] = ""

    # Merge nodes which have the same name
    unique_node_names = np.unique(list(nx.get_node_attributes(G, 'name').values()))
    node_name_mapping = {name: idx for idx, name in enumerate(unique_node_names)}

    node_mapping = {k: node_name_mapping[v] for k, v in nx.get_node_attributes(G, 'name').items()}
    G = nx.relabel_nodes(G, node_mapping)

    # Relabel nodes
    G = nx.convert_node_labels_to_integers(G)

    return G
