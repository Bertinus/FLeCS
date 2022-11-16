from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import itertools
import os
from flecs.utils import get_project_root
from typing import Set, List, Dict, Union, Tuple
from flecs.data.calcium_signaling_pathway import load_calcium_signaling_pathway
from flecs.data.grn_db_loaders import (
    get_realnet_graph,
    get_regulondb_graph,
    get_string_graph,
    get_composite_graph,
)
from flecs.data.random_graph import get_graph_from_adj_mat, get_random_adjacency_mat

# Base types
NodeIdx = int
EdgeIdx = Tuple[int, int]
NodeType = str
EdgeType = str
RelationType = Tuple[str, str, str]
AttributeName = str
Attribute = Union[str, torch.Tensor]
AttributeList = Union[List[str], torch.Tensor]

# Types
Data = Dict[AttributeName, Attribute]
NodeData = Dict[NodeIdx, Data]
EdgeData = Dict[EdgeIdx, Data]
SetData = Dict[AttributeName, AttributeList]
NodeSetData = Dict[NodeType, SetData]
RelationSetData = Dict[RelationType, SetData]


class InteractionGraph(nx.DiGraph):
    def __init__(self, graph: nx.DiGraph):
        """Holds interaction data about a Cell, s.a. Gene Regulatory Networks/Pathways.
        Inherits from nx.DiGraph and is formatted such that:
        - Nodes and edges are typed, based on a "type" attribute
        - Nodes are ordered based on their type
        - All nodes/edges of a given type have the same set of attributes
        - Node/Edge attributes are either strings or torch.Tensor.

        Once instantiated, the structure of the graph is frozen.

        Args:
            graph (nx.DiGraph): Graph used for initialization.
        """

        # Sort the nodes
        graph = self.sorted_by_node_type(graph)

        super().__init__(graph)

        # Assert that all nodes of a given type have exactly the same set of attributes
        self._check_attribute_consistency_within_node_types()

        # Assert that all edges of a given type have exactly the same set of attributes
        self._check_attribute_consistency_within_edge_types()

        # Assert that all node and edge attributes are either strings or tensors
        self._check_attributes_type()

        # Freeze the graph
        nx.freeze(self)

    @property
    def node_types(self) -> List[NodeType]:
        """TODO
        (``List[NodeType]``) List of length (n_nodes) which contains the types of all the nodes.
        """
        return list(nx.get_node_attributes(self, "type").values())

    @property
    def edge_types(self) -> List[EdgeType]:
        """TODO
        (``List[EdgeType]``) List of length (n_edges) which contains the types of all the edges.
        """
        return list(nx.get_edge_attributes(self, "type").values())

    @property
    def unique_node_types(self) -> List[NodeType]:
        return list(np.unique(self.node_types))

    @property
    def unique_edge_types(self) -> List[EdgeType]:
        return list(np.unique(self.edge_types))

    def node_data(self, n_type: str = None) -> NodeData:
        """TODO
        Args:
            n_type (str): node type to restrict to

        Returns:
             NodeData: Dictionary mapping nodes of type n_type (or all nodes if n_type is None) to their data
             dictionaries (mapping attribute names to attribute values).
        """

        def to_be_included(n: Tuple):
            if n_type is not None:
                return n[1]["type"] == n_type
            else:
                return True

        return {n[0]: n[1] for n in self.nodes(data=True) if to_be_included(n)}

    def edge_data(
        self, e_type: str = None, src_type: str = None, tgt_type: str = None
    ) -> EdgeData:
        """TODO
        Args:
            e_type (str): edge type to restrict to
            src_type (str): type of the source node
            tgt_type (str): type of the target node

        Returns:
            Dict[Tuple[int, int], DataDict]: Dictionary mapping edges of type e_type to their data dictionaries
            (mapping attribute names to attribute values). Entries can be further restricted to edges with source node
            of type src_type and target node of type tgt_type.
        """

        def to_be_included(e: Tuple):
            res = e[2]["type"] == e_type if e_type is not None else True
            if src_type is not None:
                res = res and self.node_types[e[0]] == src_type
            if tgt_type is not None:
                res = res and self.node_types[e[1]] == tgt_type
            return res

        return {(e[0], e[1]): e[2] for e in self.edges(data=True) if to_be_included(e)}

    def get_formatted_node_data(self) -> NodeSetData:
        """Formats the data of nodes.

        For all node types n_type, a single dictionary is built. It is such that values
        have length equal to the number of nodes of type n_type. The dictionary also
        contains a key "idx" listing the indices of the nodes of type n_type.

        Returns:
            FormattedDataDict: Keys correspond to node types. The values associated
            with key n_type is a dict containing the data of the nodes of type n_type.
        """
        formatted_node_data = {}
        for n_type in self.unique_node_types:
            formatted_node_data[n_type] = self.format_type_data_dict(
                self.node_data(n_type)
            )

        return formatted_node_data

    def get_formatted_edge_data(self) -> RelationSetData:
        """TODO"""
        formatted_edge_data = {}
        for src_type, e_type, tgt_type in itertools.product(
            self.unique_node_types, self.unique_edge_types, self.unique_node_types
        ):
            e_data_dict = self.edge_data(e_type, src_type=src_type, tgt_type=tgt_type)
            if e_data_dict:
                formatted_edge_data[
                    src_type, e_type, tgt_type
                ] = self.format_type_data_dict(e_data_dict)

        return formatted_edge_data

    @staticmethod
    def format_type_data_dict(set_data_dict: Union[NodeData, EdgeData]) -> SetData:
        """Formats a data dictionary.

        Args:
            set_data_dict (Dict[Union[int, Tuple[int, int]], Data]): Dictionary
                mapping node/edge indices to their Data.

        Returns:
            FormattedDataDict. Keys are attribute names. Values have length equal to
                the number of nodes/edges found in set_data_dict. The dictionary also
                contains a key "idx" listing the indices of the nodes/edges.

        """
        formatted_tdata = {"idx": []}

        # Loop over nodes/edges
        for idx, idx_data in set_data_dict.items():
            formatted_tdata["idx"].append(torch.Tensor([idx]))
            # Loop over attributes of the node/edge and add them to formatted_tdata
            for k, v in idx_data.items():
                if k not in formatted_tdata.keys():
                    formatted_tdata[k] = []
                formatted_tdata[k].append(v)

        # Remove the key "type"
        formatted_tdata.pop("type", None)

        # Concatenate lists of tensors
        for k, v in formatted_tdata.items():
            if isinstance(v[0], torch.Tensor):
                formatted_tdata[k] = torch.cat(v, dim=0)

        return formatted_tdata

    def _check_attribute_consistency_within_node_types(self):
        """Checks that all nodes of a given type have the same set of attributes."""
        for n_type in self.unique_node_types:
            node_data_list = list(self.node_data(n_type).values())
            reference_attr_names = set(node_data_list.pop(0).keys())
            for n_data in node_data_list:
                assert set(n_data.keys()) == reference_attr_names

    def _check_attribute_consistency_within_edge_types(self):
        """Checks that all edges of a given type have the same set of attributes."""
        for e_type in self.unique_edge_types:
            edge_data_list = list(self.edge_data(e_type).values())
            reference_attr_names = set(edge_data_list.pop(0).keys())
            for e_data in edge_data_list:
                assert set(e_data.keys()) == reference_attr_names

    def _check_attributes_type(self) -> None:
        def assert_attributes_are_valid(d: dict):
            assert isinstance(d["type"], str)
            for v in d.values():
                if isinstance(v, torch.Tensor):
                    assert v.shape[0] == 1
                else:
                    assert isinstance(v, str)

        # Check node attributes
        for n_d in self.node_data().values():
            assert_attributes_are_valid(n_d)

        # Check edge attributes
        for e_d in self.edge_data().values():
            assert_attributes_are_valid(e_d)

    @classmethod
    def sorted_by_node_type(cls, g: nx.DiGraph) -> nx.DiGraph:
        """
        Sorts nodes of a graph according to their type.

        Args (nx.DiGraph):
            g: graph with "type" as global node and edge attribute.

        Returns:
            nx.DiGraph: Copy of graph g in which the node ordering is based on node type. Node labels are overwritten
            and set to range(0, n_nodes).

        """
        # Make sure that "type" is a node and edge attribute
        assert "type" in cls.global_node_attr_names(g)
        assert "type" in cls.global_edge_attr_names(g)

        g = nx.convert_node_labels_to_integers(g)
        ordered_nodes = np.argsort(list(nx.get_node_attributes(g, "type").values()))

        g_sorted = nx.DiGraph()
        for n in ordered_nodes:
            g_sorted.add_node(n, **g.nodes(data=True)[n])
        g_sorted.add_edges_from(g.edges(data=True))

        g_sorted = nx.convert_node_labels_to_integers(g_sorted)

        return g_sorted

    @staticmethod
    def global_node_attr_names(g: nx.DiGraph) -> Set[str]:
        """
        A node attribute is considered global if all nodes in the graph have this attribute.

        Args:
            g (nx.DiGraph): graph.

        Returns: Set of names of the global node attributes of graph g.
        """
        node_data_dict = dict(g.nodes(data=True))
        node_attr_name_list = [set(v.keys()) for v in node_data_dict.values()]
        return set.intersection(*node_attr_name_list)

    @staticmethod
    def global_edge_attr_names(g: nx.DiGraph) -> Set[str]:
        """
        An edge attribute is considered global if all edges in the graph have this attribute.

        Args:
            g (nx.DiGraph): graph.

        Returns: Set of names of the global edge attributes of graph g.
        """
        edge_data_dict = {(e[0], e[1]): e[2] for e in list(g.edges(data=True))}
        edge_attr_name_list = [set(v.keys()) for v in edge_data_dict.values()]
        return set.intersection(*edge_attr_name_list)

    def to_digraph(self):
        return nx.DiGraph(self)

    def __str__(self):
        string = self.__repr__()
        string += "Nodes:\n"
        for i in list(self.nodes):
            string += "\t node " + str(i) + " " + str(self.nodes[i]) + "\n"
        string += "Edges:\n"
        for i in self.edges.data():
            string += "\t edge " + str(i) + "\n"
        return string

    def __repr__(self):
        string = (
            "GraphData. "
            + str(self.number_of_nodes())
            + " nodes and "
            + str(self.number_of_edges())
            + " edges.\n"
        )
        string += (
            str(len(self.unique_node_types))
            + " different types of nodes: "
            + str(self.unique_node_types)
            + ".\n"
        )
        string += (
            str(len(self.unique_edge_types))
            + " different types of edges: "
            + str(self.unique_edge_types)
            + ".\n"
        )

        return string

    def draw(self) -> None:
        """
        Method to draw the GRN, using the circular layout.
        """
        pos = nx.circular_layout(self)
        node_colors = [self.unique_node_types.index(i) for i in self.node_types]

        nx.draw_networkx_nodes(self, pos, node_color=node_colors, cmap=plt.cm.autumn)

        width = [1.0] * self.number_of_edges()
        edge_colors = [self.unique_edge_types.index(i) for i in self.edge_types]

        nx.draw_networkx_edges(
            self, pos, edge_color=edge_colors, edge_cmap=plt.cm.tab10, width=width
        )

        # Display node index
        nx.draw_networkx_labels(self, pos=pos)
        plt.axis("off")

    def draw_with_spring_layout(self) -> None:
        """
        Method to draw the GRN, using the spring layout.
        """
        pos = nx.spring_layout(self)
        node_colors = [self.unique_node_types.index(i) for i in self.node_types]
        edge_colors = [self.unique_edge_types.index(i) for i in self.edge_types]

        nx.draw(
            self,
            pos=pos,
            node_color=node_colors,
            edge_color=edge_colors,
            cmap=plt.cm.autumn,
            edge_cmap=plt.cm.tab10,
        )
        nx.draw_networkx_labels(self, pos=pos)


def available_realnet_tissue_type_files():
    """Returns all high-level RealNet networks downloaded."""
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
    interaction_type,
    realnet_tissue_type_file=None,
    tf_only=False,
    subsample_edge_prop=1.0,
    n_nodes=None,
    avg_num_parents=None,
):
    """Loads a InteractionGraph instance given an valid ineraction dataset string.
    Args:
        interaction_type (str): Specified the desired dataset. See INTERACTION_TYPES
            for valid entries.
        realnet_tissue_type_file (str): For `fantom5` data only, specifies the input
            file to be used when loading the graph (see flecs/datasets/RealNet/
            Network_compendium/Tissue-specific_regulatory_networks_FANTOM5-v1 for valid
            entries).
        tf_only (bool): Whether to include only transcription factors (TODO: which datasets)?
        subsample_edge_prob (float): When 0 < x < 1, used to sample a subset of the edges
            when instantiating the InteractionGraph. TODO: which datasets?
        n_nodes (int): Number of nodes in a synthetic graph ("test" / "random").
        avg_num_parents (int): Mean number of parents of each node in a synthetic graph
            (`test` / `random`).

    Returns: An InteractionGraph instance.
    """
    INTERACTION_TYPES = [
        "test",
        "calcium_pathway",
        "regulon_db",
        "encode",
        "fantom5",
        "string",
        "composite",
        "random",
    ]
    assert interaction_type in INTERACTION_TYPES

    if interaction_type == "test":
        nx_graph = load_calcium_signaling_pathway()
        # Add a dummy numerical node attribute.
        node_attr = {}
        for node in nx_graph.nodes(data=True):
            if node[1]["type"] == "gene":
                node_attr[node[0]] = {
                    "basal_expression": torch.tensor(np.random.exponential())[None]
                }

        nx.set_node_attributes(nx_graph, node_attr)

        # Add a dummy numerical edge attribute.
        edge_attr = {}
        for edge in nx_graph.edges(data=True):
            if edge[2]["type"] == "activation":
                edge_attr[(edge[0], edge[1])] = {
                    "strength": torch.tensor(np.random.randn())[None]
                }

        nx.set_edge_attributes(nx_graph, edge_attr)
        return InteractionGraph(nx_graph)

    elif interaction_type == "calcium_pathway":
        return InteractionGraph(load_calcium_signaling_pathway())

    elif interaction_type == "regulon_db":
        return InteractionGraph(get_regulondb_graph(tf_only=tf_only))

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
        return InteractionGraph(encode_graph)

    elif interaction_type == "fantom5":
        if realnet_tissue_type_file not in available_realnet_tissue_type_files():
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
        return InteractionGraph(fantom5_graph)

    elif interaction_type == "string":
        string_graph = get_string_graph(
            path_to_file=os.path.join(
                "STRING", "9606.protein.physical.links.detailed.v11.5.txt.gz"
            ),
            experimental_only=True,
            subsample_edge_prop=subsample_edge_prop,
        )
        return InteractionGraph(string_graph)

    elif interaction_type == "composite":
        string_graph = get_string_graph(
            path_to_file=os.path.join(
                "STRING", "9606.protein.physical.links.detailed.v11.5.txt.gz"
            ),
            experimental_only=True,
            subsample_edge_prop=subsample_edge_prop,
        )

        if realnet_tissue_type_file not in available_realnet_tissue_type_files():
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

        composite_graph = get_composite_graph(fantom5_graph, string_graph)
        return InteractionGraph(composite_graph)

    elif interaction_type == "random":
        assert n_nodes is not None, "Please specify 'n_nodes'."
        assert avg_num_parents is not None, "Please specify 'avg_num_parents'."
        random_graph = get_graph_from_adj_mat(
            get_random_adjacency_mat(n_nodes=n_nodes, avg_num_parents=avg_num_parents)
        )
        return InteractionGraph(random_graph)