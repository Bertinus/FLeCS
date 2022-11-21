from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from typing import List, Dict, Union, Tuple

# Base types
NodeIdx = int
EdgeIdx = Tuple[int, int]
NodeType = str
RelationType = str
EdgeType = Tuple[str, str, str]
AttributeName = str
Attribute = Union[str, torch.Tensor]
AttributeList = Union[List[str], torch.Tensor]

# Types
Data = Dict[AttributeName, Attribute]
NodeData = Dict[NodeIdx, Data]
EdgeData = Dict[EdgeIdx, Data]
SetData = Dict[AttributeName, AttributeList]
NodeSetData = Dict[NodeType, SetData]
EdgeSetData = Dict[EdgeType, SetData]


class InteractionData(nx.DiGraph):
    def __init__(self, graph: nx.DiGraph):
        """
        Class which holds interaction data about a Cell, s.a. Gene Regulatory Networks/Pathways.
        It inherits from `nx.DiGraph` and is formatted such that:

        * Nodes and edges are typed, based on a "type" attribute
        * Nodes are ordered based on their type
        * All nodes/edges of a given type have the same set of attributes
        * Node/Edge attributes are either strings or torch.Tensor.

        Once instantiated, the structure of the graph is frozen.

        Args:
            graph (nx.DiGraph): Graph used for initialization.
        """
        self._edge_types = None

        # Sort the nodes
        graph = self.sorted_by_node_type(graph)

        super().__init__(graph)

        # Assert that all node and edge attributes are either strings or tensors
        self._check_attributes_dtype()
        # Assert that all nodes of a given type have exactly the same set of attributes
        self._check_attribute_consistency_within_node_types()

        # Assert that all edges of a given type have exactly the same set of attributes
        self._check_attribute_consistency_within_edge_types()

        # Freeze the graph
        nx.freeze(self)

    @property
    def node_types(self) -> List[NodeType]:
        """
        (``List[NodeType]``) List of length (n_nodes) which contains the types of all the nodes.
        """
        return list(nx.get_node_attributes(self, "type").values())

    @property
    def relation_types(self) -> List[RelationType]:
        """
        (``List[RelationType]``) List of length (n_edges) which contains the relation types for all the edges.
        """
        return list(nx.get_edge_attributes(self, "type").values())

    @property
    def unique_node_types(self) -> List[NodeType]:
        return list(np.unique(self.node_types))

    @property
    def unique_relation_types(self) -> List[RelationType]:
        return list(np.unique(self.relation_types))

    @property
    def unique_edge_types(self) -> List[EdgeType]:
        if self._edge_types is None:
            node_types = self.node_types
            relation_types = self.relation_types

            self._edge_types = [
                (node_types[e[0]], relation_types[e_idx], node_types[e[1]])
                for e_idx, e in enumerate(self.edges())
            ]
        return list(set(self._edge_types))

    def node_data(self, n_type: str = None) -> NodeData:
        """
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
        self, r_type: str = None, src_type: str = None, tgt_type: str = None
    ) -> EdgeData:
        """
        Args:
            r_type (str): relation type to restrict to
            src_type (str): type of the source node
            tgt_type (str): type of the target node

        Returns:
            EdgeData: Dictionary mapping the indices of edges of type (src_type, r_type, tgt_type) to their data
                dictionaries (mapping attribute names to attribute values).
        """

        node_types = self.node_types

        def to_be_included(e: Tuple):
            res = e[2]["type"] == r_type if r_type is not None else True
            if src_type is not None:
                res = res and node_types[e[0]] == src_type
            if tgt_type is not None:
                res = res and node_types[e[1]] == tgt_type
            return res

        return {(e[0], e[1]): e[2] for e in self.edges(data=True) if to_be_included(e)}

    def get_formatted_node_data(self) -> NodeSetData:
        """Formats the data of nodes.

        For all node types n_type, a single dictionary is built. It is such that values
        have length equal to the number of nodes of type n_type. The dictionary also
        contains a key "idx" listing the indices of the nodes of type n_type.

        Returns:
            NodeSetData: Keys correspond to node types. The values associated
            with key n_type is a dict containing the data of the nodes of type n_type.
        """
        formatted_node_data = {}
        for n_type in self.unique_node_types:
            formatted_node_data[n_type] = self.format_type_data_dict(
                self.node_data(n_type)
            )

        return formatted_node_data

    def get_formatted_edge_data(self) -> EdgeSetData:
        """Formats the data of edges.

        For all edge types e_type=(src_type, r_type, tgt_type), a single dictionary is built. It is such that values
        have length equal to the number of edges of type e_type. The dictionary also contains a key "idx" listing the
        indices of the edges.

        Returns:
            EdgeSetData: Keys correspond to edge types. The value associated
            with key e_type=(src_type, r_type, tgt_type) is a dict containing the data of the corresponding edges.
        """
        formatted_edge_data = {}
        for e_type in self.unique_edge_types:
            e_data_dict = self.edge_data(
                src_type=e_type[0], r_type=e_type[1], tgt_type=e_type[2]
            )
            formatted_edge_data[e_type] = self.format_type_data_dict(e_data_dict)

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
        """
        Checks that all nodes of a given type have the same set of attributes.
        """
        for n_type in self.unique_node_types:
            node_data_list = list(self.node_data(n_type).values())
            reference_attr_names = set(node_data_list.pop(0).keys())
            for n_data in node_data_list:
                assert set(n_data.keys()) == reference_attr_names

    def _check_attribute_consistency_within_edge_types(self):
        """
        Checks that all edges of a given type have the same set of attributes.
        """
        for e_type in self.unique_edge_types:
            edge_data_list = list(
                self.edge_data(
                    src_type=e_type[0], r_type=e_type[1], tgt_type=e_type[2]
                ).values()
            )
            reference_attr_names = set(edge_data_list.pop(0).keys())
            for e_data in edge_data_list:
                assert set(e_data.keys()) == reference_attr_names

    def _check_attributes_dtype(self):
        """
        Make sure node/edges are associated with a string "type" attribute, and that other attributes are either
        strings or torch Tensors with dimension 1 along the first axis.
        """

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

        Args:
            g: graph with "type" as global node and edge attribute.

        Returns:
            nx.DiGraph: Copy of graph g in which the node ordering is based on node type. Node labels are overwritten
            and set to range(0, n_nodes).

        """
        # Make sure that "type" is a node/edge attribute of all nodes/edges
        assert len(nx.get_node_attributes(g, "type")) == g.number_of_nodes()
        assert len(nx.get_edge_attributes(g, "type")) == g.number_of_edges()

        g = nx.convert_node_labels_to_integers(g)
        ordered_nodes = np.argsort(list(nx.get_node_attributes(g, "type").values()))

        g_sorted = nx.DiGraph()
        for n in ordered_nodes:
            g_sorted.add_node(n, **g.nodes(data=True)[n])
        g_sorted.add_edges_from(g.edges(data=True))

        g_sorted = nx.convert_node_labels_to_integers(g_sorted)

        return g_sorted

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
        edge_colors = [self.unique_relation_types.index(i) for i in self.relation_types]

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
        edge_colors = [self.unique_relation_types.index(i) for i in self.relation_types]

        nx.draw(
            self,
            pos=pos,
            node_color=node_colors,
            edge_color=edge_colors,
            cmap=plt.cm.autumn,
            edge_cmap=plt.cm.tab10,
        )
        nx.draw_networkx_labels(self, pos=pos)
