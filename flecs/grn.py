from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List


########################################################################################################################
# GRN Abstract class
########################################################################################################################


class GRN(nx.DiGraph, ABC):
    """
    Class which represents the Gene Regulatory Network (GRN) of the Cell.
    """

    def __init__(self, **kwargs):
        """
        Initialize the GRN and make sure that the object contains a 'is_TF' node attribute, which indicates the nodes
        that are transcription factors.
        """
        _graph = self.load_grn(**kwargs)
        super().__init__(_graph)
        assert "is_TF" in self.node_attr_name_list

    @classmethod
    @abstractmethod
    def load_grn(cls, **kwargs) -> nx.DiGraph:
        """
        Method used to load the Gene Regulatory Network as a nx.DiGraph object. Either from a database,
        or from a random initializer.
        """

    @property
    def n_nodes(self) -> int:
        return self.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.number_of_edges()

    @property
    def tedges(self) -> torch.Tensor:
        """
        Edges in the GRN as a torch Tensor.

        :return: torch Tensor of shape (n_edges, 2)
        """
        return torch.tensor(list(self.edges()))

    def get_node_attr(self, attr_name: str) -> torch.Tensor:
        return self.get_attr(attr_name, "node")

    def get_edge_attr(self, attr_name: str) -> torch.Tensor:
        return self.get_attr(attr_name, "edge")

    def get_attr(self, attr_name: str, attr_type: str) -> torch.Tensor:
        assert attr_type in ["node", "edge"]

        attr_list = (
            self.node_attr_name_list
            if attr_type == "node"
            else self.edge_attr_name_list
        )
        get_attr_fn = (
            nx.get_node_attributes if attr_type == "node" else nx.get_edge_attributes
        )

        if attr_name not in attr_list:
            raise ValueError(
                "{} is not a {} attribute of the GRN".format(attr_name, attr_type)
            )

        attr = torch.cat(
            [t[:, None, :] for t in list(get_attr_fn(self, attr_name).values())],
            dim=1,
        )

        return attr

    def set_node_attr(self, attr_name: str, attr_values: torch.Tensor) -> None:
        assert type(attr_values) is torch.Tensor
        assert len(attr_values.shape) >= 3
        assert attr_values.shape[1] == self.n_nodes

        nx.set_node_attributes(
            self,
            dict(zip(range(self.n_nodes), attr_values.swapaxes(0, 1))),
            name=attr_name,
        )

    def set_edge_attr(self, attr_name: str, attr_values: torch.Tensor) -> None:
        assert type(attr_values) is torch.Tensor
        assert len(attr_values.shape) >= 3
        assert attr_values.shape[1] == self.n_edges

        nx.set_edge_attributes(
            self, dict(zip(self.edges, attr_values.swapaxes(0, 1))), name=attr_name
        )

    @property
    def state(self) -> torch.Tensor:
        return self.get_node_attr("state")

    @state.setter
    def state(self, state: torch.Tensor):
        self.set_node_attr("state", state)

    @property
    def tf_indices(self) -> List[int]:
        """
        List of the indices of the nodes which are transcription factors
        :return: List of integers containing the node indices
        """
        tf_idx = list(np.where(self.get_node_attr("is_TF"))[0])

        tf_idx = [int(i) for i in tf_idx]

        return tf_idx

    @property
    def edge_attr_name_list(self) -> List[str]:
        return list(list(self.edges(data=True))[0][-1].keys())

    @property
    def node_attr_name_list(self) -> List[str]:
        return list(list(self.nodes(data=True))[0][-1].keys())

    def __str__(self):
        string = "DirectedAcyclicGraph. "
        string += str(self.n_nodes) + " nodes. " + str(self.n_edges) + " edges. "
        string += "Nodes:\n"
        for i in list(self.nodes):
            string += "\t node " + str(i) + " " + str(self.nodes[i]) + "\n"
        string += "Edges:\n"
        for i in self.edges.data():
            string += "\t edge " + str(i) + "\n"
        return string

    def draw(self):
        pos = nx.circular_layout(self)

        nx.draw_networkx_nodes(
            self, pos, node_color=self.get_node_attr("is_TF"), cmap=plt.cm.autumn
        )

        width = [0.3] * self.number_of_edges()
        nx.draw_networkx_edges(self, pos, width=width)

        # Display node index above the node
        nx.draw_networkx_labels(self, pos=pos)
        plt.axis("off")

    def draw_with_spring_layout(self):
        pos = nx.spring_layout(self)
        nx.draw(
            self,
            pos=pos,
            node_color=self.get_node_attr("is_TF"),
            cmap=plt.cm.autumn,
        )
        nx.draw_networkx_labels(self, pos=pos)


########################################################################################################################
# GRN classes
########################################################################################################################


class RandomGRN(GRN):
    """
    Gene Regulatory Network initialized at random
    """

    def __init__(self, n_nodes: int, av_num_parents: float):
        """

        :param n_nodes: Number of nodes.
        :param av_num_parents: Average number of parents per node.
        """
        assert n_nodes > 0
        assert av_num_parents >= 0

        super(RandomGRN, self).__init__(n_nodes=n_nodes, av_num_parents=av_num_parents)

    @classmethod
    def load_grn(cls, n_nodes, av_num_parents) -> nx.DiGraph:
        adj_mat = cls.get_random_adjacency_mat(n_nodes, av_num_parents)

        nx_graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
        for n1, n2, d in nx_graph.edges(
            data=True
        ):  # By default an edge attr is created. We delete it.
            d.clear()

        is_tf = torch.ones((1, n_nodes, 1, 1))

        nx.set_node_attributes(
            nx_graph, dict(zip(range(n_nodes), is_tf.swapaxes(0, 1))), name="is_TF"
        )

        return nx_graph

    @classmethod
    def get_random_adjacency_mat(cls, n_nodes: int, av_num_parents: float) -> np.array:
        """
        Returns a random adjacency matrix of size (n_nodes, n_nodes) where each node has on average
            'av_num_parents' parents. Moreover, we make sure that each node has at least one parent

        Note that the resulting graph may not be acyclic, and self loops are allowed.

        :param n_nodes: Number of nodes
        :param av_num_parents: Average number of parents for each node
        :return: Numpy array of shape (n_nodes, n_nodes)
        """
        adj_mat = np.random.choice(
            [0, 1],
            size=(n_nodes, n_nodes),
            p=[
                float(n_nodes - av_num_parents) / n_nodes,
                float(av_num_parents) / n_nodes,
            ],
        )

        # Make sure that each gene has at least one parent
        for j in range(n_nodes):
            if (adj_mat[:, j] == 0).all():
                adj_mat[np.random.randint(n_nodes), j] = 1

        return adj_mat
