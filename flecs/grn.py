from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

########################################################################################################################
# GRN Abstract class
########################################################################################################################


class GRN(nx.DiGraph, ABC):
    """
    Abstract class representing the Gene Regulatory Network (GRN) of the Cell.

    It inherits from the ``networkx.DiGraph`` class. We refer to the nodes of the graph as genes for clarity, even if
    in practice some nodes could represent other entities (such as drugs, small molecules etc). Edges typically
    represent a regulation relationship between a regulator gene (or transcription factor, TF) and a regulated gene.
    """

    def __init__(self, **kwargs):
        """
        Initializes the GRN and makes sure that the object contains a 'is_TF' gene attribute, which indicates the genes
        that are transcription factors.

        Args:
            **kwargs: keyword arguments passed to the method ``self.load_grn``.
        """
        _graph = self.load_grn(**kwargs)
        super().__init__(_graph)
        assert "is_TF" in self.gene_attr_name_list

    @classmethod
    @abstractmethod
    def load_grn(cls, **kwargs) -> nx.DiGraph:
        """
        Abstract method used to load the Gene Regulatory Network as a ``networkx.DiGraph`` object.
        Either from a database, or from a random initializer.

        Returns:
            networkx.DiGraph: Graph from which the GRN is initialized.
        """

    @property
    def n_genes(self) -> int:
        """
        (``int``) Number of genes.
        """
        return self.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """
        (``int``) Number of edges.
        """
        return self.number_of_edges()

    @property
    def tedges(self) -> torch.Tensor:
        """
        (``torch.Tensor``) Edges in the GRN. Shape (n_edges, 2).
        """
        return torch.tensor(list(self.edges()))

    def get_gene_attr(self, attr_name: str) -> torch.Tensor:
        """
        Gets the values of the attribute ``attr_name`` from all genes, and returns it as a single torch.Tensor.

        Args:
            attr_name (str): Name of the gene attribute.

        Returns:
            torch.Tensor: Tensor corresponding to the ``attr_name`` attribute.
                Shape (n_cells, n_genes, *attr_dim).

        """
        return self.get_attr(attr_name, "gene")

    def get_edge_attr(self, attr_name: str) -> torch.Tensor:
        """
        Gets the values of the attribute ``attr_name`` from all edges, and returns it as a single torch.Tensor.

        Args:
            attr_name (str): Name of the edge attribute.

        Returns:
            torch.Tensor: Tensor corresponding to the ``attr_name`` attribute.
                Shape (n_cells, n_edges, *attr_dim).

        """
        return self.get_attr(attr_name, "edge")

    def get_attr(self, attr_name: str, attr_type: str) -> torch.Tensor:
        """

        Args:
            attr_name (str): Name of the attribute.
            attr_type (str): either "gene" or "edge".

        Returns:
            torch.Tensor: Tensor corresponding to the ``attr_name`` attribute.

        """
        assert attr_type in ["gene", "edge"]

        attr_list = (
            self.gene_attr_name_list
            if attr_type == "gene"
            else self.edge_attr_name_list
        )
        get_attr_fn = (
            nx.get_gene_attributes if attr_type == "gene" else nx.get_edge_attributes
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

    def set_gene_attr(self, attr_name: str, attr_values: torch.Tensor) -> None:
        """
        Sets the values of the attribute ``attr_name`` for all genes.

        Args:
            attr_name (str): Name of the gene attribute.
            attr_values (torch.Tensor): Values for the gene attribute. Shape (n_cells, n_genes, *attr_dim).

        """
        assert type(attr_values) is torch.Tensor
        assert len(attr_values.shape) >= 3
        assert attr_values.shape[1] == self.n_genes

        nx.set_gene_attributes(
            self,
            dict(zip(range(self.n_genes), attr_values.swapaxes(0, 1))),
            name=attr_name,
        )

    def set_edge_attr(self, attr_name: str, attr_values: torch.Tensor) -> None:
        """
        Sets the values of the attribute ``attr_name`` for all edges.

        Args:
            attr_name (str): Name of the edge attribute.
            attr_values (torch.Tensor): Values for the edge attribute. Shape (n_cells, n_edges, *attr_dim).

        """
        assert type(attr_values) is torch.Tensor
        assert len(attr_values.shape) >= 3
        assert attr_values.shape[1] == self.n_edges

        nx.set_edge_attributes(
            self, dict(zip(self.edges, attr_values.swapaxes(0, 1))), name=attr_name
        )

    @property
    def state(self) -> torch.Tensor:
        """
        (``torch.Tensor``) State stored as a gene attribute.
        """
        return self.get_gene_attr("state")

    @state.setter
    def state(self, state: torch.Tensor):
        self.set_gene_attr("state", state)

    @property
    def tf_indices(self) -> List[int]:
        """
        Method returning the list of the indices of the genes which are transcription factors (TFs).

        Returns:
             List[int]: List of the indices of the genes which are transcription factors.

        """
        tf_idx = list(np.where(self.get_gene_attr("is_TF"))[0])

        tf_idx = [int(i) for i in tf_idx]

        return tf_idx

    @property
    def edge_attr_name_list(self) -> List[str]:
        """
        (``List[str]``) List containing the names of the edge attributes.
        """
        return list(list(self.edges(data=True))[0][-1].keys())

    @property
    def gene_attr_name_list(self) -> List[str]:
        """
        (``List[str]``) List containing the names of the gene attributes.
        """
        return list(list(self.nodes(data=True))[0][-1].keys())

    def __str__(self):
        string = "DirectedAcyclicGraph. "
        string += str(self.n_genes) + " genes. " + str(self.n_edges) + " edges. "
        string += "Genes:\n"
        for i in list(self.nodes):
            string += "\t gene " + str(i) + " " + str(self.nodes[i]) + "\n"
        string += "Edges:\n"
        for i in self.edges.data():
            string += "\t edge " + str(i) + "\n"
        return string

    def draw(self):
        """
        Method to draw the GRN, using the circular layout.
        """
        pos = nx.circular_layout(self)

        nx.draw_networkx_genes(
            self, pos, node_color=self.get_gene_attr("is_TF"), cmap=plt.cm.autumn
        )

        width = [0.3] * self.number_of_edges()
        nx.draw_networkx_edges(self, pos, width=width)

        # Display gene index above the gene
        nx.draw_networkx_labels(self, pos=pos)
        plt.axis("off")

    def draw_with_spring_layout(self):
        """
        Method to draw the GRN, using the spring layout.
        """
        pos = nx.spring_layout(self)
        nx.draw(
            self,
            pos=pos,
            node_color=self.get_gene_attr("is_TF"),
            cmap=plt.cm.autumn,
        )
        nx.draw_networkx_labels(self, pos=pos)


########################################################################################################################
# GRN classes
########################################################################################################################


class RandomGRN(GRN):
    """
    Class implementing a Gene Regulatory Network (GRN) initialized at random. It inherits from the ``GRN`` class.

    The elements of the adjacency matrix are sampled independently at random in {0, 1}, given the number of genes, and
    the average number of parents per gene. The resulting graph can contain cycles and self loops. In addition, the
    initializer makes sure that each gene has at least one parent.
    """

    def __init__(self, n_genes: int, av_num_parents: float):
        """

        Args:
            n_genes (int): Number of genes.
            av_num_parents (float): Average number of parents per gene.
        """
        assert n_genes > 0
        assert av_num_parents >= 0

        super(RandomGRN, self).__init__(n_genes=n_genes, av_num_parents=av_num_parents)

    @classmethod
    def load_grn(cls, n_genes, av_num_parents) -> nx.DiGraph:
        """
        Generates a random graph as a ``networkx.DiGraph`` object.

        Args:
            n_genes (int): Desired number of genes in the graph.
            av_num_parents (float): Average number of parents per gene.

        Returns:
            networkx.DiGraph: Random graph.
        """
        adj_mat = cls.get_random_adjacency_mat(n_genes, av_num_parents)

        nx_graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
        for n1, n2, d in nx_graph.edges(
            data=True
        ):  # By default an edge attr is created. We delete it.
            d.clear()

        is_tf = torch.ones((1, n_genes, 1, 1))

        nx.set_gene_attributes(
            nx_graph, dict(zip(range(n_genes), is_tf.swapaxes(0, 1))), name="is_TF"
        )

        return nx_graph

    @classmethod
    def get_random_adjacency_mat(cls, n_genes: int, av_num_parents: float) -> np.array:
        """
        Generates a random adjacency matrix. Each gene has on average ``av_num_parents`` parents.
        Moreover, each gene has at least one parent.

        Args:
            n_genes (int): Number of genes.
            av_num_parents (float): Average number of parents per gene.

        Returns:
            numpy.array: Adjacency matrix. Shape (n_genes, n_genes).
        """
        adj_mat = np.random.choice(
            [0, 1],
            size=(n_genes, n_genes),
            p=[
                float(n_genes - av_num_parents) / n_genes,
                float(av_num_parents) / n_genes,
            ],
        )

        # Make sure that each gene has at least one parent
        for j in range(n_genes):
            if (adj_mat[:, j] == 0).all():
                adj_mat[np.random.randint(n_genes), j] = 1

        return adj_mat
