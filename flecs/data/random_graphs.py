import numpy as np
import networkx as nx


########################################################################################################################
# Graph initializers
########################################################################################################################


def get_graph_from_adj_mat(adj_mat: np.array) -> nx.DiGraph:
    """
    Given an adjacency matrix, initializes a networkx DiGraph with an empty "type" attribute for nodes and edges.

    Args:
        adj_mat: Numpy array of shape (n_nodes, n_nodes)

    Returns:
        graph initialized according to the adjacency matrix
    """

    assert len(adj_mat.shape) == 2 and adj_mat.shape[0] == adj_mat.shape[1]
    graph = nx.DiGraph()
    n_nodes = adj_mat.shape[0]

    # Initialize all nodes
    for i in range(n_nodes):
        graph.add_node(i, type="")

    # Initialize all edges
    edge_list = np.where(adj_mat)
    edge_list = [(edge_list[0][i], edge_list[1][i]) for i in range(len(edge_list[0]))]
    graph.add_edges_from(edge_list, type="")

    return graph


def get_random_adjacency_mat(n_nodes: int, avg_num_parents: int) -> np.array:
    """
    Computes a random adjacency matrix of size (n_nodes, n_nodes) where each node has on average
        'av_num_parents' parents. Moreover, we make sure that each node has at least one parent.

    Note that the resulting graph may not be acyclic, and self loops are allowed.

    Args:
        n_nodes: Number of nodes
        avg_num_parents: Average number of parents for each node

    Returns:
        Numpy array of shape (n_nodes, n_nodes)
    """
    adj_mat = np.random.choice(
        [0, 1],
        size=(n_nodes, n_nodes),
        p=[
            float(n_nodes - avg_num_parents) / n_nodes,
            float(avg_num_parents) / n_nodes,
        ],
    )

    # Make sure that each gene has at least one parent
    for j in range(n_nodes):
        if (adj_mat[:, j] == 0).all():
            adj_mat[np.random.randint(n_nodes), j] = 1

    return adj_mat
