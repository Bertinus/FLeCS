import networkx as nx
import pytest
import torch

from flecs.grn import RandomGRN


@pytest.fixture
def my_grn():
    grn = RandomGRN(n_genes=10, av_num_parents=3)
    return grn


def test_initialize_graph(my_grn):
    assert my_grn.n_genes == 10
    assert isinstance(my_grn, nx.DiGraph)


def test_all_genes_have_at_leat_one_parent(my_grn):

    edges = my_grn.tedges
    assert edges.shape[0] == my_grn.n_edges

    assert len(torch.unique(edges[:, 1])) == my_grn.n_genes


def test_set_gene_attr(my_grn):

    gene_attr = torch.ones((3, my_grn.n_genes, 2, 4))
    my_grn.set_gene_attr("test_gene_attr", gene_attr)

    for gene in my_grn.genes:
        print(
            torch.equal(
                my_grn.genes(data=True)[gene]["test_gene_attr"], torch.ones((3, 2, 4))
            )
        )


def test_get_gene_attr(my_grn):

    gene_attr = torch.ones((3, my_grn.n_genes, 2, 4))
    my_grn.set_gene_attr("test_gene_attr", gene_attr)

    gotten_gene_attr = my_grn.get_gene_attr("test_gene_attr")

    assert torch.equal(gotten_gene_attr, gene_attr)


def test_set_edge_attr(my_grn):

    edge_attr = torch.ones((3, my_grn.n_edges, 2, 4))
    my_grn.set_edge_attr("test_edge_attr", edge_attr)

    for edge in my_grn.edges:
        print(
            torch.equal(
                my_grn.get_edge_data(*edge)["test_edge_attr"], torch.ones((3, 2, 4))
            )
        )


def test_get_edge_attr(my_grn):

    edge_attr = torch.ones((3, my_grn.n_edges, 2, 4))
    my_grn.set_edge_attr("test_edge_attr", edge_attr)

    gotten_edge_attr = my_grn.get_edge_attr("test_edge_attr")

    assert torch.equal(gotten_edge_attr, edge_attr)


def test_get_non_existing_attr(my_grn):
    with pytest.raises(ValueError):
        my_grn.get_gene_attr("non_existing")


def test_state(my_grn):
    state = torch.ones((3, my_grn.n_genes, 2, 4))
    my_grn.state = state

    assert torch.equal(my_grn.state, state)


def test_tf_list(my_grn):
    assert isinstance(my_grn.tf_indices[0], int)
    assert len(my_grn.tf_indices) == my_grn.n_genes


def test_draw(my_grn):
    print(my_grn)
    my_grn.draw()
    my_grn.draw_with_spring_layout()
