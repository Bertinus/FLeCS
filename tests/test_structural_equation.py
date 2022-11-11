import pytest
import torch
from torch.distributions.normal import Normal

from flecs.edge_set import EdgeSet
from flecs.node_set import NodeSet


@pytest.fixture
def my_se():
    pass


def test_edge_heads(my_se):
    assert torch.equal(my_se.edge_heads, torch.LongTensor([1, 2, 3, 0]))


def test_edge_tails(my_se):
    assert torch.equal(my_se.edge_tails, torch.LongTensor([0, 1, 2, 2]))


def test_production_rates(my_se):

    state = torch.Tensor([0, 1, 2, 3])[None, :, None]
    w = my_se.weights.tensor[0]
    expected_pr = torch.sigmoid(torch.cat([2 * w[3], 0 * w[0], 1 * w[1], 2 * w[2]]))

    assert torch.isclose(
        my_se.get_production_rates(state).reshape(-1), expected_pr.reshape(-1)
    ).all()


def test_decay_rates(my_se):

    state = torch.Tensor([0, 1, 2, 3])[None, :, None]
    d = my_se.gene_decay.tensor[0]
    expected_dr = torch.cat([0 * d[0], 1 * d[1], 2 * d[2], 3 * d[3]])

    assert torch.isclose(
        my_se.get_decay_rates(state).reshape(-1), expected_dr.reshape(-1)
    ).all()


def test_attribute_dict(my_se):

    assert "weights" in my_se.attribute_dict
    assert "gene_decay" in my_se.attribute_dict

    assert my_se.attribute_dict["weights"].tensor.shape == (1, my_se.n_edges, 1)
    assert my_se.attribute_dict["gene_decay"].tensor.shape == (1, my_se.n_genes, 1)


def test_set_attribute(my_se):
    my_se.set_attribute("new_gene_attribute", GeneAttribute(dim=(3,)))

    assert "new_gene_attribute" in my_se.attribute_dict


def test_derivatives(my_se):
    state = torch.Tensor([0, 1, 2, 3])[None, :, None]

    w = my_se.weights.tensor[0]
    expected_pr = torch.sigmoid(torch.cat([2 * w[3], 0 * w[0], 1 * w[1], 2 * w[2]]))

    d = my_se.gene_decay.tensor[0]
    expected_dr = torch.cat([0 * d[0], 1 * d[1], 2 * d[2], 3 * d[3]])

    assert torch.isclose(
        my_se.get_derivatives(state).reshape(-1),
        (expected_pr - expected_dr).reshape(-1),
    ).all()


def test_representation(my_se):
    print(my_se)


def test_to_device(my_se):
    my_se.to(torch.device("cpu"))
