import pytest
import torch

from flecs.cell import Cell
from flecs.grn import RandomGRN
from flecs.structural_equation import SigmoidLinearSE


@pytest.fixture
def my_cell():
    grn = RandomGRN(10, 3)
    linear_se = SigmoidLinearSE()

    return Cell(grn=grn, structural_equation=linear_se)


def test_set_cell_state(my_cell):
    my_cell.state = torch.ones((3, 10, 1))

    assert torch.equal(my_cell.state, torch.ones((3, 10, 1)))


def test_set_cell_state_wrong_dimension(my_cell):

    with pytest.raises(AssertionError):
        my_cell.state = torch.ones((3, 9, 1))


def test_derivatives(my_cell):
    my_cell.state = torch.ones((3, 10, 1))
    assert my_cell.get_derivatives(my_cell.state).shape == (3, 10, 1)
    assert my_cell.get_production_rates(my_cell.state).shape == (3, 10, 1)
    assert my_cell.get_decay_rates(my_cell.state).shape == (3, 10, 1)


def test_get_parameters(my_cell):
    assert my_cell.get_node_parameter("gene_decay").tensor.shape == (1, 10, 1)
    assert my_cell.get_edge_parameter("weights").tensor.shape == (
        1,
        my_cell.grn.n_edges,
        1,
    )
    assert my_cell.get_parameter("weights").tensor.shape == (1, my_cell.grn.n_edges, 1)


def test_initialization(my_cell):
    assert "weights" in my_cell.grn.edge_attr_name_list
    assert "gene_decay" in my_cell.grn.node_attr_name_list


def test_n_edges(my_cell):
    assert my_cell.n_edges == my_cell.grn.n_edges


def test_syn_se_from_grn(my_cell: Cell):

    my_cell.grn.set_edge_attr(
        "new_edge_attr", 3 * torch.ones((1, my_cell.grn.n_edges, 2, 2))
    )
    my_cell.grn.set_node_attr(
        "new_node_attr", 4 * torch.ones((1, my_cell.grn.n_nodes, 5, 5))
    )

    my_cell.sync_se_from_grn()

    assert torch.equal(
        my_cell.get_edge_parameter("new_edge_attr").tensor,
        3 * torch.ones((1, my_cell.grn.n_edges, 2, 2)),
    )
    assert torch.equal(
        my_cell.get_node_parameter("new_node_attr").tensor,
        4 * torch.ones((1, my_cell.grn.n_nodes, 5, 5)),
    )
