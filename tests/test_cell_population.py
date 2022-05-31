import pytest
import torch

from flecs.cell_population import CellPopulation
from flecs.grn import RandomGRN
from flecs.structural_equation import SigmoidLinearSE


@pytest.fixture
def my_cells():
    grn = RandomGRN(10, 3)
    linear_se = SigmoidLinearSE()

    return CellPopulation(grn=grn, structural_equation=linear_se)


def test_set_cell_state(my_cells):
    my_cells.state = torch.ones((3, 10, 1))

    assert torch.equal(my_cells.state, torch.ones((3, 10, 1)))


def test_set_cell_state_wrong_dimension(my_cells):

    with pytest.raises(AssertionError):
        my_cells.state = torch.ones((3, 9, 1))


def test_derivatives(my_cells):
    my_cells.state = torch.ones((3, 10, 1))
    assert my_cells.get_derivatives(my_cells.state).shape == (3, 10, 1)
    assert my_cells.get_production_rates(my_cells.state).shape == (3, 10, 1)
    assert my_cells.get_decay_rates(my_cells.state).shape == (3, 10, 1)


def test_get_attributes(my_cells):
    assert my_cells.get_gene_attribute("gene_decay").tensor.shape == (1, 10, 1)
    assert my_cells.get_edge_attribute("weights").tensor.shape == (
        1,
        my_cells._grn.n_edges,
        1,
    )
    assert my_cells.get_attribute("weights").tensor.shape == (1, my_cells._grn.n_edges, 1)


def test_initialization(my_cells):
    assert "weights" in my_cells._grn.edge_attr_name_list
    assert "gene_decay" in my_cells._grn.gene_attr_name_list


def test_n_edges(my_cells):
    assert my_cells.n_edges == my_cells._grn.n_edges


def test_syn_se_from_grn(my_cells):

    my_cells._grn.set_edge_attr(
        "new_edge_attr", 3 * torch.ones((1, my_cells._grn.n_edges, 2, 2))
    )
    my_cells._grn.set_gene_attr(
        "new_gene_attr", 4 * torch.ones((1, my_cells._grn.n_genes, 5, 5))
    )

    my_cells.sync_se_from_grn()

    assert torch.equal(
        my_cells.get_edge_attribute("new_edge_attr").tensor,
        3 * torch.ones((1, my_cells._grn.n_edges, 2, 2)),
    )
    assert torch.equal(
        my_cells.get_gene_attribute("new_gene_attr").tensor,
        4 * torch.ones((1, my_cells._grn.n_genes, 5, 5)),
    )
