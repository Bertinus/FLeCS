import networkx as nx
import pytest
import torch
from torch.distributions.normal import Normal
from copy import copy

from flecs.cell_population import TestCellPop
from flecs.data.interaction_data import load_interaction_data
from flecs.sets import EdgeSet, NodeSet


@pytest.fixture
def my_cells():
    return TestCellPop()


@pytest.fixture
def my_grn():
    return load_interaction_data("random", n_nodes=10, avg_num_parents=3)


@pytest.fixture
def my_nodeset_shape():
    return [10, 2]


@pytest.fixture
def my_edgeset_shape():
    return [10, 2, 2, 4]


def test_initialize_with_tensor(my_cells, my_edgeset_shape, my_nodeset_shape):
    # Ensures that edgeset and nodeset tensor shapes do not change after assignment.
    es = EdgeSet(torch.rand(my_edgeset_shape), {"Values": torch.rand(my_edgeset_shape)})
    ns = NodeSet(
        my_cells,
        idx_low=0,
        idx_high=2,
        attribute_dict={"Genes": torch.rand(my_nodeset_shape)}
    )

    assert es.Values.shape == my_edgeset_shape
    assert ns.Genes.shape == my_nodeset_shape


def test_wrong_dimensions(my_cells, my_edgeset_shape, my_nodeset_shape):
    # Edgeset tensors second element should be should size 2.
    my_edgeset_shape_corrput = copy(my_edgeset_shape)
    my_edgeset_shape_corrput[1] = 3

    with pytest.raises(AssertionError):
        es = EdgeSet(
            torch.rand(my_edgeset_shape_corrput),
            {"Values": torch.rand(my_edgeset_shape_corrput)}
        )

    # Edgeset tensor shape should be length 2.
    my_edgeset_shape_corrput = copy(my_edgeset_shape)
    my_edgeset_shape_corrput.append(3)

    with pytest.raises(AssertionError):
        es = EdgeSet(
            torch.rand(my_edgeset_shape_corrput),
            {"Values": torch.rand(my_edgeset_shape_corrput)}
        )

    # TODO: This does not fail no matter what I try...
    my_nodeset_shape_corrput = copy(my_nodeset_shape)
    my_nodeset_shape_corrput[0] += 1

    ns = NodeSet(
        my_cells,
        idx_low=0,
        idx_high=100,
        attribute_dict={"Genes": torch.rand(my_nodeset_shape_corrput)}
    )


def test_sample_prior_dist(my_cells):

    my_cells["gene"].init_param(name="alpha", dist=Normal(0, 1))
    assert my_cells["gene"].alpha.shape == (my_cells.n_cells, len(my_cells["gene"]), 1)
    assert (my_cells["gene"].alpha != 0).any()

    my_cells["gene"].init_param(name="alpha", dist=Normal(0, 1), shape=(my_cells.n_cells, len(my_cells["gene"]), 15))
    assert my_cells["gene"].alpha.shape == (my_cells.n_cells, len(my_cells["gene"]), 15)
    assert (my_cells["gene"].alpha != 0).any()


def test_set_cell_state(my_cells):
    my_cells.state = torch.ones((3, 10, 1))
    assert torch.equal(my_cells.state, torch.ones((3, 10, 1)))


def test_set_cell_state_wrong_dimension(my_cells):
    # TODO: This should not work - should check shape and raise error!
    with pytest.raises(AssertionError):
        my_cells.state = torch.ones((3, 9, 1))


def test_derivatives():
    for n_cells in [1, 3]:
        my_cells = TestCellPop(n_cells=n_cells)
        test_shape = (n_cells, 60, 1)  #  Last two parameters fixed by TestCellPop.

        my_cells.state = torch.ones(test_shape)
        assert my_cells.get_derivatives(my_cells.state).shape == test_shape
        assert my_cells.get_production_rates().shape == test_shape
        assert my_cells.get_decay_rates().shape == test_shape


def test_get_attributes(my_cells):
    for n_cells in [1, 3]:
        my_cells = TestCellPop(n_cells=n_cells)

        for node_type in my_cells.node_types:
            n_nodes = my_cells[node_type].idx_high - my_cells[node_type].idx_low + 1
            test_shape = (n_cells, n_nodes, 1)  # Last parameter fixed by TestCellPop.

            for node_attr in my_cells[node_type].element_level_attr_dict.keys():
                attr_shape = getattr(my_cells[node_type], node_attr).shape
                assert attr_shape[0] == test_shape[0]
                assert attr_shape[1] == test_shape[1]
                # TODO: can we handle flexible attribute sizes?
                # TODO: shouldn't the first dimensions match when n_cells=3?
