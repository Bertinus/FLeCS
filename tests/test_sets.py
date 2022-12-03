import pytest
from flecs.sets import EdgeSet, NodeSet
import torch


def test_sets():
    es = EdgeSet(torch.rand((10, 2)), {"rate": torch.rand((10,))})

    for i in range(10):
        assert es.get_edges(i)[0].numel() == 2
        assert es.get_edges(i)[1]["rate"].numel() == 1


def test_remove_edges():
    es = EdgeSet(torch.rand((5, 2)), {"rate": torch.rand((5,))})

    with pytest.raises(AssertionError):
        es.remove_edges(torch.tensor(0))

    with pytest.raises(IOError):
        es.remove_edges(torch.tensor([0]))

    # 5 edges - 2 edges = 3 edges.
    es.remove_edges(torch.tensor([0, 1, 1, 0, 0]))
    assert es.edges.shape[0] == 3


def test_add_edges():
    es = EdgeSet(torch.rand((5, 2)), {"rate": torch.rand((5,))})
    es.add_edges(torch.rand((2, 2)), {"rate": torch.rand((2,))})
    assert list(es.edges.shape) == [7, 2]

    # Check for mismatches in dimensions.
    with pytest.raises(AssertionError):
        es.add_edges(torch.rand((2, 2)), {"rate": torch.rand((3,))})

    with pytest.raises(AssertionError):
        es.add_edges(torch.rand((3, 2)), {"rate": torch.rand((2,))})

    with pytest.raises(AssertionError):
        es.add_edges(torch.rand((2, 2, 3)), {"rate": torch.rand((2,))})

    with pytest.raises(AssertionError):
        es.add_edges(torch.rand((2, 2)), {"rate": torch.rand((2, 2))})


# TODO: Add node tests / set tests (not sure what's worth testing)...