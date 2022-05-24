from flecs.parameter import NodeParameter, EdgeParameter
from torch.distributions.normal import Normal
import pytest
import torch


@pytest.fixture
def my_parameters():
    prior_dist = Normal(0, 1)
    return NodeParameter(dim=(1, 2), prior_dist=prior_dist), EdgeParameter(
        dim=(1, 2), prior_dist=prior_dist
    )


def test_print_parameter(my_parameters):
    for param in my_parameters:
        print(param)


def test_initialize_with_tensor():
    prior_dist = Normal(0, 1)
    param = NodeParameter(
        dim=(1, 2), prior_dist=prior_dist, tensor=torch.ones(10, 3, 1, 2)
    )

    assert torch.equal(param.tensor, torch.ones(10, 3, 1, 2))


def test_wrong_dimensions():
    prior_dist = Normal(0, 1)
    with pytest.raises(ValueError):
        NodeParameter(dim=(1, 2), prior_dist=prior_dist, tensor=torch.ones(10, 3, 2, 2))


def test_sample_prior_dist():
    prior_dist = Normal(0, 1)
    param = NodeParameter(dim=(1, 2), prior_dist=prior_dist)

    param.initialize_from_prior_dist(12)

    assert param.tensor.shape[1] == 12
    assert (param.tensor != 0).any()


def test_prior_dist_not_defined():
    param = NodeParameter(dim=(1, 2))

    with pytest.raises(RuntimeError):
        param.initialize_from_prior_dist(12)
