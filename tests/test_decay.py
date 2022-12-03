import torch
from flecs.decay import exponential_decay


def test_exponential_decay():

    class DummyNode:
        def __init__(self):
            self.state = torch.ones(10)

    obj = {"node_type": DummyNode()}
    for alpha in [0.1, 0.5, 0.9, 0.99]:
        res = exponential_decay(obj, "node_type", alpha=alpha)
        assert torch.isclose(torch.mean(res), torch.tensor(alpha))
