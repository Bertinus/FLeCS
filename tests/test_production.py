import pytest
import torch

from flecs.production import efficient_inplace_message_passing, SimpleConv
from flecs.cell_population import TestCellPop


@pytest.fixture
def my_cells():
    return TestCellPop(n_cells=3)


def test_message_passing():
    pass  # TODO: good test case for edgeset?


if __name__ == "__main__":
    test_message_passing()