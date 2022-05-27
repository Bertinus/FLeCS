import pytest
import torch
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from flecs.cell import Cell
from flecs.grn import RandomGRN
from flecs.mutation import BernoulliMutation, GaussianMutation
from flecs.parameter import EdgeParameter, NodeParameter
from flecs.structural_equation import SigmoidLinearSE
from flecs.trajectory import simulate_deterministic_trajectory


@pytest.fixture
def my_cell():
    grn = RandomGRN(10, 3)
    linear_se = SigmoidLinearSE(
        gene_decay=NodeParameter(dim=(1,), prior_dist=Gamma(10, 10)),
        weights=EdgeParameter(dim=(1,), prior_dist=Normal(1, 1)),
    )

    cell = Cell(grn=grn, structural_equation=linear_se)

    cell.state = 10 * torch.ones((12, 10, 1))

    return cell


def test_duplication(my_cell):
    my_gaussian_mutation = GaussianMutation(0.1)

    my_gaussian_mutation.duplicate_and_mutate_attribute(my_cell, "weights")

    assert my_cell.get_parameter("weights").tensor.shape[0] == 12
    assert (
        my_cell.structural_equation.edge_parameter_dict["weights"].tensor.shape[0] == 12
    )
    assert my_cell.grn.get_edge_attr("weights").shape[0] == 12


def test_double_duplication(my_cell):
    my_gaussian_mutation = GaussianMutation(0.1)

    my_gaussian_mutation.duplicate_and_mutate_attribute(my_cell, "weights")

    with pytest.raises(RuntimeError):
        my_gaussian_mutation.duplicate_and_mutate_attribute(my_cell, "weights")


def test_bernoulli_mutation_proba_zero(my_cell):

    my_bernoulli_mutation = BernoulliMutation(p=0.0)
    my_bernoulli_mutation.duplicate_and_mutate_attribute(my_cell, "weights")

    assert torch.equal(
        my_cell.get_parameter("weights").tensor[0],
        my_cell.get_parameter("weights").tensor[-1],
    )


def test_bernoulli_mutation_with_proba_ones(my_cell):
    my_bernoulli_mutation = BernoulliMutation(p=1.0)
    my_bernoulli_mutation.duplicate_and_mutate_attribute(my_cell, "weights")

    assert torch.equal(
        my_cell.get_parameter("weights").tensor,
        torch.zeros((12, my_cell.grn.n_edges, 1)),
    )


def test_simulate_trajectory_with_mutations(my_cell):
    my_cell.state = torch.ones((12, 10, 1))

    my_gaussian_mutation = GaussianMutation(0.5)
    my_gaussian_mutation.duplicate_and_mutate_attribute(my_cell, "weights")

    cell_traj = simulate_deterministic_trajectory(my_cell, torch.linspace(0, 1, 100))

    assert cell_traj.shape[1] == 12
    assert not torch.equal(cell_traj[:, 0], cell_traj[:, 1])
