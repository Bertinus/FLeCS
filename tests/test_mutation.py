import pytest
import torch
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from flecs.cell_population import CellPopulation
from flecs.grn import RandomGRN
from flecs.mutation import BernoulliMutation, GaussianMutation
from flecs.attribute import EdgeAttribute, GeneAttribute
from flecs.structural_equation import SigmoidLinearSE
from flecs.trajectory import simulate_deterministic_trajectory


@pytest.fixture
def my_cells():
    grn = RandomGRN(10, 3)
    linear_se = SigmoidLinearSE(
        gene_decay=GeneAttribute(dim=(1,), prior_dist=Gamma(10, 10)),
        weights=EdgeAttribute(dim=(1,), prior_dist=Normal(1, 1)),
    )

    cells = CellPopulation(grn=grn, structural_equation=linear_se)

    cells.state = 10 * torch.ones((12, 10, 1))

    return cells


def test_duplication(my_cells):
    my_gaussian_mutation = GaussianMutation(0.1)

    my_gaussian_mutation.duplicate_and_mutate_attribute(my_cells, "weights")

    assert my_cells.get_attribute("weights").tensor.shape[0] == 12
    assert (
        my_cells.structural_equation.edge_attribute_dict["weights"].tensor.shape[0] == 12
    )
    assert my_cells._grn.get_edge_attr("weights").shape[0] == 12


def test_double_duplication(my_cells):
    my_gaussian_mutation = GaussianMutation(0.1)

    my_gaussian_mutation.duplicate_and_mutate_attribute(my_cells, "weights")

    with pytest.raises(RuntimeError):
        my_gaussian_mutation.duplicate_and_mutate_attribute(my_cells, "weights")


def test_bernoulli_mutation_proba_zero(my_cells):

    my_bernoulli_mutation = BernoulliMutation(p=0.0)
    my_bernoulli_mutation.duplicate_and_mutate_attribute(my_cells, "weights")

    assert torch.equal(
        my_cells.get_attribute("weights").tensor[0],
        my_cells.get_attribute("weights").tensor[-1],
    )


def test_bernoulli_mutation_with_proba_ones(my_cells):
    my_bernoulli_mutation = BernoulliMutation(p=1.0)
    my_bernoulli_mutation.duplicate_and_mutate_attribute(my_cells, "weights")

    assert torch.equal(
        my_cells.get_attribute("weights").tensor,
        torch.zeros((12, my_cells._grn.n_edges, 1)),
    )


def test_simulate_trajectory_with_mutations(my_cells):
    my_cells.state = torch.ones((12, 10, 1))

    my_gaussian_mutation = GaussianMutation(0.5)
    my_gaussian_mutation.duplicate_and_mutate_attribute(my_cells, "weights")

    cell_traj = simulate_deterministic_trajectory(my_cells, torch.linspace(0, 1, 100))

    assert cell_traj.shape[1] == 12
    assert not torch.equal(cell_traj[:, 0], cell_traj[:, 1])
