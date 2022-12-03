import pytest
import torch
from copy import deepcopy

from flecs.cell_population import TestCellPop
from flecs.mutation import apply_bernoulli_mutation, apply_gaussian_mutation
from flecs.trajectory import simulate_deterministic_trajectory


@pytest.fixture
def my_cells():
    return TestCellPop(n_cells=3)


def test_bernoulli_mutation(my_cells):
    N_CELLS = 10000
    P_DROPS = [0.1, 0.5, 0.9]

    for p_drop in P_DROPS:

        cells_to_test = deepcopy(my_cells)

        s1 = cells_to_test["gene"].basal_expression
        apply_bernoulli_mutation(
            cells_to_test["gene"],
            "basal_expression",
            p=p_drop,
            n_cells=N_CELLS)
        s2 = cells_to_test["gene"].basal_expression

        mean_n_of_rm_elems = torch.mean((s1 != s2[1:, ...]).sum(-1) / s1.numel())
        assert torch.isclose(mean_n_of_rm_elems, torch.tensor(p_drop), rtol=0.01)

    assert s1[0, ...].numel() == s2[0, ...].numel()
    assert s2.shape[0] == N_CELLS


def test_gaussian_mutation(my_cells):
    N_CELLS = 10000
    SIGMAS = [0.1, 1, 3]

    for sigma in SIGMAS:

        cells_to_test = deepcopy(my_cells)

        s1 = cells_to_test["gene"].basal_expression
        apply_gaussian_mutation(
            cells_to_test["gene"],
            "basal_expression",
            sigma=sigma,
            n_cells=N_CELLS)
        s2 = cells_to_test["gene"].basal_expression

        # TODO: test for this.
        #mean_n_of_rm_elems = torch.mean((s1 != s2[1:, ...]).sum(-1) / s1.numel())
        #assert torch.isclose(mean_n_of_rm_elems, torch.tensor(sigma), rtol=0.01)

    assert s1[0, ...].numel() == s2[0, ...].numel()
    assert s2.shape[0] == N_CELLS


def test_double_duplication(my_cells):
    apply_bernoulli_mutation(my_cells["gene"], "basal_expression", p=0.1, n_cells=10)

    with pytest.raises(RuntimeError):
        apply_bernoulli_mutation(
            my_cells["gene"], "basal_expression", p=0.1, n_cells=10)


def test_bernoulli_mutation_proba_zero(my_cells):
    apply_bernoulli_mutation(my_cells["gene"], "basal_expression", p=0, n_cells=10)

    assert torch.equal(
        my_cells["gene"].basal_expression[0, ...],
        my_cells["gene"].basal_expression[-1, ...],
    )


def test_bernoulli_mutation_with_proba_ones(my_cells):
    apply_bernoulli_mutation(my_cells["gene"], "basal_expression", p=1, n_cells=10)

    assert torch.equal(my_cells["gene"].basal_expression,
        torch.zeros(my_cells["gene"].basal_expression.shape),
    )


def test_simulate_trajectory_with_mutations(my_cells):
    apply_bernoulli_mutation(my_cells["gene"], "basal_expression", p=0.5, n_cells=10)
    cell_traj = simulate_deterministic_trajectory(my_cells, torch.linspace(0, 1, 100))

    # Trajectories [0] and [-1] are not equal.
    assert torch.sum(cell_traj[0, ...] == cell_traj[-1, ...]) == 0
