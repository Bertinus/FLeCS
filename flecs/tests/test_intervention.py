from flecs.cell import Cell
from flecs.parameter import NodeParameter, EdgeParameter
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from flecs.grn import RandomGRN
from flecs.structural_equation import SigmoidLinearSE
import torch
import pytest
import copy
from flecs.intervention import KnockoutIntervention, DrugLinearIntervention


@pytest.fixture
def my_cell():
    grn = RandomGRN(10, 3)
    linear_se = SigmoidLinearSE(
        gene_decay=NodeParameter(dim=(1,), prior_dist=Gamma(10, 10)),
        weights=EdgeParameter(dim=(1,), prior_dist=Normal(1, 1)),
    )

    cell = Cell(grn=grn, structural_equation=linear_se)

    cell.state = torch.ones((12, 10, 1))

    return cell


def test_ko_intervention(my_cell):
    my_ko_intervention = KnockoutIntervention()
    edges_before_intervention = copy.deepcopy(my_cell.edges)

    interv_node = int(edges_before_intervention[0, 0])

    my_ko_intervention.intervene(my_cell, interv_node)

    expected_number_of_removed_edges = (
        edges_before_intervention[:, 0] == interv_node
    ).sum()

    assert (
        len(my_cell.edges)
        == len(edges_before_intervention) - expected_number_of_removed_edges
    )

    my_ko_intervention.reset(my_cell)

    assert len(my_cell.edges) == len(edges_before_intervention)


def test_double_ko_intervention(my_cell):
    my_ko_intervention = KnockoutIntervention()
    edges_before_intervention = copy.deepcopy(my_cell.edges)

    # First KO
    interv_node_1 = int(edges_before_intervention[0, 0])
    expected_number_of_removed_edges_1 = (my_cell.edges[:, 0] == interv_node_1).sum()
    my_ko_intervention.intervene(my_cell, interv_node_1)

    # Second KO
    interv_node_2 = int(my_cell.edges[0, 0])
    expected_number_of_removed_edges_2 = (my_cell.edges[:, 0] == interv_node_2).sum()
    my_ko_intervention.intervene(my_cell, interv_node_2)

    assert (
        len(my_cell.edges)
        == len(edges_before_intervention)
        - expected_number_of_removed_edges_1
        - expected_number_of_removed_edges_2
    )

    my_ko_intervention.reset(my_cell)

    assert len(my_cell.edges) == len(edges_before_intervention)


def test_perform_ko_intervention_twice(my_cell):
    my_ko_intervention = KnockoutIntervention()
    edges_before_intervention = copy.deepcopy(my_cell.edges)

    # First KO
    interv_node_1 = int(edges_before_intervention[0, 0])
    my_ko_intervention.intervene(my_cell, interv_node_1)

    with pytest.raises(ValueError):
        my_ko_intervention.intervene(my_cell, interv_node_1)


def test_perform_drug_intervention(my_cell):

    my_drug_intervention = DrugLinearIntervention()
    direct_effects = torch.ones(10)

    production_rates_before_intervention = my_cell.get_production_rates(my_cell.state)
    my_drug_intervention.intervene(my_cell, direct_effects)
    production_rates_after_intervention = my_cell.get_production_rates(my_cell.state)

    assert torch.isclose(
        production_rates_before_intervention + 1,
        production_rates_after_intervention,
        rtol=1e-2,
        atol=1e-2,
    ).all()


def test_reset_drug_intervention(my_cell):

    my_drug_intervention = DrugLinearIntervention()
    direct_effects = torch.ones(10)

    production_rates_before_intervention = my_cell.get_production_rates(my_cell.state)
    my_drug_intervention.intervene(my_cell, direct_effects)
    my_drug_intervention.reset(my_cell)

    production_rates_after_intervention_and_reset = my_cell.get_production_rates(
        my_cell.state
    )

    assert torch.isclose(
        production_rates_before_intervention,
        production_rates_after_intervention_and_reset,
        rtol=1e-2,
        atol=1e-2,
    ).all()


def test_double_drug_intervention(my_cell):

    my_drug_intervention = DrugLinearIntervention()
    direct_effects = torch.ones(10)

    production_rates_before_intervention = my_cell.get_production_rates(my_cell.state)
    my_drug_intervention.intervene(my_cell, direct_effects)
    my_drug_intervention.intervene(my_cell, 3 * direct_effects)

    production_rates_after_double_intervention = my_cell.get_production_rates(
        my_cell.state
    )

    my_drug_intervention.reset(my_cell)

    production_rates_after_intervention_and_reset = my_cell.get_production_rates(
        my_cell.state
    )

    assert torch.isclose(
        production_rates_before_intervention + 4,
        production_rates_after_double_intervention,
        rtol=1e-2,
        atol=1e-2,
    ).all()

    assert torch.isclose(
        production_rates_before_intervention,
        production_rates_after_intervention_and_reset,
        rtol=1e-2,
        atol=1e-2,
    ).all()
