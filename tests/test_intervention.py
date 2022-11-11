import copy

import pytest
import torch
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from flecs.cell_population import CellPopulation, TestCellPop
from flecs.intervention import Intervention


@pytest.fixture
def my_cells():
    pass


def test_intervention(my_cells):
    my_ko_intervention = Intervention()
    edges_before_intervention = copy.deepcopy(my_cells.edges)

    # First KO
    interv_gene_1 = int(edges_before_intervention[0, 0])
    expected_number_of_removed_edges_1 = (my_cells.edges[:, 0] == interv_gene_1).sum()
    my_ko_intervention.intervene(my_cells, interv_gene_1)

    # Second KO
    interv_gene_2 = int(my_cells.edges[0, 0])
    expected_number_of_removed_edges_2 = (my_cells.edges[:, 0] == interv_gene_2).sum()
    my_ko_intervention.intervene(my_cells, interv_gene_2)

    assert (
        len(my_cells.edges)
        == len(edges_before_intervention)
        - expected_number_of_removed_edges_1
        - expected_number_of_removed_edges_2
    )

    my_ko_intervention.reset(my_cells)

    assert len(my_cells.edges) == len(edges_before_intervention)


def test_intervention_twice(my_cells):
    my_ko_intervention = Intervention()
    edges_before_intervention = copy.deepcopy(my_cells.edges)

    # First KO
    interv_gene_1 = int(edges_before_intervention[0, 0])
    my_ko_intervention.intervene(my_cells, interv_gene_1)

    with pytest.raises(ValueError):
        my_ko_intervention.intervene(my_cells, interv_gene_1)


def test_reset_drug_intervention(my_cells):

    my_drug_intervention = DrugLinearIntervention()
    direct_effects = torch.ones(10)

    production_rates_before_intervention = my_cells.get_production_rates(my_cells.state)
    my_drug_intervention.intervene(my_cells, direct_effects)
    my_drug_intervention.reset(my_cells)

    production_rates_after_intervention_and_reset = my_cells.get_production_rates(
        my_cells.state
    )

    assert torch.isclose(
        production_rates_before_intervention,
        production_rates_after_intervention_and_reset,
        rtol=1e-2,
        atol=1e-2,
    ).all()
