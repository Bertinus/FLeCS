from typing import Union

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

from flecs.cell_population import CellPopulation
from flecs.sets import EdgeSet, NodeSet


def duplicate_attribute(obj: Union[CellPopulation, NodeSet, EdgeSet], attr_name: str, n_cells: int):

    assert n_cells > 1
    attr_value = obj.element_level_attr_dict[attr_name]

    if attr_value.shape[0] != 1:
        raise RuntimeError(
            "Cannot duplicate an attribute which has already been duplicated."
        )

    duplicated_attr_value = torch.cat([attr_value] * n_cells, dim=0)

    obj.__setattr__(attr_name, duplicated_attr_value)


def apply_bernoulli_mutation(obj: Union[CellPopulation, NodeSet, EdgeSet], attr_name: str, p: float, n_cells: int):

    duplicate_attribute(obj, attr_name, n_cells)
    noise_dist = Bernoulli(1 - p)
    attr_value = obj.element_level_attr_dict[attr_name]
    obj.__setattr__(attr_name, attr_value * noise_dist.sample(attr_value.shape))


def apply_gaussian_mutation(obj: Union[CellPopulation, NodeSet, EdgeSet], attr_name: str, sigma: float, n_cells: int):

    duplicate_attribute(obj, attr_name, n_cells)
    noise_dist = Normal(0, sigma)
    attr_value = obj.element_level_attr_dict[attr_name]
    obj.__setattr__(attr_name, attr_value + noise_dist.sample(attr_value.shape))
