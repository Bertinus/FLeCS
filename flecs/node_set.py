import torch
from typing import Dict


class NodeSet:
    """
    srt of proteins, genes, small mol, oligos.
    NodeSet should never overlap idx.

    Node attribute is an array. e.g., decay rate for each gene.
    + for ODE solver, state of cell must be in a single tensor.
    + but we want the total state to encompass multiple nodesets
    + so we index the super cell so we can pass a nodeset tensor to the ODE solver.
    """
    def __init__(self, super_cell, idx_low: int, idx_high: int, attribute_dict: Dict[str, torch.Tensor] = None):
        self._super_cell = super_cell  # parent <- this is a cell population.
        self._idx_low = idx_low  #  range of rows in the state for this node set.
        self._idx_high = idx_high  #

        # Initialize node attributes
        self.attribute_dict = {}
        for attr_name, attr_value in attribute_dict.items():
            self[attr_name] = attr_value

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.attribute_dict[key]

    def __setitem__(self, key: str, value: torch.Tensor):
        assert isinstance(value, torch.Tensor)
        # Make sure the attribute value has the right dimension
        if len(value.shape) == 1:
            value = value[None, :]
        assert value.shape[:2] == (1, len(self))
        self.attribute_dict[key] = value

    def keys(self):
        return self.attribute_dict.keys()

    @property
    def state(self) -> torch.Tensor:
        return self._super_cell.state[:, self._idx_low: self._idx_high + 1]

    @state.setter
    def state(self, state: torch.Tensor):
        assert state.shape == self.state.shape
        self._super_cell.state[:, self._idx_low: self._idx_high + 1] = state

    @property
    def decay_rate(self) -> torch.Tensor:
        return self._super_cell.decay_rates[:, self._idx_low: self._idx_high + 1]

    @decay_rate.setter
    def decay_rate(self, decay_rate: torch.Tensor):
        assert decay_rate.shape == self.decay_rate.shape
        self._super_cell.decay_rates[:, self._idx_low: self._idx_high + 1] = decay_rate

    @property
    def production_rate(self) -> torch.Tensor:
        return self._super_cell.production_rates[:, self._idx_low: self._idx_high + 1]

    @production_rate.setter
    def production_rate(self, production_rate: torch.Tensor):
        assert production_rate.shape == self.production_rate.shape
        self._super_cell.production_rates[:, self._idx_low: self._idx_high + 1] = production_rate

    def __len__(self):
        return self._idx_high - self._idx_low + 1

    def __repr__(self):
        return "NodeSet(idx_low=" + str(self._idx_low) + ", idx_high=" + str(self._idx_high) + ", " + \
               str(self.attribute_dict) + ")"
