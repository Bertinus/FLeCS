import torch
from abc import ABC, abstractmethod
from flecs.node_set import NodeSet
from flecs.edge_set import EdgeSet
from typing import Tuple, Dict, Union


class CellPopulation(ABC):
    def __init__(self):
        self.state = torch.zeros((0, 0, 0))
        self.production_rates = torch.zeros((0, 0, 0))
        self.decay_rates = torch.zeros((0, 0, 0))
        self._node_set_dict: Dict[str, NodeSet] = {}
        self._edge_set_dict: Dict[Tuple[str, str, str], EdgeSet] = {}

    def __getitem__(
        self, key: Union[str, Tuple[str, str, str]]
    ) -> Union[NodeSet, EdgeSet]:
        if type(key) is tuple:
            return self._edge_set_dict[key]
        else:
            return self._node_set_dict[key]

    def __setitem__(
        self, key: Union[str, Tuple[str, str, str]], value: Union[NodeSet, EdgeSet]
    ):
        if type(key) is tuple:
            assert isinstance(value, EdgeSet)
            assert key not in self._edge_set_dict
            self._edge_set_dict[key] = value
        else:
            assert isinstance(value, NodeSet)
            assert key not in self._node_set_dict
            self._node_set_dict[key] = value

    @property
    def n_cells(self) -> int:
        return self.state.shape[0]

    @property
    def n_nodes(self) -> int:
        return sum([len(node_set) for node_set in self._node_set_dict.values()])

    @property
    def node_types(self):
        return list(self._node_set_dict.keys())

    @property
    def edge_types(self):
        return list(self._edge_set_dict.keys())

    @abstractmethod
    def get_production_rates(self):
        pass

    @abstractmethod
    def get_decay_rates(self):
        pass

    def get_derivatives(self, state):
        self.state = state
        return self.get_production_rates() - self.get_decay_rates()

    def __repr__(self):
        return "CellPopulation. " + str(self.n_nodes) + " nodes and " + str(self.n_cells) + " cells."

    def __str__(self):
        s = "CellPopulation. " + str(self.n_nodes) + " nodes and " + str(self.n_cells) + " cells.\n"
        s += "\t NodeSets:\n"
        for k, v in self._node_set_dict.items():
            s += "\t\t" + k + ": " + str(v) + "\n"
        s += "\t EdgeSets:\n"
        for k, v in self._edge_set_dict.items():
            s += "\t\t" + str(k) + ": " + str(v) + "\n"

        return s
