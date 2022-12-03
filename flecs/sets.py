from __future__ import annotations
import torch
from typing import Dict
import flecs.cell_population as cp


class Set(torch.nn.Module):

    def is_element_level_attr(self, v):
        """Element-level attributes have shape [n_cells, n_edges/nodes, ...]."""
        return (
            isinstance(v, torch.Tensor) and len(v.shape) > 1 and v.shape[1] == len(self)
        )

    def is_object_level_attr(self, v):
        """
        Object-level attributes have any other shape than [n_cells, n_edges/nodes, ...].
        """
        return (
            isinstance(v, torch.Tensor) and not (len(v.shape) > 1 and v.shape[1] == len(self))
        )

    @property
    def element_level_attr_dict(self):
        return {k: v for k, v in self.__dict__.items() if self.is_element_level_attr(v)}

    @property
    def object_level_attr_dict(self):
        return {k: v for k, v in self.__dict__.items() if self.is_object_level_attr(v)}

    def init_param(self, name: str, dist: torch.distributions.Distribution, shape=None):
        if shape is None:
            shape = (1, len(self), 1)
        self.__setattr__(name, dist.sample(shape))

    def __len__(self):
        raise NotImplementedError


class NodeSet(Set):
    def __init__(
        self,
        super_cell: cp.CellPopulation,
        idx_low: int,
        idx_high: int,
        attribute_dict: Dict[str, torch.Tensor] = None,
    ):
        """
        Class responsible for representing nodes of a given type (e.g. "genes" or
        "protein complexes", "oligonucleotides", "small molecules").

        Its attribute "state" points to a subset of the state of the cell "super_cell".
        The subset is defined by the range [idx_low, idx_high] along the second axis.

        Similarly, the decay_rate and production_rate attributes point to subsets of the
        corresponding attributes of "super_cell".

        Multiple nodesets should *not* have overlapping idx ranges.

        Args:
            super_cell (obj): The cell this NodeSet belongs to.
            idx_low (int): Beginning index of this NodeSet in the NodeTensor.
            idx_high: Ending index of this NodeSet in the NodeTensor. Note this code
                corrects for the fact that arrays are indexed using half intervals by
                adding +1 to idx_high for all operations.
            attribute_dict (dict): Dict of node attributes. Each node attribute is an
                array e.d., decay rate for each gene. This is done because for the ODE
                solver, the state of cell must be in a single tensor. We want the total
                state to encompass multiple nodesets, so we index the super cell so we
                can pass a nodeset tensor to the ODE solver.
        """
        super().__init__()
        self._super_cell = super_cell
        self.idx_low = idx_low
        self.idx_high = idx_high

        # Initialize attributes
        for attr_name, attr_value in attribute_dict.items():
            if len(attr_value.shape) == 1 and attr_value.shape[0] == len(self):
                attr_value = attr_value[None, :]
            self.__setattr__(attr_name, attr_value)

    @property
    def state(self) -> torch.Tensor:
        return self._super_cell.state[:, self.idx_low: self.idx_high + 1]

    @state.setter
    def state(self, state: torch.Tensor):
        assert state.shape == self.state.shape
        self._super_cell.state[:, self.idx_low: self.idx_high + 1] = state

    @property
    def decay_rate(self) -> torch.Tensor:
        return self._super_cell.decay_rates[:, self.idx_low: self.idx_high + 1]

    @decay_rate.setter
    def decay_rate(self, decay_rate: torch.Tensor):
        assert decay_rate.shape == self.decay_rate.shape
        self._super_cell.decay_rates[:, self.idx_low: self.idx_high + 1] = decay_rate

    @property
    def production_rate(self) -> torch.Tensor:
        return self._super_cell.production_rates[:, self.idx_low: self.idx_high + 1]

    @production_rate.setter
    def production_rate(self, production_rate: torch.Tensor):
        assert production_rate.shape == self.production_rate.shape
        self._super_cell.production_rates[
            :, self.idx_low: self.idx_high + 1
        ] = production_rate

    def __len__(self):
        return self.idx_high - self.idx_low + 1

    def __repr__(self):
        return "NodeSet(idx_low={}, idx_high={}, {})".format(
            self.idx_low, self.idx_high, self.element_level_attr_dict
        )


class EdgeSet(Set):
    def __init__(
        self,
        edges: torch.Tensor = None,
        attribute_dict: Dict[str, torch.Tensor] = None,
    ):
        """
        Class responsible for representing edges of a given type. An edge type is
            defined by a tuple (source_node_type, interaction_type, target_node_type).
            Examples of node types would be "gene", "codes", or "protein". Example
            interaction types would be "inhibits", "facilitates".

        An `attribute_dict` Tensor shape is [n_cells, n_edges, ...]. n_cells=1 is
            permissable for attributes being shared across all cells.

        Args:
            edges: shape (n_edges, 2).
                The first column corresponds to the indices of the source nodes in
                cell[source_node_type]. The second column corresponds to the indices of
                the target nodes in cell[target_node_type].
            attribute_dict: str attribute and Tensor of associated per-edge values. The
                tensor.
        """
        super().__init__()

       # Initialize edge indices
        if edges is None:
            edges = torch.zeros((0, 2)).long()
        assert edges.shape[1] == 2 and len(edges.shape) == 2
        self.edges = edges.long()

        # Initialize attributes
        for attr_name, attr_value in attribute_dict.items():
            if len(attr_value.shape) == 1 and attr_value.shape[0] == len(self):
                attr_value = attr_value[None, :]
            self.__setattr__(attr_name, attr_value)

    @property
    def tails(self):
        """Returns children."""
        return self.edges[:, 0]

    @property
    def heads(self):
        """Returns parents."""
        return self.edges[:, 1]

    def add_edges(
        self, edges: torch.Tensor, attribute_dict: Dict[str, torch.Tensor] = None
    ):
        """Adds provided edges with optional attibutes to the graph.
        Args:
            edges (tensor): A set of (src, target) pairs.
            attribute_dict (dict): Optional attributes for each edge.
        """
        if attribute_dict is None:
            attribute_dict = {}

        # Make sure edge_attr_dict has the right set of keys
        assert attribute_dict.keys() == self.element_level_attr_dict.keys()

        for attr_name in attribute_dict:
            # Make sure the values of edge_attr_dict have the right dimension
            if len(attribute_dict[attr_name].shape) == 1:
                attribute_dict[attr_name] = attribute_dict[attr_name][None, :]
            assert attribute_dict[attr_name].shape[1] == len(edges)

            self.__setattr__(
                attr_name,
                torch.cat(
                    (self.element_level_attr_dict[attr_name], attribute_dict[attr_name]), dim=1
                ),
            )

        self.edges = torch.cat((self.edges, edges))

    def remove_edges(self, indices: torch.Tensor):
        """Removes edges specified by indices.
        Args:
            indices: shape (n_edges) boolean.
        """
        assert len(indices.shape) > 0, "indices should not be singleton."
        if self.edges.shape[0] != len(indices):
            raise IOError("indices should be a boolean matrix the length of all edges in this set.")

        to_be_kept = torch.logical_not(indices)

        for attr_name, attr_value in self.element_level_attr_dict.items():
            self.__setattr__(attr_name, attr_value[:, to_be_kept])

        self.edges = self.edges[to_be_kept]

    def get_edges(self, indices):
        """Retrieves edges indexed by indices, with their attributes.
        Args:
            indices: shape (n_edges) boolean.

        Returns: (edges, attibutes) tuple, as indexed by `indices`.
        """
        edges = self.edges[indices]

        edge_attr_dict = {}
        for attr_name, attr in self.element_level_attr_dict.items():
            edge_attr_dict[attr_name] = attr[:, indices]

        return edges, edge_attr_dict

    def out_edges(self, gene_idx: int):
        return self.edges[:, 0] == gene_idx

    def in_edges(self, gene_idx: int):
        return self.edges[:, 1] == gene_idx

    def __len__(self):
        return len(self.edges)

    def __repr__(self):
        return "EdgeSet({}, {})".format(str(self.edges), str(self.element_level_attr_dict))
