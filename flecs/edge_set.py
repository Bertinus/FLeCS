import torch
from typing import Dict


class EdgeSet:
    """???"""
    def __init__(self, edges: torch.Tensor = None, attribute_dict: Dict[str, torch.Tensor] = None):
        """
        Class responsible for representing edges of a given type. An edge type is defined by a tuple
        (source_node_type, interaction_type, target_node_type).

        Example:  ("gene", "codes", "protein").

        Its attribute "state" points to a subset of the state of the cell "super_cell". The subset is defined by the
        range [idx_low, idx_high] along the second axis.


        Similarly, the decay_rate and production_rate attributes point to subsets of the corresponding attributes of
        "super_cell".

        Args:
            edges: shape (n_edges, 2).
                The first column corresponds to the indices of the source nodes in cell[source_node_type].
                The second column corresponds to the indices of the target nodes in cell[target_node_type].
            attribute_dict:
        """
        if edges is None:
            edges = torch.zeros((0, 2)).long()
        if attribute_dict is None:
            attribute_dict = {}

        # Initialize edge indices
        assert edges.shape[1] == 2 and len(edges.shape) == 2
        self.edges = edges.long()

        # Initialize edge attributes
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

    def __getattr__(self, item):
        if item in self.attribute_dict.keys():
            return self.attribute_dict[item]
        else:
            raise AttributeError

    def keys(self):
        return self.attribute_dict.keys()

    @property
    def tails(self):
        return self.edges[:, 0]

    @property
    def heads(self):
        return self.edges[:, 1]

    def add_edges(self, edges: torch.Tensor, attribute_dict: Dict[str, torch.Tensor] = None):
        if attribute_dict is None:
            attribute_dict = {}

        # Make sure edge_attr_dict has the right set of keys
        assert attribute_dict.keys() == self.attribute_dict.keys()

        self.edges = torch.cat((self.edges, edges))

        for attr_name in attribute_dict:
            # Make sure the values of edge_attr_dict have the right dimension
            if len(attribute_dict[attr_name].shape) == 1:
                attribute_dict[attr_name] = attribute_dict[attr_name][None, :]
            assert attribute_dict[attr_name].shape[:2] == (1, len(edges))

            self.attribute_dict[attr_name] = torch.cat(
                (self.attribute_dict[attr_name], attribute_dict[attr_name]), dim=1
            )

    def remove_edges(self, indices: torch.Tensor):
        """
        Args:
            indices: shape (n_edges) boolean.

        Returns:
        """
        to_be_kept = torch.logical_not(indices)
        self.edges = self.edges[to_be_kept]

        for attr_name, attr_value in self.attribute_dict.items():
            self.attribute_dict[attr_name] = attr_value[:, to_be_kept]

    def get_edges(self, indices):
        """
        Args:
            indices: shape (n_edges) boolean.

        Returns:
        """
        edges = self.edges[indices]

        edge_attr_dict = {}
        for attr_name, attr in self.attribute_dict.items():
            edge_attr_dict[attr_name] = attr[:, indices]

        return edges, edge_attr_dict

    def out_edges(self, gene_idx: int):
        return self.edges[:, 0] == gene_idx

    def in_edges(self, gene_idx: int):
        return self.edges[:, 1] == gene_idx

    def init_param(self, name: str, dist: torch.distributions.Distribution, shape=None):
        if shape is None:
            shape = (len(self),)
        self[name] = dist.sample(shape)

    def __len__(self):
        return len(self.edges)

    def __repr__(self):
        return "EdgeSet(" + str(self.edges) + ", " + str(self.attribute_dict) + ")"
