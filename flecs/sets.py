import torch
from typing import Dict, Tuple
from typing import Union, List
from flecs.production import SimpleConv


class Set(torch.nn.Module):
    def initialize_attributes_from_dict(self, attribute_dict):
        if attribute_dict is not None:
            for attr_name, attr_value in attribute_dict.items():
                if isinstance(attr_value, list):
                    assert len(attr_value) == len(self)
                else:
                    assert isinstance(attr_value, torch.Tensor)
                    if len(attr_value.shape) == 1 and attr_value.shape[0] == len(self):
                        attr_value = attr_value[None, :, None]

                self.__setattr__(attr_name, attr_value)

    def is_element_level_attr(self, v):
        is_tensor_element_level_attr = (
            isinstance(v, torch.Tensor) and len(v.shape) > 1 and v.shape[1] == len(self)
        )
        is_list_element_level_attr = isinstance(v, list) and len(v) == len(self)

        return is_tensor_element_level_attr or is_list_element_level_attr

    @property
    def element_level_attr_dict(self):
        non_param_dict = {k: v for k, v in self.__dict__.items() if self.is_element_level_attr(v) and k != "edges"}
        param_dict = {k: v for k, v in self.__dict__['_parameters'].items()}
        return {**non_param_dict, **param_dict}

    def init_param(self, name: str, dist: torch.distributions.Distribution, shape=None, requires_grad=True):
        if shape is None:
            shape = (1, len(self), 1)

        param = torch.nn.Parameter(dist.sample(shape), requires_grad=requires_grad)

        self.__setattr__(name, param)

    def __len__(self):
        raise NotImplementedError


class NodeSet(Set):
    def __init__(
        self,
        super_cell,
        idx_low: int,
        idx_high: int,
        attribute_dict: Dict[str, Union[List[str], torch.Tensor]] = None,
    ):
        """
        Class responsible for representing nodes of a given type (e.g. "genes",
        "proteins", "protein complexes", "small molecules").

        Its attribute "state" points to a subset of the state of the cell "super_cell".
        The subset is defined by the range [idx_low, idx_high] along the second axis. This allows us to easily access
        the state corresponding to a specific node type, while storing the state of all
        node types in a single torch.Tensor, eventually enabling efficient integration with ODE solvers.

        Similarly, the decay_rates and production_rates attributes point to subsets of the
        corresponding attributes of "super_cell".

        Node-level attributes can be defined in the form of Tensors whose dimension along the second axis is equal to
        the number of nodes. One can easily access all the node-level attributes through the `element_level_attr_dict`
        property.

        Args:
            super_cell (CellPopulation): The cell this NodeSet belongs to.
            idx_low: Beginning index of this NodeSet in the state of super_cell.
            idx_high: Ending index of this NodeSet in the state of super_cell.
            attribute_dict: Dict of node attributes.
        """
        super().__init__()
        # The _super_cell attribute is a list to avoid that the supercell object gets registered as a child module
        self._super_cell = [super_cell]
        self.idx_low = idx_low
        self.idx_high = idx_high

        # Initialize attributes
        self.initialize_attributes_from_dict(attribute_dict)

    @property
    def state(self) -> torch.Tensor:
        """
        (`torch.Tensor`) State of the nodes included in this NodeSet.
        """
        return self._super_cell[0].state[:, self.idx_low : self.idx_high]

    @state.setter
    def state(self, state: Union[torch.Tensor, float]):
        if isinstance(state, torch.Tensor):
            assert state.shape == self.state.shape
        self._super_cell[0].state[:, self.idx_low : self.idx_high] = state

    @property
    def decay_rates(self) -> torch.Tensor:
        """
        (`torch.Tensor`) Decay rates of the nodes included in this NodeSet.
        """
        return self._super_cell[0].decay_rates[:, self.idx_low : self.idx_high]

    @decay_rates.setter
    def decay_rates(self, decay_rates: torch.Tensor):
        assert decay_rates.shape == self.decay_rates.shape
        self._super_cell[0].decay_rates[:, self.idx_low : self.idx_high] = decay_rates

    @property
    def production_rates(self) -> torch.Tensor:
        """
        (`torch.Tensor`) Production rates of the nodes included in this NodeSet.
        """
        return self._super_cell[0].production_rates[:, self.idx_low : self.idx_high]

    @production_rates.setter
    def production_rates(self, production_rates: torch.Tensor):
        assert production_rates.shape == self.production_rates.shape
        self._super_cell[0].production_rates[
            :, self.idx_low : self.idx_high
        ] = production_rates

    def __len__(self):
        return self.idx_high - self.idx_low

    def __repr__(self):
        return "NodeSet(idx_low={}, idx_high={}, node attributes: {})".format(
            self.idx_low, self.idx_high, list(self.element_level_attr_dict.keys())
        )


class EdgeSet(Set):
    def __init__(
        self,
        edges: torch.Tensor = None,
        attribute_dict: Dict[str, Union[List[str], torch.Tensor]] = None,
    ):
        """
        Class responsible for representing edges of a given type. An edge type is
            defined by a tuple (source_node_type, interaction_type, target_node_type).

            * Examples of node types include "proteins", "small molecules", "gene/RNA".
            * Examples of interaction types include "inhibits", "activates", "catalyzes", "codes for".

        Edge-level attributes can be defined in the form of Tensors whose dimension along the second axis is equal to
        the number of edges. One can easily access all the edge-level attributes through the `element_level_attr_dict`
        property.

        Args:
            edges: shape (n_edges, 2). The first column corresponds to the indices of the source nodes in
                cell[source_node_type]. The second column corresponds to the indices of
                the target nodes in cell[target_node_type].
            attribute_dict: dictionary  which maps attribute names to Tensors containing attribute values for all edges.
        """
        super().__init__()
        # Initialize edge indices
        if edges is None:
            edges = torch.zeros((0, 2)).long()
        assert edges.shape[1] == 2 and len(edges.shape) == 2
        self.edges = edges.long()

        # Initialize attributes
        self.initialize_attributes_from_dict(attribute_dict)

    @property
    def tails(self) -> torch.Tensor:
        """
        (`torch.Tensor`) Returns the parents of all edges as a Tensor of shape (n_edges).
        """
        return self.edges[:, 0]

    @property
    def heads(self) -> torch.Tensor:
        """
        (`torch.Tensor`) Returns the children of all edges as a Tensor of shape (n_edges).
        """
        return self.edges[:, 1]

    def add_edges(
        self, edges: torch.Tensor, attribute_dict: Dict[str, torch.Tensor] = None
    ):
        """
        Adds provided edges to the EdgeSet. The keys of `attribute_dict` must match the keys of
        `self.element_level_attr_dict`.

        Args:
            edges (tensor): A set of (src, target) pairs. The indices must correspond to the indices of the
                source/target nodes in the source/target NodeSets.
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
                    (
                        self.element_level_attr_dict[attr_name],
                        attribute_dict[attr_name],
                    ),
                    dim=1,
                ),
            )

        assert len(self.edges.shape) == len(edges.shape), \
            "edge tensor to be added must have the same dimensions as the current edge set."
        self.edges = torch.cat((self.edges, edges))

    def remove_edges(self, indices: torch.Tensor):
        """
        Removes edges specified by indices.

        Args:
            indices: Boolean Tensor of shape (n_edges).
        """
        to_be_kept = torch.logical_not(indices)

        for attr_name, attr_value in self.element_level_attr_dict.items():
            if isinstance(attr_value, torch.nn.parameter.Parameter):
                new_attr_value = torch.nn.Parameter(attr_value[:, to_be_kept])
            else:
                new_attr_value = attr_value[:, to_be_kept]
            self.__setattr__(attr_name, new_attr_value)

        self.edges = self.edges[to_be_kept]

    def get_edges(self, indices) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Retrieves edges indexed by indices, with their attributes.

        Args:
            indices: Boolean Tensor of shape (n_edges).

        Returns:
            edges (torch.Tensor): edge indices as a Tensor of shape (n_retrieved_edges, 2).
            edge_attr_dict (Dict[str, torch.Tensor]): dictionary which maps attribute names to Tensors whose dimension
                along the second axis is equal to the number of retrieved edges.
        """
        edges = self.edges[indices]

        edge_attr_dict = {}
        for attr_name, attr in self.element_level_attr_dict.items():
            edge_attr_dict[attr_name] = attr[:, indices]

        return edges, edge_attr_dict

    def out_edges(self, node_idx: int):
        """
        Retrieves all the outgoing edges of a given node.

        Args:
            node_idx: index of the node of interest.

        Returns:
            torch.Tensor: Boolean tensor of shape (n_edges) which indicates the edges that are outgoing the node of
                interest.
        """
        return self.edges[:, 0] == node_idx

    def in_edges(self, node_idx: int):
        """
        Retrieves all the incoming edges of a given node.

        Args:
            node_idx: index of the node of interest.

        Returns:
            torch.Tensor: Boolean tensor of shape (n_edges) which indicates the edges that are incoming the node of
                interest.
        """
        return self.edges[:, 1] == node_idx

    def init_edge_conv(self, name, out_channels):
        self.__setattr__(name, SimpleConv())

        self.init_param(name=name + "_weights",
                        dist=torch.distributions.Normal(0, 0.01),
                        shape=(1, len(self), out_channels))

    def conv(self, x, name, tgt_nodeset_len=None):
        conv = self.__getattr__(name)
        weights = self.__getattr__(name + "_weights")

        if tgt_nodeset_len is None:
            tgt_nodeset_len = x.shape[1]

        return conv(x=x, edge_index=self.edges.T, edge_weight=weights, tgt_nodeset_len=tgt_nodeset_len)

    def __len__(self):
        return len(self.edges)

    def __repr__(self):
        return "EdgeSet({} edges, edge attributes: {})".format(
            len(self.edges),
            list({k: v for k, v in self.element_level_attr_dict.items() if k != "edges"}.keys())
        )


if __name__ == "__main__":

    from flecs.cell_population import CellPopulation
    import networkx as nx
    import torch_scatter

    class MyCellPop(CellPopulation):
        def __init__(self):
            g = nx.DiGraph()
            g.add_node(1, type="gene", name="G1", dummy_value=1.)
            g.add_node(2, type="gene", name="G2", dummy_value=2.)

            g.add_edge(1, 1, type="regulates", weight=.01)
            g.add_edge(2, 2, type="regulates", weight=1.)
            g.add_edge(1, 2, type="regulates", weight=-1.)

            super().__init__(g, scale_factor_state_prior=0.1)

        def compute_production_rates(self):
            edges = self["gene", "regulates", "gene"]
            genes = self["gene"]

            parent_indices = edges.tails
            children_indices = edges.heads

            messages = edges.weight * genes.state[:, parent_indices]

            genes.production_rates = torch.sigmoid(torch_scatter.scatter(messages, children_indices, dim=1))

        def compute_decay_rates(self):
            # All tracked molecules have an exponential decay rate
            ALPHA = 0.1
            self['gene'].decay_rates = ALPHA * self["gene"].state


    my_cell_pop = MyCellPop()

    print("toto")
