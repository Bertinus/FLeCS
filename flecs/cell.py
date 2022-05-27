import torch

from flecs.grn import GRN
from flecs.parameter import EdgeParameter, NodeParameter
from flecs.structural_equation import StructuralEquation


class Cell:
    """
    Class responsible for the interaction between the GRN object and the StructuralEquation object.

    Example:
        A simple example of initialization:
        ```
        from flecs.cell import Cell
        from flecs.grn import RandomGRN
        from flecs.structural_equation import SigmoidLinearSE

        grn = RandomGRN(n_nodes=10, av_num_parents=3)
        linear_se = SigmoidLinearSE()

        my_cell = Cell(grn=grn, structural_equation=linear_se)
        ```

    Attributes:
        grn (GRN): Gene Regulatory Network of the cell.
        structural_equation (StructuralEquation): Structural Equation of the cell.
    """

    def __init__(self, grn: GRN, structural_equation: StructuralEquation):
        self.grn = grn
        self.structural_equation = structural_equation
        self.structural_equation.initialize_given_structure(
            self.grn.n_nodes, self.grn.tedges
        )
        """
        Args:
            grn (GRN): Gene Regulatory Network of the cell
            structural_equation (StructuralEquation): Structural Equation of the cell
        """
        self._state = torch.zeros((0, self.grn.n_nodes, 1))

        self.sync_grn_from_se()

    @property
    def state(self):
        """(``torch.Tensor``): State of the cell. Shape (n_cells, n_nodes, *state_dim)"""
        return self._state

    @property
    def n_cells(self):
        """(``int``) Number of cells."""
        return self._state.shape[0]

    @property
    def n_nodes(self):
        """(``int``) Number of nodes."""
        return self.structural_equation.n_nodes

    @property
    def n_edges(self):
        """(``int``) Number of edges."""
        return self.structural_equation.n_edges

    @property
    def edges(self):
        """(``torch.Tensor``) Edges. Shape (n_edges, 2)."""
        return self.structural_equation.edges

    @state.setter
    def state(self, state: torch.Tensor):
        self.grn.state = state
        self._state = state

    def get_derivatives(self, state):
        """
        Returns the time derivative of the state, as computed by ``self.structural_equation``.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_nodes, *state_dim)

        Returns:
            torch.Tensor: Time derivative of the state. Shape (n_cells, n_nodes, *state_dim).

        """
        return self.structural_equation.get_derivatives(state)

    def get_production_rates(self, state):
        """
        Returns the production rates, as computed by ``self.structural_equation``.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_nodes, *state_dim)

        Returns:
            torch.Tensor: Production rates. Shape (n_cells, n_nodes, *state_dim)

        """
        return self.structural_equation.get_production_rates(state)

    def get_decay_rates(self, state):
        """
        Returns the decay rates, as computed by ``self.structural_equation``.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_nodes, *state_dim)

        Returns:
            torch.Tensor: Decay rates. Shape (n_cells, n_nodes, *state_dim)

        """
        return self.structural_equation.get_decay_rates(state)

    def get_node_parameter(self, param_name) -> NodeParameter:
        """
        Gets the node parameter ``parameter_name``.

        Args:
            param_name (str): Name of the parameter.

        Returns:
            NodeParameter: Parameter.

        """
        return self.structural_equation.node_parameter_dict[param_name]

    def get_edge_parameter(self, param_name):
        """
        Gets the edge parameter ``parameter_name``.

        Args:
            param_name (str): Name of the parameter.

        Returns:
            EdgeParameter: Parameter.

        """
        return self.structural_equation.edge_parameter_dict[param_name]

    def get_parameter(self, param_name):
        """
        Gets the parameter ``parameter_name``.

        Args:
            param_name (str): Name of the parameter.

        Returns:
            Parameter: Parameter.

        """
        return self.structural_equation.parameter_dict[param_name]

    def sync_grn_from_se(self) -> None:
        """
        Updates the values of the node and edge attributes of the GRN based on the values of the node and edge
        parameters contained in the structural equation.

        If necessary, the relevant node and edge attributes are added to the GRN.
        """
        for param_name in self.structural_equation.node_parameter_dict:
            param_tensor = self.structural_equation.node_parameter_dict[
                param_name
            ].tensor
            self.grn.set_node_attr(param_name, param_tensor)

        for param_name in self.structural_equation.edge_parameter_dict:
            param_tensor = self.structural_equation.edge_parameter_dict[
                param_name
            ].tensor
            self.grn.set_edge_attr(param_name, param_tensor)

    def sync_se_from_grn(self) -> None:
        """
        Updates the values of the node and edge parameters contained in the structural equation based on the node and
        edge attributes of the GRN.

        If necessary, the relevant node and edge parameters are added to the structural equation.
        """
        self.sync_se_from_grn_nodes()
        self.sync_se_from_grn_edges()

    def sync_se_from_grn_nodes(self) -> None:
        """
        Updates the values of the node parameters contained in the structural equation based on the node attributes of
        the GRN.

        If necessary, the relevant node parameters are added to the structural equation.
        """
        for param_name in self.grn.node_attr_name_list:
            param_tensor = self.grn.get_node_attr(param_name)

            if param_name in self.structural_equation.node_parameter_dict:
                self.structural_equation.node_parameter_dict[
                    param_name
                ].tensor = param_tensor
            else:
                new_node_parameter = NodeParameter(dim=param_tensor.shape[2:])
                new_node_parameter.tensor = param_tensor

                self.structural_equation.set_parameter(param_name, new_node_parameter)

    def sync_se_from_grn_edges(self) -> None:
        """
        Updates the values of the edge parameters contained in the structural equation based on the edge attributes of
        the GRN.

        If necessary, the relevant edge parameters are added to the structural equation.
        """
        self.structural_equation.edges = self.grn.tedges

        for param_name in self.grn.edge_attr_name_list:
            param_tensor = self.grn.get_edge_attr(param_name)

            if param_name in self.structural_equation.edge_parameter_dict:
                self.structural_equation.edge_parameter_dict[
                    param_name
                ].tensor = param_tensor
            else:
                new_edge_parameter = EdgeParameter(dim=param_tensor.shape[2:])
                new_edge_parameter.tensor = param_tensor

                self.structural_equation.set_parameter(param_name, new_edge_parameter)
