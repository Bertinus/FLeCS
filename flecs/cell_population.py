import torch

from flecs.grn import GRN
from flecs.attribute import EdgeAttribute, GeneAttribute
from flecs.structural_equation import StructuralEquation


class CellPopulation:
    """
    Class representing a population of cells.

    Example:
        A simple example of initialization:
        ```
        from flecs.cell import Cell
        from flecs.grn import RandomGRN
        from flecs.structural_equation import SigmoidLinearSE

        grn = RandomGRN(n_genes=10, av_num_parents=3)
        linear_se = SigmoidLinearSE()

        my_cell = Cell(grn=grn, structural_equation=linear_se)
        ```

    Attributes:
        grn (GRN): Gene Regulatory Network of the cell.
        structural_equation (StructuralEquation): Structural Equation of the cell.
    """

    def __init__(self, grn: GRN, structural_equation: StructuralEquation):
        """
        Args:
            grn (GRN): Gene Regulatory Network of the cell
            structural_equation (StructuralEquation): Structural Equation of the cell
        """
        self._grn = grn
        self.structural_equation = structural_equation
        self.structural_equation.initialize_given_structure(
            self._grn.n_genes, self._grn.tedges
        )
        self._state = torch.zeros((0, self._grn.n_genes, 1))

        self.sync_grn_from_se()

    @property
    def state(self):
        """(``torch.Tensor``): State of the cell. Shape (n_cells, n_genes, *state_dim)"""
        return self._state

    @property
    def n_cells(self):
        """(``int``) Number of cells."""
        return self._state.shape[0]

    @property
    def n_genes(self):
        """(``int``) Number of genes."""
        return self._state.shape[1]

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
        self._grn.state = state
        self._state = state

    def get_derivatives(self, state):
        """
        Returns the time derivative of the state, as computed by ``self.structural_equation``.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Time derivative of the state. Shape (n_cells, n_genes, *state_dim).

        """
        return self.structural_equation.get_derivatives(state)

    def get_production_rates(self, state):
        """
        Returns the production rates, as computed by ``self.structural_equation``.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Production rates. Shape (n_cells, n_genes, *state_dim)

        """
        return self.structural_equation.get_production_rates(state)

    def get_decay_rates(self, state):
        """
        Returns the decay rates, as computed by ``self.structural_equation``.

        Args:
            state (torch.Tensor): State of the cell. Shape (n_cells, n_genes, *state_dim)

        Returns:
            torch.Tensor: Decay rates. Shape (n_cells, n_genes, *state_dim)

        """
        return self.structural_equation.get_decay_rates(state)

    def get_gene_attribute(self, attr_name) -> GeneAttribute:
        """
        Gets the gene attribute ``attribute_name``.

        Args:
            attr_name (str): Name of the attribute.

        Returns:
            GeneAttribute: Attribute.

        """
        return self.structural_equation.gene_attribute_dict[attr_name]

    def get_edge_attribute(self, attr_name):
        """
        Gets the edge attribute ``attribute_name``.

        Args:
            attr_name (str): Name of the attribute.

        Returns:
            EdgeAttribute: Attribute.

        """
        return self.structural_equation.edge_attribute_dict[attr_name]

    def get_attribute(self, attr_name):
        """
        Gets the attribute ``attribute_name``.

        Args:
            attr_name (str): Name of the attribute.

        Returns:
            Attribute: Attribute.

        """
        return self.structural_equation.attribute_dict[attr_name]

    def sync_grn_from_se(self) -> None:
        """
        Updates the values of the gene and edge attributes of the GRN based on the values of the gene and edge
        attributes contained in the structural equation.

        If necessary, the relevant gene and edge attributes are added to the GRN.
        """
        for attr_name in self.structural_equation.gene_attribute_dict:
            attr_tensor = self.structural_equation.gene_attribute_dict[
                attr_name
            ].tensor
            self._grn.set_gene_attr(attr_name, attr_tensor)

        for attr_name in self.structural_equation.edge_attribute_dict:
            attr_tensor = self.structural_equation.edge_attribute_dict[
                attr_name
            ].tensor
            self._grn.set_edge_attr(attr_name, attr_tensor)

    def sync_se_from_grn(self) -> None:
        """
        Updates the values of the gene and edge attributes contained in the structural equation based on the gene and
        edge attributes of the GRN.

        If necessary, the relevant gene and edge attributes are added to the structural equation.
        """
        self.sync_se_from_grn_genes()
        self.sync_se_from_grn_edges()

    def sync_se_from_grn_genes(self) -> None:
        """
        Updates the values of the gene attributes contained in the structural equation based on the gene attributes of
        the GRN.

        If necessary, the relevant gene attributes are added to the structural equation.
        """
        for attr_name in self._grn.gene_attr_name_list:
            attr_tensor = self._grn.get_gene_attr(attr_name)

            if attr_name in self.structural_equation.gene_attribute_dict:
                self.structural_equation.gene_attribute_dict[
                    attr_name
                ].tensor = attr_tensor
            else:
                new_gene_attribute = GeneAttribute(dim=attr_tensor.shape[2:])
                new_gene_attribute.tensor = attr_tensor

                self.structural_equation.set_attribute(attr_name, new_gene_attribute)

    def sync_se_from_grn_edges(self) -> None:
        """
        Updates the values of the edge attributes contained in the structural equation based on the edge attributes of
        the GRN.

        If necessary, the relevant edge attributes are added to the structural equation.
        """
        self.structural_equation.edges = self._grn.tedges

        for attr_name in self._grn.edge_attr_name_list:
            attr_tensor = self._grn.get_edge_attr(attr_name)

            if attr_name in self.structural_equation.edge_attribute_dict:
                self.structural_equation.edge_attribute_dict[
                    attr_name
                ].tensor = attr_tensor
            else:
                new_edge_attribute = EdgeAttribute(dim=attr_tensor.shape[2:])
                new_edge_attribute.tensor = attr_tensor

                self.structural_equation.set_attribute(attr_name, new_edge_attribute)
