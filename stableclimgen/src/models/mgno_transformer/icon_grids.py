import torch
import torch.nn as nn

from ..modules import transformer_modules as helpers

from ...utils.grid_utils_icon import get_distance_angle,sequenize
from .pos_embedding import PositionEmbedder


def get_nh_indices(adjc_global: torch.Tensor, global_level: int, global_cell_indices: torch.Tensor = None, local_cell_indices: torch.Tensor = None) -> tuple:
    """
    Calculate the neighborhood indices and corresponding mask for given cell indices.

    :param adjc_global: The global adjacency tensor.
    :param global_level: The global hierarchical level.
    :param global_cell_indices: Indices of cells at the global level. Defaults to None.
    :param local_cell_indices: Indices of cells at the local level. Defaults to None.
    :return: A tuple containing neighborhood indices and a mask.
    """
    if global_cell_indices is not None:
        # Derive local cell indices from global cell indices and level
        local_cell_indices = global_cell_indices // 4**global_level

    # Get neighborhood indices and mask using helper function
    local_cell_indices_nh, mask = helpers.get_nh_of_batch_indices(local_cell_indices, adjc_global)

    return local_cell_indices_nh, mask

    
def gather_nh_data(x: torch.Tensor, local_cell_indices_nh: torch.Tensor, batch_sample_indices: torch.Tensor, sampled_level: int, global_level: int) -> torch.Tensor:
    """
    Gather neighborhood data from the input tensor based on neighborhood indices.

    :param x: Input data tensor with dimensions [batch, cells, vertices, features].
    :param local_cell_indices_nh: Neighborhood indices for local cells.
    :param batch_sample_indices: Indices for batch sampling.
    :param sampled_level: The level at which data is sampled.
    :param global_level: The global hierarchical level.
    :return: A tensor with gathered neighborhood data.
    """
    # Ensure input tensor x has at least 4 dimensions
    if x.dim() < 4:
        x = x.unsqueeze(dim=-1)

    # Extract shape parameters
    b, n, nv, e = x.shape
    nh = local_cell_indices_nh.shape[-1]

    # Compute batch indices adjusted for hierarchical levels
    local_cell_indices_nh_batch = local_cell_indices_nh - (batch_sample_indices * 4**(sampled_level - global_level)).view(-1, 1, 1)

    # Gather data based on neighborhood indices and reshape
    return torch.gather(
        x.view(b, -1, nv, e), 1, 
        index=local_cell_indices_nh_batch.view(b, -1, 1, 1).repeat(1, 1, nv, e)
    ).view(b, n, nh, nv, e)


def get_relative_positions(coords1, coords2, polar=False, periodic_fov=None):
    
    if coords2.dim() > coords1.dim():
        coords1 = coords1.unsqueeze(dim=-1)

    if coords1.dim() > coords2.dim():
        coords2 = coords2.unsqueeze(dim=-2)

    if coords1.dim() == coords2.dim():
        coords1 = coords1.unsqueeze(dim=-1)
        coords2 = coords2.unsqueeze(dim=-2)

    distances, phis = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1], base="polar" if polar else "cartesian", periodic_fov=periodic_fov)

    return distances.float(), phis.float()


class GridLayer(nn.Module):
    """
    A neural network module representing a grid layer with specific functionalities
    like coordinate transformations, neighborhood gathering, and relative positioning.

    Attributes:
        global_level (int): The global hierarchical level.
        coord_system (str): The coordinate system, e.g., "polar".
        periodic_fov (Any): The periodic field of view.
        coordinates (torch.Tensor): The coordinates of the grid.
        adjc (torch.Tensor): Adjacency tensor for the grid.
        adjc_mask (torch.Tensor): Mask indicating adjacency relations.
        fov_mask (torch.Tensor): Mask for field of view.
        min_dist (torch.Tensor): Minimum distance in the relative coordinates.
        max_dist (torch.Tensor): Maximum distance in the relative coordinates.
        mean_dist (torch.Tensor): Mean distance in the relative coordinates.
        median_dist (torch.Tensor): Median distance in the relative coordinates.
    """
    
    def __init__(self, global_level: int, adjc: torch.Tensor, adjc_mask: torch.Tensor, coordinates: torch.Tensor, coord_system: str = "polar", periodic_fov=None) -> None:
        """
        Initializes the GridLayer with given parameters.

        :param global_level: The global level for hierarchy.
        :param adjc: The adjacency tensor.
        :param adjc_mask: Mask for the adjacency tensor.
        :param coordinates: The coordinates of grid points.
        :param coord_system: The coordinate system. Defaults to "polar".
        :param periodic_fov: The periodic field of view. Defaults to None.
        """
        super().__init__()

        # Initialize attributes
        self.global_level = global_level
        self.coord_system = coord_system
        self.periodic_fov = periodic_fov

        # Register buffers for coordinates and adjacency information
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("adjc", adjc, persistent=False)
        # Create mask where adjacency is false
        self.register_buffer("adjc_mask", adjc_mask == False, persistent=False)
        # Mask for the field of view
        self.register_buffer("fov_mask", ((adjc_mask == False).sum(dim=-1) == adjc_mask.shape[1]).view(-1, 1), persistent=False)

        # Sample distances for statistical analysis
        n_samples = torch.min(torch.tensor([self.adjc.shape[0] - 1, 100]))
        nh_samples = self.adjc[:n_samples]
        coords_nh = self.get_coordinates_from_grid_indices(nh_samples)
        # Calculate relative distances
        dists = get_relative_positions(coords_nh, coords_nh, polar=True)[0]

        # Compute distance statistics
        self.min_dist = dists[dists > 1e-10].min()
        self.max_dist = dists[dists > 1e-10].max()
        self.mean_dist = dists[dists > 1e-10].mean()
        self.median_dist = dists[dists > 1e-10].median()

    def get_nh(self, x: torch.Tensor, local_indices: torch.Tensor, sample_dict: dict, relative_coordinates: bool = True, coord_system: str = None, mask: torch.Tensor = None) -> tuple:
        """
        Get the neighborhood data, mask, and coordinates.

        :param x: Input data tensor.
        :param local_indices: Indices for local neighborhoods.
        :param sample_dict: Dictionary with sample information.
        :param relative_coordinates: Whether to use relative coordinates. Defaults to True.
        :param coord_system: Coordinate system to use. Defaults to None.
        :param mask: Optional mask tensor. Defaults to None.
        :return: A tuple containing neighborhood data, mask, and coordinates.
        """
        # Get neighborhood indices and adjacency mask
        indices_nh, adjc_mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))
        
        # Gather neighborhood data
        x = gather_nh_data(x, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))

        if mask is not None:
            # Combine provided mask with adjacency mask
            mask = gather_nh_data(mask, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))
            mask = torch.logical_or(mask.squeeze(dim=-1), adjc_mask.unsqueeze(dim=-1))
        else:
            # Use adjacency mask if no mask is provided
            mask = adjc_mask.unsqueeze(dim=-1).repeat_interleave(x.shape[-2], dim=-1)

        # Get coordinates (relative or absolute)
        if relative_coordinates:
            coords = self.get_relative_coordinates_from_grid_indices(indices_nh, coord_system=coord_system)
        else:
            coords = self.get_coordinates_from_grid_indices(indices_nh)

        return x, mask, coords

    def get_coordinates_from_grid_indices(self, local_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve coordinates based on grid indices.

        :param local_indices: Indices for grid points.
        :return: Coordinates corresponding to the indices.
        """
        coords = self.coordinates[:, local_indices]
        return coords

    def get_relative_coordinates_from_grid_indices(self, local_indices: torch.Tensor, coords: torch.Tensor = None, coord_system: str = None) -> torch.Tensor:
        """
        Get relative coordinates from grid indices.

        :param local_indices: Indices for local points.
        :param coords: Precomputed coordinates. Defaults to None.
        :param coord_system: Coordinate system to use. Defaults to None.
        :return: Relative coordinates.
        """
        if coord_system is None:
            coord_system = self.coord_system

        if coords is None:
            coords = self.get_coordinates_from_grid_indices(local_indices)

        # Compute relative distances and angles
        coords_rel = get_distance_angle(
            coords[0, :, :, [0]], coords[1, :, :, [0]], coords[0], coords[1],
            base=coord_system, periodic_fov=self.periodic_fov
        )

        return coords_rel

    def get_relative_coordinates_cross(self, local_indices: torch.Tensor, coords: torch.Tensor, coord_system: str = None) -> torch.Tensor:
        """
        Get cross relative coordinates between reference and provided points.

        :param local_indices: Reference indices.
        :param coords: Provided coordinates.
        :param coord_system: Coordinate system to use. Defaults to None.
        :return: Cross relative coordinates.
        """
        if coord_system is None:
            coord_system = self.coord_system

        # Get reference coordinates
        coords_ref = self.get_coordinates_from_grid_indices(local_indices)

        if coords_ref.dim() < 4:
            coords_ref = coords_ref.unsqueeze(dim=-1)

        # Compute cross relative distances and angles
        coords_rel = get_distance_angle(
            coords_ref[0, :, :, [0]], coords_ref[1, :, :, [0]], coords[0], coords[1],
            base=coord_system, periodic_fov=self.periodic_fov
        )

        return coords_rel

    def get_sections(self, x: torch.Tensor, local_indices: torch.Tensor, section_level: int = 1, relative_coordinates: bool = True, return_indices: bool = True, coord_system: str = None) -> tuple:
        """
        Divide the input into sections based on hierarchical levels.

        :param x: Input tensor.
        :param local_indices: Local indices for sections.
        :param section_level: Level of the section. Defaults to 1.
        :param relative_coordinates: Use relative coordinates. Defaults to True.
        :param return_indices: Return indices or not. Defaults to True.
        :param coord_system: Coordinate system to use. Defaults to None.
        :return: A tuple containing sectioned data, mask, coordinates, and indices (optional).
        """
        # Sequenize indices and data
        indices = sequenize(local_indices, max_seq_level=section_level)
        x = sequenize(x, max_seq_level=section_level)

        # Get coordinates
        if relative_coordinates:
            coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
        else:
            coords = self.get_coordinates_from_grid_indices(indices)

        # Field of view mask
        mask = self.fov_mask[indices]

        if return_indices:
            return x, mask, coords, indices
        else:
            return x, mask, coords

    def get_position_embedding(self, indices: torch.Tensor, nh: bool, pos_embedder, coord_system: str, batch_dict: dict = None, section_level: int = None) -> torch.Tensor:
        """
        Compute position embeddings for the grid points.

        :param indices: Indices for which to compute embeddings.
        :param nh: Whether to use neighborhood information.
        :param pos_embedder: The position embedder object or function.
        :param coord_system: The coordinate system.
        :param batch_dict: Batch information. Defaults to None.
        :param section_level: Level of the section. Defaults to None.
        :return: Position embeddings.
        """
        if nh:
            if isinstance(pos_embedder, PositionEmbedder):
                indices = get_nh_indices(self.adjc, local_cell_indices=indices, global_level=int(self.global_level))[0]
                rel_coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
                pos_embeddings = pos_embedder(rel_coords[0], rel_coords[1])
            else:
                pos_embeddings = pos_embedder(self, indices, batch_dict)
        else:
            if isinstance(pos_embedder, PositionEmbedder):
                indices = sequenize(indices, max_seq_level=section_level)
                coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
                pos_embeddings = pos_embedder(coords[0], coords[1])
            else:
                pos_embeddings = pos_embedder(self, indices)

        return pos_embeddings