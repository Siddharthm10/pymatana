import torch
from geom_2d_functions import get_convex_hull_2d_batch
import torch.nn as nn


def get_arc_nodes_2d(
    convex_hull: torch.Tensor,
    crystal_center: torch.Tensor,
):
    """
    Get the arc nodes for the spiral sampling.
    The second node controls the curvature of the spiral.
    """

    # find the closest point in the convex hull (excluding the crystal center) to the center of the crystal

    distances = torch.norm(convex_hull - crystal_center, dim=1)
    indices = torch.argsort(distances)
    nodes = convex_hull[indices][1:3]

    rads = torch.atan2(
        nodes[:, 1] - crystal_center[1],
        nodes[:, 0] - crystal_center[0],
    )
    rads = rads + 2 * torch.pi * (rads < 0)
    distances = distances[indices][1:3]
    return torch.stack([distances, rads], dim=1)


def get_sampling_arc_coordinates_2d(nodes, n=100):
    """Get spiral sampling coordinates in 2D."""
    ends = nodes[nodes[:, 1].argsort()]
    angles = torch.linspace(ends[0, 1], ends[1, 1], n)
    radius = torch.linspace(ends[0, 0], ends[1, 0], n)

    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    return torch.stack((x, y), dim=-1), angles




def get_data_samples_on_arc_2d(
    data_batch: torch.Tensor,
    crystal_centers: torch.Tensor,
    crystal_id: int,
    hull_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    n_samples,
    fov_dict: dict,
):
    arc_nodes = get_arc_nodes_2d(
        hull_batch[crystal_id][mask_batch[crystal_id]],
        crystal_centers[crystal_id],
    )
    sampling_points, sampling_rads = get_sampling_arc_coordinates_2d(
        arc_nodes,
        n=n_samples,
    )
    sampling_points = sampling_points + crystal_centers[crystal_id]

    # Prepare the input and grid for sampling
    input_field = (
        data_batch[crystal_id]
        .T.view(
            1, 1, int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
        )
        .to(torch.float32)
    )
    # The grid contains the coordinates of the points where the input scalar
    # field is sampled. The grid is in the range [-1, 1] for both x and y.

    grid = (
        sampling_points.unsqueeze(0).unsqueeze(0).to(torch.float32)
        / torch.tensor(fov_dict["size in mm"])
        * 2
    )

    #
    samples = nn.functional.grid_sample(
        input_field, grid, mode="bilinear", align_corners=True
    ).squeeze()
    return samples, sampling_rads, sampling_points