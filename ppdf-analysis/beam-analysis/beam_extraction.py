from typing import Dict

from geometry_2d_utils import local_max_1d
from torch import Tensor, arange, argwhere, atan2, cat, cos
from torch import empty as empty_tensor
from torch import (
    float32,
    gradient,
    linspace,
    logical_and,
    meshgrid,
    norm,
    pi,
    sin,
    stack,
    tensor,
    zeros,
    zeros_like,
)
from torch.nn import functional as F


def get_arc_nodes_2d(
    convex_hull: Tensor,
    crystal_center: Tensor,
):
    """
    Get the arc nodes for the spiral sampling.
    The second node controls the curvature of the spiral.
    """

    # find the closest point in the convex hull (excluding the crystal center) to the center of the crystal

    distances = norm(convex_hull - crystal_center, dim=1)
    indices = distances.argsort()
    nodes = convex_hull[indices][1:3]

    rads = atan2(
        nodes[:, 1] - crystal_center[1],
        nodes[:, 0] - crystal_center[0],
    )
    rads = rads + 2 * pi * (rads < 0)
    distances = distances[indices][1:3]
    return stack([distances, rads], dim=1)


def get_sampling_arc_coordinates_2d(nodes, n=100):
    """Get spiral sampling coordinates in 2D."""
    ends = nodes[nodes[:, 1].argsort()]
    angles = linspace(ends[0, 1], ends[1, 1], n)
    radius = linspace(ends[0, 0], ends[1, 0], n)

    x = radius * cos(angles)
    y = radius * sin(angles)
    return stack((x, y), dim=-1), angles


def sample_ppdf_on_arc_2d_local(
    ppdf_data_2d: Tensor,
    detector_center: Tensor,
    hull_2d: Tensor,
    fov_dict: dict,
    n_samples: int = 1024,
):
    arc_nodes = get_arc_nodes_2d(
        hull_2d,
        detector_center,
    )
    # Get the sampling points and angles
    sampling_points, sampling_rads = get_sampling_arc_coordinates_2d(
        arc_nodes,
        n=n_samples,
    )
    sampling_points = sampling_points + detector_center

    # Prepare the input and grid for sampling

    # 2D PPDF field
    ppdf_field = ppdf_data_2d.T.view(
        1, 1, int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
    ).to(float32)

    # The grid contains the coordinates of the points where the input scalar
    # field is sampled. The grid is in the range [-1, 1] for both x and y.

    grid = (
        sampling_points.unsqueeze(0).unsqueeze(0).to(float32)
        / fov_dict["size in mm"]
        * 2
    )

    sampled_ppdf = F.grid_sample(
        ppdf_field, grid, mode="bilinear", align_corners=True
    ).squeeze()
    return sampled_ppdf, sampling_rads, sampling_points


def angular_edges_on_arc(
    sampled_data: Tensor,
    sampling_rads: Tensor,
    threshold: float = 0.01,
) -> Tensor:
    n_samples = sampled_data.shape[0]
    # Normalize the data
    normalized_data = sampled_data / sampled_data.max()
    # Threshold the data
    thresholded_data = zeros_like(normalized_data).masked_fill_(
        normalized_data > threshold, 1
    )
    n = 2
    thresholded_data_padded = cat(
        [
            zeros(n),
            thresholded_data,
            zeros(n),
        ]
    )
    # calculate the gradient
    gradient_amplitude_padded = gradient(thresholded_data_padded, dim=0)[0].abs()
    gradient_amplitude_padded_normalized = (
        gradient_amplitude_padded / gradient_amplitude_padded.max()
    )
    eps = 1e-8
    gradient_local_max = local_max_1d(gradient_amplitude_padded_normalized, size=9)
    edges_indices = argwhere(
        logical_and(
            gradient_local_max > 0.01,
            gradient_amplitude_padded_normalized >= gradient_local_max - eps,
        )
    ).squeeze()
    edges_indices = edges_indices - n
    edges_indices = edges_indices[edges_indices < n_samples]

    sampling_rad_step = sampling_rads[1] - sampling_rads[0]
    sampling_rads_padded = cat(
        [
            -arange(1, n + 1).flip(0) * sampling_rad_step + sampling_rads[0],
            sampling_rads,
            arange(1, n + 1) * sampling_rad_step + sampling_rads[-1],
        ]
    )
    return sampling_rads_padded[edges_indices].sort().values


def pixels_coordinates(
    fov_dict: Dict,
) -> Tensor:

    pixel_indices = stack(
        meshgrid(
            arange(0, int(fov_dict["n pixels"][0])),
            arange(0, int(fov_dict["n pixels"][1])),
            indexing="ij",
        ),
        dim=2,
    ).view(-1, 2)
    return (
        pixel_indices.to(dtype=float32) * fov_dict["mm per pixel"]
        - fov_dict["size in mm"] * 0.5
        + fov_dict["center coordinates in mm"]
    )


def pixels_to_detector_unit_rads(
    pixel_coordinates: Tensor,
    detector_unit_center: Tensor,
) -> Tensor:

    pixel_rads = atan2(
        pixel_coordinates[:, 1] - detector_unit_center[1],
        pixel_coordinates[:, 0] - detector_unit_center[0],
    )
    pixel_rads = pixel_rads + 2 * pi * (pixel_rads < 0)
    return pixel_rads


def extract_beams_properties_2d(
    normalized_ppdf: Tensor,
    pixels_coordinates: Tensor,
    pixels_rads: Tensor,
    rads_edges: Tensor,
    fov_dict: Dict,
) -> Tensor:

    n_fov_points = pixels_coordinates.shape[0]
    n_rad_intervals = rads_edges.shape[0] - 1
    pixels_rads_expanded = pixels_rads.unsqueeze(1).expand(-1, n_rad_intervals)

    rads_edges_expanded = rads_edges.unsqueeze(0).expand(n_fov_points, -1)

    rad_interval_masks = (pixels_rads_expanded >= rads_edges_expanded[:, :-1]) & (
        pixels_rads_expanded < rads_edges_expanded[:, 1:]
    )  # (n_fov_points, n_rad_intervals)

    normalized_ppdf_expanded = (
        normalized_ppdf.view(-1).unsqueeze(1).expand(-1, n_rad_intervals)
    )
    sum_template = normalized_ppdf_expanded.clone().masked_fill_(~rad_interval_masks, 0)
    normalized_ppdf_mean = sum_template.sum(dim=0) / rad_interval_masks.sum(dim=0)

    # Discard the range with mean < 0.01
    beam_masks = rad_interval_masks[:, normalized_ppdf_mean > 0.01]
    beam_n_pixels = beam_masks.sum(dim=0)
    # Get the weighted center of the beams



    # weighted_center = (
    #     normalized_ppdf.view(-1)[in_range_mask].view(-1, 1).expand(-1, 2)
    #     * pixel_coordinates[in_range_mask]
    # ).sum(dim=0) / (normalized_ppdf.view(-1)[in_range_mask].sum() + 1e-8)

    # beams_weighted_centers = torch.cat(
    #     [
    #         beams_weighted_centers,
    #         weighted_center.unsqueeze(0),
    #     ],
    #     dim=0,
    # )
    # beams_masks = torch.cat(
    #     [
    #         beams_masks,
    #         in_range_mask.unsqueeze(0),
    #     ],
    #     dim=0,
    # )
    # beams_rads_edges = torch.cat(
    #     [
    #         beams_rads_edges,
    #         rads_edges[i : i + 2].unsqueeze(0),
    #     ],
    #     dim=0,
    # )
