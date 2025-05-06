from typing import Dict

from torch import Tensor, arange, argwhere, atan2, cat, cos, diff
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
    where,
    zeros,
    zeros_like,
)
from torch.nn import functional as F

from geometry_2d_utils import local_max_1d


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
    if ends[1, 1] - ends[0, 1] > pi:
        ends[1, 1] = ends[1, 1] - 2 * pi
        angles = linspace(ends[1, 1], ends[0, 1], n)
        radius = linspace(ends[1, 0], ends[0, 0], n)
    else:
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

    relative_sampled_ppdf = sampled_data / sampled_data.max()
    threshold = 0.01
    thresholded_relative_sampled_ppdf = zeros_like(
        relative_sampled_ppdf
    ).masked_fill_(relative_sampled_ppdf > threshold, 1)

    forward_diff_abs = diff(
        thresholded_relative_sampled_ppdf, prepend=tensor([0.0])
    ).abs()
    radian_edges = sampling_rads[forward_diff_abs > 0.5]
    return radian_edges


def beams_weighted_center(
    beam_masks: Tensor,
    fov_pts: Tensor,
    ppdf_data: Tensor,
    beams_relative_sensitivities: Tensor,
) -> Tensor:
    """
    Compute the weighted center of the beams based on the pixel data and beam masks.

    Parameters
    ----------
    beam_masks: Tensor
      - The masks for the beams.
      - `shape`: (n_pixels, n_beams)
      - `dtype`: `bool`


    fov_pts: Tensor
      The field of view points.

    ppdf_data: Tensor
      The pixel data probability distribution function.

    beams_relative_sensitivities: Tensor
      The means of the beams.
    """
    n_beams = beam_masks.shape[1]
    fov_pts_expanded = fov_pts.unsqueeze(1).expand(-1, n_beams, -1)
    ppdf_data_expanded = (
        ppdf_data.view(-1).unsqueeze(1).unsqueeze(1).expand(-1, n_beams, 2)
    )
    weighted_centers = (
        (fov_pts_expanded * ppdf_data_expanded)
        .masked_fill_(~beam_masks.unsqueeze(2).expand(-1, -1, 2), 0)
        .sum(dim=0)
    )
    beam_n_pixels = beam_masks.sum(dim=0)
    weighted_centers = weighted_centers / (
        beam_n_pixels * beams_relative_sensitivities
    ).unsqueeze(1)
    return weighted_centers


def full_width_half_maximum_1d_batch(
    data_x_batch: Tensor,
    data_y_batch: Tensor,
):
    """
    Calculate the full width at half maximum (FWHM) for each curve in the batch.
    The FWHM is defined as the distance between the two points where the curve
    crosses half of the maximum value.

    The function returns the x-coordinates of the two points and the FWHM value.

    Parameters
    ----------

    data_x_batch: Tensor
      The x-coordinates of the data points.
      `shape`: (n_batch, n_points)
      `dtype`: `float32`

    data_y_batch: Tensor
      The y-coordinates of the data points.
      `shape`: (n_batch, n_points)
      `dtype`: `float32`

    Returns
    ----------

    fwhm_batch: Tensor
      The FWHM values for each curve in the batch.
      `shape`: (n_batch,)
      `dtype`: `float32`

    x_bounds_batch: Tensor
        The x-coordinates of the two points where the curve crosses half of the maximum.
        `shape`: (n_batch, 2)
        `dtype`: `float32`
    """

    n_batch = data_x_batch.shape[0]
    n_points = data_x_batch.shape[1]

    # Find the indices of the points where the curve crosses half of the maximum
    half_max = (
        (data_y_batch.max(dim=1).values * 0.5).unsqueeze(1).expand(-1, n_points)
    )
    x_above_half_max_batch = (
        where(data_y_batch >= half_max, data_x_batch, 0).sort(dim=1).values
    )
    x_upper_lower_batch = x_above_half_max_batch[:, [0, -1]]
    fwhm_batch = x_upper_lower_batch[:, 1] - x_upper_lower_batch[:, 0]
    return fwhm_batch, x_upper_lower_batch


def beam_sampling_line_batch(
    detector_unit_center: Tensor,
    beam_center_batch: Tensor,
    n_samples: int = 1024,
    length: float = 32.0,
):
    """
    Get the sampling line for the beams in batch.
    The sampling line is perpendicular to the beam axis and passes through the
    beam center.

    Parameters
    ----------

    detector_unit_center: Tensor
      The center of the detector unit.
      `shape`: (2,)
      `dtype`: `float32`

    beam_center_batch: Tensor
      The weighted centers of the beams.
      `shape`: (n_beams, 2)
      `dtype`: `float32`

    n_samples: int
      The number of samples to take along the sampling line.

    length: float
      The length of the sampling line.

    Returns
    -------
    sampling_points_batch: Tensor
      The sampling points along the sampling line.
      `shape`: (n_beams, n_samples, 2)
      `dtype`: `float32`

    sampling_distance: Tensor
      The distances along the sampling line.
      `shape`: (n_samples,)
      `dtype`: `float32`
    """
    n_beams = beam_center_batch.shape[0]
    beam_axis_rad = atan2(
        beam_center_batch[:, 1] - detector_unit_center[1],
        beam_center_batch[:, 0] - detector_unit_center[0],
    )
    beam_axis_rad = beam_axis_rad + 2 * pi * (beam_axis_rad < 0)
    sampling_line_rad = beam_axis_rad + pi * 0.5
    kx = cos(sampling_line_rad)
    ky = sin(sampling_line_rad)
    sampling_distance = linspace(-length / 2, length / 2, n_samples)
    sampling_points_batch = stack((kx, ky), dim=1).unsqueeze(1).expand(
        -1, n_samples, -1
    ) * sampling_distance.view(1, n_samples, 1).expand(
        n_beams, -1, 2
    ) + beam_center_batch.unsqueeze(
        1
    ).expand(
        -1, n_samples, -1
    )
    return sampling_points_batch, sampling_distance


def beam_samples_on_points_batch(
    beam_data_2d_batch: Tensor,
    beam_sampling_points_batch: Tensor,
    fov_dict: dict,
):
    """
    Sample the beams data on the sampling points in batch.

    Parameters
    ----------
    beam_data_2d_batch: Tensor
      The PPDFs data in 2D.
      `shape`: (n_beams, n_pixels_x, n_pixels_y)
      `dtype`: `float32`

    beam_sampling_points_batch: Tensor
        The sampling points for the beams.
        `shape`: (n_beams, n_samples, 2)
        `dtype`: `float32`

    fov_dict: Dict, (optional)
        The field of view dictionary.
        `shape`: (n_pixels_x, n_pixels_y)
        `dtype`: `float32`

    n_samples: int
        The number of samples to take along the sampling line.

    Returns
    -------
    sampled_beams_batch: Tensor
      The sampled beams data.
      `shape`: (n_beams, n_samples)
      `dtype`: `float32`
    """
    # Prepare the input and grid for sampling
    # 2D PPDF field

    beam_field = beam_data_2d_batch.view(
        beam_data_2d_batch.shape[0],
        1,
        beam_data_2d_batch.shape[1],
        beam_data_2d_batch.shape[2],
    ).swapaxes(
        -1, -2
    )  # (n_beams, 1, n_pixels_x, n_pixels_y)

    # The grid contains the coordinates of the points where the input scalar
    # field is sampled. The grid is in the range [-1, 1] for both x and y.

    grid = (
        beam_sampling_points_batch.unsqueeze(1) / fov_dict["size in mm"] * 2
    )  # (n_beams, 1, n_samples, 2)

    sampled_beams_batch = F.grid_sample(
        beam_field, grid, mode="bilinear", align_corners=True
    ).squeeze()
    return sampled_beams_batch


def beams_basic_properties(
    ppdf_2d: Tensor,
    pixels_coordinates: Tensor,
    pixels_rads: Tensor,
    rads_edges: Tensor,
    detector_unit_center: Tensor,
    threshold: float = 0.01,
) -> Dict[str, Tensor]:

    # Basic properties of the beams
    relative_ppdf = ppdf_2d / ppdf_2d.max()
    n_fov_points = pixels_coordinates.shape[0]
    n_rad_intervals = rads_edges.shape[0] - 1
    pixels_rads_expanded = pixels_rads.unsqueeze(1).expand(-1, n_rad_intervals)

    rad_interval_edges = stack(
        [
            rads_edges[:-1],
            rads_edges[1:],
        ],
        dim=1,
    )  # (n_rad_intervals, 2)
    rads_edges_expanded = rads_edges.unsqueeze(0).expand(n_fov_points, -1)

    # Get the radian intervals masks
    rad_interval_masks = (
        pixels_rads_expanded >= rads_edges_expanded[:, :-1]
    ) & (
        pixels_rads_expanded < rads_edges_expanded[:, 1:]
    )  # (n_fov_points, n_rad_intervals)

    relative_ppdf_expanded = (
        relative_ppdf.view(-1).unsqueeze(1).expand(-1, n_rad_intervals)
    )
    sum_template = relative_ppdf_expanded.clone().masked_fill_(
        ~rad_interval_masks, 0
    )
    intervals_relative_sensitivities = sum_template.sum(
        dim=0
    ) / rad_interval_masks.sum(dim=0)

    # Discard the range with mean < 0.01
    beams_masks = rad_interval_masks[
        :, intervals_relative_sensitivities > threshold
    ]
    beams_relative_sensitivities = intervals_relative_sensitivities[
        intervals_relative_sensitivities > threshold
    ]
    beam_n_pixels = beams_masks.sum(dim=0)

    # Get the weighted center of the beams
    beams_weighted_centers = beams_weighted_center(
        beams_masks,
        pixels_coordinates,
        relative_ppdf,
        beams_relative_sensitivities,
    )
    beams_sensitivities = beams_relative_sensitivities * ppdf_2d.max()

    # Calculate the beam regions radian edges
    beam_rads_edges = rad_interval_edges[
        intervals_relative_sensitivities > threshold
    ]
    # Swap the axes of the beam masks
    beams_masks = beams_masks.swapaxes(0, 1)

    # Number of beams
    n_beams = beams_masks.shape[0]

    # Calculate the beam axis angle in radian
    beam_axis_rad = atan2(
        beams_weighted_centers[:, 1] - detector_unit_center[1],
        beams_weighted_centers[:, 0] - detector_unit_center[0],
    )
    beam_axis_rad = beam_axis_rad + 2 * pi * (beam_axis_rad < 0)
    return {
        "masks": beams_masks,
        "weighted centers": beams_weighted_centers,
        "rads edges": beam_rads_edges,
        "axial angle": beam_axis_rad,
        "sensitivities": beams_sensitivities,
        "relative sensitivities": beams_relative_sensitivities,
        "number of pixels": beam_n_pixels,
        "number of beams": tensor([n_beams]),
    }


def beams_line_properties(
    beams_basic_properties: Dict[str, Tensor],
    detector_unit_center: Tensor,
    ppdf_2d: Tensor,
    fov_dict: Dict,
    line_n_samples: int = 4096,
):
    # Get the beam sampling lines
    (beam_sp_ptx_batch, beam_sp_distance) = beam_sampling_line_batch(
        detector_unit_center,
        beams_basic_properties["weighted centers"],
        n_samples=line_n_samples,
        length=64.0,
    )

    n_beams = int(beams_basic_properties["number of beams"].item())
    beams_masks = beams_basic_properties["masks"]

    beams_data_2d_batch = (
        ppdf_2d.view(-1)
        .unsqueeze(0)
        .expand(n_beams, -1)
        .clone()
        .masked_fill_(~beams_masks, 0)
    ).view(
        n_beams,
        int(fov_dict["n pixels"][0]),
        int(fov_dict["n pixels"][1]),
    )

    sampled_beams_batch = beam_samples_on_points_batch(
        beams_data_2d_batch, beam_sp_ptx_batch, fov_dict
    )
    fwhm_batch, x_bounds_batch = full_width_half_maximum_1d_batch(
        beam_sp_distance.view(1, -1).expand(-1, line_n_samples),
        sampled_beams_batch,
    )
    return {
        "masks": beams_masks,
        "fwhm": fwhm_batch,
        "fwhm distance bounds": x_bounds_batch,
        "number of beams": tensor([n_beams]),
        "sampling points": beam_sp_ptx_batch,
        "sampling distance": beam_sp_distance,
        "sampled data": sampled_beams_batch,
    }


def beams_properties_2d(
    ppdf_2d: Tensor,
    pixels_coordinates: Tensor,
    pixels_rads: Tensor,
    rads_edges: Tensor,
    detector_unit_center: Tensor,
    fov_dict: dict,
    threshold: float = 0.01,
    line_n_samples: int = 4096,
) -> Dict[str, Tensor]:

    # Basic properties of the beams
    relative_ppdf = ppdf_2d / ppdf_2d.max()
    n_fov_points = pixels_coordinates.shape[0]
    n_rad_intervals = rads_edges.shape[0] - 1
    pixels_rads_expanded = pixels_rads.unsqueeze(1).expand(-1, n_rad_intervals)

    rad_interval_edges = stack(
        [
            rads_edges[:-1],
            rads_edges[1:],
        ],
        dim=1,
    )  # (n_rad_intervals, 2)
    rads_edges_expanded = rads_edges.unsqueeze(0).expand(n_fov_points, -1)

    # Get the radian intervals masks
    rad_interval_masks = (
        pixels_rads_expanded >= rads_edges_expanded[:, :-1]
    ) & (
        pixels_rads_expanded < rads_edges_expanded[:, 1:]
    )  # (n_fov_points, n_rad_intervals)

    relative_ppdf_expanded = (
        relative_ppdf.view(-1).unsqueeze(1).expand(-1, n_rad_intervals)
    )
    sum_template = relative_ppdf_expanded.clone().masked_fill_(
        ~rad_interval_masks, 0
    )
    intervals_relative_sensitivities = sum_template.sum(
        dim=0
    ) / rad_interval_masks.sum(dim=0)

    # Discard the range with mean < 0.01
    beams_masks = rad_interval_masks[
        :, intervals_relative_sensitivities > threshold
    ]
    beams_relative_sensitivities = intervals_relative_sensitivities[
        intervals_relative_sensitivities > threshold
    ]
    beam_n_pixels = beams_masks.sum(dim=0)

    # Get the weighted center of the beams
    beams_weighted_centers = beams_weighted_center(
        beams_masks,
        pixels_coordinates,
        relative_ppdf,
        beams_relative_sensitivities,
    )
    beams_sensitivities = beams_relative_sensitivities * ppdf_2d.max()

    # Calculate the beam regions radian edges
    beam_rads_edges = rad_interval_edges[
        intervals_relative_sensitivities > threshold
    ]

    # Get the beam sampling lines
    (beam_sp_ptx_batch, beam_sp_distance) = beam_sampling_line_batch(
        detector_unit_center,
        beams_weighted_centers,
        n_samples=line_n_samples,
        length=64.0,
    )

    # Swap the axes of the beam masks
    beams_masks = beams_masks.swapaxes(0, 1)
    # Number of beams
    n_beams = beams_masks.shape[0]

    beams_data_2d_batch = (
        ppdf_2d.view(-1)
        .unsqueeze(0)
        .expand(n_beams, -1)
        .clone()
        .masked_fill_(~beams_masks, 0)
    ).view(
        n_beams,
        int(fov_dict["n pixels"][0]),
        int(fov_dict["n pixels"][1]),
    )

    sampled_beams_batch = beam_samples_on_points_batch(
        beams_data_2d_batch, beam_sp_ptx_batch, fov_dict
    )
    fwhm_batch, x_bounds_batch = full_width_half_maximum_1d_batch(
        beam_sp_distance.view(1, -1).expand(-1, line_n_samples),
        sampled_beams_batch,
    )
    # Calculate the beam axis angle in radian
    beam_axis_rad = atan2(
        beams_weighted_centers[:, 1] - detector_unit_center[1],
        beams_weighted_centers[:, 0] - detector_unit_center[0],
    )
    beam_axis_rad = beam_axis_rad + 2 * pi * (beam_axis_rad < 0)
    return {
        "masks": beams_masks,
        "weighted centers": beams_weighted_centers,
        "rads edges": beam_rads_edges,
        "axial angle": beam_axis_rad,
        "sensitivities": beams_sensitivities,
        "relative sensitivities": beams_relative_sensitivities,
        "n pixels": beam_n_pixels,
        "fwhm": fwhm_batch,
        "fwhm distance bounds": x_bounds_batch,
        "number of beams": tensor([n_beams]),
        "sampling points": beam_sp_ptx_batch,
        "sampling distance": beam_sp_distance,
        "sampled data": sampled_beams_batch,
    }
