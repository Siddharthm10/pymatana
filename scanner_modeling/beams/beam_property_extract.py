from typing import Dict, Tuple
from torch import ones, bool as torch_bool

from torch import Tensor, arange, argwhere, atan2, cat, cos, diff
from torch import empty as empty_tensor, uint16 as uint16_tensor
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

from .._geometry_2d._utils import local_max_1d, fov_pixels_to_crystal_rads


def get_arc_nodes_2d(
    convex_hull: Tensor,
    crystal_center: Tensor,
):
    """ """

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
    """
    Get spiral sampling coordinates in 2D.
    """
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


def get_beams_angle_radian(
    beams_centers: Tensor,
    reference_point: Tensor,
) -> Tensor:
    n_beams = beams_centers.shape[0]
    reference_point_expanded = reference_point.unsqueeze(0).expand(n_beams, -1)
    angles = atan2(
        reference_point_expanded[:, 1] - beams_centers[:, 1],
        reference_point_expanded[:, 0] - beams_centers[:, 0],
    )
    angles = angles + 2 * pi * (angles < 0)
    return angles


def get_beams_weighted_center(
    beams_masks: Tensor,
    pixels_xys: Tensor,
    ppdf_data: Tensor,
) -> Tensor:
    """
    Compute the weighted center of the beams based on the pixel data and beam masks.

    Parameters
    ----------
    beam_masks: Tensor
      - The masks for the beams.
      - `shape`: (n_pixels, n_beams)
      - `dtype`: `bool`


    pixels_xys: Tensor
      The field of view points.

    ppdf_data: Tensor
      The pixel data probability distribution function.

    beams_relative_sensitivities: Tensor
      The means of the beams.
    """
    n_beams = beams_masks.shape[0]
    pixels_xys_expanded = pixels_xys.unsqueeze(0).expand(n_beams, -1, -1)
    ppdf_expanded = ppdf_data.view(-1).unsqueeze(0).unsqueeze(2).expand(n_beams, -1, 2)

    xy_weighted_sum = (
        (pixels_xys_expanded * ppdf_expanded)
        .masked_fill_(~beams_masks.unsqueeze(2).expand(-1, -1, 2), 0)
        .sum(dim=1)
    )
    total_weights = (
        (ppdf_data.view(-1).unsqueeze(0).expand(n_beams, -1))
        .clone()
        .masked_fill_(~beams_masks, 0)
        .sum(dim=1)
    )
    return xy_weighted_sum / total_weights.unsqueeze(1)


def get_beams_basic_properties(
    beams_masks: Tensor,
    pixels_xys: Tensor,
    ppdf_data: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute the weighted center of the beams based on the pixel data and beam masks.

    Parameters
    ----------
    beam_masks: Tensor
      - The masks for the beams.
      - `shape`: (n_pixels, n_beams)
      - `dtype`: `bool`


    pixels_xys: Tensor
      The field of view points.

    ppdf_data: Tensor
      The pixel data probability distribution function.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        - `beams_centers`: Tensor
            The weighted centers of the beams.
        - `beams_sensitivity_batch`: Tensor
            The sensitivity of the beams.
        - `beams_area_batch`: Tensor
            The area of the beams.
    """
    n_beams = beams_masks.shape[0]
    pixels_xys_expanded = pixels_xys.unsqueeze(0).expand(n_beams, -1, -1)
    ppdf_expanded = ppdf_data.view(-1).unsqueeze(0).unsqueeze(2).expand(n_beams, -1, 2)

    xy_weighted_sum = (
        (pixels_xys_expanded * ppdf_expanded)
        .masked_fill_(~beams_masks.unsqueeze(2).expand(-1, -1, 2), 0)
        .sum(dim=1)
    )
    beams_sensitivity_batch = (
        (ppdf_data.view(-1).unsqueeze(0).expand(n_beams, -1))
        .clone()
        .masked_fill_(~beams_masks, 0)
        .sum(dim=1)
    )
    beams_area_batch = beams_masks.sum(dim=1)
    return (
        xy_weighted_sum / beams_sensitivity_batch.unsqueeze(1),
        beams_sensitivity_batch,
        beams_area_batch,
    )


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

    n_samples = data_x_batch.shape[1]

    # Find the indices of the points where the curve crosses half of the maximum
    half_max = (data_y_batch.max(dim=1).values * 0.5).unsqueeze(1).expand(-1, n_samples)
    x_above_half_max_batch = (
        where(data_y_batch >= half_max, data_x_batch, 0).sort(dim=1).values
    )
    x_upper_lower_batch = x_above_half_max_batch[:, [0, -1]]
    fwhm_batch = x_upper_lower_batch[:, 1] - x_upper_lower_batch[:, 0]
    return fwhm_batch, x_upper_lower_batch


def get_beam_sample_line_batch(
    crystal_center: Tensor, beams_centers: Tensor, n_samples: int = 513
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Get the unit vectors of the sampling lines for the beams in batch.
    The sampling line is perpendicular to the beam axis and passes through the
    beam center.
    Parameters
    ----------
    crystal_center: Tensor
      The center of the detector unit.
      `shape`: (2,)
      `dtype`: `float32`
    beams_centers: Tensor
      The weighted centers of the beams.
      `shape`: (n_beams, 2)
      `dtype`: `float32`
    Returns
    -------
    sample_line_unit_vectors: Tensor
      The unit vectors of the sampling lines.
      `shape`: (n_beams, 2)
      `dtype`: `float32`
    """

    n_beams = beams_centers.shape[0]
    beam_axis_rad = fov_pixels_to_crystal_rads(
        beams_centers,
        crystal_center,
    )
    sample_line_rad = beam_axis_rad + pi * 0.5
    sample_linear_space_1d_batch = (
        (
            arange(-(n_samples - 1) // 2, (n_samples - 1) // 2 + 1)
            .unsqueeze(0)
            .expand(n_beams, -1)
        )
        / (n_samples - 1)
        * 2
    )
    sample_xy_base_2d_batch = (
        sample_linear_space_1d_batch.unsqueeze(-1).expand(-1, -1, 2)
    ) * (
        stack((cos(sample_line_rad), sin(sample_line_rad)), dim=1)
        .unsqueeze(1)
        .expand(-1, n_samples, -1)
    )

    return (
        sample_linear_space_1d_batch,
        sample_xy_base_2d_batch,
        beams_centers.unsqueeze(1).expand(-1, n_samples, -1),
    )


def get_beams_width(
    beams_centers: Tensor,
    crystal_center: Tensor,
    beams_masks: Tensor,
    ppdf_2d: Tensor,
    fov_dict: Dict,
    initial_length: float = 64.0,
    line_n_samples: int = 1025,
):
    """
    Compute the Full Width at Half Maximum (FWHM) of beams and their sampled data
    along specified sampling lines.

    Parameters
    ----------
    beams_centers : Tensor
        A tensor of shape `(n_beams, 2)` representing the coordinates of the beam centers.
    detector_unit_center : Tensor
        A tensor of shape `(2,)` representing the center of the detector unit.
    beams_masks : Tensor
        A boolean tensor of shape `(n_beams, n_pixels_x * n_pixels_y)` indicating
        the valid regions for each beam.
    ppdf_2d : Tensor
        A 2D tensor of shape `(n_pixels_x, n_pixels_y)` representing the pixelated
        probability density function (PPDF).
    fov_dict : Dict
        A dictionary containing field-of-view (FOV) metadata, including the number
        of pixels in each dimension (`"n pixels"`).
    line_n_samples : int, optional
        The number of samples to take along each sampling line. Default is 4096.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        - `beams_fwhm` : Tensor
            A tensor of shape `(n_beams,)` containing the Full Width at Half Maximum
            (FWHM) for each beam.
        - `x_bounds_batch` : Tensor
            A tensor of shape `(n_beams, 2)` containing the x-coordinate bounds for
            each beam's FWHM.
        - `sampled_beams_data` : Tensor
            A tensor of shape `(n_beams, line_n_samples)` containing the sampled beam
            data along the sampling lines.
        - `beam_sp_distance` : Tensor
            A tensor of shape `(line_n_samples,)` representing the distances along
            the sampling lines.
    """

    n_beams = int(beams_centers.shape[0])

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
    sample_linspace_1d_batch, sample_xy_base_2d_batch, beams_centers_batch = (
        get_beam_sample_line_batch(
            crystal_center, beams_centers, n_samples=line_n_samples
        )
    )

    lines_length_batch = ones(n_beams) * initial_length
    sample_distance_1d_batch = sample_linspace_1d_batch * lines_length_batch.view(-1, 1)

    beams_fwhm_batch = lines_length_batch + 1
    x_bounds_batch = zeros((n_beams, 2), dtype=float32)
    sampled_beams_data_batch = empty_tensor((n_beams, line_n_samples), dtype=float32)
    sample_xy_batch = (
        sample_xy_base_2d_batch * lines_length_batch.view(-1, 1, 1)
        + beams_centers_batch
    )
    mask = ones(n_beams, dtype=torch_bool)

    divide_flag = True
    while divide_flag:
        sample_xy_batch[mask] = (
            sample_xy_base_2d_batch * lines_length_batch.view(-1, 1, 1)
            + beams_centers_batch
        )[mask]
        sampled_beams_data_batch[mask] = beam_samples_on_points_batch(
            beams_data_2d_batch[mask], sample_xy_batch[mask], fov_dict
        )

        beams_fwhm_batch[mask], x_bounds_batch[mask] = full_width_half_maximum_1d_batch(
            sample_distance_1d_batch[mask],
            sampled_beams_data_batch[mask],
        )

        mask = beams_fwhm_batch * 2.0 < lines_length_batch

        divide_flag = mask.any()
        lines_length_batch[mask] = lines_length_batch[mask] * 0.5
        sample_distance_1d_batch[mask] = sample_linspace_1d_batch[
            mask
        ] * lines_length_batch[mask].view(-1, 1)

    return (
        beams_fwhm_batch,
        sample_xy_batch,
        x_bounds_batch,
        sampled_beams_data_batch,
        sample_distance_1d_batch,
    )


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


def angular_edges_on_arc(
    sampled_data: Tensor,
    sampling_rads: Tensor,
    threshold: float = 0.01,
) -> Tensor:

    relative_sampled_ppdf = sampled_data / sampled_data.max()
    threshold = 0.01
    thresholded_relative_sampled_ppdf = zeros_like(relative_sampled_ppdf).masked_fill_(
        relative_sampled_ppdf > threshold, 1
    )

    forward_diff_abs = diff(
        thresholded_relative_sampled_ppdf, prepend=tensor([0.0])
    ).abs()
    radian_edges = sampling_rads[forward_diff_abs > 0.5]
    return radian_edges


def get_beams_boundaries_radians(
    sample_ppdf: Tensor, sample_rads: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Calculate the boundaries of beams in radians based on the sampled PPDF data.
    Args:
        sample_ppdf (Tensor): The sampled PPDF data.
        sample_rads (Tensor): The sampling radians.
    Returns:
        Tensor: The boundaries of the beams in radians.
    """
    n_samples = int(sample_ppdf.shape[0])
    relative_sample_ppdf = sample_ppdf / sample_ppdf.max()
    factor_average = 4
    factor_minima = 32
    sample_local_minima = -F.max_pool1d(
        -relative_sample_ppdf.unsqueeze(0).unsqueeze(0),
        kernel_size=n_samples // factor_minima + 1,
        stride=1,
        padding=n_samples // factor_minima // 2,
    ).squeeze()
    sample_local_average = F.avg_pool1d(
        relative_sample_ppdf.unsqueeze(0).unsqueeze(0),
        kernel_size=n_samples // factor_average + 1,
        stride=1,
        padding=n_samples // factor_average // 2,
    ).squeeze()

    sample_no_peak = relative_sample_ppdf.clone().where(
        relative_sample_ppdf < sample_local_average, sample_local_average
    )
    sample_avg_no_peak = F.avg_pool1d(
        sample_no_peak.unsqueeze(0).unsqueeze(0),
        kernel_size=n_samples // factor_average + 1,
        stride=1,
        padding=n_samples // factor_average // 2,
    ).squeeze()
    baseline = stack((sample_local_minima, sample_avg_no_peak)).mean(dim=0)
    rectified_sample = zeros_like(relative_sample_ppdf).masked_fill_(
        relative_sample_ppdf > baseline, 1
    )
    # Find the up and down slopes of the rectified sample
    up_slope_mask = (rectified_sample - cat([tensor([0]), rectified_sample[:-1]])) > 0.5
    down_slope_mask = (
        rectified_sample - cat([rectified_sample[1:], tensor([0])])
    ) > 0.5
    return (
        stack(
            (
                sample_rads[up_slope_mask],
                sample_rads[down_slope_mask],
            ),
            dim=1,
        ),
        baseline,
        rectified_sample,
    )


def get_beams_masks(
    pixels_rads: Tensor,
    beam_bounds: Tensor,
) -> Tensor:
    n_beams = beam_bounds.shape[0]
    pixels_rads_expanded = pixels_rads.unsqueeze(0).expand(n_beams, -1)
    # Get beams masks
    return logical_and(
        pixels_rads_expanded >= beam_bounds[:, 0].view(-1, 1),
        pixels_rads_expanded < beam_bounds[:, 1].view(-1, 1),
    )


def get_beams_combined_mask(
    beams_masks: Tensor,
) -> Tensor:
    """
    Get the combined mask for the beams.
    The combined mask is uint16 tensor with the same shape as the beams masks.
    The value of the combined mask is the beam id.
    """
    n_beams = beams_masks.shape[0]
    beams_masks_valued = (
        arange(1, n_beams + 1).view(-1, 1).expand(-1, beams_masks.shape[1])
    )
    return (
        (
            beams_masks_valued.clone()
            .masked_fill_(~beams_masks, 0)
            .to(dtype=uint16_tensor)
        )
        .sum(dim=0)
        .unsqueeze(0)
    )
