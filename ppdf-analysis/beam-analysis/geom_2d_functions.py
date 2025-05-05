import torch
import torch.nn as nn


def get_three_p_cross_batch(points_batch: torch.Tensor) -> torch.Tensor:
    # points_batch.shape = (n_batch, 3, 2)
    # return tensor of shape (n_batch,)
    return (
        points_batch[:, 1, 0] * (points_batch[:, 2, 1] - points_batch[:, 0, 1])
        + points_batch[:, 2, 0]
        * (points_batch[:, 0, 1] - points_batch[:, 1, 1])
        + points_batch[:, 0, 0]
        * (points_batch[:, 1, 1] - points_batch[:, 2, 1])
    )


def get_vertices_sorted_by_xy_2d_batch(
    vertices: torch.Tensor, main_axis: int = 0
) -> torch.Tensor:
    # Get unique x values
    unique_x, unique_x_index = torch.unique(
        vertices[:, :, 0], return_inverse=True, sorted=True
    )
    # Get unique y values
    unique_y, unique_y_index = torch.unique(
        vertices[:, :, 1], return_inverse=True, sorted=True
    )
    # Sort the indices based on x and y values
    indices_by_xy = torch.argsort(unique_x_index * 100 + unique_y_index)
    return vertices.gather(1, indices_by_xy.unsqueeze(-1).expand(-1, -1, 2))


def get_vertices_sorted_by_rad_raw_2d_batch(
    vertices: torch.Tensor, ref_point: torch.Tensor
) -> torch.Tensor:
    # sort the vertices by angle to point ref_point
    n_points = vertices.shape[1]
    rads = torch.atan2(
        vertices[:, :, 1] - ref_point[:, :, 1].expand(-1, n_points),
        vertices[:, :, 0] - ref_point[:, :, 0].expand(-1, n_points),
    )
    order = torch.argsort(rads, dim=1)
    return vertices.gather(dim=1, index=order.unsqueeze(-1).expand(-1, -1, 2))


def iter_point_convex_hull_2d_batch(
    point_batch: torch.Tensor,
    index_batch: torch.Tensor,
    convex_hull_batch: torch.Tensor,
):
    # point_batch.shape = (n_batch, 2)
    convex_hull_batch[index_batch[:, 0], index_batch[:, 1]] = point_batch
    index_batch[:, 1] += 1
    filter_index = torch.stack(
        [index_batch[:, 1] - 3, index_batch[:, 1] - 2, index_batch[:, 1] - 1],
        dim=1,
    ).flatten()

    last_three_points_batch = convex_hull_batch[
        index_batch[:, 0].repeat_interleave(3), filter_index
    ].reshape(-1, 3, 2)
    boolean_mask = get_three_p_cross_batch(last_three_points_batch) <= 0

    convex_hull_batch[
        index_batch[boolean_mask, 0], index_batch[boolean_mask, 1] - 2
    ] = point_batch[boolean_mask]
    index_batch[boolean_mask, 1] -= 1


def get_convex_hull_2d_batch(points_batch: torch.Tensor):
    n_batch = points_batch.shape[0]
    n_points = points_batch.shape[1]
    # sort the points by x and y
    points_batch = get_vertices_sorted_by_xy_2d_batch(points_batch)
    # sort the points by angle reference to the first point
    points_batch_sorted_by_rad = get_vertices_sorted_by_rad_raw_2d_batch(
        points_batch[:, 1:], points_batch[:, :1]
    )
    points_batch = torch.cat(
        (points_batch[:, :1], points_batch_sorted_by_rad), dim=1
    )

    hull_batch = torch.empty((n_batch, n_points, 2), dtype=torch.float32)

    hull_batch[:, :2] = points_batch[:, :2]
    index_batch = torch.stack(
        (
            torch.arange(n_batch, dtype=torch.int32),
            torch.ones(n_batch, dtype=torch.int32) * 2,
        ),
        dim=1,
    )

    for i in torch.arange(2, n_points, dtype=torch.int32):
        iter_point_convex_hull_2d_batch(
            points_batch[:, i], index_batch, hull_batch
        )

    mask_batch = torch.where(
        torch.arange(n_points, dtype=torch.int32)
        .view(1, -1)
        .expand(n_batch, -1)
        < index_batch[:, 1].view(-1, 1).expand(-1, n_points),
        True,
        False,
    )
    return hull_batch, mask_batch


def get_point_fov_hull_2d_batch(point_batch, fov_dict):
    """
    Compute the convex hull of the field of view (FOV) for a batch of given point.
    """
    batch_size = point_batch.shape[0]
    fov_corners = torch.tensor(
        [
            [
                -int(fov_dict["size in mm"][0]) / 2,
                -int(fov_dict["size in mm"][1]) / 2,
            ],
            [
                int(fov_dict["size in mm"][0]) / 2,
                -int(fov_dict["size in mm"][1]) / 2,
            ],
            [
                int(fov_dict["size in mm"][0]) / 2,
                int(fov_dict["size in mm"][1]) / 2,
            ],
            [
                -int(fov_dict["size in mm"][0]) / 2,
                int(fov_dict["size in mm"][1]) / 2,
            ],
        ]
    ).to(torch.float32)

    point_batch = torch.cat(
        (
            point_batch.unsqueeze(1).expand(-1, 1, -1),
            fov_corners.unsqueeze(0).expand(1, -1, -1).repeat(batch_size, 1, 1),
        ),
        dim=1,
    ).to(torch.float32)
    hull_batch, mask_batch = get_convex_hull_2d_batch(point_batch)
    return hull_batch, mask_batch


def get_local_mean_1d(data: torch.Tensor, kernel_size: int = 3):
    """
    Get the local mean of a 1D tensor with a given factor.
    """
    m = nn.Conv1d(
        1, 1, kernel_size, stride=1, padding=kernel_size // 2, bias=False
    )
    m.weight.data.fill_(1.0 / 3.0)

    return m(data.view(1, 1, -1).to(torch.float32)).squeeze()


def get_local_max_1d(data: torch.Tensor, size: int = 3):
    """
    Get the local max of a 1D tensor with a given factor.
    """
    data_padded = torch.nn.functional.pad(
        data.view(1, 1, -1), (size // 2, size // 2), mode="constant", value=0
    ).squeeze()
    return torch.max(data_padded.unfold(0, size, 1), dim=1).values
