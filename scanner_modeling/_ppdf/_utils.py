from torch import Tensor, stack, where


def smooth_ppdfs(ppdf_data_2d: Tensor, threshold: float = 1) -> Tensor:
    """
    Smooth outliers in a 2D PPDF data tensor.

    Parameters
    ----------

    ppdf_data_2d (Tensor): The 2D PPDF data tensor.
    - shape (n_pixels_x, n_pixels_y)

    threshold (float): The threshold for smoothing outliers.

    Return
    ------
    Tensor: The smoothed PPDF data tensor.
    """

    out_data = ppdf_data_2d.clone()

    out_data[ppdf_data_2d > threshold] = 0.0
    # ppdf_n_pixels_x = int(ppdf_data_2d.shape[1])
    # outlier_indices = (ppdf_data_2d > threshold).argwhere()
    # idx_1 = where(outlier_indices[:, 2] - 1 > 0, outlier_indices[:, 2] - 1, 1)
    # idx_2 = where(
    #     outlier_indices[:, 2] + 1 < ppdf_n_pixels_x,
    #     outlier_indices[:, 2] + 1,
    #     outlier_indices[:, 2] - 1,
    # )
    # idx_3 = where(outlier_indices[:, 1] - 1 > 0, outlier_indices[:, 1] - 1, 1)
    # idx_4 = where(
    #     outlier_indices[:, 1] + 1 < ppdf_n_pixels_x,
    #     outlier_indices[:, 1] + 1,
    #     outlier_indices[:, 1] - 1,
    # )

    # out_data[outlier_indices[:, 0], outlier_indices[:, 1], outlier_indices[:, 2]] = (
    #     stack(
    #         (
    #             ppdf_data_2d[
    #                 outlier_indices[:, 0],
    #                 outlier_indices[:, 1],
    #                 idx_1,
    #             ],
    #             ppdf_data_2d[
    #                 outlier_indices[:, 0],
    #                 outlier_indices[:, 1],
    #                 idx_2
    #             ],
    #             ppdf_data_2d[
    #                 outlier_indices[:, 0],
    #                 idx_3,
    #                 outlier_indices[:, 2],
    #             ],
    #             ppdf_data_2d[
    #                 outlier_indices[:, 0],
    #                 idx_4,
    #                 outlier_indices[:, 2],
    #             ],
    #         ),
    #         dim=0,
    #     ).mean(dim=0)
    # )
    # return out_data
    return out_data
