from typing import Dict, Tuple
from torch import Tensor
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def plot_2d_ppdf_mpl(
    ppdf_data: Tensor, ax: Axes, fov_dict: Dict, **kwargs
) -> AxesImage:
    mpl_imshow_extent = (
        -int(fov_dict["size in mm"][0] / 2),
        int(fov_dict["size in mm"][0] / 2),
        -int(fov_dict["size in mm"][1] / 2),
        int(fov_dict["size in mm"][1] / 2),
    )

    im = ax.imshow(
        ppdf_data.view(
            int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
        )
        .cpu()
        .numpy()
        .T,
        origin="lower",
        extent=mpl_imshow_extent,
        aspect="equal",
        vmin=0,
        cmap=kwargs.get("cmap", "hot_r"),
        **kwargs,
    )
    return im
