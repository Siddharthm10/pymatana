from torch.nn import functional as F
from torch import max as torch_max, tensor, Tensor
from typing import Sequence


def local_max_1d(data: Tensor, size: int = 3):
    """
    Get the local max of a 1D tensor with a given factor.
    """
    data_padded = F.pad(
        data.view(1, 1, -1), (size // 2, size // 2), mode="constant", value=0
    ).squeeze()
    return torch_max(data_padded.unfold(0, size, 1), dim=1).values


def fov_tensor_dict(
    n_pixels: Sequence[int] = (512, 512),
    mm_per_pixel: Sequence[float] = (0.25, 0.25),
    center_coordinates: Sequence[float] = (0.0, 0.0),
) -> dict:
    """
    Create a dictionary with the FOV information.
    """
    fov_dict = {
        "n pixels": tensor([512, 512]),
        "mm per pixel": tensor([0.25, 0.25]),
        "center coordinates in mm": tensor([0.0, 0.0]),
    }
    fov_dict["size in mm"] = fov_dict["n pixels"] * fov_dict["mm per pixel"]
    return fov_dict
