import numpy as np
import h5py
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
)

def read_data(fname):
    sysmats = np.empty((864, 128, 128))
    with h5py.File(fname, "r") as f:
        sysmats = np.copy(f["system matrix"][:])
    return sysmats


def update_progress(progress, task, n=1):
    progress.update(task, advance=n)


def get_filename(ifname: str, suffix: str, ext: str):
    fname_wo_ext = ifname.split(".")[0]
    return fname_wo_ext + suffix + "." + ext


def get_angle(cuboids_fname: str, fit_params_fname: str):
    fit_params = np.load(fit_params_fname)["fit params"]
    xtal_cuboids = np.load(cuboids_fname)["crystal cuboids"]

    xtal_x_c = xtal_cuboids[:, :, 0, 0]
    xtal_y_c = xtal_cuboids[:, :, 0, 1]
    # print(xtal_x_c.shape)
    xtal_y_c = xtal_y_c.reshape(6, 144, 1).repeat(15, axis=2)
    xtal_x_c = xtal_x_c.reshape(6, 144, 1).repeat(15, axis=2)
    vec_x = np.zeros_like(xtal_x_c)
    vec_x[xtal_x_c < 0] = -1
    vec_x[xtal_x_c > 0] = 1
    fit_params = fit_params.reshape(6, 144, 15, 3)
    vec_y = fit_params[:, :, :, 0] * vec_x
    index_up = np.isinf(fit_params[:, :, :, 0]) * (vec_y > 0)
    index_down = np.isinf(fit_params[:, :, :, 0]) * (vec_y < 0)
    vec_y[index_up] = np.inf
    vec_y[index_down] = -np.inf
    sampling_rad = np.where(
        fit_params[:, :, :, 2] != 0, np.arctan2(vec_x, vec_y), np.nan
    )
    return sampling_rad
    # sampling_rad[sampling_rad < 0] += 2 * np.pi
    # print(sampling_rad.shape)
    # output_dict = {"sampling angles": sampling_rad}
    # np.savez_compressed("sampling_angles_rad.npz", **output_dict)


if __name__ == "__main__":
    pbar = Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        # TextColumn("[progress.percentage]{task.completed} of {task.total}"),
        console=None,
    )
    task_1 = pbar.add_task("[cyan]Processing rotation angle", total=24)
    with pbar:
        sampling_rad = np.zeros((24, 6, 144, 15))
        for i in range(24):
            cuboids_fname = f"scanner_cuboids_data/scanner_cuboids_{i:03d}.npz"
            fit_params_fname = f"data/system_matrix_{i:03d}_slopes.npz"
            sampling_rad[i] = get_angle(cuboids_fname, fit_params_fname)

            output_dict = {"sampling angles": sampling_rad}
            # np.savez_compressed(
            #     f"sampling_angles_rad_{i:02d}.npz", **output_dict
            # )
            pbar.update(task_1, advance=1)
        print(sampling_rad.shape)
        np.savez_compressed("data/sampling_angles_rad_24rots.npz", **output_dict)
