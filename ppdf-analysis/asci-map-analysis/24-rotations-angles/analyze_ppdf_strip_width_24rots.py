import numpy as np
import h5py
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
)

import skimage as ski
import scipy as sp
import time


def read_data(fname):
    sysmats = np.empty((864, 128, 128))
    with h5py.File(fname, "r") as f:
        sysmats = np.copy(f["system matrix"][:])
    return sysmats


def update_progress(progress, task_id, n=1):
    progress.update(task_id, advance=n)


def get_cuts_on_boarder(pars):
    s = pars[0]
    i = pars[1]
    if s == 0:
        return [[-64, 64], [i - 64, i - 64]]
    if s == np.inf:
        return [[i - 64, i - 64], [-64, 64]]
    x = np.array([-64, 64, -i / s - 64, (128 - i) / s - 64])
    y = np.array([i - 64, 128 * s + i - 64, -64, 64])
    mask = (x >= -64) & (x <= 64) & (y >= -64) & (y <= 64)
    return [x[mask], y[mask]]


def get_beam_center(pars):
    ends = get_cuts_on_boarder(pars)
    return np.mean(ends, axis=1)


def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def get_ppdf_width(matrix_fname, fit_params_fname, progress, task):
    sysmats = read_data(matrix_fname)
    fit_params = np.load(fit_params_fname)["fit params"]
    output = np.zeros((864, 15))
    for det_index in range(864):
        det_fit_pars = fit_params[det_index]
        for idx_beam, pars in enumerate(det_fit_pars):
            if pars[2] == 0:
                continue
            center = get_beam_center(pars)
            # perpendicular unit vector
            puv = (
                np.array([1, 0])
                if pars[0] == np.inf
                else np.array([pars[0], -1]) / np.linalg.norm([pars[0], -1])
            )

            # profile line ends y and x
            ple_y = np.array([center[1] + puv[1] * 10, center[1] - puv[1] * 10])
            ple_x = np.array([center[0] + 10 * puv[0], center[0] - 10 * puv[0]])

            beam_profile = ski.measure.profile_line(
                sysmats[det_index],
                (ple_x[0] + 64, ple_y[0] + 64),
                (ple_x[1] + 64, ple_y[1] + 64),
                linewidth=1,
                mode="constant",
                cval=0,
            )
            beam_profile_intepolated = sp.ndimage.zoom(beam_profile, 4)
            # Calculate Full Width at Half Maximum (FWHM)
            half_max = np.max(beam_profile_intepolated) / 2
            indices = np.where(beam_profile_intepolated > half_max)[0]
            fwhm = (indices[-1] - indices[0]) / 4
            output[det_index, idx_beam] = fwhm
        update_progress(progress, task)
    return output


if __name__ == "__main__":

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        "Elapsed:",
        TimeElapsedColumn(),
    ) as progress:
        task_2 = progress.add_task("[cyan]Processing rotations", total=24)
        task_1 = progress.add_task("[green]Processing detector uints", total=864)
        ppdf_widths = np.zeros((24, 864, 15))
        for i in range(24):
            progress.reset(task_1)
            matrix_fname = f"data/system_matrix_{i:03d}.hdf5"
            fit_params_fname = f"data/system_matrix_{i:03d}_slopes.npz"
            ppdf_widths[i] = get_ppdf_width(
                matrix_fname, fit_params_fname, progress, task_1
            )
            update_progress(progress, task_2)
        dict_output = {"ppdf widths": ppdf_widths}
        np.savez_compressed("ppdf_beam_width.npz", **dict_output)
