import numpy as np
import pyvista as pv
import yaml

# import torch
import h5py
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    Column,
)
import matplotlib.pyplot as plt
import skimage as ski
import scipy


def get_output_filename(ifname: str, suffix: str, ext: str):
    fname_wo_ext = ifname.split(".")[0]
    return fname_wo_ext + suffix + "." + ext


def get_slope(fname: str, pbar: Progress, task_id: int):
    sysmats = np.empty((864, 128, 128))
    all_params = np.zeros((864, 15, 3))
    filtered_ppdfs = np.zeros((864, 128, 128))
    n_strips_per_ppdf = np.zeros((864, 11))
    with h5py.File(fname, "r") as f:
        sysmats = np.copy(f["system matrix"][:])
    for idp, ppdf in enumerate(sysmats):
        image = ppdf.T
        thresh = ski.filters.threshold_li(image)
        objects = ski.measure.label(image > thresh)
        large_objects = objects
        if len(np.unique(objects)) > 2:
            large_objects = ski.morphology.remove_small_objects(objects, min_size=24)
        unique_lables = np.unique(large_objects)
        unique_lables = unique_lables[np.nonzero(unique_lables)]
        n_strips_per_ppdf[idp] = len(unique_lables)
        ppdf_filt = np.zeros_like(large_objects)
        for idl, label in enumerate(unique_lables):
            ppdf_filt[large_objects == label] = idl + 1
        del large_objects, objects
        for idl, label in enumerate(unique_lables):
            x, y = np.where(ppdf_filt == idl + 1)
            unique_x, unique_index_x = np.unique(x, return_inverse=True)
            unique_y, unique_index_y = np.unique(y, return_inverse=True)
            n_unique_x = unique_x.shape[0]
            n_unique_y = unique_y.shape[0]
            if n_unique_x > n_unique_y:
                x_c = unique_x
                y_c = np.zeros_like(x_c)
                for id, unique in enumerate(unique_x):
                    y_c[id] = np.mean(y[x == unique])
                ntrim = int(np.floor(n_unique_x * 0.03))
            else:
                y_c = unique_y
                x_c = np.zeros_like(y_c)
                for id, unique in enumerate(unique_y):
                    x_c[id] = np.mean(x[y == unique])
                ntrim = int(np.floor(n_unique_y * 0.03))

            ntrim = np.max([ntrim, 1])
            x_c = x_c[ntrim:-ntrim]
            y_c = y_c[ntrim:-ntrim]
            y_c = y_c + 0.5
            x_c = x_c + 0.5
            slope = 0
            intercept = 0
            r = 0

            if len(x_c) > 1:
                if np.unique(x_c).shape[0] < 3 and np.unique(y_c).shape[0] > 120:
                    slope = 0
                    intercept = x_c[0]
                    r = 1
                elif np.all(y_c == y_c[0]):
                    slope = np.inf
                    intercept = y_c[0]
                    r = 1
                else:
                    slope, intercept, r, p, se = scipy.stats.linregress(y_c, x_c)
            if np.abs(r) < 0.75:
                continue
            all_params[idp, idl] = np.array([slope, intercept, r])
            filtered_ppdfs[idp] = ppdf_filt
        pbar.update(task_id, advance=1)

    n_strips_per_ppdf = np.array(n_strips_per_ppdf)
    dict_to_save = {
        "fit params": all_params,
        "filtered ppdfs": filtered_ppdfs,
        "n strips per ppdf": n_strips_per_ppdf,
    }
    print(f'{"Max n_strips_per_ppdf":40s}: {np.max(n_strips_per_ppdf)}')
    ofname = get_output_filename(fname, "_slopes", "npz")
    np.savez_compressed(ofname, **dict_to_save)


pbar = Progress(
    "[progress.description]{task.description}",
    SpinnerColumn(),
    BarColumn(),
    MofNCompleteColumn(),
    # TextColumn("[progress.percentage]{task.completed} of {task.total}"),
    console=None,
)
task_2 = pbar.add_task("[cyan]Processing PPDFs...", total=864, completed=0)
task_1 = pbar.add_task("[cyan]Processing matrices...", total=24, completed=0)

with pbar:
    for i in range(24):
        pbar.update(task_2, completed=0)
        infname = f"data/system_matrix_{i:02d}.hdf5"
        try:
            h5f = h5py.File(infname, "r")
            h5f.close()
        except Exception as e:
            print(f"File {infname} not found")
            continue
        print(get_output_filename(infname, "_slopes", "npz"))
        get_slope(infname, pbar, task_2)
        pbar.update(task_1, advance=1)
