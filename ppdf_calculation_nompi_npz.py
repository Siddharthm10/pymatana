import torch
import time
import h5py
import os
from rich.progress import Progress, BarColumn
from rich.console import Console

from raytracer_2d import (
    load_scanner_geometry_npz,
    set_default_device_as_cpu,
    get_geom_dict,
    get_ppdf,
)

if __name__ == "__main__":
    import sys

    set_default_device_as_cpu()

    fov_dict = {
        "n_pixels": torch.tensor([512, 512]),
        "mm_per_pixel": torch.tensor([0.25, 0.25]),
        "center": torch.tensor([0.0, 0.0]),
    }

    scanner_geometry_dir = "scanner_cuboids_data_24rots"
    fov_n_pixels = int(torch.prod(fov_dict["n_pixels"]))

    file_paths = [
        os.path.join(scanner_geometry_dir, f"scanner_cuboids_{i:03d}.npz")
        for i in range(24)
    ]

    # First pass to count total xtals (for progress bar only)
    total_xtals = 0
    xtal_counts = []
    for path in file_paths:
        _, xtal_verts = load_scanner_geometry_npz(path)
        count = xtal_verts.shape[0]
        xtal_counts.append(count)
        total_xtals += count

    progress = Progress(
        "{task.description}",
        BarColumn(),
        "{task.completed:03d}/{task.total:03d}",
        "[progress.percentage]{task.percentage:>3.0f}% Completed",
        transient=True,
        console=Console(),
    )
    task = progress.add_task("Computing PPDFs", total=total_xtals)

    with progress:
        for file_idx, file_path in enumerate(file_paths):
            try:
                plate_verts_2d, xtal_verts_2d = load_scanner_geometry_npz(file_path)
                geom_dict = get_geom_dict(plate_verts_2d, xtal_verts_2d, fov_dict)
                num_xtals = xtal_verts_2d.shape[0]

                output_filename = f"scanner_ppdfs_{file_idx:03d}.hdf5"
                with h5py.File(output_filename, "w") as out_h5file:
                    ppdf_dataset = out_h5file.create_dataset(
                        "ppdfs", shape=(num_xtals, fov_n_pixels), dtype="f"
                    )
                    elapsed_times = torch.zeros(num_xtals)

                    for local_idx in range(num_xtals):
                        start_time = time.time()
                        ppdf = get_ppdf(local_idx, geom_dict=geom_dict)
                        ppdf_dataset[local_idx] = ppdf.unsqueeze(0).numpy()
                        elapsed_times[local_idx] = time.time() - start_time
                        progress.update(task, advance=1)

                progress.console.print(
                    f"[green]Saved:[/green] {output_filename} — "
                    f"Avg time per xtal: {elapsed_times.mean():.4f}s"
                )
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                sys.exit(1)

    progress.console.print("[bold green]All PPDF calculations completed![/bold green]")
