if __name__ == "__main__":

    import torch
    import h5py
    import os
    import sys
    from pathlib import Path
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        SpinnerColumn,
    )

    # Get the angular bin boundaries
    n_bins = 360
    angular_bin_boundaries = torch.arange(n_bins) / 180 * torch.pi

    notebook_path = os.path.abspath(
        "".join([p for p in sys.argv if p.endswith(".ipynb")])
    )
    # Find the parent of parent directory of the notebook
    top_dir = Path(notebook_path).parents[0]
    input_dir = os.path.join(top_dir, "python_scripts", "output")

    layouts_unique_id = "e1531c3444e51439add2f18f5714fc50"
    beam_w_lim = 4
    beam_r_sensitivity_lim = 0.05

    out_dir = os.path.join(top_dir, "python_scripts", "output")
    # Initialize the histogram for ASCI map
    asci_histogram = torch.zeros(
        (512 * 512, n_bins),
        dtype=torch.int32,
    )

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        refresh_per_second=10,
    ) as progress:
        task_layouts = progress.add_task("Processing layouts...", total=24)
        for layout_idx in range(0, 24):
            # load the beams properties
            beams_properties_hdf5_filename = (
                "beams_props_" + layouts_unique_id + f"_{layout_idx:02d}.hdf5"
            )
            with h5py.File(
                os.path.join(input_dir, beams_properties_hdf5_filename), "r"
            ) as f:
                # print(f.keys())
                beams_props_tensor = torch.from_numpy(f["beam_properties"][:])  # type: ignore
                beam_props_header = f["beam_properties"].attrs["Header"]  # type: ignore

            # load the beams masks for the layout
            beams_masks_hdf5_filename = (
                "beams_masks_" + layouts_unique_id + f"_{layout_idx:02d}.hdf5"
            )
            with h5py.File(
                os.path.join(input_dir, beams_masks_hdf5_filename), "r"
            ) as beams_masks_hdf5:
                beams_masks = torch.from_numpy(beams_masks_hdf5["beam_mask"][:])  # type: ignore

            layout_beams_props = beams_props_tensor.clone()
            # Digitize the angles
            digitized_angles = torch.bucketize(
                layout_beams_props[:, 3], angular_bin_boundaries, right=False
            )
            layout_beams_props = torch.cat(
                (
                    layout_beams_props,
                    (digitized_angles - 1).unsqueeze(1).float(),
                ),
                dim=1,
            )

            nan_angle_mask = torch.isnan(layout_beams_props[:, 3])
            layout_beams_props_filtered = layout_beams_props[~nan_angle_mask]
            layout_beams_props_filtered = layout_beams_props_filtered[
                layout_beams_props_filtered[:, 4] < beam_w_lim
            ]
            layout_beams_props_filtered = layout_beams_props_filtered[
                layout_beams_props_filtered[:, 8] > beam_r_sensitivity_lim
            ]

            layout_beams_props_filtered = layout_beams_props_filtered[
                layout_beams_props_filtered[:, 9] > 0
            ]
            ppds = layout_beams_props_filtered[:, 7] / layout_beams_props_filtered[:, 9]
            layout_beams_props_filtered = layout_beams_props_filtered[ppds > 1e-5]

            # Loop through the beams, get the beam properties
            task_2 = progress.add_task(
                f"Processing beams for layout {layout_idx:02d}",
                total=int(layout_beams_props_filtered.shape[0]),
            )

            for beam_props in layout_beams_props_filtered:
                detector_idx = int(beam_props[1])
                beam_idx = int(beam_props[2])
                angle_bin_idx = int(beam_props[11])
                asci_histogram[
                    beams_masks[detector_idx] == beam_idx, angle_bin_idx
                ] += 1
                progress.update(task_2, advance=1)
            # remove the task after finishing
            progress.remove_task(task_2)
            progress.update(task_layouts, advance=1)

            # Save the ASCI histogram
            asci_histogram_filename = os.path.join(
                out_dir, f"asci_histogram_scanner_layouts_{layouts_unique_id}_{layout_idx:03d}.hdf5"
            )
            with h5py.File(asci_histogram_filename, "w") as f:
                f.create_dataset("asci_histogram", data=asci_histogram.numpy())

                f["asci_histogram"].attrs["Name"] = "ASCI histogram for beams properties"
                f["asci_histogram"].attrs["Layout unique ID"] = layouts_unique_id
                f["asci_histogram"].attrs["Layout index"] = layout_idx
                f["asci_histogram"].attrs["Beam width limit"] = beam_w_lim
                f["asci_histogram"].attrs["Beam radius sensitivity limit"] = beam_r_sensitivity_lim
                f["asci_histogram"].attrs["Number of angular bins"] = n_bins
                