import os

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from torch import cat, tensor, arange

from beam_properties import (
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_weighted_center,
    get_beam_width,
    get_beams_angle_radian,
    sample_ppdf_on_arc_2d_local,
)
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import (
    fov_tensor_dict,
    pixels_coordinates,
    pixels_to_detector_unit_rads,
)
from ppdf_io import load_ppdfs_data_from_hdf5

if __name__ == "__main__":

    scanner_layouts_dir = "../../../pymatcal/scanner_layouts"
    scanner_layouts_filename = "scanner_layouts_77faff53af5863ca146878c7c496c75e.tensor"

    ppdfs_dataset_dir = (
        "../../../../data/scanner_layouts_77faff53af5863ca146878c7c496c75e"
    )

    # Load the scanner layouts
    scanner_layouts_data, filename_unique_id = load_scanner_layouts(
        scanner_layouts_dir, scanner_layouts_filename
    )
    # Load the PPDFs data
    fov_dict = fov_tensor_dict(
        n_pixels=(512, 512),
        mm_per_pixel=(0.25, 0.25),
        center_coordinates=(0.0, 0.0),
    )
    fov_points_2d = pixels_coordinates(fov_dict)

    # Create the progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    layout_sequence = arange(0, 1)
    with progress:
        task_outer = progress.add_task(
            "Processing layouts", total=int(layout_sequence.shape[0])
        )

        # Loop through the scanner positions (0 to 23)
        for layout_idx in layout_sequence:
            # Load the scanner geometry
            plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
                layout_idx, scanner_layouts_data
            )

            # Set the PPDFs filename for a particular scanner position
            ppdfs_hdf5_filename = f"position_{layout_idx:03d}_ppdfs.hdf5"

            # Load the PPDFs data
            ppdfs = load_ppdfs_data_from_hdf5(
                ppdfs_dataset_dir, ppdfs_hdf5_filename, fov_dict
            )

            detector_unit_centers = detector_units_vertices.mean(dim=1)
            fov_corners = (
                tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
                * fov_dict["size in mm"]
                * 0.5
            )

            hull_points_batch = cat(
                (
                    fov_corners.unsqueeze(0).expand(
                        detector_units_vertices.shape[0], -1, -1
                    ),
                    detector_unit_centers.unsqueeze(1),
                ),
                dim=1,
            )
            hull_points_batch = sort_points_for_hull_batch_2d(hull_points_batch)

            # Print the shapes of the loaded data
            print(
                f"Layout {layout_idx}:\n"
                + f"Metal plates shape  : {list(plates_vertices.shape)}\n"
                + f"Detector units shape: {list(detector_units_vertices.shape)}\n"
                + f"PPDFs data shape:     {list(ppdfs.shape)}"
            )
            n_detector_units = detector_units_vertices.shape[0]

            # Create the progress bar for the inner loop
            task_inner = progress.add_task(
                f"Processing detector units for layout {layout_idx}",
                total=n_detector_units,
            )
            # Loop through the detector units
            for detector_unit_idx in range(n_detector_units):

                ppdf_data_2d = ppdfs[detector_unit_idx].view(
                    int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
                )
                # Calculate the convex hull for the detector unit
                hull_2d = convex_hull_2d(hull_points_batch[detector_unit_idx])

                # Sample the PPDFs on the arc of the convex hull
                (sampled_ppdf, sampling_rads, sampling_points) = (
                    sample_ppdf_on_arc_2d_local(
                        ppdf_data_2d,
                        detector_unit_centers[detector_unit_idx],
                        hull_2d,
                        fov_dict,
                    )
                )

                relative_sampled_ppdf = sampled_ppdf / sampled_ppdf.max()
                beams_boundaries = beams_boundaries_radians(
                    sampled_ppdf, sampling_rads, threshold=0.01
                )

                fov_points_xy = pixels_coordinates(fov_dict)
                fov_points_rads = pixels_to_detector_unit_rads(
                    fov_points_xy,
                    detector_unit_centers[detector_unit_idx],
                )
                beams_masks = get_beams_masks(
                    fov_points_rads,
                    beams_boundaries,
                )
                beams_weighted_centers = get_beams_weighted_center(
                    beams_masks,
                    fov_points_xy,
                    ppdf_data_2d,
                )
                n_beams = beams_masks.shape[0]

                beam_width = get_beam_width(
                    beams_weighted_centers,
                    detector_unit_centers[detector_unit_idx],
                    beams_masks,
                    ppdf_data_2d,
                    fov_dict,
                )
                beams_angle = get_beams_angle_radian(
                    beams_weighted_centers,
                    detector_unit_centers[detector_unit_idx],
                )
                # print(
                #     f"Detector unit {detector_unit_idx}:\n"
                #     + f"Beam width: {beam_width}\n"
                #     + f"Beams angle: {beams_angle}\n"
                #     + f"Beams boundaries: {beams_boundaries}\n"
                #     + f"Beams masks shape: {list(beams_masks.shape)}\n"
                #     + f"Beams weighted centers shape: {list(beams_weighted_centers.shape)}\n"
                # )
                if n_beams < 3:
                    print(
                        f"Detector unit {detector_unit_idx}:\n"
                        + f"Number of beams: {n_beams}\n"
                    )
                progress.update(
                    task_inner,
                    advance=1,
                )
            progress.remove_task(task_inner)
            progress.update(
                task_outer,
                advance=1,
            )
        progress.remove_task(task_outer)
    print("Processing completed.")
