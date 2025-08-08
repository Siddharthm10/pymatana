import os
import sys
from pathlib import Path

notebook_path = os.path.abspath("".join([p for p in sys.argv if p.endswith(".ipynb")]))
# Find the parent of parent directory of the notebook
top_dir = Path(notebook_path).parents[3]
# top_dir = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
# Add the parent directory to the system path
if top_dir not in sys.path:
    # add the top directory to the beginning of the path
    sys.path.insert(0, top_dir.as_posix())
print(f"sys.path[0]: {sys.path[0]}")
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from torch import arange, cat, tensor

from scanner_modeling.beams import (
    get_beams_boundaries_radians,
    get_beams_masks,
    get_beams_combined_mask,
    sample_ppdf_on_arc_2d_local,
)
from scanner_modeling.convex_hull import convex_hull_2d, sort_points_for_hull_2d
from scanner_modeling.geometry_2d import (
    fov_pixels_coordinates,
    fov_pixels_to_crystal_rads,
    fov_tensor_dict,
)
from scanner_modeling.io import (
    append_to_hdf5_dataset,
    initialize_beam_masks_hdf5,
    load_ppdfs_data_from_hdf5,
    load_scanner_geometry_from_layout,
    load_scanner_layouts,
)
from scanner_modeling.ppdf import smooth_ppdfs

if __name__ == "__main__":

    # Define the directory containing the scanner layouts
    scanner_layouts_dir = os.path.join(top_dir, "scanner_layouts")
    scanner_layouts_filename = "scanner_layouts_e1531c3444e51439add2f18f5714fc50.tensor"

    # Define the directory containing the PPDFs data
    ppdfs_dataset_dir = os.path.join(
        os.path.dirname(top_dir),
        "data_with_git_lfs",
        "scanner_layouts_ppdfs",
        "scanner_layouts_e1531c3444e51439add2f18f5714fc50",
    )

    # Load the scanner layouts
    scanner_layouts_data, layouts_unique_id = load_scanner_layouts(
        scanner_layouts_dir, scanner_layouts_filename
    )
    # Load the PPDFs data
    fov_dict = fov_tensor_dict(
        n_pixels=(512, 512),
        size_in_mm=(128, 128),
        center_coordinates=(0.0, 0.0),
    )
    fov_points_xy = fov_pixels_coordinates(fov_dict)
    fov_n_pixels_int = int(fov_dict["n pixels"].prod())

    # Create the progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[progress.completed]{task.completed}/{task.total}"),
    )

    # Define the layout sequence
    # layout_sequence = arange(0, 24)
    layout_sequence = arange(0, 1)

    with progress:
        # Create the progress bar for the outer loop
        task_outer = progress.add_task(
            "Processing layouts", total=int(layout_sequence.shape[0])
        )

        # Create the progress bar for the inner loop
        task_inner = progress.add_task(
            f"Processing detector units for layout {int(layout_sequence[0]):02d}",
            total=1,
        )

        # Loop through the scanner positions (0 to 23)
        for layout_idx in layout_sequence:
            # Initialize the HDF5 file to store the beams properties
            out_hdf5_filename = f"beams_masks_{layouts_unique_id}_{layout_idx:02d}.hdf5"
            out_dir = "output"
            out_beam_masks_hdf5_file, beam_masks_dataset = initialize_beam_masks_hdf5(
                fov_n_pixels_int, out_hdf5_filename, out_dir
            )
            # initialize_beam_masks_hdf5(
            #     out_hdf5_filename, out_dir, fov_dict["n pixels"])
            # Load the scanner geometry
            (
                plates_objects_vertices,
                crystal_objects_vertices,
                plate_objects_edges,
                crystal_objects_edges,
            ) = load_scanner_geometry_from_layout(int(layout_idx), scanner_layouts_data)

            # Set the PPDFs filename for a particular scanner position
            layouts_unique_id = "e1531c3444e51439add2f18f5714fc50"
            ppdfs_hdf5_filename = (
                "scanner_layouts_"
                + layouts_unique_id
                + f"_layout_{layout_idx:03d}_ppdfs.hdf5"
            )

            # Load the PPDFs data and remove outliers
            treated_ppdf_data_2d = smooth_ppdfs(
                load_ppdfs_data_from_hdf5(
                    ppdfs_dataset_dir, ppdfs_hdf5_filename, fov_dict
                ).view(-1, int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])),
                threshold=1,
            )
            crystal_objects_centers = crystal_objects_vertices.mean(dim=1)
            fov_corners = (
                tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
                * fov_dict["size in mm"]
                * 0.5
            )

            n_crystals = int(crystal_objects_vertices.shape[0])

            # Create a sequence of detector unit indices
            crystal_objects_idx_sequence = arange(0, n_crystals)

            # Reset the progress bar for the inner loop
            progress.reset(
                task_inner,
                total=n_crystals,
                description=f"Processing detector units for layout {int(layout_idx):02d}",
                completed=0,
                start=True,
            )

            # Loop through the detector units
            for selected_crystal_idx in crystal_objects_idx_sequence:
                crystal_center = crystal_objects_centers[selected_crystal_idx]
                fov_points_rads = fov_pixels_to_crystal_rads(
                    fov_points_xy,
                    crystal_center,
                )

                ppdf_data_2d = treated_ppdf_data_2d[selected_crystal_idx]
                # Calculate the convex hull for the crystal
                hull_points = sort_points_for_hull_2d(
                    cat(
                        (
                            fov_corners,
                            crystal_center.unsqueeze(0),
                        ),
                        dim=0,
                    )
                )
                hull_2d = convex_hull_2d(hull_points)

                # Sample the PPDFs on the arc of the convex hull
                (sampled_ppdf, sampling_rads, sampling_points) = (
                    sample_ppdf_on_arc_2d_local(
                        ppdf_data_2d,
                        crystal_center,
                        hull_2d,
                        fov_dict,
                    )
                )
                beams_boundaries, baseline, rectified_sample = (
                    get_beams_boundaries_radians(sampled_ppdf, sampling_rads)
                )

                beams_masks = get_beams_masks(
                    fov_points_rads,
                    beams_boundaries,
                )
                combined_beams_masks = get_beams_combined_mask(beams_masks)

                # Append the beams properties to the HDF5 dataset
                append_to_hdf5_dataset(beam_masks_dataset, combined_beams_masks)

                progress.update(
                    task_inner,
                    advance=1,
                )

            # Close the HDF5 file
            out_beam_masks_hdf5_file.close()
            # Print the output filename
            print(f"Beams masks saved in:\n{out_dir}/{out_hdf5_filename}")  #
            progress.update(
                task_outer,
                advance=1,
            )

    print("Processing completed.")
