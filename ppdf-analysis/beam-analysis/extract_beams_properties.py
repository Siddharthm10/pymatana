import os

from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import fov_tensor_dict
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
    fov_tensor_dict = fov_tensor_dict(
        n_pixels=(512, 512),
        mm_per_pixel=(0.25, 0.25),
        center_coordinates=(0.0, 0.0),
    )

    # Loop through the scanner positions (0 to 23)
    for layout_idx in range(1):
        # Load the scanner geometry
        plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
            layout_idx, scanner_layouts_data
        )

        # Set the PPDFs filename for a particular scanner position
        ppdfs_hdf5_filename = "position_000_ppdfs.hdf5"

        # Load the PPDFs data
        ppdfs = load_ppdfs_data_from_hdf5(
            ppdfs_dataset_dir, ppdfs_hdf5_filename, fov_tensor_dict
        )

        # Print the shapes of the loaded data
        print(
            f"Layout {layout_idx}:\n"
            + f"Metal plates shape  : {list(plates_vertices.shape)}\n"
            + f"Detector units shape: {list(detector_units_vertices.shape)}\n"
            + f"PPDFs data shape:     {list(ppdfs.shape)}"
        )
        