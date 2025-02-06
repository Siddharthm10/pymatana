# Angular Sampling Completeness Index (ASCI) Map Analysis

| **Review Date** | 2025-02-06 |
|-----------------|------------|
| **Author** | [Fang Han](mailto:fhanonline@gmail.com)|

> [!NOTE]
> This directory contains the code used to generate the ASCI maps in the 2025 SNMMI abstract submission\
 **_"Angular sampling completeness index and width of projection probability density function strips as spatial resolution metrics in self-collimation single photon emission tomography"_**.

## How to run the scripts

> [!Warning]
> The scripts are not guaranteed to be out-of-the-box runnable. \
They are provided as a reference for the methods used in the analysis. \
Please read the scripts and modify them as needed to suit your needs.

### Prerequisites

- Python 3.10 or later
- NumPy
- SciPy
- Matplotlib
- Jupyter Lab or Jupyter Notebook
- scikit-image

### Steps

1. Run `generate_3d_cuboids.py` to generate the cuboids for the 24 rotations.
1. Run `evaluate_sampling_slopes_24rots.py` to generate the slopes data.
1. Run `evaluate_sampling_angles_24rots.py` to generate the angles data.
1. Run `analyze_ppdf_strip_width_24rots.py` to generate the strip width data.
1. Run `plot_asci_map_24rots.ipynb` to generate the ASCI maps


## Description of the scripts

### `evaluate_sampling_slopes_24rots.py`

The script `evaluate_sampling_slopes.py` reads in the PPDF matrices and performs linear regression on the PPDF beams to extract the slopes. The slopes data are saved in `.npz` files.

### `evaluate_sampling_angles_24rots.py`

The script `evaluate_sampling_angles.py` reads in the slopes data and calculates the angles from where the ppdf beams are sampled. The angles data are saved in a `.npz` file.

### `analyze_ppdf_strip_width_24rots.py`

The script `analyze_ppdf_strip_width.py` reads in the PPDF matrices and calculates the strip widths. The strip width data are saved in a `.npz` file.

### `plot_asci_map_24rots.ipynb`

The Jupyter notebook `plot_asci_map_24rots.ipynb` reads in the angles and strip width data and generates the ASCI maps.

### `generate_3d_cuboids.py`

The script `generate_3d_cuboids.py` generates the cuboids for the 24 rotations.
