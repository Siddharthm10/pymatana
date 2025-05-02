HOW TO GET PPDS GRAPHS. 


STEPS
1. Generate your cuboids data. Use this link as reference - https://github.com/spebt/pymatana/blob/main/ppdf-analysis/asci-map-analysis/24-rotations-angles/generate_3d_cuboids.py
2. For 24 rotations, the data is alredy uploaded in the "scanner_cuboids_data_24rots" folder. If you wish to work with 24 rotations, you may skip this step.
3. Based on the scanner geometry data generated, the next step is to generate system matrices. Leverage the "ppdf_parallel_npz.py" file to generate the ppdfs via MPI support. If you do not have access to MPI parallelizations, you can work with the "ppdf_calculation_nompi_npz.py" file, but based on my analysis, atleast for 24 rotations, the Estimated time for completion goes upto 3 days, so MPI support is highly encouraged to be used.
4. Once you run the above codes, a folder should come up called sysmats with all the system matrices indexed based on numbers.
5. The next step is to run the "evaluate_beam_params_24rots.py" code file that will help compute all the beam parameters present in all the ppdfs we generated. The output after running this file will be saved in a folder called "npzs" that will have the corresponding beam parameters. Note: If you directly clone this repo, you may already see the pre-computed beam params for 24 rotations, if you wish to compute this for some different combination then you can delete the pre-computed params.
6. Based on our research, for PPDS metric, we will need the volume of our beams, so next step will be to run the "calculate_beam_volume_24rots.py" file that will help do this. The results of this will be stored in a folder called "beam_volumes". Note: If you directly clone this repo, you may already see the pre-computed beam volumes for 24 rotations, if you wish to compute this for some different combination then you can delete the pre-computed volumes.
7. The next step is to run the "calculate_ppds_maps.py" file that will help us compute the PPDS metric and plot it. The output after running this file will be stored in the "results" directory. Note: If you directly clone this repo, you may already see the pre-computed PPDS Map for 24 rotations, if you wish to compute this for some different combination then you can delete the pre-computed PPDS Map.

Added Note
The slurm_scripts folder contains the slurm script I leveraged to run the MPI Support system matrix generation code file on UB's CCR Cluster. 

Questions ?
Feel free to reach out on htripathi6@gmail.com for any implementation issues or doubts you may have while working on this. 
