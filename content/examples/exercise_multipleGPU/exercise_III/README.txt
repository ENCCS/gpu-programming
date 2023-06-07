Exercise III: Implement GPU-aware support

Note: The MPI-OpenACC code is stored in the folder "mpiacc_gpuaware"
      The MPI-OpenMP code is stored in te folder "mpiomp_gpuaware"

3.1 Incorporate the OpenACC directive *host_data use_device()* to pass a device pointer to an MPI function. (see lines 106 and 132; and lines 160, 161, 164, 165)

3.2 Incorporate the OpenMP directive *data use_device_ptr()* to pass a device pointer to an MPI function. (see lines 106 and 132; and lines 159, 160, 163, 164)

3.3 Compile and run the code on multiple GPUs.
To compile: "./compile.sh"

To submit a job: "sbatch script.slurm"
