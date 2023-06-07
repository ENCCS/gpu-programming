Exercise II: Apply the traditional MPI combined with either OpenACC or OpenMP

Note: The MPI-OpenACC code is stored in the folder "mpiacc"
      The MPI-OpenMP code is stored in te folder "mpiomp"

2.1  Incoporate the OpenACC directives *update host()* and *update device()* before and after calling an MPI function, respectively. (see lines 106 and 133)

2.2. Incorporate the OpenMP directives *update device() from()* and *update device() to()* before and after calling an MPI function, respectively. (see lines 106 and 133)

2.3 Compile and run the code on multiple GPUs.
To compile: "./compile.sh"

To submit a job: "sbatch script.slurm"

**Note: The OpenACC directive *update host()* is used to transfer data from GPU to CPU within a data region; while the directive *update device()* is used to transfer the data from CPU to GPU.

**Note: The OpenMP directive *update device() from()* is used to transfer data from GPU to CPU within a data region; while the directive *update device() to()* is used to transfer the data from CPU to GPU.
