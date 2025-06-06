# CUDA and HIP examples

This folder contains several examples in CUDA and HIP that can be used for the hands on based on the episode [Non-portable kernel-based models](https://enccs.github.io/gpu-programming/9-non-portable-kernel-models/). 
## Compiling on LUMI
First load the modules:
```
module load LUMI
module load partition/G
module load rocm/6.0.3
``` 
This will load the newest ROCm on LUMI.
Now the code can be compiled:
```
hipcc -O2 --offload-arch=gfx90a <code>.cpp
``` 
Alternatively, when available,  one could use `Makefile` set to use the [cray compilers](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/#using-hipcc). 
## Running
In order to execute the HIP application on GPU nodes we submit it to the partition `dev-g` (`-p` flag). We have specify as well the running options like number of GPUs needed, MPI tasks and CPU core per MPI task. Below we have an example of a job with 2 GPUs (`--gpus`, 1 node (`-N`), 2 MPI tasks (`-n`)  and 7 cores per MPI task (`-c`):

``` 
srun -p dev-g --gpus 2 -N 1 -n 2 -c 7 --time=00:10:00 --account=project_465001310 ./a.out
``` 
Modify this according to the needs of the job. Note that the modules should be loaded in the terminal which is used for launching the job.

If a reservation is available add `--reservation=<res_name>` to use the specific nodes dedicated to the course.

## Exercise instructions
The best way to learn programming is to get our hands dirty. Use the example codes in this folder to reproduce the problems presented in the [Non-portable kernel-based models](https://enccs.github.io/gpu-programming/9-non-portable-kernel-models/) episode.

Here are some suggestions for playing around:
* check the GPU assignment in the "Hello World" example. 
    - try  1 MPI task with multiple GPUs and set the code to use something different from the default `device 0`. Use the set device function `hipSetDevice(nd)` before the `hipGetDevice(&device)`.
    - try P nodes with n(<=8)  MPI tasks per node with each MPI task being assigned a different GPU.
* check the vector addition with device memory and with unified memory
* implement the matrix transpose and compute the effective bandwidths achieved on LUMI GPUs
* implement a code using 1 GPU and do a reduction on a vector
* based on the [CUDA blog streams](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) tray to implement a code doing vector additions using streams overlap data transfers and computations.
* based on the [CUDA blog reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) try to implement a code doing a sum reduction.
* check on LUMI the memory bandwidth in the [memory transpose](https://github.com/ENCCS/gpu-programming/tree/main/content/examples/cuda-hip/hip/04_matrix_transpose) example.
