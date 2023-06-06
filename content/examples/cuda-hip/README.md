This folder contains several examples in CUDA and HIP that can be used for the hands on based on the episode [Non-portable kernel-based models](https://enccs.github.io/gpu-programming/9-non-portable-kernel-models/). 
## Compiling on LUMI
First load the modules:
```
module load LUMI/22.08
module load partition/G
module load rocm/5.3.3
``` 
This will load the newest ROCm on LUMI.
Alternatively one could use `Makefile` in the folder which are set to use the [cray compilers](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/#using-hipcc). 
## Running
In order to execute the HIP application on GPU nodes we submit it to the partition `standard-g` (`-p` flag). We have specify as well the running options like number of gpus needed, mpi tasks and CPU core per MPI taks. Below we have an example of a job with 2GPUs, 1 node, 2 MPI tasks and 4 cores per MPI task:

``` 
srun -p standard-g --gpus 2 -N 1 -n 2 -c 4 --time=00:10 --account=project_465000485 ./a.out
``` 
Modify this according to the neeeds of the job. Note that the modules should be loaded in the terminal which is used for launching the job.

## Exercises instructions
The best way to learn programming is to get our hands dirty. Use the example codes in this folder to repoduce the problems presented in the [Non-portable kernel-based models](https://enccs.github.io/gpu-programming/9-non-portable-kernel-models/) episode.

Here are some suggestions for playing around:
* check the GPU assignment in the "Hello World" example. 
    - try  1 MPI taks with multiple GPUs and set the code to use something different from the default `device 0`
    - try P nodes with N(<=8)  MPI tasks per node with each MPI task being assigned a different GPU.
* check the vector addition with device memory and with unified memory
* implement the matrix transpose and compute the effective bandwidths achieved on LUMI GPUs
* implement a code using 1 GPU and do a reduction on a vector
* based on the [CUDA blog](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) try to implement a code using [HIP streams](https://docs.amd.com/bundle/4.5-HIP-API/page/group___stream.html)  to perform the vector addition problem or reduction.
