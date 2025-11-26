#!/bin/bash 

# load modules
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm/6.0.3

hipcc --offload-arch=gfx90a -o executable.hip.exe vec_add_cuda.cu.hip
