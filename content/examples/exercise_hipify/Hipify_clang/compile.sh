#!/bin/bash 

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

hipcc --offload-arch=gfx90a -o executable.hip.exe vec_add_cuda.cu.hip
