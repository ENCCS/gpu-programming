#!/bin/bash 

module load CrayEnv
module load PrgEnv-cray
module load rocm
module load craype-accel-amd-gfx90a

hipcc --offload-arch=gfx90a -o executable.hip.exe vec_add_cuda.cu.hip
