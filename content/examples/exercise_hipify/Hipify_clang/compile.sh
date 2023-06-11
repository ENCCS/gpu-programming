#!/bin/bash 

module load CrayEnv
module load PrgEnv-cray
module load rocm/5.2.3
module load craype-accel-amd-gfx90a

hipcc --offload-arch=gfx90a -o executable.hip.exe vec_add_cuda.cu.hip
