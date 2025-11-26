#!/bin/bash -l

#load modules
module purge
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

# compile with cc
cc -fopenmp -o executable.omp.exe openMP_code.c
