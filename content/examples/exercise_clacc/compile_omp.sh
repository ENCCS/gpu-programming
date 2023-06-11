#!/bin/bash -l

#load modules
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm/5.2.3

#compile with cc
cc -fopenmp -o executable.omp.exe openMP_code.c
