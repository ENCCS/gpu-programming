#!/bin/bash -l

#load modules
module purge
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

# OpenACC is supported only by the Cray Fortran compiler
# C/C++ compilers have NO support for OpenACC
cc -o executable.acc.exe openACC_code.c
