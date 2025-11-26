#!/bin/bash -l

# load modules
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm/6.0.3

# OpenACC is supported only by the Cray Fortran compiler
# C/C++ compilers have NO support for OpenACC
cc -o executable.acc.exe openACC_code.c
