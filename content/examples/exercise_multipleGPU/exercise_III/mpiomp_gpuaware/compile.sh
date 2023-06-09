#!/bin/bash 

#load modules
ml CrayEnv
ml PrgEnv-cray
ml cray-mpich
ml rocm
ml craype-accel-amd-gfx90a

#compile
ftn -homp -o laplace.gpuaware.mpiomp.exe laplace_gpuaware_mpiomp.f90
rm *.acc.s
rm *.acc.o
