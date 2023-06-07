#!/bin/bash 

#load modules
ml CrayEnv
ml PrgEnv-cray
ml cray-mpich
ml rocm
ml craype-accel-amd-gfx90a

#compile
ftn -hacc -o laplace.gpuaware.mpiacc.exe laplace_gpuaware_mpiacc.f90
rm *.acc.s
rm *.acc.o
