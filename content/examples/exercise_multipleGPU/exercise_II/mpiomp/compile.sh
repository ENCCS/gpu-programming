#!/bin/bash 

#load modules
ml CrayEnv
ml PrgEnv-cray
ml cray-mpich
ml rocm
ml craype-accel-amd-gfx90a

#compile
ftn -homp -o laplace.mpiomp.exe laplace_mpiomp.f90 
rm *.acc.s
rm *.acc.o
