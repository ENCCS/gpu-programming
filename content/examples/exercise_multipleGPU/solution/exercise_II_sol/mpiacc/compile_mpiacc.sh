#!/bin/bash 

#load modules
ml CrayEnv
ml PrgEnv-cray
ml cray-mpich
ml rocm
ml craype-accel-amd-gfx90a

#compile
ftn -hacc -o laplace.mpiacc.exe laplace_mpiacc.f90 
rm *.acc.s
rm *.acc.o
