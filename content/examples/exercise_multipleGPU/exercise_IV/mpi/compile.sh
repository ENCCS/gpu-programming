#!/bin/bash 

#load modules
ml CrayEnv
ml PrgEnv-cray
ml cray-mpich
ml rocm

#compile
ftn -o laplace.mpi.exe laplace_mpi.f90
