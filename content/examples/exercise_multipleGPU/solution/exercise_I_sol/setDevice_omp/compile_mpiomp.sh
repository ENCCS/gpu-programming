#!/bin/bash 

#load modules
ml CrayEnv
ml PrgEnv-cray
ml cray-mpich
ml rocm
ml craype-accel-amd-gfx90a

#compile
ftn -homp -o assignDevice.omp.exe assignDevice_omp.f90
