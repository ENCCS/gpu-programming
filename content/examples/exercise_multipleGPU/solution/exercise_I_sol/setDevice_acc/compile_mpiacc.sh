#!/bin/bash 

#load modules
ml CrayEnv
ml PrgEnv-cray
ml cray-mpich
ml rocm
ml craype-accel-amd-gfx90a

#compile
ftn -hacc -o assignDevice.acc.exe assignDevice_acc.f90
