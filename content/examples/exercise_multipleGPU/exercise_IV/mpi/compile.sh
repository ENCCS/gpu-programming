#!/bin/bash 

#Load the LUMI software stack
module load LUMI/24.03 partition/G
module load cpeCray

#compile
ftn -o laplace.mpi.exe laplace_mpi.f90
