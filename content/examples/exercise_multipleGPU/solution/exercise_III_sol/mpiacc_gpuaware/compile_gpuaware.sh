#!/bin/bash 

#Load the LUMI software stack
module load LUMI/24.03 partition/G
module load cpeCray

#compile
ftn -hacc -o laplace.gpuaware.mpiacc.exe laplace_gpuaware_mpiacc.f90
