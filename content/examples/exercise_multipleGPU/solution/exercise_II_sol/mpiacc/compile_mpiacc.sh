#!/bin/bash 

#Load the LUMI software stack
module load LUMI/24.03 partition/G
module load cpeCray

#compile
ftn -hacc -o laplace.mpiacc.exe laplace_mpiacc.f90 
