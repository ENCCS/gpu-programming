#!/bin/bash 

#load modules
module load LUMI/24.03 partition/G
module load cpeCray

#compile
ftn -homp -o assignDevice.omp.exe assignDevice_omp.f90
