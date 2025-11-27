#!/bin/bash -l
#SBATCH --job-name=mpi
#SBATCH --account=project_465002387
#SBATCH --time=00:05:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -o %x-%j.out

#Load the LUMI software stack
module load LUMI/24.03 partition/G
module load cpeCray

time srun  ./laplace.mpi.exe
