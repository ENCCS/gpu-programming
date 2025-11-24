#!/bin/bash -l
#SBATCH --job-name=mpi
#SBATCH --account=project_465002387
#SBATCH --time=00:05:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -o %x-%j.out

module load CrayEnv
module load PrgEnv-cray
module load cray-mpich
module load rocm

time srun  ./laplace.mpi.exe
