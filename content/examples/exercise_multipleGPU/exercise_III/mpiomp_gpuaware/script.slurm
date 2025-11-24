#!/bin/bash -l
#SBATCH --job-name=gpuaware-mpiomp
#SBATCH --account=project_465002387
#SBATCH --time=00:05:00
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH -o %x-%j.out

module load CrayEnv
module load PrgEnv-cray
module load cray-mpich
module load craype-accel-amd-gfx90a
module load rocm

#To enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

time srun  ./laplace.gpuaware.mpiomp.exe
echo
