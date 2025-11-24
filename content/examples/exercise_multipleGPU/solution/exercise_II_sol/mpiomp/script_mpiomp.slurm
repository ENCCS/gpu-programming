#!/bin/bash -l
#SBATCH --job-name=mpiomp
#SBATCH --account=project_465002387
#SBATCH --time=00:05:00
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
##SBATCH --gpus-per-task=4
#SBATCH -o %x-%j.out

module load CrayEnv
module load PrgEnv-cray
module load cray-mpich
module load craype-accel-amd-gfx90a
module load rocm

time srun ./laplace.mpiomp.exe 
echo 
