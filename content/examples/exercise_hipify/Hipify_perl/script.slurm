#!/bin/bash -l
#SBATCH --job-name=test.hip
#SBATCH --account=project_465002387
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH -o %x-%j.out

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm/6.0.3

srun ./executable.hip.exe
