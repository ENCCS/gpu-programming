#!/bin/bash -l
#SBATCH --job-name=test.omp
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
module load rocm

export LD_LIBRARY_PATH=/scratch/project_465002387/clacc/llvm-project/install/lib:$LD_LIBRARY_PATH

time srun ./executable.omp.exe
