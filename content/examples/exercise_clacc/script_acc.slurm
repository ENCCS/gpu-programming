#!/bin/bash -l
#SBATCH --job-name=test.acc
#SBATCH --account=project_465002387
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH -o %x-%j.out

module purge
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm/6.0.3

#set env. var for clang
#export PATH=/scratch/project_465002387/clacc/llvm-project/install/bin:$PATH

export LD_LIBRARY_PATH=/scratch/project_465002387/clacc/llvm-project/install/lib:$LD_LIBRARY_PATH

time srun ./executable.acc.exe
