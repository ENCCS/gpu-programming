#!/bin/bash -l
#SBATCH --job-name=test.acc
#SBATCH --account=project_465000485
#SBATCH --time=00:10:00
#SBATCH --partition=eap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH -o %x-%j.out

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

#set env. var for clang
#export PATH=/scratch/project_465000485/Clacc/llvm-project/install/bin:$PATH

export LD_LIBRARY_PATH=/scratch/project_465000485/Clacc/llvm-project/install/lib:$LD_LIBRARY_PATH

time srun ./executable.acc.exe
