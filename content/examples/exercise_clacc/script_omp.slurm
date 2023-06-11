#!/bin/bash -l
#SBATCH --job-name=test.omp
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
module load rocm/5.2.3

export LD_LIBRARY_PATH=/scratch/project_465000485/Clacc/llvm-project/install/lib:$LD_LIBRARY_PATH

time srun ./executable.omp.exe
