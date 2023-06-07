#!/bin/bash -l
#SBATCH --job-name=setDevice_acc
#SBATCH --account=project_465000485
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g
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

srun ./assignDevice.acc.exe | sort
echo 
