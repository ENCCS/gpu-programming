#!/bin/bash -l
#SBATCH --job-name=gpuaware-mpiacc
#SBATCH --account=project_465002387
#SBATCH --time=00:05:00
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH -o %x-%j.out

#Load the LUMI software stack
module load LUMI/24.03 partition/G
module load cpeCray

#To enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

#8GPUs#time srun --cpu-bind=map_cpu:57,49,41,33,25,17,1,9 ./laplace.gpuaware.mpiacc.exe
#4GPUs#time srun --cpu-bind=map_cpu:57,49,41,33 ./laplace.gpuaware.mpiacc.exe

time srun ./laplace.gpuaware.mpiacc.exe

rocm-smi --showtoponuma
