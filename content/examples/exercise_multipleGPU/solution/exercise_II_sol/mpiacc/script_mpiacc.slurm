#!/bin/bash -l
#SBATCH --job-name=mpiacc
#SBATCH --account=project_465002387
#SBATCH --time=00:05:00
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
##SBATCH --gpus-per-task=4
#SBATCH -o %x-%j.out

#Load the LUMI software stack
module load LUMI/24.03 partition/G
module load cpeCray

time srun ./laplace.mpiacc.exe 
echo 
