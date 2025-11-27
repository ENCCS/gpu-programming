#!/bin/bash -l
#SBATCH --job-name=setDevice_acc
#SBATCH --account=project_465002387
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
##SBATCH --gpus-per-task=4
#SBATCH -o %x-%j.out

#Load the LUMI software stack
module load LUMI/24.03 partition/G
module load cpeCray

srun ./assignDevice.acc.exe | sort
echo 
