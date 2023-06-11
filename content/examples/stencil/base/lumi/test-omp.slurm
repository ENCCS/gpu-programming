#!/bin/bash -l
#SBATCH --job-name=stencil-omp
#SBATCH --output=job.o%j
#SBATCH --error=job.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:00
#SBATCH --account=project_465000485

# a) Test scaling at inreasing time steps
for N in 1 8 32
do
	export OMP_NUM_THREADS=$N
	for T in 500 5000
	do
		./stencil 2000 2000 $T
	done
done

# b) Test scaling at increasing grid sizes
for N in 1 8 32
do
        export OMP_NUM_THREADS=$N
	for S in 1000 2000 4000 8000
	do
                ./stencil $S $S 500
        done
done

