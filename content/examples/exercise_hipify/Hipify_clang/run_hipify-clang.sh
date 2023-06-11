#!/bin/bash

MyDir=/scratch/project_465000485
MyContainer=${MyDir}/Container/cuda_11.4.3-devel-ubuntu20.04.sif

module load rocm/5.2.3

singularity exec -B $PWD,/opt:/opt ${MyContainer} bash -c \
          "export PATH=/opt/rocm-5.2.3/bin:$PATH && \
	  hipify-clang vec_add_cuda.cu -o vec_add_cuda.cu.hip --cuda-path=/usr/local/cuda-11.4 -I /usr/local/cuda-11.4/include"
