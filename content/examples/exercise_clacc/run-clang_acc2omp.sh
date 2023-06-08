#!/bin/bash -l

#set env. var for clang
export PATH=/scratch/project_465000485/Clacc/llvm-project/install/bin:$PATH

export LD_LIBRARY_PATH=/scratch/project_465000485/Clacc/llvm-project/install/lib:$LD_LIBRARY_PATH

clang -fopenacc-print=omp -fopenacc-structured-ref-count-omp=no-ompx-hold openACC_code.c > openMP_code.c
