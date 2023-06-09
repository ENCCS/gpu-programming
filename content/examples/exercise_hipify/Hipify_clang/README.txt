#Converting CUDA code to HIP with hipify-clang

The folder "/exercise_hipify/Hipify_clang" contains a cuda example named "vec_add_cuda.cu", in addition to the following script
 "run_hipify-clang"; "compile.sh" and "script.slurm"

#The script "run_hipify-clang" converts CUDA code to HIP code

#The script "compile.sh" compiles the generated HIP code

#The script "script.slurm" is Slurm job


Here are the step-by-step guide to convert CUDA code to HIP


- Step 0: copy the folder /project/project_465000485/exercise_hipify/Hipify_clang to your path

- Step 1: Convert CUDA code "vec_add_cuda.cu" to HIP code "vec_add_cuda.cu.hip"
  ./run_hipify-clang

  This generates the HIP code "vec_add_cuda.cu.hip"

- Step 3: Compile the generated HIP code "vec_add_cuda.cu.hip"
  ./compile.sh

  This generates the executable named "executable.hip.exe"

- Step 4: Submit a job
  sbatch script.slurm
