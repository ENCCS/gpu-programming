#Converting CUDA code to HIP with hipify-perl

The folder "/exercise_hipify/Hipify_perl" contains a cuda example named "vec_add_cuda.cu", in addition to the following script
"generate_hipify-perl.sh"; "run_hipify-perl"; "compile.sh" and "script.slurm"

#The script "generate_hipify-perl.sh" generates "hipify-perl" file

#The script "run_hipify-perl" converts CUDA code to HIP code

#The script "compile.sh" compiles the generated HIP code

#The script "script.slurm" is Slurm job


Here are the step-by-step guide to convert CUDA code to HIP


- Step 0: copy the folder /project/project_465002387/exercise_hipify/Hipify_perl to your path

- Step 1: Generate "hipify-perl"
  cd exercise_hipify/Hipify_perl
  ./generate_hipify-perl.sh

  This generates a file named "hipify-perl"

- Step 2: Convert CUDA code "vec_add_cuda.cu" to HIP code "vec_add_cuda.cu.hip"
  ./run_hipify-perl

  This generates the HIP code "vec_add_cuda.cu.hip"

- Step 3: Compile the generated HIP code "vec_add_cuda.cu.hip"
  ./compile.sh

  This generates the executable named "executable.hip.exe"

- Step 4: Submit a job
  sbatch script.slurm
