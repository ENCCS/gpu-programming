#Converting OpenACC code to OpenMP with clacc

The folder /exercise_clacc contains a OpenACC code (named openACC_code.c) and the scripts - run-clang_acc2omp.sh; compile_omp.sh and script_omp.slurm

#The script "run-clang_acc2omp.sh" is used for converting OpenACC code to OpenMP

#The script "compile_omp.sh" is for compiling the generated OpenMP code

#The script "script_omp.slurm" is for submitting a Slurm job


Here are the step-by-step guide to convert OpenACC code to OpenMP


- Step 0: copy the folder /project/project_465002387/exercise_clacc to your path

- Step 1: Convert OpenACC code to OpenMP
  cd exercise_clacc
  ./run-clang_acc2omp.sh

  This generate a code named "openMP_code.c"

- Step 2: Compile the generated OpenMP "openMP_code.c"
  ./compile_omp.sh

  This generates the executable named "executable.omp.exe"  

- Step 3: Submit a job
  sbatch script_omp.slurm
