.. translating-gpu-application:

Translating GPU-accelerated applications
========================================

.. questions::

   - q1
   - q2

.. objectives::

   - To learn about how to use the ````hipify-perl```` and ````hipify-clang```` tools to translate CUDA sources to HIP sources.
   - To learn about how to use the ````clacc```` tool to convert OpenACC C application to OpenMP offloading.
   - To learn about how to compile the generated HIP and OpenMP codes.

.. instructor-note::

   - 20 min teaching
   - 15 min exercises

Summary 
-------

We present an overview of different tools that enable converting CUDA and OpenACC codes to HIP and OpenMP, respectively. This conversion 
process enables an application to target various GPU architectures, specifically, NVIDIA and AMD GPUs. Here we focus on
`hipify <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html>`_ and `clacc <https://csmd.ornl.gov/project/clacc>`_ tools. This guide is adapted from the `NRIS documentation <https://documentation.sigma2.no/code_development/guides/cuda_translating-tools.html>`_

Translating CUDA to HIP with Hipify
-----------------------------------

In this section, we cover the use of ````hipify-perl```` and ````hipify-clang```` tools to translate a CUDA code to HIP. 

Hipify-perl
~~~~~~~~~~~

The ````hipify-perl```` tool is a script based on perl that translates CUDA syntax into HIP syntax 
(see .e.g. `here <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl>`_ for more details). 
For instance, in a CUDA code that incorporates the CUDA functions `cudaMalloc` and `cudaDeviceSynchronize`, the tool will 
substitute ````cudaMalloc```` with the HIP function ````hipMalloc````. Similarly the CUDA function `cudaDeviceSynchronize` will be substituted with the HIP function `hipDeviceSynchronize`. We list below the basic steps to run ````hipify-perl```` on LUMI-G.

- **Step 1**: Generating ````hipify-perl```` script

.. code-block::

         $ module load rocm
         $ hipify-clang --perl

- **Step 2**: Running the generated ````hipify-perl````

.. code-block::

         $ hipify-perl program.cu > program.cu.hip

- **Step 3**: Compiling with ````hipcc```` the generated HIP code

.. code-block::

         $ hipcc --offload-arch=gfx90a -o program.hip.exe program.cu.hip

Despite the simplicity of the use of ````hipify-perl````, the tool might not be suitable for large applications, as it relies heavily 
on substituting CUDA strings with HIP strings (e.g. it substitutes ``*cuda*`` with ``*hip*``). In addition, ````hipify-perl```` lacks the ability of `distinguishing device/host function calls <https://docs.amd.com/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl>`_. The alternative here is to use the ````hipify-clang```` tool as we shall describe in the next section.

Hipify-clang
~~~~~~~~~~~~

As described in the `HIPIFY documentation <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl>`_, the ````hipify-clang```` tool is based on clang for translating CUDA sources into HIP sources. The tool is more robust for translating CUDA codes compared to the ````hipify-perl```` tool. Furthermore, it facilitates the analysis of the code by providing assistance.

In short, ````hipify-clang```` requires ````LLVM+CLANG```` and ````CUDA````. Details about building ````hipify-clang```` can be found 
`here <https://github.com/ROCm-Developer-Tools/HIPIFY>`_. Note that ````hipify-clang```` is available on LUMI-G. The issue however might be related to the installation of CUDA-toolkit. To avoid any eventual issues with the installation procedure we opt for CUDA singularity 
container. Here we present a step-by-step guide for running ````hipify-clang````:

- **Step 1**: Pulling a CUDA singularity container e.g.

.. code-block::

         $ singularity pull docker://nvcr.io/nvidia/cuda:11.4.3-devel-ubuntu20.04

- **Step 3**: Loading a rocm module and launching the CUDA singularity

.. code-block::

         $ module load rocm
         $ singularity shell -B $PWD,/opt:/opt cuda_11.4.0-devel-ubuntu20.04.sif
         
where the current directory ````$PWD```` in the host is mounted to that of the container, and the directory ````/opt```` in the host 
is mounted to the that inside the container.

- **Step 4**: Setting the environment variable ````$PATH````
In order to run ````hipify-clang```` from inside the container, one can set the environment variable ````$PATH```` that defines tha path to look for the binary ````hipify-clang````.

.. code-block::

         $ export PATH=/opt/rocm-5.2.3/bin:$PATH

Note that the rocm version we used is ````rocm-5.2.3````.

- **Step 5**: Running ````hipify-clang```` from inside the singularity container

.. code-block::

         $ hipify-clang program.cu -o hip_program.cu.hip --cuda-path=/usr/local/cuda-11.4 -I /usr/local/cuda-11.4/include

Here the cuda path and the path to the ``*includes*`` and ``*defines*`` files should be specified. The CUDA source code and the generated output code are `program.cu` and `hip_program.cu.hip`, respectively.

The syntax for the compilation process of the generated hip code is similar to the one described in the previous section
(see the **Step 3** in the hipify-perl section).

Translating OpenACC to OpenMP with Clacc
----------------------------------------

`Clacc <https://github.com/llvm-doe-org/llvm-project/tree/clacc/main>`_ is a tool to translate an OpenACC application to OpenMP offloading with the Clang/LLVM compiler environment. Note that the tool is specific to OpenACC C, while OpenACC fortran is already supported on AMD GPU. As indicated in the `GitHub repository <https://github.com/llvm-doe-org/llvm-project/tree/clacc/main>`_ 
the compiler ````Clacc```` is the ````Clang````'s executable in the subdirectory ````\bin```` of the ````\install```` directory as described below.

In the following we present a step-by-step guide for building and using `Clacc`:

Building ````Clacc````
~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Building and installing `Clacc <https://github.com/llvm-doe-org/llvm-project/tree/clacc/main>`_.

.. code-block::

         $ git clone -b clacc/main https://github.com/llvm-doe-org/llvm-project.git
         $ cd llvm-project
         $ mkdir build && cd build
         $ cmake -DCMAKE_INSTALL_PREFIX=../install     \
            -DCMAKE_BUILD_TYPE=Release            \
            -DLLVM_ENABLE_PROJECTS="clang;lld"    \
            -DLLVM_ENABLE_RUNTIMES=openmp         \
            -DLLVM_TARGETS_TO_BUILD="host;AMDGPU" \
            -DCMAKE_C_COMPILER=gcc                \
            -DCMAKE_CXX_COMPILER=g++              \
            ../llvm
         $ make
         $ make install

**Step 2**: Setting up environment variables to be able to work from the ````/install```` directory, which is the simplest way. We assume that the ````/install```` directory is located in the path ````/project/project_xxxxxx/Clacc/llvm-project````. 
For more advanced usage, which includes for instance modifying ````Clacc````, we refer readers to
`"Usage from Build directory" <https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md>`_

.. code-block::

         $ export PATH=/project/project_xxxxxx/Clacc/llvm-project/install/bin:$PATH
         $ export LD_LIBRARY_PATH=/project/project_xxxxxx/Clacc/llvm-project/install/lib:$LD_LIBRARY_PATH

**Step 3**: Source to source conversion of the `openACC_code.c` code to be printed out to the file `openMP_code.c`:

.. code-block:: 

         $ clang -fopenacc-print=omp -fopenacc-structured-ref-count-omp=no-ompx-hold openACC_code.c > openMP_code.c

Here the flag ````-fopenacc-structured-ref-count-omp=no-ompx-hold```` is introduced to disable the ````ompx_hold```` map type modifier, which is used by the OpenACC ````copy```` clause translation. The ````ompx_hold```` is an OpenMP extension that might not be supported yet by other compilers.

**Step 4** Compiling the code with the `cc compiler wrapper <https://docs.lumi-supercomputer.eu/development/compiling/prgenv/>`_

.. code-block::

         module load CrayEnv
         module load PrgEnv-cray
         module load craype-accel-amd-gfx90a
         module load rocm

         cc -fopenmp -o executable openMP_code.c


Conclusion
----------

We have presented an overview of the usage of available tools to convert CUDA codes to HIP, and OpenACC codes to OpenMP 
offloading. In general the translation process for large applications might be incomplete and thus 
requires manual modification to complete the porting process. It is however worth noting that the accuracy of the translation process 
requires that applications are written correctly according to the CUDA and OpenACC syntaxes.

Relevant links
--------------

`Hipify GitHub <https://github.com/ROCm-Developer-Tools/HIPIFY>`_

`HIPify Reference Guide v5.1 <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html>`_

`HIP example <https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition>`_

`Porting CUDA to HIP <https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP>`_

`Clacc Main repository README <https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md>`_
