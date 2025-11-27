.. _gpu-porting:

Preparing code for GPU porting
==============================

.. questions::

   - What are the key steps involved in porting code to take advantage of GPU parallel processing capability?
   - How can I identify the computationally intensive parts of my code that can benefit from GPU acceleration?
   - What are the considerations for refactoring loops to suit the GPU architecture and improve memory access patterns?
   - Are there any tools that can translate automatically between different frameworks?

.. objectives::

   - Getting familiarized the steps involved in porting code to GPUs to take advantage of parallel processing capabilities.
   - Giving some idea about refactoring loops and modifying operations to suit the GPU architecture and improve memory access patterns.
   - Learn to use automatic translation tools to port from CUDA to HIP and from OpenACC to OpenMP

.. instructor-note::

   - 30 min teaching
   - 20 min exercises

Porting from CPU to GPU
-----------------------

When porting code to take advantage of the parallel processing capability of GPUs, several steps need to be followed and some additional work is required before writing actual parallel code to be executed on the GPUs:

* **Identify Targeted Parts**: Begin by identifying the parts of the code that contribute significantly to the execution time. These are often computationally intensive sections such as loops or matrix operations. The Pareto principle suggests that roughly 10% of the code accounts for 90% of the execution time.

* **Equivalent GPU Libraries**: If the original code uses CPU libraries like BLAS, FFT, etc, it's crucial to identify the equivalent GPU libraries. For example, `cuBLAS` or `hipBLAS` can replace CPU-based BLAS libraries. Utilizing GPU-specific libraries ensures efficient GPU utilization.

* **Refactor Loops**: When porting loops directly to GPUs, some refactoring is necessary to suit the GPU architecture. This typically involves splitting the loop into multiple steps or modifying operations to exploit the independence between iterations and improve memory access patterns. Each step of the original loop can be mapped to a kernel, executed by multiple GPU threads, with each thread corresponding to an iteration.

* **Memory Access Optimization**: Consider the memory access patterns in the code. GPUs perform best when memory access is coalesced and aligned. Minimizing global memory accesses and maximizing utilization of shared memory or registers can significantly enhance performance. Review the code to ensure optimal memory access for GPU execution.

Discussion
^^^^^^^^^^
 .. challenge:: How would this be ported?
     
    Inspect the following Fortran code (if you don't read Fortran: do-loops == for-loops)

    .. code-block:: Fortran
    
        k2 = 0
        do i = 1, n_sites
          do j = 1, n_neigh(i)
            k2 = k2 + 1
            counter = 0 
            counter2 = 0
            do n = 1, n_max
              do np = n, n_max
                do l = 0, l_max
                  if( skip_soap_component(l, np, n) )cycle
                  
                  counter = counter+1
                  do m = 0, l
                    k = 1 + l*(l+1)/2 + m
                    counter2 = counter2 + 1 
                    multiplicity = multiplicity_array(counter2)
                    soap_rad_der(counter, k2) = soap_rad_der(counter, k2) + multiplicity * real( cnk_rad_der(k, n, k2) * conjg(cnk(k, np, i)) + cnk(k, n, i) * conjg(cnk_rad_der(k, np, k2)) )
                    soap_azi_der(counter, k2) = soap_azi_der(counter, k2) + multiplicity * real( cnk_azi_der(k, n, k2) * conjg(cnk(k, np, i)) + cnk(k, n, i) * conjg(cnk_azi_der(k, np, k2)) )
                    soap_pol_der(counter, k2) = soap_pol_der(counter, k2) + multiplicity * real( cnk_pol_der(k, n, k2) * conjg(cnk(k, np, i)) + cnk(k, n, i) * conjg(cnk_pol_der(k, np, k2)) )
                  end do
                end do
              end do
            end do
          
            soap_rad_der(1:n_soap, k2) = soap_rad_der(1:n_soap, k2) / sqrt_dot_p(i) - soap(1:n_soap, i) / sqrt_dot_p(i)**3 * dot_product( soap(1:n_soap, i), soap_rad_der(1:n_soap, k2) )
            soap_azi_der(1:n_soap, k2) = soap_azi_der(1:n_soap, k2) / sqrt_dot_p(i) - soap(1:n_soap, i) / sqrt_dot_p(i)**3 * dot_product( soap(1:n_soap, i), soap_azi_der(1:n_soap, k2) )
            soap_pol_der(1:n_soap, k2) = soap_pol_der(1:n_soap, k2) / sqrt_dot_p(i) - soap(1:n_soap, i) / sqrt_dot_p(i)**3 * dot_product( soap(1:n_soap, i), soap_pol_der(1:n_soap, k2) )
        
            if( j == 1 )then
              k3 = k2
            else
              soap_cart_der(1, 1:n_soap, k2) = dsin(thetas(k2)) * dcos(phis(k2)) * soap_rad_der(1:n_soap, k2) - dcos(thetas(k2)) * dcos(phis(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2) - dsin(phis(k2)) / rjs(k2) * soap_azi_der(1:n_soap, k2)
              soap_cart_der(2, 1:n_soap, k2) = dsin(thetas(k2)) * dsin(phis(k2)) * soap_rad_der(1:n_soap, k2) - dcos(thetas(k2)) * dsin(phis(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2) + dcos(phis(k2)) / rjs(k2) * soap_azi_der(1:n_soap, k2)
              soap_cart_der(3, 1:n_soap, k2) = dcos(thetas(k2)) * soap_rad_der(1:n_soap, k2) + dsin(thetas(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2)
              soap_cart_der(1, 1:n_soap, k3) = soap_cart_der(1, 1:n_soap, k3) - soap_cart_der(1, 1:n_soap, k2)
              soap_cart_der(2, 1:n_soap, k3) = soap_cart_der(2, 1:n_soap, k3) - soap_cart_der(2, 1:n_soap, k2)
              soap_cart_der(3, 1:n_soap, k3) = soap_cart_der(3, 1:n_soap, k3) - soap_cart_der(3, 1:n_soap, k2)
            end if
          end do
        end do

   Some steps at first glance:

      * the code could (has to) be splitted in 3 kernels. Why? 
      * check if there are any variables that could lead to false dependencies between iterations, like the index `k2`
      * is it efficient for GPUs to split the work over the index `i`? What about the memory access? Note the arrays are `2D` in Fortran
      * is it possible to collapse some loops? Combining nested loops can reduce overhead and improve memory access patterns, leading to better GPU performance.
      * what is the best memory access in a GPU? Review memory access patterns in the code. Minimize global memory access by utilizing shared memory or registers where appropriate. Ensure memory access is coalesced and aligned, maximizing GPU memory throughput


.. keypoints::

   - Identify equivalent GPU libraries for CPU-based libraries and utilizing them to ensure efficient GPU utilization.
   - Importance of identifying the computationally intensive parts of the code that contribute significantly to the execution time.
   - The need to refactor loops to suit the GPU architecture.
   - Significance of memory access optimization for efficient GPU execution, including coalesced and aligned memory access patterns.


Porting between different GPU frameworks
----------------------------------------

You might also find yourself in a situation where you need to port a code from one particular 
GPU framework to another. This section gives an overview of different tools that enable converting CUDA and 
OpenACC codes to HIP and OpenMP, respectively. This conversion process enables an application to target various 
GPU architectures, specifically, NVIDIA and AMD GPUs. Here we focus on
`hipify <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html>`__ and 
`clacc <https://csmd.ornl.gov/project/clacc>`__ tools. 
This guide is adapted from the `NRIS documentation <https://documentation.sigma2.no/code_development/guides/cuda_translating-tools.html>`__.

Translating CUDA to HIP with Hipify
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we cover the use of ``hipify-perl`` and ``hipify-clang`` tools to translate a CUDA code to HIP. 

Hipify-perl
~~~~~~~~~~~

The ``hipify-perl`` tool is a script based on perl that translates CUDA syntax into HIP syntax 
(see .e.g. `here <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl>`__ for more details). 
For instance, in a CUDA code that incorporates the CUDA functions ``cudaMalloc``` and ``cudaDeviceSynchronize``, the tool will substitute ``cudaMalloc`` with the HIP function ``hipMalloc``. Similarly the CUDA function ``cudaDeviceSynchronize`` will be substituted with the HIP function ``hipDeviceSynchronize``. We list below the basic steps to run ``hipify-perl`` on LUMI-G.

- **Step 1**: Generating ``hipify-perl`` script

  .. code-block:: console
  
           $ module load rocm
           $ hipify-clang --perl

- **Step 2**: Running the generated ``hipify-perl``

  .. code-block:: console
  
           $ hipify-perl program.cu > program.cu.hip

- **Step 3**: Compiling with ``hipcc`` the generated HIP code

  .. code-block:: console
  
           $ hipcc --offload-arch=gfx90a -o program.hip.exe program.cu.hip

Despite the simplicity of the use of ``hipify-perl``, the tool might not be suitable for large applications, as it relies heavily on substituting CUDA strings with HIP strings (e.g. it substitutes ``*cuda*`` with ``*hip*``). 
In addition, ``hipify-perl`` lacks the ability of `distinguishing device/host function calls <https://docs.amd.com/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl>`_. 
The alternative here is to use the ``hipify-clang`` tool as will be described in the next section.

Hipify-clang
~~~~~~~~~~~~

As described in the `HIPIFY documentation <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl>`_, 
the ``hipify-clang`` tool is based on clang for translating CUDA sources into HIP sources. 
The tool is more robust for translating CUDA codes compared to the ``hipify-perl`` tool. 
Furthermore, it facilitates the analysis of the code by providing assistance.

In short, ``hipify-clang`` requires ``LLVM+CLANG`` and ``CUDA``. Details about building ``hipify-clang`` can be found `here <https://github.com/ROCm-Developer-Tools/HIPIFY>`__. Note that ``hipify-clang`` is available on LUMI-G. 
The issue however might be related to the installation of CUDA-toolkit. 
To avoid any eventual issues with the installation procedure we opt for CUDA singularity container. Here we present a step-by-step guide for running ``hipify-clang``:

- **Step 1**: Pulling a CUDA singularity container e.g.

  .. code-block:: console
  
           $ singularity pull docker://nvcr.io/nvidia/cuda:11.4.3-devel-ubuntu20.04

- **Step 2**: Loading a rocm module and launching the CUDA singularity

  .. code-block:: console
  
           $ module load rocm
           $ singularity shell -B $PWD,/opt:/opt cuda_11.4.0-devel-ubuntu20.04.sif
         
  where the current directory ``$PWD`` in the host is mounted to that of the container, and the directory ``/opt`` in the host is mounted to the that inside the container.

- **Step 3**: Setting the environment variable ``$PATH``.
  In order to run ``hipify-clang`` from inside the container, one can set the environment variable ``$PATH`` that defines tha path to look for the binary ``hipify-clang``.

  .. code-block:: console
  
           $ export PATH=/opt/rocm-5.2.3/bin:$PATH

  Note that the rocm version we used is ````rocm-5.2.3````.

- **Step 4**: Running ````hipify-clang```` from inside the singularity container

  .. code-block:: console
  
           $ hipify-clang program.cu -o hip_program.cu.hip --cuda-path=/usr/local/cuda-11.4 -I /usr/local/cuda-11.4/include
  
  Here the cuda path and the path to the ``*includes*`` and ``*defines*`` files should be specified. The CUDA source code and the generated output code are `program.cu` and `hip_program.cu.hip`, respectively.
  
  The syntax for the compilation process of the generated hip code is similar to the one described in the previous section (see the **Step 3** in the hipify-perl section).
  
Exercises for how to use ``Hipify-perl`` and ``Hipify-clang`` tools can be accessed `here <examples/exercise_hipify>`_.  

Translating OpenACC to OpenMP with Clacc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Clacc <https://github.com/llvm-doe-org/llvm-project/tree/clacc/main>`_ is a tool to translate an OpenACC 
application to OpenMP offloading with the Clang/LLVM compiler environment. 
Note that the tool is specific to OpenACC C, while OpenACC fortran is already supported on AMD GPU. 
As indicated in the `GitHub repository <https://github.com/llvm-doe-org/llvm-project/tree/clacc/main>`_ the compiler ``Clacc`` is the ``Clang``'s executable in the subdirectory ``\bin`` of the ``\install`` directory as described below.

In the following we present a step-by-step guide for building and using `Clacc`:

- **Step 1**: Building and installing `Clacc <https://github.com/llvm-doe-org/llvm-project/tree/clacc/main>`_.

  .. code-block:: console
  
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

- **Step 2**: Setting up environment variables to be able to work from the ``/install`` directory, which is the simplest way. We assume that the ``/install`` directory is located in the path ``/project/project_xxxxxx/Clacc/llvm-project``. 
For more advanced usage, which includes for instance modifying ``Clacc``, we refer readers to `"Usage from Build directory" <https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md>`_

  .. code-block:: console
  
           $ export PATH=/project/project_xxxxxx/Clacc/llvm-project/install/bin:$PATH
           $ export LD_LIBRARY_PATH=/project/project_xxxxxx/Clacc/llvm-project/install/lib:$LD_LIBRARY_PATH

- **Step 3**: Source to source conversion of the `openACC_code.c` code to be printed out to the file `openMP_code.c`:

  .. code-block:: console
  
           $ clang -fopenacc-print=omp -fopenacc-structured-ref-count-omp=no-ompx-hold openACC_code.c > openMP_code.c
  
  Here the flag ``-fopenacc-structured-ref-count-omp=no-ompx-hold`` is introduced to disable the ``ompx_hold`` map type modifier, which is used by the OpenACC ``copy`` clause translation. The ``ompx_hold`` is an OpenMP extension that might not be supported yet by other compilers.

- **Step 4** Compiling the code with the `cc compiler wrapper <https://docs.lumi-supercomputer.eu/development/compiling/prgenv/>`_

  .. code-block::
  
           module load CrayEnv
           module load PrgEnv-cray
           module load craype-accel-amd-gfx90a
           module load rocm
  
           cc -fopenmp -o executable openMP_code.c

Exercises for how to use ``Clacc`` tool can be accessed `here <examples/exercise_clacc>`_.

Translating CUDA to SYCL/DPC++ with SYCLomatic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intel offers a tool for CUDA-to-SYCL code migration, included in the Intel oneAPI Basekit.

It is not installed on LUMI, but the general workflow is similar to the HIPify Clang and also requires an existing CUDA installation:

  .. code-block:: console

           $ dpct program.cu 
           $ cd dpct_output/
           $ icpx -fsycl program.dp.cpp

SYCLomatic can migrate larger projects by using ``-in-root`` and ``-out-root`` flags to process directories recursively. It can also
use compilation database (supported by CMake and other build systems) to deal with more complex project layouts.

Please note that the code generated by SYCLomatic relies on oneAPI-specific extensions, and thus cannot be directly used with other
SYCL implementations, such as hipSYCL. The ``--no-incremental-migration`` flag can be added to ``dpct`` command to minimize, but not
completely avoid, the use of this compatibility layer. That would require manual effort, since some CUDA concepts cannot be directly
mapped to SYCL.

Additionally, CUDA applications might assume certain hardware behavior, such as 32-wide warps. If the target hardware is different
(e.g., AMD MI250 GPUs, used in LUMI, have warp size of 64), the algorithms might need to be adjusted manually.

Conclusion
^^^^^^^^^^

This concludes a brief overview of the usage of available tools to convert CUDA codes to HIP and SYCL, and OpenACC codes to OpenMP offloading. In general the translation process for large applications might be incomplete and thus requires manual modification to complete the porting process. It is however worth noting that the accuracy of the translation process requires that applications are written correctly according to the CUDA and OpenACC syntaxes.

See also
--------

- `Hipify GitHub <https://github.com/ROCm-Developer-Tools/HIPIFY>`_
- `HIPify Reference Guide v5.1 <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html>`_
- `HIP example <https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition>`_
- `Porting CUDA to HIP <https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP>`_
- `Clacc Main repository README <https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md>`_
- `SYCLomatic main mage <https://www.intel.com/content/www/us/en/developer/articles/technical/syclomatic-new-cuda-to-sycl-code-migration-tool.html>`_
- `SYCLomatic documentation <https://oneapi-src.github.io/SYCLomatic/get_started/index.html>`_

.. keypoints::

   - Useful tools exist to automatically translate tools from CUDA to HIP and SYCL and from OpenACC to OpenMP, but they may require manual modifications.
   
   
