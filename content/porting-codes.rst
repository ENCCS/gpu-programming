.. translating-gpu-application:

Translating GPU-accelerated applications
========================================

.. questions::

   - q1
   - q2

.. objectives::

   - To learn about how to use the `hipify-perl` and `hipify-clang` tools to translate CUDA sources to HIP sources.
   - To learn about how to use the `clacc` tool to convert OpenACC application to OpenMP offloading.
   - To learn about how to compile the generated HIP and OpenMP codes.

.. instructor-note::

   - 20 min teaching
   - 15 min exercises

Summary 
-------

We present an overview of different tools that enable converting CUDA and OpenACC codes to HIP and OpenMP, respectively. This conversion 
process enables an application to target various GPU architectures, specifically, NVIDIA and AMD GPUs. Here we focus on
`hipify <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html>`_ and `clacc <https://csmd.ornl.gov/project/clacc>`_ tools. 
These tools have been tested on the supercomputer `LUMI-G <https://lumi-supercomputer.eu/lumi_supercomputer/>`_ 
in which the GPUs are of type `AMD MI250X GPU <https://www.amd.com/en/products/server-accelerators/instinct-mi250x>`_.

Translating CUDA to HIP with Hipify
-----------------------------------

In this section, we cover the use of `hipify-perl` and `hipify-clang` tools to translate a CUDA application to HIP. 
This guide is adapted from the `NRIS documentation <https://documentation.sigma2.no/code_development/guides/cuda_translating-tools.html>`_

Hipify-perl
~~~~~~~~~~~

The `hipify-perl` tool is a script based on perl that translates CUDA syntax into HIP syntax 
(see .e.g. `here <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl>`_ for more details). 
As an example, in a CUDA code that makes use of CUDA functions `cudaMalloc` and `cudaDeviceSynchronize`, the tool will 
replace `cudaMalloc` by the HIP function `hipMalloc`. Similarly for the CUDA function `cudaDeviceSynchronize`, which will be 
replaced by `hipDeviceSynchronize`. We list below the basic steps to run `hipify-perl`.

- **Step 1**: loading modules

On LUMI-G, the following modules need to be loaded:

```console
$module load CrayEnv
```

```console
$module load rocm
```
- **Step 2**: generating `hipify-perl` script

```console
$hipify-clang --perl
```
- **Step 3**: running `hipify-perl`

```console
$perl hipify-perl program.cu > program.cu.hip
```
- **Step 4**: compiling with `hipcc` the generated HIP code

```console
$hipcc --offload-arch=gfx90a -o program.hip.exe program.cu.hip
```
Despite the simplicity of the use of `hipify-perl`, the tool might not be suitable for large applications, as it relies heavily 
on substituting CUDA strings with HIP strings (e.g. it replaces *cuda* with *hip*). In addition, `hipify-perl` lacks the ability 
of [distinguishing device/host function calls](https://docs.amd.com/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl). 
The alternative here is to use `hipify-clang` as we shall describe in the next section.

Hipify-clang
~~~~~~~~~~~~

As described `here <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html#perl>`_, the `hipify-clang` tool
is based on clang for translating CUDA sources into HIP sources. The tool is more robust for translating CUDA codes compared to the 
`hipify-perl` tool. Furthermore, it facilitates the analysis of the code by providing assistance.

In short, `hipify-clang` requires `LLVM+CLANG` and `CUDA`. Details about building `hipify-clang` can be found 
`here <https://github.com/ROCm-Developer-Tools/HIPIFY>`_. Note that `hipify-clang` is available on LUMI-G. The issue however might be 
related to the installation of CUDA-toolkit. To avoid any eventual issues with the installation procedure we opt for CUDA singularity 
container. Here we present a step-by-step guide for running `hipify-clang`:

- **Step 1**: pulling a CUDA singularity container e.g.

```console
$singularity pull docker://nvcr.io/nvidia/cuda:11.4.0-devel-ubi8
```
- **Step 2**: loading a ROCM module before launching the container.

```console
$module load rocm
```

During our testing, we used the rocm version `rocm-5.0.2`.

- **Step 3**: launching the container

```console
$singularity shell -B $PWD,/opt:/opt cuda_11.4.0-devel-ubuntu20.04.sif
```

where the current directory `$PWD` in the host is mounted to that of the container, and the directory `/opt` in the host 
is mounted to the that inside the container.

- **Step 4**: setting the environment variable `$PATH`
In order to run `hipify-clang` from inside the container, one can set the environment variable `$PATH` that defines tha path to look 
for the binary `hipify-clang`.

```console
$export PATH=/opt/rocm-5.0.2/bin:$PATH
```

- **Step 5**: running `hipify-clang`

```console
$hipify-clang program.cu -o hip_program.cu.hip --cuda-path=/usr/local/cuda-11.4 -I /usr/local/cuda-11.4/include
```

Here the cuda path and the path to the *includes* and *defines* files should be specified. The CUDA source code and the generated 
output code are `program.cu` and `hip_program.cu.hip`, respectively.

- **Step 6**: the syntax for the compilation process of the generated hip code is similar to the one described in the previous section
(see the hipify-perl section).

Translate OpenACC to OpenMP with Clacc
--------------------------------------

`Clacc <https://github.com/llvm-doe-org/llvm-project/tree/clacc/main>`_ is a tool to translate an OpenACC application to OpenMP offloading 
with the Clang/LLVM compiler environment. Note that the tool is specific to OpenACC C. OpenACC fortran is already supported on AMD GPU. 
As indicated in the `GitHub repository <https://github.com/llvm-doe-org/llvm-project/tree/clacc/main>` 
the compiler `Clacc` is the `Clang`'s executable in the subdirectory `\bin` of the `\install` directory as described below.

In the following we present a step-by-step guide for building and using `Clacc`:

**Step 1.1**: Loading the following modules to be able to build `Clacc` (For LUMI-G):

```console
module load CrayEnv
module load rocm
```
**Step 1.2**: Building and installing `Clacc`.

```console
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
```
**Step 1.3**: Setting up environment variables to be able to work from the `/install` directory, which is the simplest way. 
For more advanced usage, which includes for instance modifying `Clacc`, we refer readers to
`"Usage from Build directory" <https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md>`_

```console
$ export PATH=`pwd`/../install/bin:$PATH
$ export LD_LIBRARY_PATH=`pwd`/../install/lib:$LD_LIBRARY_PATH
```
**Step 2.1**: To compile the translated OpenMP code, one needs first to load these modules:

```console
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
```
**Step 2.2**: Compiling & running an OpenACC code on a CPU-host:
```console
$ clang -fopenacc openACC_code.c && ./executable
```
**Step 2.3** Compiling & run an OpenACC code on AMD-GPU:
```console
$ clang -fopenacc -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a openACC_code.c && ./executable
```
**Step 2.4**
Source to source mode with `OpenMP` port printed out to the console:
```console
$ clang -fopenacc-print=omp OpenACC_code.c
```
**Step 2.5** Compiling the code with the `cc compiler wrapper <https://docs.lumi-supercomputer.eu/development/compiling/prgenv/>`_
```console
cc -fopenmp -o executable OpenMP_code.c
```

Conclusion
----------

We have presented an overview of the usage of available tools to convert CUDA codes to HIP , and OpenACC codes to OpenMP 
offloading. In general the translation process for large applications might cover about 80% of the source code and thus 
requires manual modification to complete the porting process. It is however worth noting that the accuracy of the translation process 
requires that applications are written correctly according to the CUDA and OpenACC syntaxes.

Relevant links
--------------

`Hipify GitHub <https://github.com/ROCm-Developer-Tools/HIPIFY>`_

`HIPify Reference Guide v5.1 <https://docs.amd.com/en-US/bundle/HIPify-Reference-Guide-v5.1/page/HIPify.html>`_

`HIP example <https://github.com/olcf-tutorials/simple_HIP_examples/tree/master/vector_addition>`_

`Porting CUDA to HIP <https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP>`_

`Clacc Main repository README <https://github.com/llvm-doe-org/llvm-project/blob/clacc/main/README.md>`_
