.. _gpu-ecosystem:

The GPU hardware and software ecosystem
=======================================

.. questions::

   - q1
   - q2

.. objectives::

   - o1
   - o2

.. instructor-note::

   - 20 min teaching
   - 0 min exercises


Overview of GPU hardware
------------------------

.. figure:: img/hardware/CPUAndGPU.png
    :align: center

    A comparison of the CPU and GPU architecture.
    CPU (left) has complex core structure and pack several cores on a single chip.
    GPU cores are very simple in comparison, they also share data and control between each other.
    This allows to pack more cores on a single chip, thus achieving very high compute density.

.. admonition:: In short
   :class: dropdown

   - Accelerators offer high performance due to their scalability and high density of compute elements.
   - They have separate circuit boards connected to CPUs via PCIe bus, with their own memory.
   - CPUs copy data from their own memory to the GPU memory, execute the program, and copy the results back.
   - GPUs run thousands of threads simultaneously, quickly switching between them to hide memory operations.
   - Effective data management and access pattern is critical on the GPU to avoid running out of memory.

One of the most important features that allows the accelerators to reach this high performance is their scalability.
Computational cores on accelerators are usually grouped into multiprocessors.
The multiprocessors share the data and logical elements.
This allows to achieve a very high density of a compute elements on a GPU.
This also allows for better scaling: more multiprocessors means more raw performance and this is very easy to achieve with more transistors available.

Accelerators are a separate main circuit board with the processor, memory, power management, etc.
It is connected to the motherboard with CPUs via PCIe bus.
Having its own memory means that the data has to be copied to and from it.
CPU acts as a main processor, controlling the execution workflow.
It copies the data from its own memory to the GPU memory, executes the program and copies the results back.
GPUs runs tens of thousands of threads simultaneously on thousands of cores and does not do much of the data management.
With many cores trying to access the memory simultaneously and with little cache available, the accelerator can run out of memory very quickly.
This makes the data management and its access pattern is essential on the GPU.
Accelerators like to be overloaded with the number of threads, because they can switch between threads very quickly.
This allows to hide the memory operations: while some threads wait, others can compute.


How do GPUs differ from CPUs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CPUs and GPUs were designed with different goals in mind. While the CPU 
is designed to excel at executing a sequence of operations, called a thread, 
as fast as possible and can execute a few tens of these threads in parallel, 
the GPU is designed to excel at executing many thousands of them in parallel. 
GPUs were initially developed for highly-parallel task of graphic processing 
and therefore designed such that more transistors are devoted to data processing 
rather than data caching and flow control. More transistors dedicated to 
data processing is beneficial for highly parallel computations; the GPU can 
hide memory access latencies with computation, instead of relying on large data caches 
and complex flow control to avoid long memory access latencies, 
both of which are expensive in terms of transistors.




.. list-table::  
   :widths: 100 100
   :header-rows: 1

   * - CPU
     - GPU
   * - General purpose
     - Highly specialized for parallelism
   * - Good for serial processing
     - Good for parallel processing
   * - Great for task parallelism
     - Great for data parallelism
   * - Low latency per thread
     - High-throughput
   * - Large area dedicated cache and control
     - Hundreds of floating-point execution units



GPU platforms
-------------

GPUs come together with software stacks or APIs that  work in conjunction with the hardware and give a standard way for the software to interact with the GPU hardware. They  are used by software developers to write code that can take advantage of the parallel processing power of the GPU, and they provide a standard way for software to interact with the GPU hardware. Typically, they provide access to low-level functionality, such as memory management, data transfer between the CPU and the GPU, and the scheduling and execution of parallel processing tasks on the GPU. They may also provide higher level functions and libraries optimized for specific HPC  workloads, like linear algebra or fast Fourier transforms. Finally, in order to facilitate the developers to optimize and write correct codes, debugging  and profiling tools are also included. 

*Nvidia*, *AMD*, and *Intel* are the major companies which design and produces GPUs for HPC providing each its own suit **CUDA**, **ROCm**, and respectively **OneAPI**. This way they can offer optimization, differentiation (offering unique features tailored to their devices), vendor lock-in, licensing, and royalty fees, which can result in better performance, profitability, and customer loyalty. 
There are also cross-platform APIs such **DirectCompute** (only for Windows operating system), **OpenCL**, and **SYCL**.


CUDA
^^^^

**Compute Unified Device Architecture** is the parallel computing platform from Nvidia. The CUDA API provides a comprehensive set of functions and tools for developing high-performance applications that run on NVIDIA GPUs. It consists of two main components: the CUDA Toolkit and the CUDA driver. The toolkit provides a set of libraries, compilers, and development tools for programming and optimizing CUDA applications, while the driver is responsible for communication between the host CPU and the GPU. CUDA is designed to work with programming languages such as C, C++, and Fortran.

CUDA API provides many highly optimize libraries such as: **cuBLAS** (for linear algebra operatiosn, such a dense matrix multiplication), **cuFFT** (for performing fast Fourier transforms), **cuRAND** (for generating pseudo-randonm numbers), **cuSPARSE** (for sparse matrices operations). Using these libraries, developers can quickly and easily accelerate complex computations on NVIDIA GPUs without having to write low-level GPU code themselves.

There are several compilers that can be used for developing and eecuting code on Nvidia GPUs: **nvcc**. The latest versions are based on the widely used LLVM open source compiler infrastructure. nvcc produces optimized code for NVIDIA GPUs and drives a supported host compiler for AMD, Intel, OpenPOWER, and Arm CPUs.

In addition to this are provided **nvc** (C11 compiler), **nvc++** (C++17 compiler), and  **nvfortran** (ISO Fortran 2003 compiler). These compilers can as well create code for execution on the Nvidia GPUs, and also support GPU and multicore CPU programming with parallel language features, OpeanACC and OpenMP.


When programming mistakes are inevitable they have to be fixed as soon as possible. The CUDA toolkit includes the command line tool **cuda-gdb** which can be used to find errors in the code. It is an extension to GDB, the GNU Project debugger.  The existing GDB debugging features are inherently present for debugging the host code, and additional features have been provided to support debugging CUDA device code, allowing simultaneous debugging of both GPU and CPU code within the same application. The tool provides developers with a mechanism for debugging CUDA applications running on actual hardware. This enables developers to debug applications without the potential variations introduced by simulation and emulation environments.

In addition to this the command line tool **compute-sanitizer** can be used to look esclusively for memory access problems: unallocated buffers, out of bounds accesses, race conditions, and unitilized variables. 

Finally, in order to utilize the GPUs at maximum some performance analysis tools. Nvidia provides NVIDIA Nsight Systems and NVIDIA Nsight Compute tools for helping the developers to optimize their applications. The former, NVIDIA Nsight Systems, is a system-wide performance analysis tool that  provides detailed metrics on both CPU and GPU usage, memory bandwidth, and other system-level metrics. The latter, NVIDIA Nsight Compute, is a kernel-level performance analysis tool that allows developers to analyze the performance of individual CUDA kernels. It provides detailed metrics on kernel execution, including memory usage, instruction throughput, and occupancy. These tools have grahical which can be used for all steps of the performance analysis, however on supercomputers it is recommended to use the command line interface for collecting the information needed and then visualize and analyse the results using the graphical interface on personal computers.

Apart from what was presented above there are many others tools and features provided by Nvidia. The CUDA eco-system is very well developed. 

.. admonition:: In short
   :class: dropdown

   - CUDA: Nvidia's parallel computing platform
      - Components: CUDA Toolkit & CUDA driver
      - Supports C, C++, and Fortran languages
   - CUDA API Libraries: cuBLAS, cuFFT, cuRAND, cuSPARSE
      - Accelerate complex computations on GPUs
   - Compilers: nvcc, nvc, nvc++, nvfortran
      - Support GPU and multicore CPU programming
      - Compatible with OpenACC and OpenMP
   - Debugging tools: cuda-gdb, compute-sanitizer
      - Debug GPU and CPU code simultaneously
      - Identify memory access issues
   - Performance analysis tools: NVIDIA Nsight Systems, NVIDIA Nsight Compute
      - Analyze system-wide and kernel-level performance
      - Optimize CPU and GPU usage, memory bandwidth, instruction throughput
   - Comprehensive CUDA ecosystem with extensive tools and features

ROCm
^^^^
ROCm is an open software platform allowing researchers to tap the power of AMD accelerators. The ROCm platform is built on the foundation of open portability, supporting environments across multiple accelerator vendors and architectures. In some way it is very similar to CUDA API. It contains libraries, compilers, and development tools for programming and optimizing programs for AMD GPUs. for debugging provide the command line tool `roc-gdb`, while for perforance analysis `roc-prof` and `roctracer`.  

In order to produce code for the AMD GPUs one can Heterogeneous-Computing Interface for Portability (HIP). HIP provides source portability to either Nvidia or AMD plaform. It  provides the hipcc compiler driver, which will call the appropriate toolchain depending on the desired platform. On the AMD ROCm platform, HIP provides a header and runtime library built on top of the HIP-Clang compiler. On an Nvidia platform, HIP provides header file which translate from the HIP runtime APIs to CUDA runtime APIs. The header file contains mostly inlined functions and thus has very low overhead. The code is then compiled with nvcc, the standard C++ compiler provided with the CUDA.

On AMD platforms libraries are prefixed by `roc`, which can be called directly from HIP. In order to make portable call one call the libraries using `hip` prefixed wrappers. These can be used at no perfomrance cost and insure that a HIP code can be use on other platforms with no changes. 

OneAPI
^^^^^^



GPUs in general support different features, even among the same producers. In general newer cards come with extra features and sometimes old features are not supported anymore. It is important when compiling to create binaries targeting the specific architecture when compiling. A binary built for a newer card will not run on older devices, while a binary build for older devices might not run efficiently on newer architectures. In CUDA the compute capability which is targeted is specified by the `-arch=sm_XY`, where `X` specifies the major architecture and it is between 1 and 9, and `Y`the minor. When using HIP on Nvidia platforms one needs to use compiling optoin `--gpu-architecture=sm_XY`, while on AMD platforms  `--offload-arch=gfxabc`( where `abc` is the architecture code such as `90a` for the MI200 series or `908` for MI100 series).


Hardware
^^^^^^^^


.. list-table:: Hardware
   :widths: 25 25 50
   :header-rows: 1

   * - Nvidia
     - AMD
     - Intel
   * - streaming processor/streaming core
     - SIMD lane
     - processing element
   * - SIMT unit
     - SIMD unit
     - 
   * - streaming multiprocessor (SMP)
     - computing unit (CU)
     - execution unit (EU)



Summary
-------

- GPUs are highly parallel devices that can execute certain parts of the program in many parallel threads.
- CPU controls the works flow and makes all the allocations and data transfers.
- In order to use the GPU efficiently, one has to split their the problem  in many parts that can run simultaneously.


.. keypoints::

   - k1
   - k2
