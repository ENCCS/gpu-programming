.. _intro-to-gpu-prog-models:

Introduction to GPU programming models
======================================

.. questions::

   - What are the key differences between different GPU programming approaches?
   - How should I choose which framework to use for my project?

.. objectives::

   - Understand basic examples in different GPU programming frameworks
   - Perform a quick cost-benefit analysis in the context of own code projects

.. instructor-note::

   - X min teaching
   - X min exercises

There are different ways to use GPUs for computations. In the best case, when someone has already written the code, one only needs to set the parameters and initial configurations in order to get started. Or in some cases the problem is in such a way that it is only needed to use a library to solve the most intensive part of the code. 
However these are quite limited cases and in general some programming might be needed. There are several GPU programming software environments and APIs available such as, **directive-based models**, **non-portable kernel-based models**, and **portable kernel-based models**, as well as high-level frameworks and libraries.

Standard C++/Fortran
--------------------

Programs written in standard C++ and Fortran languages can now take advantage of NVIDIA GPUs without
depending of any external library. This is possible thanks to the `NVIDIA SDK <https://developer.nvidia.com/hpc-sdk>`_
suite of compilers that translates and optimizes the code for running on GPUs. Guidelines for writing C++ code
can be found `here <https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/>`_ while
those for Fortran code can be found `here <https://developer.nvidia.com/blog/accelerating-fortran-do-concurrent-with-gpus-and-the-nvidia-hpc-sdk/>`_.
The performance of these two approaches is promising as it can be seen in the examples provided in those
guidelines.

Directive-based programming
---------------------------

A fast and cheap way is to use **directive based** approaches. In this case the existing *serial* code is annotated with *hints* which indicate to the compiler which loops and regions should be executed on the GPU. In the absence of the API the directives are treated as comments and the code will just be executed as a usual serial code. This approach is focused on productivity and easy usage (but to the detriment of performance), and allows employing accelerators with minimum programming effort by adding parallelism to existing code without the need to write accelerator-specific code. There are two common ways to program using directives, namely **OpenAcc** and **OpenMP**.


OpenACC
~~~~~~~~

`OpenACC <https://www.openacc.org/>`_ is  developed by a consortium formed in 2010 with the goal of developing a standard, portable, and scalable programming model for accelerators, including GPUs. Members of the OpenACC consortium include GPU vendors, such as NVIDIA and AMD, as well as leading supercomputing centers, universities, and software companies. Until recently it was supporting only Nvidia GPUs, but now there is effort to support more and more devices and architectures.

OpenMP
~~~~~~~

`OpenMP <https://www.openmp.org/>`_ started a multi-platform, shared-memory parallel programming API for multi-core CPUs and added relatively recently support for GPU offloading. It aims to support various types of GPUs. 

In theory the directive based approaches should work with both C/C++ and FORTRAN codes and third party extensions are available for other languages. 

Non-portable kernel-based models (native programming models)
------------------------------------------------------------

When doing direct GPU programming the developer has a large level of control by writing low-level code that directly communicates with the GPU and its hardware. Theoretically direct GPU programming methods provide the ability to write low-level, GPU-accelerated code that can provide significant performance improvements over CPU-only code. However, they also require a deeper understanding of the GPU architecture and its capabilities, as well as the specific programming method being used.

CUDA
~~~~

`CUDA <https://developer.nvidia.com/cuda-toolkit>`_ is a parallel computing platform and API developed by NVIDIA. It is historically the first mainstream GPU programming framework. It allows developers to write C++-like code that is executed on the GPU. CUDA provides a set of libraries and tools for low-level GPU programming and provides a performance boost for demanding computationally-intensive applications. While there is an extensive ecosystem, CUDA is limited to the NVIDIA hardware.

HIP
~~~

`HIP <https://github.com/ROCm-Developer-Tools/HIP>`_ (Heterogeneous Interface for Portability) is an API developed by AMD that provides a low-level interface for GPU programming. HIP is designed to provide a single source code that can be used on both NVIDIA and AMD GPUs. It is based on the CUDA programming model and provides an almost identical programming interface to CUDA.


Portable kernel-based models (cross-platform portability ecosystems)
--------------------------------------------------------------

Cross-platform portability ecosystems typically provide a higher-level abstraction layer which provide a convenient and portable programming model for GPU programming. They can help reduce the time and effort required to maintain and deploy GPU-accelerated applications. The goal of these ecosystems is achieving performance portability with a single-source application. In C++, the most notable cross-platform portability ecosystems are `Alpaka <https://alpaka.readthedocs.io/>`_, `Kokkos <https://github.com/kokkos/kokkos>`_, `OpenCL <https://www.khronos.org/opencl/>`_ (C and C++ APIs), `RAJA <https://github.com/LLNL/RAJA>`_, and `SYCL <https://www.khronos.org/sycl/>`_.

Kokkos
~~~~~~

`Kokkos <https://github.com/kokkos/kokkos>`_ is an open-source performance portable programming model for heterogeneous parallel computing that has been so far mostly developed at Sandia National Laboratories. It is a C++-based ecosystem that provides a programming model for developing efficient and scalable parallel applications that run on many-core architectures such as CPUs, GPUs, and FPGAs. The Kokkos ecosystem consists of several components, such as the Kokkos core library, which provides parallel execution and memory abstraction, the Kokkos kernels library, which provides math kernels for linear algebra and graph algorithms, and the Kokkos tools library, which provides profiling and debugging tools. Kokkos components integrate well with other software libraries and technologies, such as MPI and OpenMP. Furthermore, the project collaborates with other projects, in order to provide interoperability and standardization for portable C++ programming.


OpenCL
~~~~~~

`OpenCL <https://www.khronos.org/opencl/>`_ (Open Computing Language) is a cross-platform, open-standard API for general-purpose parallel computing on CPUs, GPUs and FPGAs. It supports a wide range of hardware from multiple vendors. OpenCL provides a low-level programming interface for GPU programming and enables developers to write programs that can be executed on a variety of platforms. Unlike programming models such as CUDA, HIP, Kokkos, and SYCL, OpenCL uses a separate-source model. Recent versions of the OpenCL standard added C++ support for both API and the kernel code, but the C-based interface is still more widely used.

SYCL
~~~~

`SYCL <https://www.khronos.org/sycl/>`_ is a royalty-free, open-standard C++ programming model for multi-device programming. It provides a high-level, single-source programming model for heterogeneous systems, including GPUs. Originally SYCL was developed on top of OpenCL, however it is not limited to just that. It can be implemented on top of other low-level heterogeneous computing APIs, such as CUDA or HIP, enabling developers to write programs that can be executed on a variety of platforms. Note that while SYCL is relatively high-level model, the developers are still required to write GPU kernels explicitly.


High-level language support
---------------------------

WRITEME: General paragraph about modern GPU libraries for high-level languages:

- Python
- Julia
- SYCL


Cost-benefit analysis
---------------------

WRITEME begin

- how to choose between frameworks?
- depends on:

  - specifics of the problem at hand
  - whether starting from scratch or from existing code
  - background knowledge of programmer
  - how much time can be invested
  - performance needs

WRITEME end


Summary
-------

Each of these GPU programming environments has its own strengths and weaknesses, and the best choice for a given project will depend on a range of factors, including the hardware platforms being targeted, the type of computation being performed, and the developer's experience and preferences. High-level and productivity-focused APIs provide a simplified programming model and  maximize code portability, while low-level and performance-focused APIs provide a high level of control over the GPU's hardware but also require more coding effort and expertise.




.. keypoints::

   - k1
   - k2