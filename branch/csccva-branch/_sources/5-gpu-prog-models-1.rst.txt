.. _gpu-prog-models-1:

GPU programming models 1 (intro)
================================

.. questions::

   - q1
   - q2

.. objectives::

   - o1
   - o2

.. instructor-note::

   - X min teaching
   - X min exercises

There are different way to use GPUs for computations. In the best case someone already wrote a code it only requires setting the parameters and initial configurations. Or in some cases the problem is in such a way that it is only needed to use a library to solve the most intensive part of the code. 
However these are quite limited cases and in general some programming might be needed. There are several GPU programming software environments and APIs available such as, **directive based**, **high level frameworks**, or direct GPU programming. 


Directive based programming
---------------------------

A very fast and cheap way is to use **directive based** approaches. In this case the existing *serial* code is annotated with *hints* which indicate to the compiler which loops and regions should be executed on the GPU. In the absence of the API the directives are treated as comments and the code will just be executed as a usual serial code.  This approach is focused on productivity and easy usage, getting performance with minimum programming effort  by adding parallelism to existing code without the need to write GPU-specific code. There are two possible ways to program using directives, namely **OpenAcc** and **OpenMP**.


OpenACC
~~~~~~~~

`OpenACC <https://www.openacc.org/>`_ is  developed by a consortium formed in 2010 with the goal of developing a standard, portable, and scalable programming model for accelerators, including GPUs. Members of the OpenACC consortium include GPU vendors, such as NVIDIA and AMD, as well as leading supercomputing centers, universities, and software companies. Until recently it was supporting only Nvidia GPUs, but now there is effort to support more and more devices and architectures.

OpenMP
~~~~~~~

`OpenMP <https://www.openmp.org/>`_ started a multi-platform, shared-memory parallel programming API for multi-core CPUs and added relatively recently support for GPU offloading. It aims to support various types of GPUs. 

In theory the directive based approaches should work with both C/C++ and FORTRAN codes and third party extensions are available for other languages. 

Cross-platform portability ecosystems (portable kernels)
--------------------------------------------------------

Cross-platform portability ecosystems provide a higher-level abstraction layer which provide a convenient and portable programming model for GPU programming, and they can help reduce the time and effort required to maintain and deploy GPU-accelerated applications. The goal of these ecosystems is performance portability with a single-source application. In C++, the most notable cross-platform portability ecosystems are `Alpaka <https://alpaka.readthedocs.io/>`_, `Kokkos <https://github.com/kokkos/kokkos>`_, `RAJA <https://github.com/LLNL/RAJA>`_, and `SYCL <https://www.khronos.org/sycl/>`_.

Kokkos
~~~~~~

`Kokkos <https://github.com/kokkos/kokkos>`_ is an open-source performance portable programming model for heterogeneous parallel computing that was developed at Sandia National Laboratories. It is a C++ library that provides a high-level programming model for developing efficient and scalable parallel applications that run on many-core processors such as GPUs and CPUs. Kokkos provides a variety of abstractions, including parallel algorithms, data parallelism, and task-based parallelism, which enable developers to write portable and performant code for GPU and CPU systems. Kokkos also integrates well with other software libraries and technologies, such as MPI and OpenMP.

SYCL
~~~~

`SYCL <https://www.khronos.org/sycl/>`_ is a royalty-free, open-standard C++ programming model for multi-device programming. It provides a high-level, single-source programming model for heterogeneous systems, including GPUs. Originally SYCL was developed on top of OpenCL, however it is not limited to just that. It can be implemented on top of other low-level heterogeneous computing APIs, such as CUDA or HIP, enabling developers to write programs that can be executed on a variety of platforms. Note that while SYCL is relatively high-level model, the developers are still required to write GPU kernels explicitly.

OpenCL
~~~~~~

`OpenCL <https://www.khronos.org/opencl/>`_ (Open Computing Language) is a cross-platform, open-standard API for general-purpose parallel computing on GPUs and CPUs. It supports a wide range of hardware, including GPUs from multiple vendors. OpenCL provides a low-level programming interface for GPU programming and enables developers to write programs that can be executed on a variety of platforms. Unlike CUDA, HIP, and SYCL, OpenCL uses separate-source model. Recent versions of the OpenCL standard added C++ support for both API and the kernel code, but the C-based interface is more widely used.

Native programming models (non-portable kernels)
------------------------------------------------

When doing direct GPU programming the developer has a large level of control by writing low-level code that directly communicates with the GPU and its hardware. Theoretically direct GPU programming methods provide the ability to write low-level, GPU-accelerated code that can provide significant performance improvements over CPU-only code. However, they also require a deeper understanding of the GPU architecture and its capabilities, as well as the specific programming method being used.

CUDA
~~~~

`CUDA <https://developer.nvidia.com/cuda-toolkit>`_ is a parallel computing platform and API developed by NVIDIA. It is historically the first mainstream GPU programming framework. It allows developers to write C++-like code that is executed on the GPU. CUDA provides a set of libraries and tools for low-level GPU programming and provides a performance boost for demanding computationally-intensive applications. While there is an extensive ecosystem, CUDA is limited to the NVIDIA hardware.

HIP
~~~

`HIP <https://github.com/ROCm-Developer-Tools/HIP>`_ (Heterogeneous Interface for Portability) is an API developed by AMD that provides a high-level interface for GPU programming. HIP is designed to provide a single source code that can be used on both NVIDIA and AMD GPUs. It is based on the CUDA programming model and provides a similar programming interface to CUDA.

Each of these GPU programming environments has its own strengths and weaknesses, and the best choice for a given project will depend on a range of factors, including the hardware platforms being targeted, the type of computation being performed, and the developer's experience and preferences. High-level and productivity-focused APIs provide a simplified programming model and  maximize code portability, while low-level and performance-focused APIs provide a high level of control over the GPU's hardware but also require more coding effort and expertise.


.. keypoints::

   - k1
   - k2
