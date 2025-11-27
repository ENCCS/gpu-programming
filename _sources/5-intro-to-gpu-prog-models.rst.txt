.. _intro-to-gpu-prog-models:


Introduction to GPU programming models
======================================

.. questions::

   - What are the key differences between different GPU programming approaches?
   - How should I choose which framework to use for my project?

.. objectives::

   - Understand the  basic ideas in different GPU programming frameworks
   - Perform a quick cost-benefit analysis in the context of own code projects

.. instructor-note::

   - 20 min teaching
   - 10 min discussion


There are different ways to use GPUs for computations. In the best case, when the code has already been written, one only needs to set the parameters and initial configuration in order to get started. In some other cases the problem is posed in such a way that a third-party library can be used to solve the most intensive part of the code (for example, this is increasingly the case with machine-learning workflows in Python). 
However, these cases are stil quite limited; in general, some additional programming might be needed. There are many GPU programming software environments and APIs available, which can be broadly grouped into **directive-based models**, **non-portable kernel-based models**, and **portable kernel-based models**, as well as high-level frameworks and libraries (including attempts at language-level support).


Standard C++/Fortran
--------------------

Programs written in standard C++ and Fortran languages can now take advantage of NVIDIA GPUs without depending on any external library. This is possible thanks to the `NVIDIA SDK <https://developer.nvidia.com/hpc-sdk>`__ suite of compilers that translates and optimizes the code for running on GPUs.

- `Here <https://developer.nvidia.com/blog/developing-accelerated-code-with-standard-language-parallelism/>`_ is the series of articles on acceleration with standard language parallelism.
- Guidelines for writing C++ code can be found `here <https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/>`__, 
- while those for Fortran code can be found `here <https://developer.nvidia.com/blog/accelerating-fortran-do-concurrent-with-gpus-and-the-nvidia-hpc-sdk/>`__.

The performance of these two approaches is promising, as can be seen in the examples provided in those guidelines.


Directive-based programming
---------------------------

A fast and cheap way is to use **directive based** approaches. In this case the existing *serial* code is annotated with *hints* which indicate to the compiler which loops and regions should be executed on the GPU. In the absence of the API the directives are treated as comments and the code will just be executed as a usual serial code. This approach is focused on productivity and easy usage (but to the possible detriment of performance), and allows employing accelerators with minimal programming effort by adding parallelism to existing code without the need to write accelerator-specific code. There are two common ways to program using directives, namely **OpenACC** and **OpenMP**.


OpenACC
~~~~~~~

`OpenACC <https://www.openacc.org/>`_ is developed by a consortium formed in 2010 with the goal of developing a standard, portable, and scalable programming model for accelerators, including GPUs. Members of the OpenACC consortium include GPU vendors, such as NVIDIA and AMD, as well as leading supercomputing centers, universities, and software companies. Until recently it was supporting only NVIDIA GPUs, but now there is effort to support more devices and architectures.


OpenMP
~~~~~~

`OpenMP <https://www.openmp.org/>`_ started as a multi-platform, shared-memory parallel programming API for multi-core CPUs and relatively recently has added support for GPU offloading. OpenMP aims to support various types of GPUs, which is done through the parent compiler. 

The directive based approaches work with C/C++ and FORTRAN codes, while some third party extensions are available for other languages. 


Non-portable kernel-based models (native programming models)
------------------------------------------------------------

When doing direct GPU programming the developer has a large level of control by writing low-level code that directly communicates with the GPU and its hardware. Theoretically direct GPU programming methods provide the ability to write low-level, GPU-accelerated code that can provide significant performance improvements over CPU-only code. However, they also require a deeper understanding of the GPU architecture and its capabilities, as well as the specific programming method being used.

CUDA
~~~~

`CUDA <https://developer.nvidia.com/cuda-toolkit>`_ is a parallel computing platform and API developed by NVIDIA. It is historically the first mainstream GPU programming framework. It allows developers to write C-like code that is executed on the GPU. CUDA provides a set of libraries and tools for low-level GPU programming and provides a performance boost for demanding computationally-intensive applications. While there is an extensive ecosystem, CUDA is restricted to NVIDIA hardware. 


HIP
~~~

`HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html>`_ (Heterogeneous Interface for Portability) is an API developed by AMD that provides a low-level interface for GPU programming. HIP is designed to provide a single source code that can be used on both NVIDIA and AMD GPUs. It is based on the CUDA programming model and provides an almost identical programming interface to CUDA.

Multiple examples of CUDA/HIP code are available in the `content/examples/cuda-hip <https://github.com/ENCCS/gpu-programming/tree/main/content/examples/cuda-hip>`__ directory of this repository.


Portable kernel-based models (cross-platform portability ecosystems)
--------------------------------------------------------------------

Cross-platform portability ecosystems typically provide a higher-level abstraction layer which enables a convenient and portable programming model for GPU programming. They can help reduce the time and effort required to maintain and deploy GPU-accelerated applications. The goal of these ecosystems is to achieve performance portability with a single-source application. In C++, the most notable cross-platform portability ecosystems are `SYCL <https://www.khronos.org/sycl/>`_, `OpenCL <https://www.khronos.org/opencl/>`_ (C and C++ APIs), and `Kokkos <https://github.com/kokkos/kokkos>`_; others include `alpaka <https://alpaka.readthedocs.io/>`_ and `RAJA <https://github.com/LLNL/RAJA>`_.


OpenCL
~~~~~~

`OpenCL <https://www.khronos.org/opencl/>`_ (Open Computing Language) is a cross-platform, open-standard API for general-purpose parallel computing on CPUs, GPUs and FPGAs. It supports a wide range of hardware from multiple vendors. OpenCL provides a low-level programming interface for GPU programming and enables developers to write programs that can be executed on a variety of platforms. Unlike programming models such as CUDA, HIP, Kokkos, and SYCL, OpenCL uses a separate-source model. Recent versions of the OpenCL standard added C++ support for both API and the kernel code, but the C-based interface is still more widely used. 
The OpenCL Working Group doesn’t provide any frameworks of its own. Instead, vendors who produce OpenCL-compliant devices release frameworks as part of their software development kits (SDKs). The two most popular OpenCL SDKs are released by NVIDIA and AMD. In both cases, the development kits are free and contain the libraries and tools that make it possible to build OpenCL applications.


Kokkos
~~~~~~

`Kokkos <https://github.com/kokkos/kokkos>`_ is an open-source performance portable programming model for heterogeneous parallel computing that has been mainly developed at Sandia National Laboratories. It is a C++-based ecosystem that provides a programming model for developing efficient and scalable parallel applications that run on many-core architectures such as CPUs, GPUs, and FPGAs. The Kokkos ecosystem consists of several components, such as the Kokkos core library, which provides parallel execution and memory abstraction, the Kokkos kernel library, which provides math kernels for linear algebra and graph algorithms, and the Kokkos tools library, which provides profiling and debugging tools. Kokkos components integrate well with other software libraries and technologies, such as MPI and OpenMP. Furthermore, the project collaborates with other projects, in order to provide interoperability and standardization for portable C++ programming.

alpaka
~~~~~~

`alpaka <https://alpaka.readthedocs.io/>`_ (Abstraction Library for Parallel Kernel Acceleration) is an open-source C++ header-only library that aims to provide performance portability across heterogeneous accelerator architectures by abstracting the underlying levels of parallelism. The library is platform-independent and supports the concurrent and cooperative use of multiple devices, including host CPUs (x86, ARM, RISC-V) and GPUs from different vendors (NVIDIA, AMD, and Intel).

A key advantage of alpaka is that it requires only a single implementation of a user kernel, expressed as a function object with a standardized interface. This eliminates the need to write specialized code for different backends. The library provides a variety of accelerator backends, including CUDA, HIP, SYCL, OpenMP, and serial execution, that can be selected based on the target device. Moreover, multiple accelerator backends can even be combined to target different vendor hardware within a single application.

SYCL
~~~~

`SYCL <https://www.khronos.org/sycl/>`_ is a royalty-free, open-standard C++ programming model for multi-device programming. It provides a high-level, single-source programming model for heterogeneous systems, including GPUs. Originally SYCL was developed on top of OpenCL; however, it is no more limited to just that. It can be implemented on top of other low-level heterogeneous computing APIs, such as CUDA or HIP, enabling developers to write programs that can be executed on a variety of platforms. Note that while SYCL is relatively high-level model, the developers are still required to write GPU kernels explicitly.

While alpaka, Kokkos, and RAJA refer to specific projects, SYCL itself is only a standard, for which several implementations exist. For GPU programming, `Intel oneAPI DPC++ <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`_ (supporting Intel GPUs natively, and NVIDIA and AMD GPUs with `Codeplay oneAPI plugins <https://codeplay.com/solutions/oneapi/>`_) and `AdaptiveCpp <https://github.com/AdaptiveCpp/AdaptiveCpp/>`_ (previously also known as hipSYCL or Open SYCL, supporting NVIDIA and AMD GPUs, with experimental Intel GPU support available in combination with Intel oneAPI DPC++) are the most widely used. Other implementations of note are `triSYCL <https://github.com/triSYCL/triSYCL>`_ and `ComputeCPP <https://developer.codeplay.com/products/computecpp/ce/home/>`_.


High-level language support
---------------------------

Python
~~~~~~

Python offers support for GPU programming through multiple abstraction levels.


**CUDA Python, HIP Python and PyCUDA**

These projects are, respectively, `NVIDIA- <https://developer.nvidia.com/cuda-python>`_, `AMD- <https://rocm.docs.amd.com/projects/hip-python/en/latest/>`_ 
and `community-supported <https://documen.tician.de/pycuda/>`_ wrappers providing Python bindings to the low-level CUDA and HIP APIs. To use these approaches directly, in most cases knowledge of CUDA or HIP programming is needed. 

CUDA Python also aims to support higher-level toolkits and libraries, such as **CuPy** and **Numba**.


**CuPy**

`CuPy <https://cupy.dev/>`_ is a GPU-based data array library compatible with NumPy/SciPy. It offers a highly similar interface to NumPy and SciPy, making it easy for developers to transition to GPU computing. Code written with NumPy can often be adapted to use CuPy with minimal modifications; in most straightforward cases, one might simply replace 'numpy' and 'scipy' with 'cupy' and 'cupyx.scipy' in their Python code. 


**Numba**

`Numba <https://numba.pydata.org/>`_ is an open-source JIT compiler that translates a subset of Python and NumPy code into optimized machine code. Numba supports CUDA-capable GPUs and is able to generate code for them using several different syntax variants.
In 2021, upstream support for `AMD (ROCm) support <https://numba.readthedocs.io/en/stable/release-notes.html#version-0-54-0-19-august-2021>`_ was discontinued.
However, as of 2025, AMD has added downstream support for the Numba API through the 
`Numba HIP package <https://github.com/ROCm/numba-hip>`_.


Julia
~~~~~

Julia has first-class support for GPU programming through the following packages that target GPUs from all three major vendors:

- `CUDA.jl <https://cuda.juliagpu.org/stable/>`_ for NVIDIA GPUs
- `AMDGPU.jl <https://amdgpu.juliagpu.org/stable/>`_ for AMD GPUs
- `oneAPI.jl <https://github.com/JuliaGPU/oneAPI.jl>`_ for Intel GPUs
- `Metal.jl <https://github.com/JuliaGPU/Metal.jl>`_ for Apple M-series GPUs

``CUDA.jl`` is the most mature, ``AMDGPU.jl`` is somewhat behind but still ready for general use, while ``oneAPI.jl`` and ``Metal.jl`` are functional but might contain bugs, miss some features and provide suboptimal performance. Their respective APIs are however completely analogous and translation between libraries is straightforward.

All packages offer both high-level abstractions that require very little programming effort and a lower level approach for writing kernels for fine-grained control.


.. admonition:: In short
   :class: dropdown
   
   - **Directive-based programming:**
  
     - Existing serial code is annotated with directives to indicate which parts should be executed on the GPU.
     - OpenACC and OpenMP are common directive-based programming models.
     - Productivity and easy usage are prioritized over performance.
     - Minimum programming effort is required to add parallelism to existing code.

   - **Non-portable kernel-based models:**
  
     - Low-level code is written to directly communicate with the GPU.
     - CUDA is NVIDIA's parallel computing platform and API for GPU programming.
     - HIP is an API developed by AMD that provides a similar programming interface to CUDA for both NVIDIA and AMD GPUs.
     - Deeper understanding of GPU architecture and programming methods is needed.

   - **Portable kernel-based models:**
     
     - Higher-level abstractions for GPU programming that provide portability.
     - Examples include OpenCL, Kokkos, alpaka, RAJA, and SYCL.
     - Aim to achieve performance portability with a single-source application.
     - Can run on various GPUs and platforms, reducing the effort required to maintain and deploy GPU-accelerated applications.

   - **High-level language support:**

     - C++ and Fortran feature initiatives to support GPUs through language-standard parallelism.
     - Python libraries like PyCUDA, CuPy, and Numba offer GPU programming capabilities.
     - Julia has packages such as CUDA.jl, AMDGPU.jl, oneAPI.jl, and Metal.jl for GPU programming.
     - These approaches provide high-level abstraction and interfaces for GPU programming in the respective languages.


Summary
-------

Each of these GPU programming environments has its own strengths and weaknesses, and the best choice for a given project will depend on a range of factors, including: 

- the hardware platforms being targeted,
- the type of computation being performed, and
- the developer's experience and preferences.
 
**High-level and productivity-focused APIs** provide a simplified programming model and maximize code portability, while **low-level and performance-focused APIs** provide a high level of control over the GPU's hardware but also require more coding effort and expertise.


Exercises
---------

.. challenge:: Discussion

   - Which GPU programming frameworks have you already used previously, if any?
   - What did you find most challenging? What was most useful?

   Let us know in the main room or via comments in HackMD document.


.. keypoints::

   - GPU programming approaches can be split into 1) directive-based, 2) non-portable kernel-based, 3) portable kernel-based, and 4) high-level language support.
   - There are multiple frameworks/languages available for each approach, each with pros and cons. 

