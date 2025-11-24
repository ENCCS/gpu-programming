.. _portable-kernel-models:

Portable kernel-based models
============================

.. questions::

   - How to program GPUs with alpaka, C++ StdPar, Kokkos, OpenCL, and SYCL?
   - What are the differences between these programming models.

.. objectives::

   - Be able to use portable kernel-based models to write simple codes
   - Understand how different approaches to memory and synchronization in Kokkos and SYCL work

.. instructor-note::

   - 60 min teaching
   - 30 min exercises

The goal of the cross-platform portability ecosystems is to allow the same code to run on multiple architectures, therefore reducing code duplication. They are usually based on C++, and use function objects/lambda functions to define the loop body (i.e., the kernel), which can run on multiple architectures like CPU, GPU, and FPGA from different vendors. An exception to this is OpenCL, which originally offered only a C API (although currently also C++ API is available), and uses a separate-source model for the kernel code. However, unlike in many conventional CUDA or HIP implementations, the portability ecosystems require kernels to be written only once if one prefers to run it on CPU and GPU for example. Some notable cross-platform portability ecosystems are alpaka, Kokkos, OpenCL, RAJA, and SYCL. Kokkos, alpaka, and RAJA are individual projects whereas OpenCL and SYCL are standards followed by several projects implementing (and extending) them. For example, some notable SYCL implementations include `Intel oneAPI DPC++ <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`_, `AdaptiveCpp <https://github.com/AdaptiveCpp/AdaptiveCpp/>`_ (previously known as hipSYCL or Open SYCL), `triSYCL <https://github.com/triSYCL/triSYCL>`_, and `ComputeCPP <https://developer.codeplay.com/products/computecpp/ce/home/>`_.

C++ StdPar
^^^^^^^^^^

In C++17, the initial support for parallel execution of standard algorithms has been introduced.
Most algorithms available via the standard ``<algorithms>`` header were given an overload accepting with an `*execution policy* <https://en.cppreference.com/w/cpp/algorithm>`_ argument which allows the programmer to request parallel execution of the standard library function.
While the main goal was to allow low-effort, high-level interface to run existing algorithms like ``std::sort`` on many CPU cores, implementations are allowed to use other hardware, and functions like ``std::for_each`` or ``std::transform`` offer great flexibility in writing the algorithm.

C++ StdPar, also called Parallel STL or PSTL, could be considered similar to directive-based models, as it is very high-level and does not give the programmer fine-grained control over data movement or any access to hardware-specific features like shared (local) memory.
Even the GPU to run on is selected automatically, since standard C++ does not have the concept of a *device* (but there are vendor extensions allowing the programmer more control)
However, for applications that already relies on algorithms from C++ standard library, StdPar can be a good way to reap the performance benefits of both CPUs and GPUs with minimal code modifications.

For GPU programming, all three vendors offer their implementations of StdPar with the ability to offload code to the GPU: NVIDIA has ``nvc++``, AMD has experimental `roc-stdpar <https://github.com/ROCm/roc-stdpar>`_, and Intel offers StdPar offload with their oneAPI compiler. `AdaptiveCpp <https://github.com/AdaptiveCpp/AdaptiveCpp/>`__ offers an independent StdPar implementation, able to target devices from all three vendors. While being a part of the C++ standard, the level of support and the maturity of StdPar implementations varies a lot between different compilers: not all compilers support all algorithms, and different heuristics for mapping the algorithm to hardware and for managing data movement can have effect on performance.

StdPar compilation
~~~~~~~~~~~~~~~~~~

The build process depends a lot on the used compiler:

- AdaptiveCpp: Add ``--acpp-stdpar`` flag when calling ``acpp``.
- Intel oneAPI: Add ``-fsycl -fsycl-pstl-offload=gpu`` flags when calling ``icpx``.
- NVIDIA NVC++: Add ``-stdpar`` flag when calling ``nvc++`` (not supported with plain ``nvcc``).

StdPar programming
~~~~~~~~~~~~~~~~~~

In its simplest form, using C++ standard parallelism requires including an additional ``<execution>`` header and adding one argument to a supported standard library function.

For example, let's look at the following sequential code sorting a vector:

.. code-block:: C++

    #include <algorithm>
    #include <vector>
    
    void f(std::vector<int>& a) {
      std::sort(a.begin(), a.end());
    }

To make it run sorting on the GPU, only a minor modification is needed:

.. code-block:: C++

    #include <algorithm>
    #include <vector>
    #include <execution> // To get std::execution
    
    void f(std::vector<int>& a) {
      std::sort(
          std::execution::par_unseq, // This algorithm can be run in parallel
          a.begin(), a.end()
        );
    }

Now, when compiled with one of the supported compilers, the code will run the sorting on a GPU.

While the can initially seem very limiting, many standard algorithms, such as ``std::transform``, ``std::accumulate``, ``std::transform_reduce``, and ``std::for_each`` can run custom functions over an array, thus allowing one to offload an arbitrary algorithm, as long as it does not violate typical limitations of GPU kernels, such as not throwing any exceptions and not doing system calls.

StdPar execution policies
~~~~~~~~~~~~~~~~~~~~~~~~~

In C++, there are four different execution policies to choose from:

- ``std::execution::seq``: run algorithm serially, don't parallelize it.
- ``std::execution::par``: allow parallelizing the algorithm (as if using multiple threads),
- ``std::execution::unseq``: allow vectorizing the algorithm (as if using SIMD),
- ``std::execution::par_unseq``: allow both vectorizing and parallelizing the algorithm.

The main difference between ``par`` and ``unseq`` is related to thread progress and locks: using ``unseq`` or ``par_unseq`` requires that the algorithms does not contain mutexes and other locks between the processes, while ``par`` does not have this limitation.

For GPU, the optimal choice is ``par_unseq``, since this places the least requirement on the compiler in terms of operation ordering.
While ``par`` is also supported in some cases, it is best avoided, both due to limited compiler support and as an indication that the algorithm is likely a poor fit for the hardware.


Kokkos
^^^^^^

Kokkos is an open-source performance portability ecosystem for parallelization on large heterogeneous hardware architectures of which development has mostly taken place on Sandia National Laboratories. The project started in 2011 as a parallel C++ programming model, but have since expanded into a more broad ecosystem including Kokkos Core (the programming model), Kokkos Kernels (math library), and Kokkos Tools (debugging, profiling and tuning tools). By preparing proposals for the C++ standard committee, the project also aims to influence the ISO/C++ language standard such that, eventually, Kokkos capabilities will become native to the language standard. A more detailed introduction is found `HERE <https://www.sandia.gov/news/publications/hpc-annual-reports/article/kokkos/>`__.

The Kokkos library provides an abstraction layer for a variety of different parallel programming models, currently CUDA, HIP, SYCL, HPX, OpenMP, and C++ threads. Therefore, it allows better portability across different hardware manufactured by different vendors, but introduces an additional dependency to the software stack. For example, when using CUDA, only CUDA installation is required, but when using Kokkos with NVIDIA GPUs, Kokkos and CUDA installation are both required. Kokkos is not a very popular choice for parallel programming, and therefore, learning and using Kokkos can be more difficult compared to more established programming models such as CUDA, for which a much larger amount of search results and Stack Overflow discussions can be found.


Kokkos compilation
~~~~~~~~~~~~~~~~~~

Furthermore, one challenge with some cross-platform portability libraries is that even on the same system, different projects may require different combinations of compilation settings for the portability library. For example, in Kokkos, one project may wish the default execution space to be a CUDA device, whereas another requires a CPU. Even if the projects prefer the same execution space, one project may desire the Unified Memory to be the default memory space and the other may wish to use pinned GPU memory. It may be burdensome to maintain a large number of library instances on a single system. 

However, Kokkos offers a simple way to compile Kokkos library simultaneously with the user project. This is achieved by specifying Kokkos compilation settings (see `HERE <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Compiling.html>`__) and including the Kokkos Makefile in the user Makefile. CMake is also supported. This way, the user application and Kokkos library are compiled together. The following is an example Makefile for a single-file Kokkos project (hello.cpp) that uses CUDA (Volta architecture) as the backend (default execution space) and Unified Memory as the default memory space:

.. tabs:: 

   .. tab:: Makefile for hello.cpp

      .. code-block:: makefile

         default: build
   
         # Set compiler
         KOKKOS_PATH = $(shell pwd)/kokkos
         CXX = hipcc
         # CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
         
         # Variables for the Makefile.kokkos
         KOKKOS_DEVICES = "HIP"
         # KOKKOS_DEVICES = "Cuda"
         KOKKOS_ARCH = "VEGA90A"
         # KOKKOS_ARCH = "Volta70"
         KOKKOS_CUDA_OPTIONS = "enable_lambda,force_uvm"
         
         # Include Makefile.kokkos
         include $(KOKKOS_PATH)/Makefile.kokkos
         
         build: $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) hello.cpp
          $(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_LDFLAGS) hello.cpp $(KOKKOS_LIBS) -o hello

To build a **hello.cpp** project with the above Makefile, no steps other than cloning the Kokkos project into the current directory is required. 

Kokkos programming
~~~~~~~~~~~~~~~~~~

When starting to write a project using Kokkos, the first step is understand Kokkos initialization and finalization. Kokkos must be initialized by calling ``Kokkos::initialize(int& argc, char* argv[])`` and finalized by calling ``Kokkos::finalize()``. More details are given in `HERE <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Initialization.html>`__.

Kokkos uses an execution space model to abstract the details of parallel hardware. The execution space instances map to the available backend options such as CUDA, OpenMP, HIP, or SYCL. If the execution space is not explicitly chosen by the programmer in the source code, the default execution space ``Kokkos::DefaultExecutionSpace`` is used. This is chosen when the Kokkos library is compiled. The Kokkos execution space model is described in more detail in `HERE <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-spaces>`__.

Similarly, Kokkos uses a memory space model for different types of memory, such as host memory or device memory. If not defined explicitly, Kokkos uses the default memory space specified during Kokkos compilation as described `HERE <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-memory-spaces>`__.

The following is an example of a Kokkos program that initializes Kokkos and prints the execution space and memory space instances: 

.. tabs:: 

   .. tab:: hello.cpp
      
      .. code-block:: C++

         #include <Kokkos_Core.hpp>
         #include <iostream>
         
         int main(int argc, char* argv[]) {
           Kokkos::initialize(argc, argv);
           std::cout << "Execution Space: " << 
             typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
           std::cout << "Memory Space: " << 
             typeid(Kokkos::DefaultExecutionSpace::memory_space).name() << std::endl;
           Kokkos::finalize();
           return 0;
         }

With Kokkos, the data can be accessed either through raw pointers or through Kokkos Views. With raw pointers, the memory allocation into the default memory space can be done using ``Kokkos::kokkos_malloc(n * sizeof(int))``. Kokkos Views are a data type that provides a way to access data more efficiently in memory corresponding to a certain Kokkos memory space, such as host memory or device memory. A 1-dimensional view of type int* can be created by ``Kokkos::View<int*> a("a", n)``, where ``"a"`` is a label, and ``n`` is the size of the allocation in the number of integers. Kokkos determines the optimal layout for the data at compile time for best overall performance as a function of the computer architecture. Furthermore, Kokkos handles the deallocation of such memory automatically. More details about Kokkos Views are found `HERE <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html>`__.

Finally, Kokkos provides three different parallel operations: ``parallel_for``, ``parallel_reduce``, and ``parallel_scan``. The ``parallel_for`` operation is used to execute a loop in parallel. The ``parallel_reduce`` operation is used to execute a loop in parallel and reduce the results to a single value. The ``parallel_scan`` operation implements a prefix scan. The usage of ``parallel_for`` and ``parallel_reduce`` are demonstrated in the examples later in this chapter. More detail about the parallel operations are found `HERE <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html>`__.

Run Kokkos hello.cpp example in simple steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following should work on AMD VEGA90A devices straight out of the box (needs ROCm installation). On NVIDIA Volta V100 devices (needs CUDA installation), use the variables commented out on the Makefile.

1. ``git clone https://github.com/kokkos/kokkos.git``
2. Copy the above Makefile into the current folder (make sure the indentation of the last line is tab, and not space)
3. Copy the above hello.cpp file into the current folder
4. ``make``
5. ``./hello``


OpenCL
^^^^^^
OpenCL is a cross-platform, open-standard API for writing parallel programs that execute across heterogeneous platforms consisting of CPUs, GPUs, FPGAs and other devices. The first version of OpenCL (1.0) was released in December 2008, and the latest version of OpenCL (3.0) was released in September 2020. OpenCL is supported by a number of vendors, including AMD, ARM, Intel, NVIDIA, and Qualcomm. It is a royalty-free standard, and the OpenCL specification is maintained by the Khronos Group. OpenCL provides a low-level programming interface initially based on C, but more recently also a C++ interface has become available.

OpenCL compilation
~~~~~~~~~~~~~~~~~~
OpenCL supports two modes for compiling the programs: online and offline. Online compilation occurs at runtime, when the host program calls a function to compile the source code. Online mode allows dynamic generation and loading of kernels, but may incur some overhead due to compilation time and possible errors. Offline compilation occurs before runtime, when the source code of a kernel is compiled into a binary format that can be loaded by the host program. This mode allows faster execution and better optimization of kernels, but may limit the portability of the program, because the binary can only run on the architectures it was compiled for.

OpenCL comes bundled with several parallel programming ecosystems, such as NVIDIA CUDA and Intel oneAPI. For example, after successfully installing such packages and setting up the environment, one may simply compile an OpenCL program by the commands such as ``icx cl_devices.c -lOpenCL`` (Intel oneAPI) or ``nvcc cl_devices.c -lOpenCL`` (NVIDIA CUDA), where ``cl_devices.c`` is the compiled file. Unlike most other programming models, OpenCL stores kernels as text and compiles them for the device in runtime (JIT-compilation), and thus does not require any special compiler support: one can compile the code using simply ``gcc cl_devices.c -lOpenCL`` (or ``g++`` when using C++ API), as long as the required libraries and headers are installed in a standard locations.

The AMD compiler installed on LUMI supports both OpenCL C and C++ API, the latter with some limitations.
To compile a program, you can use the AMD compilers on a GPU partition:

.. code-block:: console

    $ module load LUMI/24.03 partition/G
    $ module load rocm/6.0.3
    $ module load PrgEnv-cray-amd
    $ CC program.cpp -lOpenCL -o program # C++ program
    $ cc program.c -lOpenCL -o program # C program


OpenCL programming
~~~~~~~~~~~~~~~~~~
OpenCL programs consist of two parts: a host program that runs on the host device (usually a CPU) and one or more kernels that run on compute devices (such as GPUs). The host program is responsible for the tasks such as managing the devices for the selected platform, allocating memory objects, building and enqueueing kernels, and managing memory objects. 

The first steps when writing an OpenCL program are to initialize the OpenCL environment by selecting the platform and devices, creating a context or contexts associated with the selected device(s), and creating a command queue for each device. A simple example of selecting the default device, creating a context and a queue associated with the device is show below.

.. tabs:: 

   .. tab:: OpenCL initialization (C++ API)
      
      .. code-block:: C++
         
         // Initialize OpenCL
         cl::Device device = cl::Device::getDefault();
         cl::Context context(device);
         cl::CommandQueue queue(context, device);

   .. tab:: OpenCL initialization (C API)
      
      .. code-block:: C
         
         // Initialize OpenCL
         cl_int err; // Error code returned by API calls
         cl_platform_id platform;
         err = clGetPlatformIDs(1, &platform, NULL);
         assert(err == CL_SUCCESS); // Checking error codes is skipped later for brevity
         cl_device_id device;
         err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
         cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
         cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);


OpenCL provides two main programming models to manage the memory hierarchy of host and accelerator devices: buffers and shared virtual memory (SVM). Buffers are the traditional memory model of OpenCL, where the host and the devices have separate address spaces and the programmer has to explicitly specify the memory allocations and how and where the memory is accessed. This can be done with class ``cl::Buffer`` and functions such as ``cl::CommandQueue::enqueueReadBuffer()``. Buffers are supported since early versions of OpenCL, and work well across different architectures. Buffers can also take advantage of device-specific memory features, such as constant or local memory.

SVM is a newer memory model of OpenCL, introduced in version 2.0, where the host and the devices share a single virtual address space. Thus, the programmer can use the same pointers to access the data from host and devices simplifying the programming effort. In OpenCL, SVM comes in different levels such as coarse-grained buffer SVM, fine-grained buffer SVM, and fine-grained system SVM. All levels allow using the same pointers across a host and devices, but they differ in their granularity and synchronization requirements for the memory regions. Furthermore, the support for SVM is not universal across all OpenCL platforms and devices, and for example, GPUs such as NVIDIA V100 and A100 only support the coarse-grained SVM buffer. This level requires explicit synchronization for memory accesses from a host and devices (using functions such as ``cl::CommandQueue::enqueueMapSVM()`` and ``cl::CommandQueue::enqueueUnmapSVM()``), making the usage of SVM less convenient. It is further noted that this is unlike the regular Unified Memory offered by CUDA, which is closer to the fine-grained system SVM level in OpenCL. 

OpenCL uses a separate-source kernel model where the kernel code is often kept in separate files that may be compiled during runtime. The model allows the kernel source code to be passed as a string to the OpenCL driver after which the program object can be executed on a specific device. Although referred to as the separate-source kernel model, the kernels can still be defined as a string in the host program compilation units as well, which may be a more convenient approach in some cases.

The online compilation with the separate-source kernel model has several advantages over the binary model, which requires offline compilation of kernels into device-specific binaries that can are loaded by the application at runtime. Online compilation preserves the portability and flexibility of OpenCL, as the same kernel source code can run on any supported device. Furthermore, dynamic optimization of kernels based on runtime information, such as input size, work-group size, or device capabilities, is possible. An example of an OpenCL kernel, defined by a string in the host compilation unit, and assigning the global thread index into a global device memory is shown below.

.. tabs:: 

   .. tab:: OpenCL kernel example
      
      .. code-block:: C++
         
         static const std::string kernel_source = R"(
           __kernel void dot(__global int *a) {
             int i = get_global_id(0);
             a[i] = i;
           }
         )";

The above kernel named ``dot`` and stored in the string ``kernel_source`` can be set to build in the host code as follows:

.. tabs:: 

   .. tab:: OpenCL kernel build example (C++ API)
      
      .. code-block:: C++
         
         cl::Program program(context, kernel_source);
         program.build({device});
         cl::Kernel kernel_dot(program, "dot");

   .. tab:: OpenCL kernel build example (C API)
      
      .. code-block:: C
         
         cl_int err;
         cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
         err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
         cl_kernel kernel_dot = clCreateKernel(program, "vector_add", &err);


SYCL
^^^^

`SYCL <https://www.khronos.org/sycl/>`__ is a royalty-free, open-standard C++ programming model for multi-device programming. It provides a high-level, single-source programming model for heterogeneous systems, including GPUs. There are several implementations of the standard. For GPU programming, `Intel oneAPI DPC++ <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`__ and `AdaptiveCpp <https://github.com/AdaptiveCpp/AdaptiveCpp/>`__ (also known as hipSYCL) are the most popular for desktop and HPC GPUs; `ComputeCPP <https://developer.codeplay.com/products/computecpp/ce/home/>`__ is a good choice for embedded devices. The same standard-compliant SYCL code should work with any implementation, but they are not binary-compatible.

The most recent version of the SYCL standard is SYCL 2020, and it is the version we will be using in this course. 

SYCL compilation
~~~~~~~~~~~~~~~~

Intel oneAPI DPC++
******************

For targeting Intel GPUs, it is enough to install `Intel oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html>`__. Then, the compilation is as simple as ``icpx -fsycl file.cpp``.

It is also possible to use oneAPI for NVIDIA and AMD GPUs. In addition to oneAPI Base Toolkit, the vendor-provided runtime (CUDA or HIP) and the corresponding `Codeplay oneAPI plugin <https://codeplay.com/solutions/oneapi/>`__ must be installed.
Then, the code can be compiled using Intel LLVM compiler bundled with oneAPI:

- ``clang++ -fsycl -fsycl-targets=nvidia_gpu_sm_86 file.cpp`` for targeting CUDA 8.6 NVIDIA GPU,
- ``clang++ -fsycl -fsycl-targets=amd_gpu_gfx90a`` for targeting GFX90a AMD GPU.

AdaptiveCpp
***********

Using AdaptiveCpp for NVIDIA or AMD GPUs also requires having CUDA or HIP installed first. Then ``acpp`` can be used for compiling the code, specifying the target devices. For example, here is how to compile the program supporting an AMD and an NVIDIA device:

- ``acpp --acpp-targets='hip:gfx90a;cuda:sm_70' file.cpp``


Using SYCL on LUMI
******************

LUMI does not have a system-wide installation of any SYCL framework, but a recent AdaptiveCpp installation is
available in CSC modules:

.. code-block:: console

    $ module load LUMI/24.03 partition/G
    $ module load rocm/6.0.3
    $ module use /appl/local/csc/modulefiles
    $ module load acpp/24.06.0

The default compilation target is preset to MI250 GPUs, so to compile a single C++ file it is enough to call ``acpp -O2 file.cpp``.

When running applications built with AdaptiveCpp, one can often see the warning "dag_direct_scheduler: Detected a requirement that is neither of discard access mode", reflecting the lack of an optimization hint when using buffer-accessor model. The warning is harmless and can be ignored.

SYCL programming
~~~~~~~~~~~~~~~~

SYCL is, in many aspects, similar to OpenCL, but uses, like Kokkos, a single-source model with kernel lambdas.

To submit a task to device, first a `sycl::queue` must be created, which is used as a way to manage the
task scheduling and execution. In the simplest case, that's all the initialization one needs:

.. code-block:: C++
    
    int main() {
      // Create an out-of-order queue on the default device:
      sycl::queue q;
      // Now we can submit tasks to q!
    }

If one wants more control, the device can be explicitly specified, or additional properties can be passed to
a queue:

.. code-block:: C++
    
    // Iterate over all available devices
    for (const auto &device : sycl::device::get_devices()) {
      // Print the device name
      std::cout << "Creating a queue on " << device.get_info<sycl::info::device::name>() << "\n";
      // Create an in-order queue for the current device
      sycl::queue q(device, {sycl::property::queue::in_order()});
      // Now we can submit tasks to q!
    }


Memory management can be done in two different ways: *buffer-accessor* model and *unified shared memory* (USM).
The choice of the memory management models also influences how the GPU tasks are synchronized.

In the *buffer-accessor* model, a ``sycl::buffer`` objects are used to represent arrays of data. A buffer is
not mapped to any single one memory space, and can be migrated between the GPU and the CPU memory
transparently. The data in ``sycl::buffer`` cannot be read or written directly, an accessor must be created.
``sycl::accessor`` objects specify the location of data access (host or a certain GPU kernel) and the access
mode (read-only, write-only, read-write).
Such approach allows optimizing task scheduling by building a directed acyclic graph (DAG) of data dependencies:
if kernel *A* creates a write-only accessor to a buffer, and then kernel *B* is submitted with a read-only
accessor to the same buffer, and then a host-side read-only accessor is requested, then it can be deduced that
*A* must complete before *B* is launched and also that the results must be copied to the host
before the host task can proceed, but the host task can run in parallel with kernel *B*.
Since the dependencies between tasks can be built automatically, by default SYCL uses *out-of-order queues*:
when two tasks are submitted to the same ``sycl::queue``, it is not guaranteed that the second one will launch
only after the first one completes.
When launching a kernel, accessors must be created:

.. code-block:: C++
    
    // Create a buffer of n integers
    auto buf = sycl::buffer<int>(sycl::range<1>(n));
    // Submit a kernel into a queue; cgh is a helper object
    q.submit([&](sycl::handler &cgh) {
      // Create write-only accessor for buf
      auto acc = buf.get_access<sycl::access_mode::write>(cgh);
      // Define a kernel: n threads execute the following lambda
      cgh.parallel_for<class KernelName>(sycl::range<1>{n}, [=](sycl::id<1> i) {
          // The data is written to the buffer via acc
          acc[i] = /*...*/
      });
    });
    /* If we now submit another kernel with accessor to buf, it will not
     * start running until the kernel above is done */

Buffer-accessor model simplifies many aspects of heterogeneous programming and prevents many synchronization-related
bugs, but it only allows very coarse control of data movement and kernel execution.

The *USM* model is similar to how NVIDIA CUDA or AMD HIP manage memory. The programmer has to explicitly allocate
the memory on the device (``sycl::malloc_device``), on the host (``sycl::malloc_host``), or in the shared memory
space (``sycl::malloc_shared``). Despite its name, unified shared memory, and the similarity to OpenCL's SVM, not
all USM allocations are shared: for example, a memory allocated by ``sycl::malloc_device`` cannot be accessed
from the host. The allocation functions return memory pointers that can be used directly, without accessors.
This means that the programmer have to ensure the correct synchronization between host and device tasks to avoid
data races. With USM, it is often convenient to use *in-order queues* with USM, instead of the default *out-of-order* queues.
More information on USM can be found in the `Section 4.8 of SYCL 2020 specification <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm>`__.

.. code-block:: C++
    
    // Create a shared (migratable) allocation of n integers
    // Unlike with buffers, we need to specify a queue (or, explicitly, a device and a context)
    int* v = sycl::malloc_shared<int>(n, q);
    // Submit a kernel into a queue; cgh is a helper object
    q.submit([&](sycl::handler &cgh) {
      // Define a kernel: n threads execute the following lambda
      cgh.parallel_for<class KernelName>(sycl::range<1>{n}, [=](sycl::id<1> i) {
          // The data is directly written to v
          v[i] = /*...*/
      });
    });
    // If we want to access v, we have to ensure that the kernel has finished
    q.wait();
    // After we're done, the memory must be deallocated
    sycl::free(v, q);

Exercise
~~~~~~~~

.. exercise:: Exercise: Implement SAXPY in SYCL

   In this exercise we would like to write (fill-in-the-blanks) a simple code doing SAXPY (vector addition).
   
   To compile and run the code interactively, first make an allocation and load the AdaptiveCpp module:

   .. code-block:: console

      $ salloc -A project_465001310 -N 1 -t 1:00:00 -p standard-g --gpus-per-node=1
      ....
      salloc: Granted job allocation 123456

      $ module load LUMI/24.03 partition/G
      $ module use /appl/local/csc/modulefiles
      $ module load rocm/6.0.3 acpp/24.06.0

   Now you can run a simple device-detection utility to check that a GPU is available (note ``srun``):

    .. code-block:: console

      $ srun acpp-info -l
      =================Backend information===================
      Loaded backend 0: HIP
        Found device: AMD Instinct MI250X
      Loaded backend 1: OpenMP
        Found device: hipSYCL OpenMP host device


   If you have not done it already, clone the repository using ``git clone https://github.com/ENCCS/gpu-programming.git`` or **update it** using ``git pull origin main``.

   Now, let's look at the example code in ``content/examples/portable-kernel-models/exercise-sycl-saxpy.cpp``:

   .. literalinclude:: examples/portable-kernel-models/exercise-sycl-saxpy.cpp
      :language: c++
      :emphasize-lines: 16,17,25,30,31,35,39,40,62


   To compile and run the code, use the following command:

   .. code-block:: console

      $ acpp -O3 exercise-sycl-saxpy.cpp -o exercise-sycl-saxpy
      $ srun ./exercise-sycl-saxpy
      Running on AMD Instinct MI250X
      Results are correct!

   The code will not compile as-is!
   Your task is to fill in missing bits indicated by ``TODO`` comments.
   You can also test your understanding using the "Bonus questions" in the code.

   If you feel stuck, take a look at the ``exercise-sycl-saxpy-solution.cpp`` file.

alpaka
^^^^^^

The `alpaka <https://github.com/alpaka-group/alpaka3>`__ library is an open-source header-only C++20 abstraction library for accelerator development.

Its aim is to provide performance portability across accelerators by abstracting the underlying levels of parallelism. The project provides a single-source C++ API that enables developers to write parallel code once and run it on different hardware architectures without modification.
The name "alpaka" comes from **A**\bstractions for **L**\evels of **P**\arallelism, **A**\lgorithms, and **K**\ernels for **A**\ccelerators.
The library is platform-independent and supports the concurrent and cooperative use of multiple devices, including host CPUs (x86, ARM, and RISC-V) and GPUs from different vendors (NVIDIA, AMD, and Intel).
A variety of accelerator backends, CUDA, HIP, SYCL, OpenMP, and serial execution, are available and can be selected based on the target device.
Only a single implementation of a user kernel is required, expressed as a function object with a standardized interface.
This eliminates the need to write specialized CUDA, HIP, SYCL, OpenMP, Intel TBB or threading code.
Moreover, multiple accelerator backends can be combined to target different vendor hardware within a single system and even within a single application.

The abstraction is based on a virtual index domain decomposed into equally sized chunks called frames.
**alpaka** provides a uniform abstraction to traverse these frames, independent of the underlying hardware.
Algorithms to be parallelized map the chunked index domain and native worker threads onto the data, expressing the computation as kernels that are executed in parallel threads (SIMT), thereby also leveraging SIMD units.
Unlike native parallelism models such as CUDA, HIP, and SYCL, **alpaka** kernels are not restricted to three dimensions.
Explicit caching of data within a frame via shared memory allows developers to fully unleash the performance of the compute device.
Additionally, **alpaka** offers primitive functions such as iota, transform, transform-reduce, reduce, and concurrent, simplifying the development of portable high-performance applications.
Host, device, mapped, and managed multi-dimensional views provide a natural way to operate on data.

Here we demonstrate the usage of **alpaka3**, which is a complete rewrite of `alpaka <https://github.com/alpaka-group/alpaka>`__.
It is planned to merge this separate codebase back into the mainline alpaka repository before the first release in Q2/Q3 of 2026.
Nevertheless, the code is well-tested and can be used for development today.


Installing alpaka on your system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ease of use, we recommend installing alpaka using CMake as described below. For other ways to use alpaka in your projects, see the `alpaka3 documentation <https://alpaka3.readthedocs.io/en/latest/basic/install.html>`__.

1. **Clone the repository**

   Clone the alpaka source code from GitHub to a directory of your choice:

   .. code-block:: bash

      git clone https://github.com/alpaka-group/alpaka3.git
      cd alpaka

2. **Set installation directory**

   Set the ``ALPAKA_DIR`` environment variable to the directory where you want to install alpaka. This can be any directory you choose where you have write access.

   .. code-block:: bash

      export ALPAKA_DIR=/path/to/your/alpaka/install/dir

3. **Build and install**

   Create a build directory and use CMake to build and install alpaka. We use ``CMAKE_INSTALL_PREFIX`` to tell CMake where to install the library.

   .. code-block:: bash

      mkdir build
      cmake -B build -S . -DCMAKE_INSTALL_PREFIX=$ALPAKA_DIR
      cmake --build build --parallel

4. **Update environment**

   To make sure that other projects can find your alpaka installation, you should add the installation directory to your ``CMAKE_PREFIX_PATH``. You can do this by adding the following line to your shell configuration file (e.g. ``~/.bashrc``):

   .. code-block:: bash

      export CMAKE_PREFIX_PATH=$ALPAKA_DIR:$CMAKE_PREFIX_PATH

   You will need to source your shell configuration file or open a new terminal for the changes to take effect.


alpaka Compilation
~~~~~~~~~~~~~~~~~~~

We recommend building your projects which use alpaka using CMake. A variety of strategies can be used to deal with building your application for a specific device or set of devices. Here we show a minimal way to get started, but this is by no means the only way to set up your projects.
Please refer to the `alpaka3 documentation <https://alpaka3.readthedocs.io/en/latest/basic/install.html>`__ for alternative ways to use alpaka in your project, including a way to make your source code agnostic to the accelerator being targeted by defining a device specification in CMake.

The following example demonstrates a ``CMakeLists.txt`` for a single-file project using alpaka3 (``main.cpp`` which is presented in the section below):

    .. code-block:: cmake
    
        cmake_minimum_required(VERSION 3.25)
        project(myAlpakaApp VERSION 1.0)
    
        # Find installed alpaka
        find_package(alpaka REQUIRED)
    
        # Build the executable
        add_executable(myAlpakaApp main.cpp)
        target_link_libraries(myAlpakaApp PRIVATE alpaka::alpaka)
        alpaka_finalize(myAlpakaApp)

Using alpaka on LUMI
********************

To load the environment for using the AMD GPUs on LUMI with HIP, one can use the following modules -

.. code-block:: console

    $ module load LUMI/24.03 partition/G
    $ module load rocm/6.0.3
    $ module load buildtools/24.03
    $ module load PrgEnv-amd
    $ module load craype-accel-amd-gfx90a
    $ export CXX=hipcc


alpaka Programming
~~~~~~~~~~~~~~~~~~

When starting with alpaka3, the first step is understanding the **device selection model**. Unlike frameworks that require explicit initialization calls, alpaka3 uses a device specification to determine which backend and hardware to use. The device specification consists of two components:

- **API**: The parallel programming interface (host, cuda, hip, oneApi)
- **Device Kind**: The type of hardware (cpu, nvidiaGpu, amdGpu, intelGpu)

Here we specify and use these at runtime to select and initialize devices. The device selection process is described in detail in the alpaka3 documentation.

alpaka3 uses an **execution space model** to abstract parallel hardware details. A device selector is created using ``alpaka::onHost::makeDeviceSelector(devSpec)``, which returns an object that can query available devices and create device instances for the selected backend.

The following example demonstrates a basic alpaka program that initializes a device and prints information about it:

    .. code-block:: cpp
        
        #include <alpaka/alpaka.hpp>
        #include <cstdlib>
        #include <iostream>
        
        namespace ap = alpaka;

        auto getDeviceSpec()
        {
            /* Select a device, possible combinations of api+deviceKind:
             * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
             * oneApi+amdGpu, oneApi+nvidiaGpu
             */
            return ap::onHost::DeviceSpec{ap::api::hip, ap::deviceKind::amdGpu};
        }
        
        int main(int argc, char** argv)
        {
            // Initialize device specification and selector
            ap::onHost::DeviceSpec devSpec = getDeviceSpec();
            auto deviceSelector = ap::onHost::makeDeviceSelector(devSpec);
            
            // Query available devices
            auto num_devices = deviceSelector.getDeviceCount();
            std::cout << "Number of available devices: " << num_devices << "\n";
            
            if (num_devices == 0) {
                std::cerr << "No devices found for the selected backend\n";
                return EXIT_FAILURE;
            }
            
            // Select and initialize the first device
            auto device = deviceSelector.makeDevice(0);
            std::cout << "Using device: " << device.getName() << "\n";
            
            return EXIT_SUCCESS;
        }

alpaka3 provides memory management abstractions through buffers and views. Memory can be allocated on host or device using ``alpaka::allocBuf<T, Idx>(device, extent)``. Data transfers between host and device are handled through ``alpaka::memcpy(queue, dst, src)``. The library automatically manages memory layouts for optimal performance on different architectures.

For parallel execution, alpaka3 provides kernel abstractions. Kernels are defined as functors or lambda functions and executed using work division specifications that define the parallelization strategy. The framework supports various parallel patterns including element-wise operations, reductions, and scans.

Tour of **alpaka** Features
***************************

Now we will quickly explore the most commonly used features of alpaka and go over some basic usage. A quick reference of commonly used alpaka features is available `here. <https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html>`__  

**General setup**: Include the consolidated header once and you are ready to start using alpaka.

.. code-block:: c++

   #include <alpaka/alpaka.hpp>

   namespace myProject
   {
       namespace ap = alpaka;
       // Your code here
   }

**Accelerator, platform, and device management**: Select devices by combining the desired API with the appropriate hardware kind using the device selector.

.. code-block:: c++

   auto devSelector = ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);
   if (devSelector.getDeviceCount() == 0)
   {
       throw std::runtime_error("No device found!");
   }
   auto device = devSelector.makeDevice(0);

**Queues and events**: Create blocking or non-blocking queues per device, record events, and synchronize work as needed.

.. code-block:: c++

   auto queue = device.makeQueue();
   auto nonBlockingQueue = device.makeQueue(ap::queueKind::nonBlocking);
   auto blockingQueue = device.makeQueue(ap::queueKind::blocking);

   auto event = device.makeEvent();
   queue.enqueue(event);
   ap::onHost::wait(event);
   ap::onHost::wait(queue);

**Memory management**: Allocate host, device, mapped, unified, or deferred buffers, create non-owning views, and move data portably with `memcpy`, `memset`, and `fill`.

.. code-block:: c++

    auto hostBuffer = ap::onHost::allocHost<DataType>(extent3D);
    auto devBuffer = ap::onHost::alloc<DataType>(device, extentMd);
    auto devMappedBuffer = ap::onHost::allocMapped<DataType>(device, extentMd);
    
    auto hostView = ap::makeView(api::host, externPtr, ap::Vec{numElements});
    auto devNonOwningView = devBuffer.getView();
    
    ap::onHost::memset(queue, devBuffer, uint8_t{0});
    ap::onHost::memcpy(queue, devBuffer, hostBuffer);
    ap::onHost::fill(queue, devBuffer, DataType{42});

**Kernel execution**: Build a `FrameSpec` manually or request one tuned for your data type, then enqueue kernels with automatic or explicit executors.

.. code-block:: c++

    constexpr uint32_t dim = 2u;
    using IdxType = size_t;
    using DataType = int;
    
    IdxType valueX, valueY;
    auto extentMD = ap::Vec{valueY, valueX};

    auto frameSpec = ap::onHost::FrameSpec{numFramesMd, frameExtentMd};
    auto tunedSpec = ap::onHost::getFrameSpec<DataType>(device, extentMd);
    
    queue.enqueue(tunedSpec, ap::KernelBundle{kernel, kernelArgs...});
    
    auto executor = ap::exec::cpuSerial;
    queue.enqueue(executor, tunedSpec, ap::KernelBundle{kernel, kernelArgs...});

**Kernel implementation**: Write kernels as functors annotated with `ALPAKA_FN_ACC`, use shared memory, synchronization, atomics, and math helpers directly inside the kernel body.

.. code-block:: c++

   struct MyKernel
   {
       ALPAKA_FN_ACC void operator()(ap::onAcc::concepts::Acc auto const& acc, auto... args) const
       {
           auto idxMd = acc.getIdxWithin(ap::onAcc::origin::grid, ap::onAcc::unit::blocks);

           auto sharedMdArray =
               ap::onAcc::declareSharedMdArray<float, ap::uniqueId()>(acc, ap::CVec<uint32_t, 3, 4>{});

           ap::onAcc::syncBlockThreads(acc);
           auto old = onAcc::atomicAdd(acc, args...);
           ap::onAcc::memFence(acc, ap::onAcc::scope::block);
           auto sinValue = ap::math::sin(args[0]);
       }
   };

   
Run alpaka3 Example in Simple Steps
***********************************


The following example works on systems with CMake 3.25+ and an appropriate C++ compiler. For GPU execution, ensure the corresponding runtime (CUDA, ROCm, or oneAPI) is installed.

1. Create a directory for your project:

   .. code-block:: bash

      mkdir my_alpaka_project && cd my_alpaka_project

2. Copy the CMakeLists.txt from above into the current folder

3. Copy the main.cpp file into the current folder

4. Configure and build:

   .. code-block:: bash

      cmake -B build -S . -Dalpaka_DEP_HIP=ON
      cmake --build build --parallel

5. Run the executable:

   .. code-block:: bash

      ./build/myAlpakaApp

.. note::

    The device specification system allows you to select the target device at CMake configuration time. The format is ``"api:deviceKind"``, where:

    - **api**: The parallel programming interface (``host``, ``cuda``, ``hip``, ``oneApi``)
    - **deviceKind**: The type of device (``cpu``, ``nvidiaGpu``, ``amdGpu``, ``intelGpu``)

    Available combinations are: ``host:cpu``, ``cuda:nvidiaGpu``, ``hip:amdGpu``, ``oneApi:cpu``, ``oneApi:intelGpu``, ``oneApi:nvidiaGpu``, ``oneApi:amdGpu``

.. warning::

    The CUDA, HIP, or Intel backends only work if the CUDA SDK, HIP SDK, or OneAPI SDK are available respectively


Expected output
***************

.. code-block:: text

   Number of available devices: 1
   Using device: [Device Name]

The device name will vary depending on your hardware (e.g., "NVIDIA A100", "AMD MI250X", or your CPU model).


Compile and Execute Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can test the **alpaka** provided examples from the `example section <#examples>`_.
The examples have hard coded the usage of the AMD ROCm platform required on LUMI.
To switch to CPU usage only you can simply replace ``ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);`` with ``ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);``

The following steps assume you have downloaded alpaka already and the path to the **alapka** source code is stored in the environment variable ``ALPAKA_DIR``.
To test the example copy the code into a file ``main.cpp``

Alternatively, `click here <https://godbolt.org/z/69exnG4xb>`__ to try the first example using in the godbolt compiler explorer.

.. tabs::

  .. tab:: HIP for AMD GPUs

    .. code-block:: bash

        # use the following in C++ code
        # auto devSelector = ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);
        # We use CC to refer to the compiler to work smoothly with the LUMI environment
        CC -I $ALPAKA_DIR/include/ -std=c++20 -x hip --offload-arch=gfx90a main.cpp
        ./a.out

  .. tab:: Host compiler for CPU

    .. code-block:: bash

        # use the following in C++ code
        # auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host, ap::deviceKind::cpu);
        # We use CC to refer to the compiler to work smoothly with the LUMI environment
        CC -I $ALPAKA_DIR/include/ -std=c++20 main.cpp
        ./a.out

  .. tab:: CUDA for NVIDIA GPUs  

    .. code-block:: bash

        # use the following in C++ code
        # auto devSelector = ap::onHost::makeDeviceSelector(ap::api::cuda, ap::deviceKind::nvidiaGpu);
        nvcc -I $ALPAKA_DIR/include/ -std=c++20 --expt-relaxed-constexpr -x cuda main.cpp
        ./a.out

  .. tab:: oneAPI SYCL for CPU

    .. code-block:: bash

        # use the following in C++ code
        # auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::cpu);
        icpx -I $ALPAKA_DIR/include/ -std=c++20 -fsycl -fsycl-targets=spir64_x86_64 main.cpp
        ./a.out

  .. tab:: oneAPI SYCL for Intel GPUs

    .. code-block:: bash

        # use the following in C++ code
        # auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::intelGpu);
        icpx -I $ALPAKA_DIR/include/ -std=c++20 -fsycl -fsycl-targets=spir64 main.cpp
        ./a.out

  .. tab:: oneAPI SYCL for AMD GPUs

    .. note::
   
        To use oneAPI Sycl with AMD or NVIDIA Gpus you must install the corresponding Codeplay oneAPI plugin as described `here <https://codeplay.com/solutions/oneapi/plugins/>`__.

    .. code-block:: bash

        # use the following in C++ code
        # auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::amdGpu);
        icpx -I $ALPAKA_DIR/include/ -std=c++20 -fsycl -fsycl-targets=amd_gpu_gfx90a main.cpp
        ./a.out

  .. tab:: oneAPI SYCL for NVIDIA GPUs

    .. note::
    
        To use oneAPI Sycl with AMD or NVIDIA Gpus you must install the corresponding Codeplay oneAPI plugin as described `here <https://codeplay.com/solutions/oneapi/plugins/>`__.
   
    .. code-block:: bash

        # use the following in C++ code
        # auto devSelector = ap::onHost::makeDeviceSelector(ap::api::oneApi, ap::deviceKind::nvidiaGpu);
        icpx -I $ALPAKA_DIR/include/ -std=c++20 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_80 main.cpp
        ./a.out


Exercise
~~~~~~~~

.. exercise:: Exercise: Write a vector add kernel in alpaka

    In this exercise we would like to write (fill-in-the-blanks) a simple kernel to add two vectors.
    
    To compile and run the code interactively, first we first need to get an allocation on a GPU node and load the modules for alpaka:

    .. code-block:: console
    
        $ srun -p dev-g --gpus 1 -N 1 -n 1 --time=00:20:00 --account=project_465002387 --pty bash
        ....
        srun: job 1234 queued and waiting for resources
        srun: job 1234 has been allocated resources

        $ module load LUMI/24.03 partition/G
        $ module load rocm/6.0.3
        $ module load buildtools/24.03
        $ module load PrgEnv-amd
        $ module load craype-accel-amd-gfx90a
        $ export CXX=hipcc

    Now you can run a simple device-detection utility to check that a GPU is available (note ``srun``):

    .. code-block:: console

      $ rocm-smi

      ======================================= ROCm System Management Interface =======================================
      ================================================= Concise Info =================================================
      Device  [Model : Revision]    Temp    Power  Partitions      SCLK    MCLK     Fan  Perf    PwrCap  VRAM%  GPU%  
              Name (20 chars)       (Edge)  (Avg)  (Mem, Compute)                                                     
      ================================================================================================================
      0       [0x0b0c : 0x00]       45.0°C  N/A    N/A, N/A        800Mhz  1600Mhz  0%   manual  0.0W      0%   0%    
              AMD INSTINCT MI200 (                                                                                    
      ================================================================================================================
      ============================================= End of ROCm SMI Log ==============================================


    Now, let's look at the code to set up the exercise:
    
    Below we use fetch content with our CMake to get started with alpaka quickly.

    .. literalinclude:: examples/portable-kernel-models/alpaka-exercise-vectorAdd-cmake.txt
        :language: cmake
        :caption: CMakeLists.txt
        :emphasize-lines: 12,19


    Below we have the main alpaka code doing a vector addition on device using a high level transform function
    
    .. literalinclude:: examples/portable-kernel-models/alpaka-exercise-vectorAdd.cpp
        :language: c++
        :caption: main.cpp
        :emphasize-lines: 35

    To set up our project, we create a folder and place our CMakeLists.txt and main.cpp in there.
    
    .. code-block:: console
    
        $ mkdir alpakaExercise && cd alpakaExercise
        $ vim CMakeLists.txt
        and now paste the CMakeLsits here (Press i, followed by Ctrl+Shift+V)
        Press esc and then :wq to exit vim
        $ vim main.cpp
        Similarly, paste the C++ code here
    
    To compile and run the code, use the following commands:

    .. code-block:: console
    
        configure step, we additionaly specify that HIP is available
        $ cmake -B build -S . -Dalpaka_DEP_HIP=ON
        build
        $ cmake --build build --parallel
        run
        $ ./build/vectorAdd
        Using alpaka device: AMD Instinct MI250X id=0
        c[0] = 1
        c[1] = 2
        c[2] = 3
        c[3] = 4
        c[4] = 5

    Now your task will be to write and launch your first alpaka kernel.
    This kernel will do the vector addition and we will use this instead of the transform helper. 

    .. solution:: Writing the vector add kernel
    
        .. literalinclude:: examples/portable-kernel-models/alpaka-exercise-vectorAdd-solution.cpp
            :language: c++
            :emphasize-lines: 5,6,7,8,9,10,11,12,13,14,15,46,48,49
    

Examples
^^^^^^^^

Parallel for with Unified Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs:: 

   .. tab:: StdPar
         .. literalinclude:: examples/portable-kernel-models/stdpar-unified-memory.cpp
            :language: C++

   .. tab:: Kokkos
         .. literalinclude:: examples/portable-kernel-models/kokkos-unified-memory.cpp
            :language: C++

   .. tab:: OpenCL
         .. literalinclude:: examples/portable-kernel-models/opencl-unified-memory.c
            :language: C

   .. tab:: SYCL
         .. literalinclude:: examples/portable-kernel-models/sycl-unified-memory.cpp
            :language: C++

   .. tab:: alpaka-algorithms
         .. literalinclude:: examples/portable-kernel-models/alpaka-algorithms-unified-memory.cpp
            :language: C++

   .. tab:: alpaka
         .. literalinclude:: examples/portable-kernel-models/alpaka-unified-memory.cpp
            :language: C++

Parallel for with GPU buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs:: 

   .. tab:: Kokkos
         .. literalinclude:: examples/portable-kernel-models/kokkos-buffers.cpp
            :language: C++

   .. tab:: OpenCL
         .. literalinclude:: examples/portable-kernel-models/opencl-buffers.cpp
            :language: C++
   
   .. tab:: SYCL
         .. literalinclude:: examples/portable-kernel-models/sycl-buffers.cpp
            :language: C++

   .. tab:: alpaka-algorithms
         .. literalinclude:: examples/portable-kernel-models/alpaka-algorithms-buffers.cpp
            :language: C++

   .. tab:: alpaka
         .. literalinclude:: examples/portable-kernel-models/alpaka-buffers.cpp
            :language: C++

   
Asynchronous parallel for kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs:: 

   .. tab:: Kokkos
         .. literalinclude:: examples/portable-kernel-models/kokkos-async-kernels.cpp
            :language: C++
   
   .. tab:: OpenCL
         .. literalinclude:: examples/portable-kernel-models/opencl-async-kernels.c
            :language: C
  
   .. tab:: SYCL
         .. literalinclude:: examples/portable-kernel-models/sycl-async-kernels.cpp
            :language: C++

   .. tab:: alpaka-algorithms
         .. literalinclude:: examples/portable-kernel-models/alpaka-algorithms-async.cpp
            :language: C++

   .. tab:: alpaka
         .. literalinclude:: examples/portable-kernel-models/alpaka-async-kernels.cpp
            :language: C++
 
Reduction
~~~~~~~~~

.. tabs:: 

   .. tab:: StdPar
         .. literalinclude:: examples/portable-kernel-models/stdpar-reduction.cpp
            :language: C++

   .. tab:: Kokkos
         .. literalinclude:: examples/portable-kernel-models/kokkos-reduction.cpp
            :language: C++

   .. tab:: OpenCL
         .. literalinclude:: examples/portable-kernel-models/opencl-reduction.cpp
            :language: C++

   .. tab:: SYCL
         .. literalinclude:: examples/portable-kernel-models/sycl-reduction.cpp
            :language: C++

   .. tab:: alpaka-algorithms
         .. literalinclude:: examples/portable-kernel-models/alpaka-algorithms-reduction.cpp
            :language: C++
 

Pros and cons of cross-platform portability ecosystems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

General observations
~~~~~~~~~~~~~~~~~~~~

    - The amount of code duplication is minimized.
    - The same code can be compiled to multiple architectures from different vendors.
    - Limited learning resources compared to CUDA (Stack Overflow, course material, documentation).

Lambda-based kernel models (Kokkos, SYCL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - Higher level of abstraction.
    - Less knowledge of the underlying architecture is needed for initial porting.
    - Very nice and readable source code (C++ API).
    - The models are relatively new and not very popular yet.

Functor-based kernel model (alpaka) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - Very good portability.
    - Higher level of abstraction.
    - Low-level API always awailable which gives more control and allows fine tuning.
    - User friendly C++ API for both the host and kernel code.
    - Small community and ecosystem.
    
Separate-source kernel models (OpenCL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - Very good portability.
    - Mature ecosystem.
    - Limited number of vendor-provided libraries.
    - Low-level API gives more control and allows fine tuning.
    - Both C and C++ APIs available (C++ API is less well supported).
    - The low-level API and separate-source kernel model are less user friendly.

C++ Standard Parallelism (StdPar, PSTL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - Very high level of abstraction.
    - Easy to speed up code which already relying on STL algorithms.
    - Very little control over hardware.
    - Support by compilers is improving, but is far from mature.

.. keypoints::

   - General code organization is similar to non-portable kernel-based models.
   - As long as no vendor-specific functionality is used, the same code can run on any GPU.
