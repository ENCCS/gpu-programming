.. _portable-kernel-models:

Portable kernel-based models
============================

.. questions::

   - How to program GPUs with Kokkos, OpenCL, and SYCL?
   - What are the differences between these programming models.

.. objectives::

   - Be able to use portable kernel-based models to write simple codes
   - Understand how different approaches to memory and synchronization in Kokkos and SYCL work

.. instructor-note::

   - 45 min teaching
   - 30 min exercises

The goal of the cross-platform portability ecosystems is to allow the same code to run on multiple architectures, therefore reducing code duplication. They are usually based on C++, and use function objects/lambda functions to define the loop body (i.e., the kernel), which can run on multiple architectures like CPU, GPU, and FPGA from different vendors. An exception to this is OpenCL, which originally offered only a C API (although currently also C++ API is available), and uses a separate-source model for the kernel code. However, unlike in many conventional CUDA or HIP implementations, the portability ecosystems require kernels to be written only once if one prefers to run it on CPU and GPU for example. Some notable cross-platform portability ecosystems are Alpaka, Kokkos, OpenCL, RAJA, and SYCL. Alpaka, Kokkos and RAJA are individual projects whereas OpenCL and SYCL are standards followed by several projects implementing (and extending) them. For example, some notable SYCL implementations include `Intel oneAPI DPC++ <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`_, `hipSYCL <https://github.com/OpenSYCL/OpenSYCL>`_ (also known as Open SYCL), `triSYCL <https://github.com/triSYCL/triSYCL>`_, and `ComputeCPP <https://developer.codeplay.com/products/computecpp/ce/home/>`_.

Kokkos
^^^^^^

Kokkos is an open-source performance portability ecosystem for parallelization on large heterogeneous hardware architectures of which development has mostly taken place on Sandia National Laboratories. The project started in 2011 as a parallel C++ programming model, but have since expanded into a more broad ecosystem including Kokkos Core (the programming model), Kokkos Kernels (math library), and Kokkos Tools (debugging, profiling and tuning tools). By preparing proposals for the C++ standard committee, the project also aims to influence the ISO/C++ language standard such that, eventually, Kokkos capabilities will become native to the language standard. A more detailed introduction is found `HERE <https://www.sandia.gov/news/publications/hpc-annual-reports/article/kokkos/>`__.

The Kokkos library provides an abstraction layer for a variety of different parallel programming models, currently CUDA, HIP, SYCL, HPX, OpenMP, and C++ threads. Therefore, it allows better portability across different hardware manufactured by different vendors, but introduces an additional dependency to the software stack. For example, when using CUDA, only CUDA installation is required, but when using Kokkos with NVIDIA GPUs, Kokkos and CUDA installation are both required. Kokkos is not a very popular choice for parallel programming, and therefore, learning and using Kokkos can be more difficult compared to more established programming models such as CUDA, for which a much larger amount of search results and Stack Overflow discussions can be found.


Kokkos compilation
~~~~~~~~~~~~~~~~~~

Furthermore, one challenge with some cross-platform portability libraries is that even on the same system, different projects may require different combinations of compilation settings for the portability library. For example, in Kokkos, one project may wish the default execution space to be a CUDA device, whereas another requires a CPU. Even if the projects prefer the same execution space, one project may desire the Unified Memory to be the default memory space and the other may wish to use pinned GPU memory. It may be burdensome to maintain a large number of library instances on a single system. 

However, Kokkos offers a simple way to compile Kokkos library simultaneously with the user project. This is achieved by specifying Kokkos compilation settings (see `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Compiling.html>`__) and including the Kokkos Makefile in the user Makefile. CMake is also supported. This way, the user application and Kokkos library are compiled together. The following is an example Makefile for a single-file Kokkos project (hello.cpp) that uses CUDA (Volta architecture) as the backend (default execution space) and Unified Memory as the default memory space:

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

When starting to write a project using Kokkos, the first step is understand Kokkos initialization and finalization. Kokkos must be initialized by calling ``Kokkos::initialize(int& argc, char* argv[])`` and finalized by calling ``Kokkos::finalize()``. More details are given in `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Initialization.html>`__.

Kokkos uses an execution space model to abstract the details of parallel hardware. The execution space instances map to the available backend options such as CUDA, OpenMP, HIP, or SYCL. If the execution space is not explicitly chosen by the programmer in the source code, the default execution space ``Kokkos::DefaultExecutionSpace`` is used. This is chosen when the Kokkos library is compiled. The Kokkos execution space model is described in more detail in `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-spaces>`__.

Similarly, Kokkos uses a memory space model for different types of memory, such as host memory or device memory. If not defined explicitly, Kokkos uses the default memory space specified during Kokkos compilation as described `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-memory-spaces>`__.

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

With Kokkos, the data can be accessed either through raw pointers or through Kokkos Views. With raw pointers, the memory allocation into the default memory space can be done using ``Kokkos::kokkos_malloc(n * sizeof(int))``. Kokkos Views are a data type that provides a way to access data more efficiently in memory corresponding to a certain Kokkos memory space, such as host memory or device memory. A 1-dimensional view of type int* can be created by ``Kokkos::View<int*> a("a", n)``, where ``"a"`` is a label, and ``n`` is the size of the allocation in the number of integers. Kokkos determines the optimal layout for the data at compile time for best overall performance as a function of the computer architecture. Furthermore, Kokkos handles the deallocation of such memory automatically. More details about Kokkos Views are found `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/View.html>`__.

Finally, Kokkos provides three different parallel operations: ``parallel_for``, ``parallel_reduce``, and ``parallel_scan``. The ``parallel_for`` operation is used to execute a loop in parallel. The ``parallel_reduce`` operation is used to execute a loop in parallel and reduce the results to a single value. The ``parallel_scan`` operation implements a prefix scan. The usage of ``parallel_for`` and ``parallel_reduce`` are demonstrated in the examples later in this chapter. More detail about the parallel operations are found `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html>`__.

Run Kokkos hello.cpp example in simple steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following should work on AMD VEGA90A devices straight out of the box (needs ROCM installation). On NVIDIA Volta V100 devices (needs CUDA installation), use the variables commented out on the Makefile.

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

    $ module load LUMI/23.03 partition/G
    $ module load rocm/5.2.3
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

`SYCL <https://www.khronos.org/sycl/>`__ is a royalty-free, open-standard C++ programming model for multi-device programming. It provides a high-level, single-source programming model for heterogeneous systems, including GPUs. There are several implementations of the standard. For GPU programming, `Intel oneAPI DPC++ <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`__ and `hipSYCL <https://github.com/OpenSYCL/OpenSYCL>`__ are the most popular for desktop and HPC GPUs; `ComputeCPP <https://developer.codeplay.com/products/computecpp/ce/home/>`__ is a good choice for embedded devices. The same standard-compliant SYCL code should work with any implementation, but they are not binary-compatible.

The most recent version of the SYCL standard is SYCL 2020, and it is the version we will be using in this course. 

SYCL compilation
~~~~~~~~~~~~~~~~

Intel oneAPI DPC++
******************

For targeting Intel GPUs, it is enough to install `Intel oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html>`__. Then, the compilation is as simple as ``icpx -fsycl file.cpp``.

It is also possible to use oneAPI for NVIDIA and AMD GPUs. In addition to oneAPI Base Toolkit, the vendor-provided runtime (CUDA or HIP) and the corresponding `Codeplay oneAPI plugin <https://codeplay.com/solutions/oneapi/>`__ must be installed.
Then, the code can be compiled using Intel LLVM compiler bundled with oneAPI:

- ``clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_86 file.cpp`` for targeting CUDA 8.6 NVIDIA GPU,
- ``clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a`` for targeting GFX90a AMD GPU.

hipSYCL
*******

Using hipSYCL for NVIDIA or AMD GPUs also requires having CUDA or HIP installed first. Then ``syclcc`` can be used for compiling the code, specifying the target devices. For example, here is how to compile the program supporting an AMD and an NVIDIA device:

- ``syclcc --hipsycl-targets='hip:gfx90a;cuda:sm_70' file.cpp``


Using SYCL on LUMI
******************

LUMI does not have a system-wide installation of any SYCL framework. For this course, an installation
of hipSYCL 0.9.4 was prepared, which can be loaded as:

.. code-block:: console

    $ module load LUMI/22.08 partition/G
    $ module load rocm/5.3.3
    $ module use /project/project_465000485/Easy_Build_Installations/modules/LUMI/22.08/partition/G/
    $ module load hipSYCL

The default compilation target is preset to MI250 GPUs, so to compile a single C++ file it is enough to call ``syclcc -O2 file.cpp``.

When running applications built with hipSYCL, one can often see the warning "[hipSYCL Warning] dag_direct_scheduler: Detected a requirement that is neither of discard access mode", reflecting the lack of an optimization hint when using buffer-accessor model. The warning is harmless and can be ignored.

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

Examples
^^^^^^^^

Parallel for with Unified Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++

         #include <Kokkos_Core.hpp>
         
         int main(int argc, char* argv[]) {
         
           // Initialize Kokkos
           Kokkos::initialize(argc, argv);
         
           {
             unsigned n = 5;
         
             // Allocate on Kokkos default memory space (Unified Memory)
             int* a = (int*) Kokkos::kokkos_malloc(n * sizeof(int));
             int* b = (int*) Kokkos::kokkos_malloc(n * sizeof(int));
             int* c = (int*) Kokkos::kokkos_malloc(n * sizeof(int));
           
             // Initialize values on host
             for (unsigned i = 0; i < n; i++)
             {
               a[i] = i;
               b[i] = 1;
             }
           
             // Run element-wise multiplication on device
             Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
               c[i] = a[i] * b[i];
             });

             // Kokkos synchronization
             Kokkos::fence();
             
             // Print results
             for (unsigned i = 0; i < n; i++)
               printf("c[%d] = %d\n", i, c[i]);
            
             // Free Kokkos allocation (Unified Memory)
             Kokkos::kokkos_free(a);
             Kokkos::kokkos_free(b);
             Kokkos::kokkos_free(c);
           }
  
           // Finalize Kokkos
           Kokkos::finalize();
           return 0;
         }

   .. tab:: OpenCL

      .. code-block:: C

         // We're using OpenCL C API here, since SVM support in C++ API is unstable on ROCm
         #define CL_TARGET_OPENCL_VERSION 220
         #include <CL/cl.h>
         #include <stdio.h>
         
         // For larger kernels, we can store source in a separate file
         static const char* kernel_source = "                                                 \
           __kernel void dot(__global const int *a, __global const int *b, __global int *c) { \
             int i = get_global_id(0);                                                        \
             c[i] = a[i] * b[i];                                                              \
           }                                                                                  \
         ";
         
         int main(int argc, char *argv[]) {
         
           // Initialize OpenCL
           cl_platform_id platform;
           clGetPlatformIDs(1, &platform, NULL);
           cl_device_id device;
           clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
           cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
           cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
         
           // Compile OpenCL program for found device.
           cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
           clBuildProgram(program, 1, &device, NULL, NULL, NULL);
           cl_kernel kernel = clCreateKernel(program, "dot", NULL);
         
           // Set problem dimensions
           unsigned n = 5;
         
           // Create SVM buffer objects on host side
           int *a = clSVMAlloc(context, CL_MEM_READ_ONLY, n * sizeof(int), 0);
           int *b = clSVMAlloc(context, CL_MEM_READ_ONLY, n * sizeof(int), 0);
           int *c = clSVMAlloc(context, CL_MEM_WRITE_ONLY, n * sizeof(int), 0);
         
           // Pass arguments to device kernel
           clSetKernelArgSVMPointer(kernel, 0, a);
           clSetKernelArgSVMPointer(kernel, 1, b);
           clSetKernelArgSVMPointer(kernel, 2, c);
         
           // Create mappings for host and initialize values
           clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, a, n * sizeof(int), 0, NULL, NULL);
           clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, b, n * sizeof(int), 0, NULL, NULL);
           for (unsigned i = 0; i < n; i++) {
             a[i] = i;
             b[i] = 1;
           }
           clEnqueueSVMUnmap(queue, a, 0, NULL, NULL);
           clEnqueueSVMUnmap(queue, b, 0, NULL, NULL);
         
           size_t globalSize = n;
           clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
         
           // Create mapping for host and print results
           clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, c, n * sizeof(int), 0, NULL, NULL);
           for (unsigned i = 0; i < n; i++)
             printf("c[%d] = %d\n", i, c[i]);
           clEnqueueSVMUnmap(queue, c, 0, NULL, NULL);
         
           // Free SVM buffers
           clSVMFree(context, a);
           clSVMFree(context, b);
           clSVMFree(context, c);
         
           return 0;
         }

   .. tab:: SYCL

      .. code-block:: C++

         #include <sycl/sycl.hpp>

         int main() {

           sycl::queue q;
           unsigned n = 5;

           // Allocate shared memory (Unified Shared Memory)
           int *a = sycl::malloc_shared<int>(n, q);
           int *b = sycl::malloc_shared<int>(n, q);
           int *c = sycl::malloc_shared<int>(n, q);

           // Initialize values on host
           for (unsigned i = 0; i < n; i++) {
             a[i] = i;
             b[i] = 1;
           }

           // Run element-wise multiplication on device
           q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
             c[i] = a[i] * b[i];
           }).wait();

           // Print results
           for (unsigned i = 0; i < n; i++) {
             printf("c[%d] = %d\n", i, c[i]);
           }

           // Free shared memory allocation (Unified Memory)
           sycl::free(a, q);
           sycl::free(b, q);
           sycl::free(c, q);

           return 0;
         }

Parallel for with GPU buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++

          #include <Kokkos_Core.hpp>
          
          int main(int argc, char* argv[]) {
          
            // Initialize Kokkos
            Kokkos::initialize(argc, argv);
          
            {
              unsigned n = 5;
          
              // Allocate space for 5 ints on Kokkos host memory space
              Kokkos::View<int*, Kokkos::HostSpace> h_a("h_a", n);
              Kokkos::View<int*, Kokkos::HostSpace> h_b("h_b", n);
              Kokkos::View<int*, Kokkos::HostSpace> h_c("h_c", n);
          
              // Allocate space for 5 ints on Kokkos default memory space (eg, GPU memory)
              Kokkos::View<int*> a("a", n);
              Kokkos::View<int*> b("b", n);
              Kokkos::View<int*> c("c", n);
            
              // Initialize values on host
              for (unsigned i = 0; i < n; i++)
              {
                h_a[i] = i;
                h_b[i] = 1;
              }
              
              // Copy from host to device
              Kokkos::deep_copy(a, h_a);
              Kokkos::deep_copy(b, h_b);
            
              // Run element-wise multiplication on device
              Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
                c[i] = a[i] * b[i];
              });

              // Copy from device to host
              Kokkos::deep_copy(h_c, c);

              // Print results
              for (unsigned i = 0; i < n; i++)
                printf("c[%d] = %d\n", i, h_c[i]);
            }
            
            // Finalize Kokkos
            Kokkos::finalize();
            return 0;
          }

   .. tab:: OpenCL

      .. code-block:: C++

          // We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
          #define CL_TARGET_OPENCL_VERSION 110
          #define CL_HPP_TARGET_OPENCL_VERSION 110
          #include <CL/cl.hpp>
          
          // For larger kernels, we can store source in a separate file
          static const std::string kernel_source = R"(
            __kernel void dot(__global const int *a, __global const int *b, __global int *c) {
              int i = get_global_id(0);
              c[i] = a[i] * b[i];
            }
          )";
          
          int main(int argc, char *argv[]) {
          
            // Initialize OpenCL
            cl::Device device = cl::Device::getDefault();
            cl::Context context(device);
            cl::CommandQueue queue(context, device);
          
            // Compile OpenCL program for found device.
            cl::Program program(context, kernel_source);
            program.build({device});
            cl::Kernel kernel_dot(program, "dot");
          
            {
              // Set problem dimensions
              unsigned n = 5;
            
              std::vector<int> a(n), b(n), c(n);
            
              // Initialize values on host
              for (unsigned i = 0; i < n; i++) {
                a[i] = i;
                b[i] = 1;
              }
            
              // Create buffers and copy input data to device.
              cl::Buffer dev_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               n * sizeof(int), a.data());
              cl::Buffer dev_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               n * sizeof(int), b.data());
              cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, n * sizeof(int));
            
              // Pass arguments to device kernel
              kernel_dot.setArg(0, dev_a);
              kernel_dot.setArg(1, dev_b);
              kernel_dot.setArg(2, dev_c);
            
              // We don't need to apply any offset to thread IDs
              queue.enqueueNDRangeKernel(kernel_dot, cl::NullRange, cl::NDRange(n), cl::NullRange);
            
              // Read result
              queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, n * sizeof(int), c.data());
            
              // Print results
              for (unsigned i = 0; i < n; i++)
                printf("c[%d] = %d\n", i, c[i]);
            }
          
            return 0;
          }


   .. tab:: SYCL

      .. code-block:: C++

         #include <sycl/sycl.hpp>
         
         int main() {

           sycl::queue q;
           unsigned n = 5;

           // Allocate space for 5 ints
           auto a_buf = sycl::buffer<int>(sycl::range<1>(n));
           auto b_buf = sycl::buffer<int>(sycl::range<1>(n));
           auto c_buf = sycl::buffer<int>(sycl::range<1>(n));

           // Initialize values
           // We should use curly braces to limit host accessors' lifetime
           //    and indicate when we're done working with them:
           {
             auto a_host_acc = a_buf.get_host_access();
             auto b_host_acc = b_buf.get_host_access();
             for (unsigned i = 0; i < n; i++) {
               a_host_acc[i] = i;
               b_host_acc[i] = 1;
             }
           }

           // Submit a SYCL kernel into a queue
           q.submit([&](sycl::handler &cgh) {
             // Create read accessors over a_buf and b_buf
             auto a_acc = a_buf.get_access<sycl::access_mode::read>(cgh);
             auto b_acc = b_buf.get_access<sycl::access_mode::read>(cgh);
             // Create write accesor over c_buf
             auto c_acc = c_buf.get_access<sycl::access_mode::write>(cgh);
             // Run element-wise multiplication on device
             cgh.parallel_for<class vec_add>(sycl::range<1>{n}, [=](sycl::id<1> i) {
                 c_acc[i] = a_acc[i] * b_acc[i];
             });
           });

           // No need to synchronize, creating the accessor for c_buf will do it automatically
           {
               const auto c_host_acc = c_buf.get_host_access();
               // Print results
               for (unsigned i = 0; i < n; i++)
                 printf("c[%d] = %d\n", i, c_host_acc[i]);
           }

           return 0;
         }

Asynchronous parallel for kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++

         #include <Kokkos_Core.hpp>
         
         int main(int argc, char* argv[]) {
         
           // Initialize Kokkos
           Kokkos::initialize(argc, argv);
         
           {
             unsigned n = 5;
             unsigned nx = 20;
         
             // Allocate on Kokkos default memory space (Unified Memory)
             int* a = (int*) Kokkos::kokkos_malloc(nx * sizeof(int));
         
             // Create 'n' execution space instances (maps to streams in CUDA/HIP)
             auto ex = Kokkos::Experimental::partition_space(
               Kokkos::DefaultExecutionSpace(), 1,1,1,1,1);
           
             // Launch 'n' potentially asynchronous kernels 
             // Each kernel has their own execution space instances
             for(unsigned region = 0; region < n; region++) {
               Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ex[region], 
                 nx / n * region, nx / n * (region + 1)), KOKKOS_LAMBDA(const int i) {
                   a[i] = region + i;
                 });
             }

             // Sync execution space instances (maps to streams in CUDA/HIP)
             for(unsigned region = 0; region < n; region++)
               ex[region].fence();

             // Print results
             for (unsigned i = 0; i < nx; i++)
               printf("a[%d] = %d\n", i, a[i]);

             // Free Kokkos allocation (Unified Memory)
             Kokkos::kokkos_free(a);
           }
           
           // Finalize Kokkos
           Kokkos::finalize();
           return 0;
         }

   .. tab:: OpenCL

      .. code-block:: C

         // We're using OpenCL C API here, since SVM support in C++ API is unstable on ROCm
         #define CL_TARGET_OPENCL_VERSION 200
         #include <CL/cl.h>
         #include <stdio.h>
         
         // For larger kernels, we can store source in a separate file
         static const char* kernel_source = "              \
                    __kernel void async(__global int *a) { \
                      int i = get_global_id(0);            \
                      int region = i / get_global_size(0); \
                      a[i] = region + i;                   \
                    }                                      \
         ";
         
         int main(int argc, char *argv[]) {
           // Initialize OpenCL
           cl_platform_id platform;
           clGetPlatformIDs(1, &platform, NULL);
           cl_device_id device;
           clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
           cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
           cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
         
           // Compile OpenCL program for found device.
           cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
           clBuildProgram(program, 1, &device, NULL, NULL, NULL);
           cl_kernel kernel = clCreateKernel(program, "async", NULL);
         
           // Set problem dimensions
           unsigned n = 5;
           unsigned nx = 20;
         
           // Create SVM buffer objects on host side
           int *a = clSVMAlloc(context, CL_MEM_WRITE_ONLY, nx * sizeof(int), 0);
         
           // Pass arguments to device kernel
           clSetKernelArgSVMPointer(kernel, 0, a);
         
           // Launch multiple potentially asynchronous kernels on different parts of the array
           for(unsigned region = 0; region < n; region++) {
             size_t offset = (nx / n) * region;
             size_t size = nx / n;
             clEnqueueNDRangeKernel(queue, kernel, 1, &offset, &size, NULL, 0, NULL, NULL);
           }
         
           // Create mapping for host and print results
           clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, a, nx * sizeof(int), 0, NULL, NULL);
           for (unsigned i = 0; i < nx; i++)
             printf("a[%d] = %d\n", i, a[i]);
           clEnqueueSVMUnmap(queue, a, 0, NULL, NULL);
         
           // Free SVM buffers
           clSVMFree(context, a);
         
           return 0;
         }

   .. tab:: SYCL

      .. code-block:: C++

         #include <sycl/sycl.hpp>
         
         int main() {

           sycl::queue q;
           unsigned n = 5;
           unsigned nx = 20;

           // Allocate shared memory (Unified Shared Memory)
           int *a = sycl::malloc_shared<int>(nx, q);

           // Launch multiple potentially asynchronous kernels on different parts of the array
           for(unsigned region = 0; region < n; region++) {
             q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
               const int iShifted = i + nx / n * region;
               a[iShifted] = region + iShifted;
             });
           }

           // Synchronize
           q.wait();

           // Print results
           for (unsigned i = 0; i < nx; i++)
             printf("a[%d] = %d\n", i, a[i]);

           // Free shared memory allocation (Unified Memory)
           sycl::free(a, q);

           return 0;
         }

Reduction
~~~~~~~~~
.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++

         #include <Kokkos_Core.hpp>
         
         int main(int argc, char* argv[]) {
         
           // Initialize Kokkos
           Kokkos::initialize(argc, argv);
         
           {
             unsigned n = 10;
             
             // Initialize sum variable
             int sum = 0;
           
             // Run sum reduction kernel
             Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int i, int &lsum) {
               lsum += i;
             }, sum);

             // Kokkos synchronization
             Kokkos::fence();

             // Print results
             printf("sum = %d\n", sum);
           }
  
           // Finalize Kokkos
           Kokkos::finalize();
           return 0;
         }

   .. tab:: OpenCL

      .. code-block:: C++

         // We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
         #define CL_TARGET_OPENCL_VERSION 110
         #define CL_HPP_TARGET_OPENCL_VERSION 110
         #include <CL/cl.hpp>
         
         // For larger kernels, we can store source in a separate file
         static const std::string kernel_source = R"(
           __kernel void reduce(__global int* sum, __local int* local_mem) {
             
             // Get work group and work item information
             int gsize = get_global_size(0); // global work size
             int gid = get_global_id(0); // global work item index
             int lsize = get_local_size(0); // local work size
             int lid = get_local_id(0); // local work item index
             
             // Store reduced item into local memory
             local_mem[lid] = gid; // initialize local memory
             barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
             
             // Perform reduction across the local work group
             for (int s = 1; s < lsize; s *= 2) { // loop over local memory with stride doubling each iteration
               if (lid % (2 * s) == 0) {
                 local_mem[lid] += local_mem[lid + s];
               }
               barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
             }
             
             if (lid == 0) { // only one work item per work group
               atomic_add(sum, local_mem[0]); // add partial sum to global sum atomically
             }
           }
         )";
          
         int main(int argc, char* argv[]) {
         
           // Initialize OpenCL
           cl::Device device = cl::Device::getDefault();
           cl::Context context(device);
           cl::CommandQueue queue(context, device);
         
           // Compile OpenCL program for found device
           cl::Program program(context, kernel_source);
           program.build({device});
           cl::Kernel kernel_reduce(program, "reduce");
         
           {
             // Set problem dimensions
             unsigned n = 10;
         
             // Initialize sum variable
             int sum = 0;
         
             // Create buffer for sum
             cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &sum);
         
             // Pass arguments to device kernel
             kernel_reduce.setArg(0, buffer); // pass buffer to device
             kernel_reduce.setArg(1, sizeof(int), NULL); // allocate local memory
         
             // Enqueue kernel
             queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(n), cl::NullRange);
         
             // Read result
             queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(int), &sum);
         
             // Print result
             printf("sum = %d\n", sum);
           }
         
           return 0;
         }


   .. tab:: SYCL

      .. code-block:: C++
      
         // We use built-in sycl::reduction mechanism in this example.
         // The manual implementation of the reduction kernel can be found in
         // the "Non-portable kernel models" chapter.

         #include <sycl/sycl.hpp>
         
         int main() {
           sycl::queue q;
           unsigned n = 10;
         
           // Initialize sum
           int sum = 0;
           {
             // Create a buffer for sum to get the reduction results
             sycl::buffer<int> sum_buf{&sum, 1};
           
             // Submit a SYCL kernel into a queue
             q.submit([&](sycl::handler &cgh) {
               // Create temporary object describing variables with reduction semantics
               auto sum_acc = sum_buf.get_access<sycl::access_mode::read_write>(cgh);
               // We can use built-in reduction primitive
               auto sum_reduction = sycl::reduction(sum_acc, sycl::plus<int>());
           
               // A reference to the reducer is passed to the lambda
               cgh.parallel_for(sycl::range<1>{n}, sum_reduction,
                               [=](sycl::id<1> idx, auto &reducer) { reducer.combine(idx[0]); });
             }).wait();
             // The contents of sum_buf are copied back to sum by the destructor of sum_buf
           }
           // Print results
           printf("sum = %d\n", sum);
         }

Pros and cons of cross-platform portability ecosystems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

General observations
~~~~~~~~~~~~~~~~~~~~

    - The amount of code duplication is minimized
    - The same code can be compiled to multiple architectures from different vendors
    - Limited learning resources compared to CUDA (Stack Overflow, course material, documentation)

Lambda-based kernel models (Kokkos, SYCL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - Higher level of abstraction 
    - Less knowledge of the underlying architecture is needed for initial porting
    - Very nice and readable source code (C++ API)
    - The models are relatively new and not very popular yet
    
Separate-source kernel models (OpenCL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Very good portability
    - Mature ecosystem 
    - Low-level API gives more control and allows fine tuning
    - Both C and C++ APIs available (C++ API is less well supported)
    - The low-level API and separate-source kernel model are less user friendly

.. keypoints::

   - General code organization is similar to non-portable kernel-based models.
   - As long as no vendor-specific functionality is used, the same code can run on any GPU.
