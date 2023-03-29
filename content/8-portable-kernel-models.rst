.. _portable-kernel-models:

Portable kernel-based models
====================================================================

The goal of the cross-platform portability ecosystems is to allow the same code to run on multiple architectures, therefore reducing code duplication. They are usually based on C++, and use function objects/lambda functions to define the loop body (ie, the kernel), which can run on multiple architectures like CPU, GPU, and FPGA from different vendors. An exception to this is OpenCL, which originally offered only a C API (although currently also C++ API is available), and uses a separate-source model for the kernel code. However, unlike in many conventional CUDA or HIP implementations, the portability ecosystems require kernels to be written only once if one prefers to run it on CPU and GPU for example. Some notable cross-platform portability ecosystems are Kokkos, OpenCL, SYCL, and RAJA. Kokkos and RAJA are individual projects whereas OpenCL and SYCL are standards followed by several projects implementing (and extending) them. For example, some notable SYCL implementations include `Intel DPC++ <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`_, `Open SYCL <https://github.com/OpenSYCL/OpenSYCL>`_ (formerly hipSYCL), `triSYCL <https://github.com/triSYCL/triSYCL>`_, and `ComputeCPP <https://developer.codeplay.com/products/computecpp/ce/home/>`_.

Kokkos
^^^^^^

Kokkos is an open-source performance portability ecosystem for parallelization on large heterogeneous hardware architectures of which development has mostly taken place on Sandia National Laboratories. The project started in 2011 as a parallel C++ programming model, but have since expanded into a more broad ecosystem including Kokkos Core (the programming model), Kokkos Kernels (math library), and Kokkos Tools (debugging, profiling and tuning tools). By preparing proposals for the C++ standard committee, the project also aims to influence the ISO/C++ language standard such that, eventually, Kokkos capabilities will become native to the language standard. A more detailed introduction is found `HERE <https://www.sandia.gov/news/publications/hpc-annual-reports/article/kokkos/>`_.

The Kokkos library provides an abstraction layer for a variety of different custom or native languages such as OpenMP, CUDA, and HIP. Therefore, it allows better portability across different hardware manufactured by different vendors, but introduces an additional dependency to the software stack. For example, when using CUDA, only CUDA installation is required, but when using Kokkos with NVIDIA GPUs, Kokkos and CUDA installation are both required. Kokkos is not a very popular choice for parallel programming, and therefore, learning and using Kokkos can be more difficult compared to more established programming models such as CUDA, for which a much larger amount of search results and stackoverflow discussions can be found.


Kokkos compilation
~~~~~~~~~~~~~~~~~~

Furthermore, one challenge with some cross-platform portability libraries is that even on the same system, different projects may require different combinations of compilation settings for the portability library. For example, in Kokkos, one project may wish the default execution space to be a CUDA device, whereas another requires a CPU. Even if the projects prefer the same execution space, one project may desire the Unified Memory to be the default memory space and the other may wish to use pinned GPU memory. It may be burdensome to maintain a large number of library instances on a single system. However, Kokkos offers a simple way to compile Kokkos library simultaneously with the user project. This is achieved by specifying Kokkos compilation settings (see `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Compiling.html>`_) and including the Kokkos Makefile in the user Makefile. CMake is also supported. This way, the user application and Kokkos library are compiled together. The following is an example Makefile for a single-file Kokkos project (hello.cpp) that uses CUDA (Volta architecture) as the backend (default execution space) and Unified Memory as the default memory space:

.. tabs:: 

   .. tab:: Makefile

      .. code-block:: makefile

         default: build
   
         # Set compiler
         KOKKOS_PATH = $(shell pwd)/kokkos
         CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
         
         # Variables for the Makefile.kokkos
         KOKKOS_DEVICES = "Cuda"
         KOKKOS_ARCH = "Volta70"
         KOKKOS_CUDA_OPTIONS = "enable_lambda,force_uvm"
         
         # Include Makefile.kokkos
         include $(KOKKOS_PATH)/Makefile.kokkos
         
         build: $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) hello.cpp
                 $(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) hello.cpp -o hello


Kokkos programming
~~~~~~~~~~~~~~~~~~

When starting to write a project using Kokkos, the first step is understand Kokkos initialization and finalization. Kokkos must be initialized by calling ``Kokkos::initialize(int& argc, char* argv[])`` and finalized by calling ``Kokkos::finalize()``. More details are given in `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Initialization.html>`_.

Kokkos uses an execution space model to abstract the details of parallel hardware. The execution space instances map to the available backend options such as CUDA, OpenMP, HIP, or SYCL. If the execution space is not explicitly chosen by the programmer in the source code, the default execution space ``Kokkos::DefaultExecutionSpace`` is used. This is chosen when the Kokkos library is compiled. The Kokkos execution space model is described in more detail in `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-spaces>`_.

Similarly, Kokkos uses a memory space model for different types of memory, such as host memory or device memory. If not defined explicitly, Kokkos uses the default memory space specified during Kokkos compilation as described `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-memory-spaces>`_.

The following is an example of a Kokkos program that initializes Kokkos and prints the execution space and memory space instances: 

.. tabs:: 

   .. tab:: C++
      
      .. code-block:: C++

         #include <Kokkos_Core.hpp>
         #include <iostream>
         
         int main(int argc, char* argv[]) {
           Kokkos::initialize(argc, argv);
           std::cout << "Execution Space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
           std::cout << "Memory Space: " << typeid(Kokkos::DefaultExecutionSpace::memory_space).name() << std::endl;
           Kokkos::finalize();
           return 0;
         }

With Kokkos, the data can be accessed either through raw pointers or through Kokkos Views. With raw pointers, the memory allocation into the default memory space can be done using ``Kokkos::kokkos_malloc(n * sizeof(int))``. Kokkos Views are a data type that provides a way to access data more efficiently in memory corresponding to a certain Kokkos memory space, such as host memory or device memory. A 1-dimensional view of type int* can be created by ``Kokkos::View<int*> a("a", n)``, where ``"a"`` is a label, and ``n`` is the size of the allocation in the number of integers. Kokkos determines the optimal layout for the data at compile time for best overall performance as a function of the computer architecture. Furthermore, Kokkos handles the deallocation of such memory automatically. More details about Kokkos Views are found `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/View.html>`_.

Finally, Kokkos provides three different parallel operations: ``parallel_for``, ``parallel_reduce``, and ``parallel_scan``. The ``parallel_for`` operation is used to execute a loop in parallel. The ``parallel_reduce`` operation is used to execute a loop in parallel and reduce the results to a single value. The ``parallel_scan`` operation is used to execute a loop in parallel and scan the results. The usage of ``parallel_for`` and ``parallel_reduce`` are demonstrated in the examples later in this chapter. More detail about the parallel operations are found `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html>`_.



OpenCL
^^^^^^


SYCL
^^^^

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
               b[i] = i;
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

      .. code-block:: C++

         // We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
         #define CL_HPP_MINIMUM_OPENCL_VERSION 200
         #define CL_HPP_TARGET_OPENCL_VERSION 200
         #include <CL/opencl.hpp>
         
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
           program.build(device);
           cl::Kernel kernel_dot(program, "dot");
         
           {
             unsigned n = 5;
           
             // Create SVM buffer object on host side 
             int *a = (int*)clSVMAlloc(context(), CL_MEM_READ_ONLY, n * sizeof(int), 0);
             int *b = (int*)clSVMAlloc(context(), CL_MEM_READ_ONLY, n * sizeof(int), 0);
             int *c = (int*)clSVMAlloc(context(), CL_MEM_WRITE_ONLY, n * sizeof(int), 0);
           
             // Pass arguments to device kernel
             clSetKernelArgSVMPointer(kernel_dot(), 0, a);
             clSetKernelArgSVMPointer(kernel_dot(), 1, b);
             clSetKernelArgSVMPointer(kernel_dot(), 2, c);
           
             // Create mappings for host and initialize values
             clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_WRITE, a, n * sizeof(int), 0, NULL, NULL);
             clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_WRITE, b, n * sizeof(int), 0, NULL, NULL);
             for (unsigned i = 0; i < n; i++) {
               a[i] = i;
               b[i] = 1;
             }
             clEnqueueSVMUnmap(queue(), a, 0, NULL, NULL);
             clEnqueueSVMUnmap(queue(), b, 0, NULL, NULL);
           
             // We don't need to apply any offset to thread IDs
             queue.enqueueNDRangeKernel(kernel_dot, cl::NullRange, cl::NDRange(n), cl::NullRange);
           
             // Create mapping for host and print results
             clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_READ, c, n * sizeof(int), 0, NULL, NULL);
             for (unsigned i = 0; i < n; i++)
               printf("c[%d] = %d\n", i, c[i]);
             clEnqueueSVMUnmap(queue(), c, 0, NULL, NULL);
           
             // Free SVM buffers
             clSVMFree(context(), a);
             clSVMFree(context(), b);
             clSVMFree(context(), c);
           }
         
           return 0;
         }

   .. tab:: SYCL

      .. code-block:: C++

         #include <sycl/sycl.hpp>

         int main(int argc, char* argv[]) {

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

   .. tab:: CUDA

      .. code-block:: C

         WRITEME

   .. tab:: HIP

      .. code-block:: C

         WRITEME

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
                h_b[i] = i;
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

              // Kokkos synchronization
              Kokkos::fence();

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
          #define CL_HPP_MINIMUM_OPENCL_VERSION 110
          #define CL_HPP_TARGET_OPENCL_VERSION 110
          #include <CL/opencl.hpp>
          
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
            program.build(device);
            cl::Kernel kernel_dot(program, "dot");
          
            {
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
         
         int main(int argc, char **argv) {

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

   .. tab:: CUDA

      .. code-block:: C

         WRITEME

   .. tab:: HIP

      .. code-block:: C

         WRITEME

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
         
             // Allocate on Kokkos default memory space (eg, GPU memory)
             Kokkos::View<int*> a("a", nx);
         
             // Create execution space instances (maps to streams in CUDA/HIP) for each region
             auto ex = Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(),1,1,1,1,1);
           
             // Launch multiple potentially asynchronous kernels in different execution space instances
             for(unsigned region = 0; region < n; region++) {
               Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ex[region], nx / n * region, nx / n * (region + 1)), KOKKOS_LAMBDA(const int i) {
                 a[i] = region + i;
               });
             }

             // Sync execution space instances (maps to streams in CUDA/HIP)
             for(unsigned region = 0; region < n; region++)
               ex[region].fence();

             // Print results
             for (unsigned i = 0; i < nx; i++)
               printf("a[%d] = %d\n", i, a[i]);
           }
           
           // Finalize Kokkos
           Kokkos::finalize();
           return 0;
         }

   .. tab:: OpenCL

      .. code-block:: C++


   .. tab:: SYCL

      .. code-block:: C++

         #include <sycl/sycl.hpp>
         
         int main(int argc, char* argv[]) {

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

   .. tab:: CUDA

      .. code-block:: C++

         WRITEME

   .. tab:: HIP

      .. code-block:: C++

         WRITEME

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
             unsigned n = 5;
             
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
         #define CL_HPP_MINIMUM_OPENCL_VERSION 200
         #define CL_HPP_TARGET_OPENCL_VERSION 200
         #include <CL/opencl.hpp>
         
         
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
           program.build(device);
           cl::Kernel kernel_reduce(program, "reduce");
         
           {
             unsigned n = 10;
         
             // Create SVM buffer for sum
             int *sum = (int*)clSVMAlloc(context(), CL_MEM_READ_WRITE, sizeof(int), 0);
         
             // Pass arguments to device kernel
             clSetKernelArgSVMPointer(kernel_reduce(), 0, sum); // pass SVM pointer to device
             kernel_reduce.setArg(1, sizeof(int), NULL); // allocate local memory
         
             // Create mapping for host and initialize sum variable
             clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_WRITE, sum, sizeof(int), 0, NULL, NULL);
             *sum = 0;
             clEnqueueSVMUnmap(queue(), sum, 0, NULL, NULL);
         
             // Enqueue kernel
             queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(n), cl::NullRange);
         
             // Create mapping for host and print result
             clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_READ, sum, sizeof(int), 0, NULL, NULL);
             printf("sum = %d\n", *sum);
             clEnqueueSVMUnmap(queue(), sum, 0, NULL, NULL);
         
             // Free SVM buffer
             clSVMFree(context(), sum);
           }
         
           return 0;
         }


   .. tab:: SYCL

      .. code-block:: C++

         #include <sycl/sycl.hpp>

         int main(int argc, char *argv[]) {
           sycl::queue q;
           unsigned n = 5;

           // Buffers with just 1 element to get the reduction results
           int* sum = sycl::malloc_shared<int>(1, q);
           *sum = 0;

           q.submit([&](sycl::handler &cgh) {
             // Create temporary objects describing variables with reduction semantics
             auto sum_reduction = sycl::reduction(sum, sycl::plus<int>());

             // A reference to the reducer is passed to the lambda
             cgh.parallel_for(sycl::range<1>{n}, sum_reduction,
                               [=](sycl::id<1> idx, auto &reducer) { reducer.combine(idx[0]); });
           }).wait();

           // Print results
           printf("sum = %d\n", *sum);
         }

   .. tab:: CUDA

      .. code-block:: C

         WRITEME

   .. tab:: HIP

      .. code-block:: C

         WRITEME

Pros and cons of cross-platform portability ecosystems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The amount of code duplication is minimized

    The same code can be compiled to multiple architectures from different vendors

    Higher level of abstraction, does not require as much knowledge of the underlying architecture

    Less matured ecosystem compared to CUDA, more uncertainty about future

    Less learning resources (stackoverflow, course material, documentation)


.. keypoints::

   - k1
   - k2
