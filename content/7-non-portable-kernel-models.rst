.. _non-portable-kernel-models:


Non-portable kernel-based models
================================

.. questions::

   - q1
   - q2

.. objectives::

   - o1
   - o2

.. instructor-note::

   - 55 min teaching
   - 30 min exercises

"Native" GPU programming
^^^^^^^^^^^^^^^^^^^^^^^^

Unlike some cross-platform portability ecosystems, such as Alpaka, Kokkos, OpenCL, RAJA, and SYCL, which cater to multiple architectures, CUDA and HIP are solely focused on GPUs. They provide extensive libraries, APIs, and compiler toolchains that optimize code execution on NVIDIA GPUs (in the case of CUDA) and both NVIDIA and AMD GPUs (in the case of HIP). Because they are developed by the device producers, these programming models provide high-performance computing capabilities and offer advanced features like shared memory, thread synchronization, and memory management specific to GPU architectures.

CUDA, initially developed by NVIDIA, has gained significant popularity and is widely used for GPU programming. It offers a comprehensive ecosystem that includes not only the CUDA programming model but also a vast collection of GPU-accelerated libraries. Developers can write CUDA kernels using C++ and seamlessly integrate them into their applications to harness the massive parallelism of GPUs.

HIP, on the other hand, is an open-source project that aims to provide a more "portable" GPU programming interface. It allows developers to write GPU code in a syntax similar to CUDA and provides a translation layer that enables the same code to run on both NVIDIA and AMD GPUs. This approach minimizes the effort required to port CUDA code to different GPU architectures and provides flexibility for developers to target multiple platforms.

By being closely tied to the GPU hardware, CUDA and HIP provide a level of performance optimization that may not be achievable with cross-platform portability ecosystems. The libraries and toolchains offered by these programming models are specifically designed to exploit the capabilities of the underlying GPU architectures, enabling developers to achieve high-performance computing.

Developers utilizing CUDA or HIP can tap into an extensive ecosystem of GPU-accelerated libraries, covering various domains, including linear algebra, signal processing, image processing, machine learning, and more. These libraries are highly optimized to take advantage of the parallelism and computational power offered by GPUs, allowing developers to accelerate their applications without having to implement complex algorithms from scratch.

As mentioned before CUDA and HIP are very similar so it makes sense to check them in the same time. 

Hello World
~~~~~~~~~~~

Below we have the most basic example of CUDA and HIP, the "Hello World" program:

.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++

         #include <Kokkos_Core.hpp>
         #include <iostream>
         
         int main() {
           Kokkos::initialize();

           int count = Kokkos::Cuda().concurrency();
           int device = Kokkos::Cuda().impl_internal_space_instance()->impl_internal_space_id();
         
           std::cout << "Hello! I'm GPU " << device << " out of " << count << " GPUs in total." << std::endl;
         
           Kokkos::finalize();
         
           return 0;
         }


   .. tab:: OpenCL

      .. code-block:: C++
      
         #include <CL/opencl.hpp>
         #include <stdio>
         int main(void) {
           cl_uint count;
           cl_device_id device;
           clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, &count);
           
           printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);
           
           return 0;
         }

   .. tab:: SYCL

      .. code-block:: C++

         #include <iostream>
         #include <sycl/sycl.hpp>
         
         int main() {
           auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
           auto count = gpu_devices.size();
           std::cout << "Hello! I'm using the SYCL device: <"
                     << gpu_devices[0].get_info<sycl::info::device::name>()
                     << ">, the first of " << count << " devices." << std::endl;
           return 0;
        }

   .. tab:: CUDA

      .. code-block:: C
      
        #include <cuda_runtime.h>
        #include <cuda.h>
        #include <stdio.h>
          
        int main(void){
          int count, device;
            
          cudaGetDeviceCount(&count);
          cudaGetDevice(&device);
            
          printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count); 
          return 0;
        }

   .. tab:: HIP

      .. code-block:: C
      
          #include <hip/hip_runtime.h>
          #include <stdio.h>
      
          int main(void){
            int count, device;
        
            hipGetDeviceCount(&count);
            hipGetDevice(&device);
        
            printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);
            return 0;


In both versions, we include the necessary headers: **cuda_runtime.h** and **cuda.h** for CUDA, and **hip_runtime.h** for HIP. These headers provide the required functionality for GPU programming.

To retrieve information about the available devices, we use the functions **<cuda/hip>GetDeviceCount** and **<cuda/hip>GetDevice**. These functions allow us to determine the total number of GPUs and the index of the currently used device. In the code examples, we default to using device 0.

As an exercise, modify the "Hello World" code to explicitly use a specific GPU. Do this by using the **<cuda/hip>SetDevice** function, which allows to set the desired GPU device. 
Note that the device number provided has to be within the range of available devices, otherwise, the program may fail to run or produce unexpected results.
To experiment with different GPUs, modify the code to include the following line before retrieving device information:

 .. code-block:: C
 
     cudaSetDevice(deviceNumber); // For CUDA  
     hipSetDevice(deviceNumber); // For HIP
 

Replace **deviceNumber** with the desired GPU device index. Run the code with different device numbers to observe the output. 


Vector addition
^^^^^^^^^^^^^^^


.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++
      
   .. tab:: OpenCL

      .. code-block:: C++
      
   .. tab:: SYCL

      .. code-block:: C++
      
   .. tab:: CUDA

      .. code-block:: C++
      
   .. tab:: HIP

      .. code-block:: C++
      
         #include <hip/hip_runtime.h>
         #include <stdio.h>
         #include <stlib.h>
         #include <math.h> 
         
         __global__ void vector_add(float *A, float *B, float *C, int n){
           
           int tid = threadIdx.x + blockIdx.x * blockDim.x;
           int stride = gridDim.x * blockDim.x;
           if(tid<n){
             C[tid] += A[tid]+B[tid];
           }
        }
        
        int main(void){ 
          const int N = 10000;
          float *Ah, *Bh, *Ch, *Cref;
          float *Ad, *Bd, *Cd;

          // Allocate the arrays on CPU
          Ah =(float*)malloc(n * sizeof(float));
          Bh =(float*)malloc(n * sizeof(float));
          Ch =(float*)malloc(n * sizeof(float));
          Cref =(float*)malloc(n * sizeof(float));
          
          // initialise data and calculate reference values on CPU
          for (i=0; i < n; i++) {
            Ah[i] = sin(i) * 2.3;
            Bh[i] = cos(i) * 1.1;
            Cref[i] = Ah[i] + Bh[i];
          }
          
          // Allocate the arrays on GPU
          hipMalloc((void**)&Ad, N * sizeof(float));
          hipMalloc((void**)&Bd, N * sizeof(float));
          hipMalloc((void**)&Cd, N * sizeof(float));
          
          // Transfer the data from CPU to GPU
          hipMemcpy(Ad, Ah, sizeof(float) * n, hipMemcpyHostToDevice);
          hipMemcpy(Bd, Bh, sizeof(float) * n, hipMemcpyHostToDevice);
          
          // define grid dimensions + launch the device kernel
          dim3 blocks, threads;
          threads=dim3(256,1,1);
          blocks=dim3((N+256-1)/256,1,1);
          
          //Launch Kernel
          // use
          //hipLaunchKernelGGL(vector_add, blocks, threads, 0, 0, Ad, Bd, Cd, N); // or
          vector_add<<< blocks, threads,0,0>>(Ad, Bd, Cd, N);
          
          // copy results back to CPU
          hipMemcpy(Ch, Cd, sizeof(float) * N, hipMemcpyDeviceToHost);
          
          printf("reference: %f %f %f %f ... %f %f\n",
                        Cref[0], Cref[1], Cref[2], Cref[3], Cref[n-2], Cref[n-1]);
          printf("   result: %f %f %f %f ... %f %f\n",
                          Ch[0],   Ch[1],   Ch[2],   Ch[3],   Ch[n-2],   Ch[n-1]);

          // confirm that results are correct
          float error = 0.0;
          float tolerance = 1e-6;
          float diff;
          for (i=0; i < n; i++) {
            diff = abs(y_ref[i] - y[i]);
            if (diff > tolerance){
              error += diff;
            }
          }
         printf("total error: %f\n", error);
         printf("  reference: %f at (42)\n", Cref[42]);
        printf("     result: %f at (42)\n",    Ch[42]);

        return 0;
      }
      
Examples
^^^^^^^^
**I was thinking about having the exact same example cases for each 4 programming models in portable and non-portable kernel chapters (duplicated), so it would be easy to compare cuda,hip,kokkos,opencl, and sycl?**

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

           // This is needed to avoid bug in coarse grain SVMAllocator::allocate()
           cl::CommandQueue::setDefault(queue);
         
           // Compile OpenCL program for found device.
           cl::Program program(context, kernel_source);
           program.build(device);
           cl::Kernel kernel_dot(program, "dot");
         
           {
             // Set problem dimensions
             unsigned n = 5;
           
             // Create SVM buffer objects on host side 
             cl::SVMAllocator<int, cl::SVMTraitReadOnly<>> svmAllocRead(context);
             int *a = svmAllocRead.allocate(n);
             int *b = svmAllocRead.allocate(n);
         
             cl::SVMAllocator<int, cl::SVMTraitWriteOnly<>> svmAllocWrite(context);
             int *c = svmAllocWrite.allocate(n);
           
             // Pass arguments to device kernel
             kernel_dot.setArg(0, a);
             kernel_dot.setArg(1, b);
             kernel_dot.setArg(2, c);
           
             // Create mappings for host and initialize values
             queue.enqueueMapSVM(a, CL_TRUE, CL_MAP_WRITE, n * sizeof(int));
             queue.enqueueMapSVM(b, CL_TRUE, CL_MAP_WRITE, n * sizeof(int));
             for (unsigned i = 0; i < n; i++) {
               a[i] = i;
               b[i] = 1;
             }
             queue.enqueueUnmapSVM(a);
             queue.enqueueUnmapSVM(b);
           
             // We don't need to apply any offset to thread IDs
             queue.enqueueNDRangeKernel(kernel_dot, cl::NullRange, cl::NDRange(n), cl::NullRange);
           
             // Create mapping for host and print results
             queue.enqueueMapSVM(c, CL_TRUE, CL_MAP_READ, n * sizeof(int));
             for (unsigned i = 0; i < n; i++)
               printf("c[%d] = %d\n", i, c[i]);
             queue.enqueueUnmapSVM(c);
           
             // Free SVM buffers
             svmAllocRead.deallocate(a, n);
             svmAllocRead.deallocate(b, n);
             svmAllocWrite.deallocate(c, n);
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

      .. code-block:: C++

         // We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
         #define CL_HPP_MINIMUM_OPENCL_VERSION 200
         #define CL_HPP_TARGET_OPENCL_VERSION 200
         #include <CL/opencl.hpp>
         
         // For larger kernels, we can store source in a separate file
         static const std::string kernel_source = R"(
           __kernel void async(__global int *a) {
             int i = get_global_id(0);
             int region = i / get_global_size(0);
             a[i] = region + i;
           }
         )";
         
         int main(int argc, char *argv[]) {
         
           // Initialize OpenCL
           cl::Device device = cl::Device::getDefault();
           cl::Context context(device);
           cl::CommandQueue queue(context, device);

           // This is needed to avoid bug in coarse grain SVMAllocator::allocate()
           cl::CommandQueue::setDefault(queue);           
         
           // Compile OpenCL program for found device.
           cl::Program program(context, kernel_source);
           program.build(device);
           cl::Kernel kernel_async(program, "async");
         
           {
             // Set problem dimensions
             unsigned n = 5;
             unsigned nx = 20;
           
             // Create SVM buffer object on host side 
             cl::SVMAllocator<int, cl::SVMTraitWriteOnly<>> svmAlloc(context);
             int *a = svmAlloc.allocate(nx);
           
             // Pass arguments to device kernel
             kernel_async.setArg(0, a);
           
             // Launch multiple potentially asynchronous kernels on different parts of the array
             for(unsigned region = 0; region < n; region++) {
               queue.enqueueNDRangeKernel(kernel_async, cl::NDRange(nx / n * region), 
                 cl::NDRange(nx / n), cl::NullRange);
             }
           
             // Create mapping for host and print results
             queue.enqueueMapSVM(a, CL_TRUE, CL_MAP_READ, nx * sizeof(int));
             for (unsigned i = 0; i < nx; i++)
               printf("a[%d] = %d\n", i, a[i]);
             queue.enqueueUnmapSVM(a);
           
             // Free SVM buffer
             svmAlloc.deallocate(a, nx);
           }
         
           return 0;
         }

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
         #define CL_HPP_MINIMUM_OPENCL_VERSION 110
         #define CL_HPP_TARGET_OPENCL_VERSION 110
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

         #include <sycl/sycl.hpp>
         
         int main(int argc, char *argv[]) {
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
               auto sum_reduction = sycl::reduction(sum_buf, cgh, sycl::plus<>());
           
               // A reference to the reducer is passed to the lambda
               cgh.parallel_for(sycl::range<1>{n}, sum_reduction,
                               [=](sycl::id<1> idx, auto &reducer) { reducer.combine(idx[0]); });
             }).wait();
             // The contents of sum_buf are copied back to sum by the destructor of sum_buf
           }
           // Print results
           printf("sum = %d\n", sum);
         }

   .. tab:: CUDA

      .. code-block:: C

         WRITEME

   .. tab:: HIP

      .. code-block:: C

         WRITEME

Pros and cons of native programming models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. keypoints::

   - k1
   - k2
