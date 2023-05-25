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

Fundamentals of GPU programming with CUDA and HIP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
~~~~~~~~~~~~~~~
To demonstrate the fundamental features of CUDA/HIP programming, let's begin with a straightforward task of element-wise vector addition. The code snippet below demonstrates how to utilize CUDA and HIP for efficiently executing this operation.

.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++
        
      
   .. tab:: OpenCL

      .. code-block:: C++
      

   .. tab:: SYCL

      .. code-block:: C++

         #include <iostream>
         #include <sycl/sycl.hpp>
         
         using namespace sycl;
         
         int main(int argc, char *argv[]) { 
         int N=10000;
         queue q{default_selector{}}; // the queue will be executed on the best device in the system 
         
         std::vector<float> Ah(N);
         std::vector<float> Bh(N);
         std::vector<float> Ch(N);
         std::vector<float> Cref(N);

         // Initialize data and calculate reference values on CPU
         for (int i = 0; i < N; i++) {
          Ah[i] = std::sin(i) * 2.3f;
          Bh[i] = std::cos(i) * 1.1f;
          Cref[i] = Ah[i] + Bh[i];
         }
         
         // Allocate the arrays on GPU
         float* Ad = malloc_device<float>(N, q);
         float* Bd = malloc_device<float>(N, q);
         float* Cd = malloc_device<float>(N, q);
         
         q.memcpy(Ad, Ah.data(), N * sizeof(double));
         q.memcpy(Cd, Ch.data(), N * sizeof(double));
         
         // Define grid dimensions and launch the device kernel
         auto threads = range<1>(256);
         
         range<1> global_size(N);
         q.submit([&](handler& h) {
             h.parallel_for(vector_add, nd_range<1>(global_size, threads), [=](nd_item<1> item) {
                  int tid = item.get_global_id(0);
                  Cd[tid] = Ad[tid] + Bd[tid];
             });
         });
         // Copy results back to CPU
         q.submit([&](handler& h) {
            h.memcpy(Ch.data(), Cd, sizeof(float) * N);
         }).wait(); // wait for all operations in the queue q to finish. 

         // Print reference and result values
        std::cout << "Reference: "<< Cref[0] << Cref[1] << Cref[2] << Cref[3] << Cref[N-2] << Cref[N-1] << " " <<<std::endl;
        std::cout << "Result   : "<< Ch[0]   << Ch[1]     << Ch[2] << Ch[3]    << Ch[N-2]  <<   Ch[N-1] << " " <<<std::endl;
        

        // Compare results and calculate the total error
        float error = 0.0f;
        float tolerance = 1e-6f;
        for (int i = 0; i < N; i++) {
        float diff = std::abs(Cref[i] - Ch[i]);
           if (diff > tolerance) {
            error += diff;
           }
        }

       std::cout << "Total error: " << error << std::endl;
       std::cout << "Reference:   " << Cref[42] << " at (42)" << std::endl;
       std::cout << "Result   :   " << Ch[42]   << " at (42)" << std::endl;

       // Free the GPU memory
       free(Ad, q);
       free(Bd, q);
       free(Cd, q);

       return 0;
    }
      
   .. tab:: CUDA

      .. code-block:: C++

        #include <stdio.h>
        #include <cuda.h>
        #inclde <cuda_runtime.h>
        #include <math.h>

        __global__ void vector_add(float *A, float *B, float *C, int n) {
          int tid = threadIdx.x + blockIdx.x * blockDim.x;
          if (tid < n) {
              C[tid] = A[tid] + B[tid];
          }
        }

        int main(void) {
          const int N = 10000;
          float *Ah, *Bh, *Ch, *Cref;
          float *Ad, *Bd, *Cd;
          int i;

          // Allocate the arrays on CPU
          Ah = (float*)malloc(N * sizeof(float));
          Bh = (float*)malloc(N * sizeof(float));
          Ch = (float*)malloc(N * sizeof(float));
          Cref = (float*)malloc(N * sizeof(float));

          // initialise data and calculate reference values on CPU
          for (i = 0; i < N; i++) {
              Ah[i] = sin(i) * 2.3;
              Bh[i] = cos(i) * 1.1;
              Cref[i] = Ah[i] + Bh[i];
          }

          // Allocate the arrays on GPU
          cudaMalloc((void**)&Ad, N * sizeof(float));
          cudaMalloc((void**)&Bd, N * sizeof(float));
          cudaMalloc((void**)&Cd, N * sizeof(float));

          // Transfer the data from CPU to GPU
          cudaMemcpy(Ad, Ah, sizeof(float) * N, cudaMemcpyHostToDevice);
          cudaMemcpy(Bd, Bh, sizeof(float) * N, cudaMemcpyHostToDevice);

          // define grid dimensions + launch the device kernel
          dim3 blocks, threads;
          threads = dim3(256, 1, 1);
          blocks = dim3((N + 256 - 1) / 256, 1, 1);

          // Launch Kernel
          vector_add<<<blocks, threads>>>(Ad, Bd, Cd, N);

          // copy results back to CPU
          cudaMemcpy(Ch, Cd, sizeof(float) * N, cudaMemcpyDeviceToHost);

          printf("reference: %f %f %f %f ... %f %f\n",
              Cref[0], Cref[1], Cref[2], Cref[3], Cref[N - 2], Cref[N - 1]);
          printf("   result: %f %f %f %f ... %f %f\n",
              Ch[0], Ch[1], Ch[2], Ch[3], Ch[N - 2], Ch[N - 1]);

          // confirm that results are correct
          float error = 0.0;
          float tolerance = 1e-6;
          float diff;
          for (i = 0; i < N; i++) {
              diff = fabs(Cref[i] - Ch[i]);
              if (diff > tolerance) {
                  error += diff;
              }
          }
          printf("total error: %f\n", error);
          printf("  reference: %f at (42)\n", Cref[42]);
          printf("     result: %f at (42)\n", Ch[42]);

          // Free the GPU arrays
          cudaFree(Ad);
          cudaFree(Bd);
          cudaFree(Cd);

          // Free the CPU arrays
          free(Ah);
          free(Bh);
          free(Ch);
          free(Cref);

          return 0;
        }

      
   .. tab:: HIP

      .. code-block:: C++
      
         #include <hip/hip_runtime.h>
         #include <stdio.h>
         #include <stlib.h>
         #include <math.h> 
         
         __global__ void vector_add(float *A, float *B, float *C, int n){
           
           int tid = threadIdx.x + blockIdx.x * blockDim.x;
           if(tid<n){
             C[tid] = A[tid]+B[tid];
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
         
         // Free the GPU arrays
         hipFree(Ad);
         hipFree(Bd);
         hipFree(Cd);

         // Free the CPU arrays
         free(Ah);
         free(Bh);
         free(Ch);
         free(Cref);

         return 0;
       }

In this case, the CUDA and HIP codes are equivalent one to one so we will only refer to the CUDA version. The CUDA and HIP programming model are host centric programming models. The main program is executed on CPU and controls all the operations, memory allocations, data transfers between CPU and GPU, and launches the kernels to be executed on the GPU. The code starts with defining the GPU kernel function called **vector_add** with attribute **___global__**. It takes three input arrays `A`, `B`, and `C` along with the array size `n`. The kernel function contains the actually code which is executed on the GPU by multiple threads in parallel.

Accelerators in general and GPUs in particular have their own dedicated memory separate from the system memory (**this could change soon! see AMD MI300 and Nvidia Hopper!**). When programming for GPUs, there are two sets of pointers involved and it's necessary to manage data movement between the host memory and the accelerator memory.  Data needs to be explicitly copied from the host memory to the accelerator memory before it can be processed by the accelerator. Similarly, results or modified data may need to be copied back from the accelerator memory to the host memory to make them accessible to the CPU. 

The main function of the code initializes the input arrays `Ah, Bh` on the CPU and computes the reference array `Cref`. It then allocates memory on the GPU for the input and output arrays `Ad, Bd`, and `Cd`  using **cudaMalloc**. The data is transferred from the CPU to the GPU using hipMemcpy, and then the GPU kernel is launched using the `<<<.>>>` syntax.  All kernels launch are asynchrouneous. After launch the control returns to the `main()` and the code proceeds to the next instructions. 

After the kernel execution, the result array `Cd` is copied back to the CPU using **cudaMemcpy**. The code then prints the reference and result arrays, calculates the error by comparing the reference and result arrays. Finally, the GPU and CPU memory are deallocated using **cudaFree** and **free** functions, respectively. 

The host functions  **cudaSetDevice**, **cudaMalloc**, **cudaMemcpy**, and **cudaFree** are blocking, i.e. the code does not continues to next instructions until the operations are completed. However this is not the defualt behaiviour,  for many operations there are asynchrounous equivalents and there are as well many library calls return the control to the `main()` after calling. This allows the developers to launch idependent operations and overlap them. 

In short, this code demonstrates how to utilize the CUDA and HIP to perform vector addition on a GPU, showcasing the steps involved in allocating memory, transferring data between the CPU and GPU, launching a kernel function, and handling the results. It serves as a starting point for GPU-accelerated computations using CUDA and HIP.

In order to practice the concepts shown above, edit the skeleton code in the repository and the code corrresponding to  setting the device, memory allocations and transfers, and the kernel execution. 

Vector Addition with Unified Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a while already GPUs upport unified memory, which allows to use the same pointer for both CPU and GPU data. This simplifies developing codes by removing the explicit data transfers. The data resides on CPU until it is neeed on GPU or viceversa. However  the data transfers still happens "under the hood" and the developer needs to construct the code to avoid unecessary transfers. Below one can see the modified vector addition codes:


.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++
      
   .. tab:: OpenCL

      .. code-block:: C++
      
   .. tab:: SYCL

      .. code-block:: C++

         #include <iostream>
         #include <sycl/sycl.hpp>
         
         using namespace sycl;
         
         int main(int argc, char *argv[]) { 
         int N=10000;
         queue q{default_selector{}}; // the queue will be executed on the best device in the system 
         
         // Allocate the arrays
         float* Ah = malloc_shared<float>(N, q);
         float* Bh = malloc_shared<float>(N, q);
         float* Ch = malloc_shared<float>(N, q);
         float* Cref = malloc_shared<float>(N, q);

         // Initialize data and calculate reference values on CPU
         for (int i = 0; i < N; i++) {
          Ah[i] = std::sin(i) * 2.3f;
          Bh[i] = std::cos(i) * 1.1f;
          Cref[i] = Ah[i] + Bh[i];
         }
         
         // Define grid dimensions and launch the device kernel
         auto threads = range<1>(256);
         
         range<1> global_size(N);
         
         q.submit([&](handler& h) {
             h.parallel_for(vector_add, nd_range<1>(global_size, threads), [=](nd_item<1> item) {
                  int tid = item.get_global_id(0);
                  Cref[tid] = Ah[tid] + Bh[tid];
             });
         }).wait(); // wait for all operations in the queue q to finish. 

         // Print reference and result values
        std::cout << "Reference: "<< Cref[0] << Cref[1] << Cref[2] << Cref[3] << Cref[N-2] << Cref[N-1] << " " <<<std::endl;
        std::cout << "Result   : "<< Ch[0]   << Ch[1]     << Ch[2] << Ch[3]    << Ch[N-2]  <<   Ch[N-1] << " " <<<std::endl;
        

        // Compare results and calculate the total error
        float error = 0.0f;
        float tolerance = 1e-6f;
        for (int i = 0; i < N; i++) {
        float diff = std::abs(Cref[i] - Ch[i]);
           if (diff > tolerance) {
            error += diff;
           }
        }

       std::cout << "Total error: " << error << std::endl;
       std::cout << "Reference:   " << Cref[42] << " at (42)" << std::endl;
       std::cout << "Result   :   " << Ch[42]   << " at (42)" << std::endl;

       // Free the GPU memory
       free(Ad, q);
       free(Bd, q);
       free(Cd, q);
       free(Cref, q);

       return 0;
    }
      
   .. tab:: CUDA

      .. code-block:: C++

        #include <stdio.h>
        #include <cuda.h>
        #inclde <cuda_runtime.h>
        #include <math.h>

        __global__ void vector_add(float *A, float *B, float *C, int n) {
          int tid = threadIdx.x + blockIdx.x * blockDim.x;
          if (tid < n) {
              C[tid] = A[tid] + B[tid];
          }
        }

        int main(void) {
          const int N = 10000;
          float *Ah, *Bh, *Ch, *Cref;
          int i;

          // Allocate the arrays using Unified Memory
          cudaMallocManaged(&Ah, N * sizeof(float));
          cudaMallocManaged(&Bh, N * sizeof(float));
          cudaMallocManaged(&Ch, N * sizeof(float));
          cudaMallocManaged(&Cref, N * sizeof(float));


          // initialise data and calculate reference values on CPU
          for (i = 0; i < N; i++) {
              Ah[i] = sin(i) * 2.3;
              Bh[i] = cos(i) * 1.1;
              Cref[i] = Ah[i] + Bh[i];
          }

          // define grid dimensions
          dim3 blocks, threads;
          threads = dim3(256, 1, 1);
          blocks = dim3((N + 256 - 1) / 256, 1, 1);

          // Launch Kernel
          vector_add<<<blocks, threads>>>(Ah, Bh, Ch, N);
          cudaDeviceSynchronize(); // Wait for the kernel to complete
          
          //At this point we want to access the data on CPU
          printf("reference: %f %f %f %f ... %f %f\n",
              Cref[0], Cref[1], Cref[2], Cref[3], Cref[N - 2], Cref[N - 1]);
          printf("   result: %f %f %f %f ... %f %f\n",
              Ch[0], Ch[1], Ch[2], Ch[3], Ch[N - 2], Ch[N - 1]);

          // confirm that results are correct
          float error = 0.0;
          float tolerance = 1e-6;
          float diff;
          for (i = 0; i < N; i++) {
              diff = fabs(Cref[i] - Ch[i]);
              if (diff > tolerance) {
                  error += diff;
              }
          }
          printf("total error: %f\n", error);
          printf("  reference: %f at (42)\n", Cref[42]);
          printf("     result: %f at (42)\n", Ch[42]);

          // Free the GPU arrays
          cudaFree(Ah);
          cudaFree(Bh);
          cudaFree(Ch);
          cudaFree(Cref);
          
          return 0;
        }

      
   .. tab:: HIP

      .. code-block:: C++ 
         
         #include <hip/hip_runtime.h>
         #include <stdio.h>
         #include <math.h>

         __global__ void vector_add(float *A, float *B, float *C, int n) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;            
            if (tid < n) {
              C[tid] = A[tid] + B[tid];
           }
         }
         
         int main(void) { 
           const int N = 10000;
           float *Ah, *Bh, *Ch, *Cref;
           // Allocate the arrays using Unified Memory  
           hipMallocManaged((void **)&Ah, N * sizeof(float));
           hipMallocManaged((void **)&Bh, N * sizeof(float));
           hipMallocManaged((void **)&Ch, N * sizeof(float));
           hipMallocManaged((void **)&Cref, N * sizeof(float));

           // Initialize data and calculate reference values on CPU
           for (int i = 0; i < N; i++) {
             Ah[i] = sin(i) * 2.3;
             Bh[i] = cos(i) * 1.1;
             Cref[i] = Ah[i] + Bh[i];
           }
           // All data at this point is on CPU

           // Define grid dimensions + launch the device kernel
           dim3 blocks, threads;
           threads = dim3(256, 1, 1);
           blocks = dim3((N + 256 - 1) / 256, 1, 1);
           
           //Launch Kernel
           // use
           //hipLaunchKernelGGL(vector_add, blocks, threads, 0, 0, Ah, Bh, Ch, N); // or
           vector_add<<<blocks, threads>>>(Ah, Bh, Ch, N);
           hipDeviceSynchronize(); // Wait for the kernel to complete

           // At this point we want to access the data on the CPU
           printf("reference: %f %f %f %f ... %f %f\n",
                 Cref[0], Cref[1], Cref[2], Cref[3], Cref[N - 2], Cref[N - 1]);
           printf("   result: %f %f %f %f ... %f %f\n",
                 Ch[0], Ch[1], Ch[2], Ch[3], Ch[N - 2], Ch[N - 1]);

           // Confirm that results are correct
           float error = 0.0;
           float tolerance = 1e-6;
           float diff;
           for (int i = 0; i < N; i++) {
           diff = fabs(Cref[i] - Ch[i]);
             if (diff > tolerance) {
               error += diff;
             }
           }
           printf("total error: %f\n", error);
           printf("  reference: %f at (42)\n", Cref[42]);
           printf("     result: %f at (42)\n", Ch[42]);

           // Free the Unified Memory arrays
           hipFree(Ah);
           hipFree(Bh);
           hipFree(Ch);
           hipFree(Cref);

           return 0;
         }

Now the arrays Ah, Bh, Ch, and Cref are using cudaMallocManaged to allocate Unified Memory. The **vector_add kernel** is launched by passing these Unified Memory pointers directly. After the kernel launch, **cudaDeviceSynchronize** is used to wait for the kernel to complete execution. Finally, **cudaFree** is used to free the Unified Memory arrays.The Unified Memory allows for transparent data migration between CPU and GPU, eliminating the need for explicit data transfers.

As an exercise modify the skeleton code for vector addition to use Unified Memory. 

Memory Optimizations
^^^^^^^^^^^^^^^^^^^^
Vector addition is a relatively simple, straight forward case. Each thread reads data from, does an addition and then saves the result. Two  adjacent threads access memory location in memory close to each other. Also the data is used only once. In practice this not the case. Also sometimes the same data is used several times resulting in additional memory accesses. 

Memory optimization is one of the most important type of optimization done to efficiently use the GPUs. Before looking how it is done in practice let's revisit some basic concepts about GPUs and execution model.  GPUa are comprised many ligth cores, the so-called Streaming Processors (SP) in CUDA, which are physically group togheter in units, i.e. Streaming Multi-Processors (SMP) in CUDA architecture (note that in AMD the equivalent is called Computing Units, while in Intel GPUs they are Execution Units). The work is done on GPUs by launching many threads each executing an instance of the same kernel. The order of execution is not defined, and the threads can only exchange information in specific conditions. Because of the way the SPs are grouped the threads are also grouped in **blocks**. Each **block** is assigned to an SMP, and can not be splitted. An SMP can have more than block residing at a moment, however there is no communications between the threads in different blocks. In addition to the SPs, each SMP contains very fast moemory which in CUDA is refered to as `shared memory`. The threads in a block can read and write to the shared memory and use it as a user controled cache. One thread can for example write to a location in the shared memory while another thread in the same block can read and use that data. In order to be sure that all threads in the block completed writing  **__syncthreads()** function has to be used to make the threads in the block  wait untill all of them reached the specific place in the kernel. Another important aspect in the GPU programming model is that the threads in the block are not executed indepentely. The threads in a block are physically grouped in warps of size 32 in CUDA or wavefronts of size 64 in ROCm devices. All memory accesses of the global GPU memory are done per warp. When data is needed for some calculations a warp loads from the GPU memory blocks of specific size (64 or 128 Bytes). These operation is very expensive, it has a latency of hundreds of cycles. This means that the threads in a warp should work with elemetns of the data located close in the memmory. In the vector addition two threads near each other, of index tid and tid+1, access elements adjacent in the GPU memory.  

The shared memory can be used to improve performance in two ways. It is possible to avoid extra reads from the memory when several threads in the same block need the same data or it can be used to improve the memory access patterns.

Matrix Transpose
^^^^^^^^^^^^^^^^
Matrix transpose is a classic example where shared memory can significantly improve the performance. The use of shared memory reduces global memory accesses and exploits the high bandwidth and low latency of shared memory.

.. figure:: img/concepts/transpose_img.png
   :align: center

First as a reference we use a simple kernel which copy the data from one array to the other. 

.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++
      
   .. tab:: OpenCL

      .. code-block:: C++
      
   .. tab:: SYCL

      .. code-block:: C++

         #include <iostream>
         #include <sycl/sycl.hpp>
         
         using namespace sycl;
         
   .. tab:: CUDA

      .. code-block:: C++

        #include <stdio.h>
        #include <cuda.h>
        #inclde <cuda_runtime.h>
        #include <math.h>
      
   .. tab:: HIP

      .. code-block:: C++ 
      
         #include <hip/hip_runtime.h>

         #include <cstdlib>
         #include <vector>

         const static int width = 4096;
         const static int height = 4096;

         __global__ void copy_kernel(float *in, float *out, int width, int height) {
            int x_index = blockIdx.x * tile_dim + threadIdx.x;
            int y_index = blockIdx.y * tile_dim + threadIdx.y;

            int index = y_index * width + x_index;

            out[index] = in[index];
        }
        
        int main() {
           std::vector<float> matrix_in;
           std::vector<float> matrix_out;

           matrix_in.resize(width * height);
           matrix_out.resize(width * height);

           for (int i = 0; i < width * height; i++) {
             matrix_in[i] = (float)rand() / (float)RAND_MAX;
           }
        
           float *d_in,*d_out;
        
           hipMalloc((void **)&d_in, width * height * sizeof(float));
           hipMalloc((void **)&d_out, width * height * sizeof(float));

           hipMemcpy(d_in, matrix_in.data(), width * height * sizeof(float),
                  hipMemcpyHostToDevice);

           printf("Setup complete. Launching kernel \n");
           int block_x = width / tile_dim;
           int block_y = height / tile_dim;
  
           // Create events
           hipEvent_t start_kernel_event;
           hipEventCreate(&start_kernel_event);
           hipEvent_t end_kernel_event;
           hipEventCreate(&end_kernel_event);

           printf("Warm up the gpu!\n");
           for(int i=1;i<=10;i++){
              copy_kernel<<<dim3(block_x, block_y),dim3(tile_dim, tile_dim)>>>(d_in, d_out, width,height);
           }

           hipEventRecord(start_kernel_event, 0);
        
           for(int i=1;i<=10;i++){
              copy_kernel<<<dim3(block_x, block_y),dim3(tile_dim, tile_dim)>>>(d_in, d_out, width,height);
           }
  
          hipEventRecord(end_kernel_event, 0);
          hipEventSynchronize(end_kernel_event);

          hipDeviceSynchronize();
          float time_kernel;
          hipEventElapsedTime(&time_kernel, start_kernel_event, end_kernel_event);

          printf("Kernel execution complete \n");
          printf("Event timings:\n");
          printf("  %.6f ms - copy \n  Bandwidth %.6f GB/s\n", time_kernel/10, 2.0*10000*(((double)(width)*      (double)height)*sizeof(float))/(time_kernel*1024*1024*1024));
 
          hipMemcpy(matrix_out.data(), d_out, width * height * sizeof(float),
                     hipMemcpyDeviceToHost);

          return 0;
        }

We note that this code does not do any calculations. Each thread reads one element and then writes it to another locations. By measuring the execution time of the kernel we can compute the effective bandwidth achieve by this kernel. We can measure the time b using **rocprof** or **cuda/hip events**. On a Nvidia V100 GPU this code achieves `717 GB/s` out of the theoretical peak `900 GB/s`. 

Now we do the first iteration of the code, a naive transpose. The reads have a nice coalesced access pattern, but the writing is now very inefficient. 

.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++
      
   .. tab:: OpenCL

      .. code-block:: C++
      
   .. tab:: SYCL

      .. code-block:: C++

         #include <iostream>
         #include <sycl/sycl.hpp>
         
         using namespace sycl;
         
   .. tab:: CUDA

      .. code-block:: C++

        #include <stdio.h>
        #include <cuda.h>
        #inclde <cuda_runtime.h>
        #include <math.h>
      
   .. tab:: HIP

      .. code-block:: C++ 
         
         __global__ void transpose_naive_kernel(float *in, float *out, int width, int height) {
            int x_index = blockIdx.x * tile_dim + threadIdx.x;
            int y_index = blockIdx.y * tile_dim + threadIdx.y;

            int in_index = y_index * width + x_index;
            int out_index = x_index * height + y_index;

           out[out_index] = in[in_index];
        }
      
Checking the index `in_index` we see that two adjacent threads (`threadIx.x, threadIdx.x+1`) access location in memory near eachother. However the writes are not. Threads access data which in a strided way. Two adjacent threads access data separated by `height` elements. This practically results in 32 memory operations, however due to under the hood optimzations the achieved bandwidth is `311 GB/s`.      

We can improve the code by readig the data in a coalesced way, save it in the shared memory row by row and then write in the global memory column by column.


.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++
      
   .. tab:: OpenCL

      .. code-block:: C++
      
   .. tab:: SYCL

      .. code-block:: C++

         #include <iostream>
         #include <sycl/sycl.hpp>
         
         using namespace sycl;
         
   .. tab:: CUDA

      .. code-block:: C++

        #include <stdio.h>
        #include <cuda.h>
        #inclde <cuda_runtime.h>
        #include <math.h>
      
   .. tab:: HIP

      .. code-block:: C++ 
         
         const static int tile_dim = 16;

         __global__ void transpose_SM_kernel(float *in, float *out, int width, int height) {
           __shared__ float tile[tile_dim][tile_dim];

           int x_tile_index = blockIdx.x * tile_dim;
           int y_tile_index = blockIdx.y * tile_dim;
           
           int in_index =(y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
           int out_index =(x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

           tile[threadIdx.y][threadIdx.x] = in[in_index];

           __syncthreads();

          out[out_index] = tile[threadIdx.x][threadIdx.y];
       }
       
We define a *tile_dim* constant to determine the size of the shared memory tile. The matrix transpose kernel uses a 2D grid of thread blocks, where each thread block operates on a `tile_dim x tile_dim` tile of the input matrix.

The kernel first loads data from the global memory into the shared memory tile. Each thread loads a single element from the input matrix into the shared memory tile. Then, a **__syncthreads()** barrier ensures that all threads have finished loading data into shared memory before proceeding.

Next, the kernel writes the transposed data from the shared memory tile back to the output matrix in global memory. Each thread writes a single element from the shared memory tile to the output matrix. 
By using shared memory, this optimized implementation reduces global memory accesses and exploits memory coalescence, resulting in improved performance compared to a naive transpose implementation.

This kernel achieved on Nvidia V100 `674 GB/s`. 

This is pretty close to the  bandwidth achieved by the simple copy kernel, but there is one more thing to improve. Shared memory is composed of banks. Each banks can service only one request at the time. Bank conflicts happen when more than 1 thread in a specific warp try to access data in bank. The bank conflicts are resolved by serializing the accesses resulting in less performance. In the above example when data is saved to the shared memory, each thread in the warp will save an element of the data in a different one. Assuming that shared memory has 16 banks after writing each bank will contain one column. At the last step when we write from the shared memory to the global memory each warp load data from the same bank. A simple way to avoid this is by just padding the temporary array. 


.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++
      
   .. tab:: OpenCL

      .. code-block:: C++
      
   .. tab:: SYCL

      .. code-block:: C++

         #include <iostream>
         #include <sycl/sycl.hpp>
         
         using namespace sycl;
         
   .. tab:: CUDA

      .. code-block:: C++

        #include <stdio.h>
        #include <cuda.h>
        #inclde <cuda_runtime.h>
        #include <math.h>
      
   .. tab:: HIP

      .. code-block:: C++ 
         
         const static int tile_dim = 16;

         __global__ void transpose_SM_nobc_kernel(float *in, float *out, int width, int height) {
           __shared__ float tile[tile_dim][tile_dim+1];

           int x_tile_index = blockIdx.x * tile_dim;
           int y_tile_index = blockIdx.y * tile_dim;
           
           int in_index =(y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
           int out_index =(x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

           tile[threadIdx.y][threadIdx.x] = in[in_index];

           __syncthreads();

          out[out_index] = tile[threadIdx.x][threadIdx.y];
       }
       
By padding the array the data is slightly shifting it resulting in no bank conflicts. The effective bandwidth for this kernel is `697 GB/s`. 

CUDA/HIP Streams
^^^^^^^^^^^^^^^^
CUDA/HIP streams are independent execution contexts, a sequence of operations that execute in issue-order on the GPU. The operations issue in different streams can be executed concurrentely. 

Consider a case which involves copying data from CPU to GPU, computations and then coying back the result to GPU. Without streams nothing can be overlap. 

.. figure:: img/concepts/StreamsTimeline.png
   :align: center

Modern GPUs can execute in the same copying in both directions and calculations. One way to improve the performance  is to divide the problem in smaller independent parts. Let's consider 5 streams and consider the case where copy in one direction and computation take the same amount of time. After the first and second stream copy data to the GPU, the GPU is practically occupied all time. Significant perfomance  improvements can be obtained by eliminating the time in which the GPU is idle , waiting for data to arrive from the CPU.  This very useful for problems where there is often communication to the CPU because the GPU memory can not fit all the problem or the application runs in a multi--gpu set up and communication is needed often.  
Note that even when streams are not explcitely used it si possible to launch all the GPU operations asnynchronous and overlap CPU operations (such I/O) and GPU operations. 

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
