.. _non-portable-kernel-models:


Non-portable kernel-based models
================================

.. questions::

   - How to program GPUs with CUDA and HIP?
   - What optimizations are possible when programming with CUDA and HIP? 

.. objectives::

   - Be able to use CUDA and HIP to write basic codes
   - Understand how the execution is done and how to do optimizations

.. instructor-note::

   - 45 min teaching
   - 30 min exercises

Fundamentals of GPU programming with CUDA and HIP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike some cross-platform portability ecosystems, such as alpaka, Kokkos, OpenCL, RAJA, and SYCL, which cater to multiple architectures, CUDA and HIP are solely focused on GPUs. They provide extensive libraries, APIs, and compiler toolchains that optimize code execution on NVIDIA GPUs (in the case of CUDA) and both NVIDIA and AMD GPUs (in the case of HIP). Because they are developed by the device producers, these programming models provide high-performance computing capabilities and offer advanced features like shared memory, thread synchronization, and memory management specific to GPU architectures.

CUDA, developed by NVIDIA, has gained significant popularity and is widely used for GPU programming. It offers a comprehensive ecosystem that includes not only the CUDA programming model but also a vast collection of GPU-accelerated libraries. Developers can write CUDA kernels using C++ and seamlessly integrate them into their applications to harness the massive parallelism of GPUs.

HIP, on the other hand, is an open-source project that aims to provide a more "portable" GPU programming interface. It allows developers to write GPU code in a syntax similar to CUDA and provides a translation layer that enables the same code to run on both NVIDIA and AMD GPUs. This approach minimizes the effort required to port CUDA code to different GPU architectures and provides flexibility for developers to target multiple platforms.

By being closely tied to the GPU hardware, CUDA and HIP provide a level of performance optimization that may not be achievable with cross-platform portability ecosystems. The libraries and toolchains offered by these programming models are specifically designed to exploit the capabilities of the underlying GPU architectures, enabling developers to achieve high performance.

Developers utilizing CUDA or HIP can tap into an extensive ecosystem of GPU-accelerated libraries, covering various domains, including linear algebra, signal processing, image processing, machine learning, and more. These libraries are highly optimized to take advantage of the parallelism and computational power offered by GPUs, allowing developers to accelerate their applications without having to implement complex algorithms from scratch.

As mentioned before, CUDA and HIP are very similar so it makes sense to cover both at the same time. 

.. callout:: Comparison to portable kernel-based models

   In code examples below, we will also show examples in the portable kernel-based frameworks Kokkos, SYCL and OpenCL, which will be covered in the next episode.

Hello World
~~~~~~~~~~~

Below we have the most basic example of CUDA and HIP, the "Hello World" program:

.. tabs:: 

   ..  group-tab:: CUDA
        .. literalinclude:: examples/non-portable-kernel-models/cuda-hello-world.c
           :language: C

   ..  group-tab:: HIP
        .. literalinclude:: examples/non-portable-kernel-models/hip-hello-world.c
           :language: C

   ..  group-tab:: Kokkos
        .. literalinclude:: examples/non-portable-kernel-models/kokkos-hello-world.cpp
           :language: C++

   ..  group-tab:: OpenCL
        .. literalinclude:: examples/non-portable-kernel-models/opencl-hello-world.c
           :language: C

   ..  group-tab:: SYCL
        .. literalinclude:: examples/non-portable-kernel-models/sycl-hello-world.cpp
           :language: C++

In both versions, we include the necessary headers: **cuda_runtime.h** and **cuda.h** for CUDA, and **hip_runtime.h** for HIP. These headers provide the required functionality for GPU programming.

To retrieve information about the available devices, we use the functions **<cuda/hip>GetDeviceCount** and **<cuda/hip>GetDevice**. These functions allow us to determine the total number of GPUs and the index of the currently used device. In the code examples, we default to using device 0.

As an exercise, modify the "Hello World" code to explicitly use a specific GPU. Do this by using the **<cuda/hip>SetDevice** function, which allows to set the desired GPU device. 
Note that the device number provided has to be within the range of available devices, otherwise, the program may fail to run or produce unexpected results.
To experiment with different GPUs, modify the code to include the following line before retrieving device information:

 .. code-block:: C
 
     cudaSetDevice(deviceNumber); // For CUDA  
     hipSetDevice(deviceNumber); // For HIP
 

Replace **deviceNumber** with the desired GPU device index. Run the code with different device numbers to observe the output (more examples for the "Hello World" program are available in the `content/examples/cuda-hip <https://github.com/ENCCS/gpu-programming/tree/main/content/examples/cuda-hip>`__ subdirectory of this lesson repository).


Vector Addition
~~~~~~~~~~~~~~~
To demonstrate the fundamental features of CUDA/HIP programming, let's begin with a straightforward task of element-wise vector addition. The code snippet below demonstrates how to utilize CUDA and HIP for efficiently executing this operation.

.. tabs:: 

   ..  group-tab:: CUDA
        .. literalinclude:: examples/non-portable-kernel-models/cuda-vector-add.cu
           :language: C++
      
   ..  group-tab:: HIP
        .. literalinclude:: examples/non-portable-kernel-models/hip-vector-add.cpp
           :language: C++
      
   ..  group-tab:: OpenCL
        .. literalinclude:: examples/non-portable-kernel-models/opencl-vector-add.c
           :language: C

   ..  group-tab:: SYCL
        .. literalinclude:: examples/non-portable-kernel-models/sycl-vector-add.cpp
           :language: C++
      
In this case, the CUDA and HIP codes are equivalent one to one so we will only refer to the CUDA version. The CUDA and HIP programming model are host centric programming models. The main program is executed on CPU and controls all the operations, memory allocations, data transfers between CPU and GPU, and launches the kernels to be executed on the GPU. The code starts with defining the GPU kernel function called **vector_add** with attribute **___global__**. It takes three input arrays `A`, `B`, and `C` along with the array size `n`. The kernel function contains the actually code which is executed on the GPU by multiple threads in parallel.

Accelerators in general and GPUs in particular usually have their own dedicated memory separate from the system memory (AMD MI300A is one exception, using the same memory for both CPU and GPU). When programming for GPUs, there are two sets of pointers involved and it's necessary to manage data movement between the host memory and the accelerator memory. Data needs to be explicitly copied from the host memory to the accelerator memory before it can be processed by the accelerator. Similarly, results or modified data may need to be copied back from the accelerator memory to the host memory to make them accessible to the CPU. 

The main function of the code initializes the input arrays `Ah, Bh` on the CPU and computes the reference array `Cref`. It then allocates memory on the GPU for the input and output arrays `Ad, Bd`, and `Cd` using **cudaMalloc**. Herein, `h` is for the 'host' (CPU) and `d` for the 'device' (GPU). The data is transferred from the CPU to the GPU using hipMemcpy, and then the GPU kernel is launched using the `<<<.>>>` syntax. All kernels launch are asynchronous. After launch the control returns to the `main()` and the code proceeds to the next instructions. 

After the kernel execution, the result array `Cd` is copied back to the CPU using **cudaMemcpy**. The code then prints the reference and result arrays, calculates the error by comparing the reference and result arrays. Finally, the GPU and CPU memory are deallocated using **cudaFree** and **free** functions, respectively. 

The host functions  **cudaSetDevice**, **cudaMalloc**, **cudaMemcpy**, and **cudaFree** are blocking, i.e. the code does not continues to next instructions until the operations are completed. However this is not the default behaviour, for many operations there are asynchronous equivalents and there are as well many library calls return the control to the `main()` after calling. This allows the developers to launch independent operations and overlap them. 

In short, this code demonstrates how to utilize the CUDA and HIP to perform vector addition on a GPU, showcasing the steps involved in allocating memory, transferring data between the CPU and GPU, launching a kernel function, and handling the results. It serves as a starting point for GPU-accelerated computations using CUDA and HIP.
More examples for the vector (array) addition program are available at `content/examples <https://github.com/ENCCS/gpu-programming/tree/main/content/examples>`_.

In order to practice the concepts shown above, edit the skeleton code in the repository and the code corresponding to setting the device, memory allocations and transfers, and the kernel execution. 


Vector Addition with Unified Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a while already GPUs support unified memory, which allows to use the same pointer for both CPU and GPU data. This simplifies developing codes by removing the explicit data transfers. The data resides on CPU until it is needed on GPU or vice-versa. However the data transfers still happens "under the hood" and the developer needs to construct the code to avoid unnecessary transfers. Below one can see the modified vector addition codes:


.. tabs:: 

   ..  group-tab:: CUDA
        .. literalinclude:: examples/non-portable-kernel-models/cuda-vector-add-unified-memory.cu
           :language: C++
      
   ..  group-tab:: HIP
        .. literalinclude:: examples/non-portable-kernel-models/hip-vector-add-unified-memory.cpp
           :language: C++

   ..  group-tab:: SYCL
        .. literalinclude:: examples/non-portable-kernel-models/sycl-vector-add-unified-memory.cpp
           :language: C++

Now the arrays `Ah`, `Bh`, `Ch`, and `Cref` are using `cudaMallocManaged` to allocate Unified Memory. The **vector_add kernel** is launched by passing these Unified Memory pointers directly. After the kernel launch, **cudaDeviceSynchronize** is used to wait for the kernel to complete execution. Finally, **cudaFree** is used to free the Unified Memory arrays. The Unified Memory allows for transparent data migration between CPU and GPU, eliminating the need for explicit data transfers.

As an exercise modify the skeleton code for vector addition to use Unified Memory. 

.. admonition:: Basics - In short
   :class: dropdown

   
   - CUDA is developed by NVIDIA, while HIP is an open-source project (from AMD) for multi-platform GPU programming.
   - CUDA and HIP are GPU-focused programming models for optimized code execution on NVIDIA and AMD GPUs.
   - CUDA and HIP are similar, allowing developers to write GPU code in a syntax similar to CUDA and target multiple platforms.
   - CUDA and HIP are programming models focused solely on GPUs
   - CUDA and HIP offer high-performance computing capabilities and advanced features specific to GPU architectures, such as shared memory and memory management.
   - They provide highly GPU-accelerated libraries in various domains like linear algebra, signal processing, image processing, and machine learning.
   - Programming for GPUs involves managing data movement between host and accelerator memory.
   - Unified Memory simplifies data transfers by using the same pointer for CPU and GPU data, but code optimization is still necessary.


Memory Optimizations
^^^^^^^^^^^^^^^^^^^^
Vector addition is a relatively simple, straight forward case. Each thread reads data from memory, does an addition and then saves the result. Two adjacent threads access memory location in memory close to each other. Also the data is used only once. In practice this not the case. Also sometimes the same data is used several times resulting in additional memory accesses. 

Memory optimization is one of the most important type of optimization done to efficiently use the GPUs. Before looking how it is done in practice let's revisit some basic concepts about GPUs and execution model.  


GPUs are comprised many light cores, the so-called Streaming Processors (SP) in CUDA, which are physically group together in units, i.e. Streaming Multi-Processors (SMP) in CUDA architecture (note that in AMD the equivalent is called Computing Units, while in Intel GPUs they are Execution Units). The work is done on GPUs by launching many threads each executing an instance of the same kernel. The order of execution is not defined, and the threads can only exchange information in specific conditions. Because of the way the SPs are grouped the threads are also grouped in **blocks**. Each **block** is assigned to an SMP, and can not be split. An SMP can have more than block residing at a moment, however there is no communications between the threads in different blocks. In addition to the SPs, each SMP contains very fast memory which in CUDA is referred to as `shared memory`. The threads in a block can read and write to the shared memory and use it as a user controlled cache. One thread can for example write to a location in the shared memory while another thread in the same block can read and use that data. In order to be sure that all threads in the block completed writing **__syncthreads()** function has to be used to make the threads in the block wait until all of them reached the specific place in the kernel. Another important aspect in the GPU programming model is that the threads in the block are not executed independently. The threads in a block are physically grouped in warps of size 32 in NVIDIA devices or wavefronts of size 32 or 64 in AMD devices (depending on device architecture). Intel devices are notable in that the warp size, called SIMD width, is highly configurable, with typical possible values of 8, 16, or 32 (depends on the hardware). All memory accesses of the global GPU memory are done per warp. When data is needed for some calculations a warp loads from the GPU memory blocks of specific size (64 or 128 Bytes). These operation is very expensive, it has a latency of hundreds of cycles. This means that the threads in a warp should work with elements of the data located close in the memory. In the vector addition two threads near each other, of index tid and tid+1, access elements adjacent in the GPU memory.  


The shared memory can be used to improve performance in two ways. It is possible to avoid extra reads from the memory when several threads in the same block need the same data (see `stencil <https://github.com/ENCCS/gpu-programming/tree/main/content/examples/stencil>`_ code) or it can be used to improve the memory access patterns like in the case of matrix transpose.

.. admonition:: Memory, Execution - In short
   :class: dropdown

   - GPUs consist of streaming processors (SPs) grouped together in units, such as Streaming Multi-Processors (SMPs) in CUDA architecture.
   - Work on GPUs is done by launching threads, with each thread executing an instance of the same kernel, and the execution order is not defined.
   - Threads are organized into blocks, assigned to an SMP, and cannot be split, and there is no communication between threads in different blocks.
   - Each SMP contains shared memory, which acts as a user-controlled cache for threads within a block, allowing efficient data sharing and synchronization.
   - The shared memory can be used to avoid extra memory reads when multiple threads in the same block need the same data or to improve memory access patterns, such as in matrix transpose operations.
   - Memory accesses from global GPU memory are performed per warp (groups of threads), and loading data from GPU memory has high latency.
   - To optimize memory access, threads within a warp should work with adjacent elements in memory to reduce latency.
   - Proper utilization of shared memory can improve performance by reducing memory reads and enhancing memory access patterns.


Matrix Transpose
~~~~~~~~~~~~~~~~
Matrix transpose is a classic example where shared memory can significantly improve the performance. The use of shared memory reduces global memory accesses and exploits the high bandwidth and low latency of shared memory.

.. figure:: img/concepts/transpose_img.png
   :align: center

First as a reference we use a simple kernel which copy the data from one array to the other. 

.. tabs:: 
         
   ..  group-tab:: CUDA
        .. literalinclude:: examples/non-portable-kernel-models/cuda-matrix-transpose-v0.cu
           :language: C++

   ..  group-tab:: HIP
        .. literalinclude:: examples/non-portable-kernel-models/hip-matrix-transpose-v0.cpp
           :language: C++
      
   ..  group-tab:: SYCL
        .. literalinclude:: examples/non-portable-kernel-models/sycl-matrix-transpose-v0.cpp
           :language: C++

We note that this code does not do any calculations. Each thread reads one element and then writes it to another locations. By measuring the execution time of the kernel we can compute the effective bandwidth achieve by this kernel. We can measure the time using **rocprof** or **cuda/hip events**. On a NVIDIA V100 GPU this code achieves `717 GB/s` out of the theoretical peak `900 GB/s`. 

Now we do the first iteration of the code, a naive transpose. The reads have a nice `coalesced` access pattern, but the writing is now very inefficient. 

.. tabs:: 

   ..  group-tab:: CUDA/HIP
        .. literalinclude:: examples/non-portable-kernel-models/cuda-matrix-transpose-v1.cu
           :language: C++
           :lines: 13-22

   ..  group-tab:: SYCL
        .. literalinclude:: examples/non-portable-kernel-models/sycl-matrix-transpose-v1.cpp
           :language: C++
           :lines: 11-19

Checking the index `in_index` we see that two adjacent threads (`threadIx.x, threadIdx.x+1`) access location in memory near each other. However the writes are not. Threads access data which in a strided way. Two adjacent threads access data separated by `height` elements. This practically results in 32 memory operations, however due to under the hood optimizations the achieved bandwidth is `311 GB/s`.

We can improve the code by reading the data in a `coalesced` way, save it in the shared memory row by row and then write in the global memory column by column.


 .. tabs:: 

   ..  group-tab:: CUDA/HIP
        .. literalinclude:: examples/non-portable-kernel-models/cuda-matrix-transpose-v2.cu
           :language: C++
           :lines: 13-30

   ..  group-tab:: SYCL
        .. literalinclude:: examples/non-portable-kernel-models/sycl-matrix-transpose-v2.cpp
           :language: C++
           :lines: 11-28

We define a **tile_dim** constant to determine the size of the shared memory tile. The matrix transpose kernel uses a 2D grid of thread blocks, where each thread block operates on a `tile_dim x tile_dim` tile of the input matrix.

The kernel first loads data from the global memory into the shared memory tile. Each thread loads a single element from the input matrix into the shared memory tile. Then, a **__syncthreads()** barrier ensures that all threads have finished loading data into shared memory before proceeding.

Next, the kernel writes the transposed data from the shared memory tile back to the output matrix in global memory. Each thread writes a single element from the shared memory tile to the output matrix. 
By using shared memory, this optimized implementation reduces global memory accesses and exploits memory coalescence, resulting in improved performance compared to a naive transpose implementation.

This kernel achieved on NVIDIA V100 `674 GB/s`. 

This is pretty close to the bandwidth achieved by the simple copy kernel, but there is one more thing to improve. 

Shared memory is composed of `banks`. Each banks can service only one request at the time. Bank conflicts happen when more than 1 thread in a specific warp try to access data in bank. The bank conflicts are resolved by serializing the accesses resulting in less performance. In the above example when data is saved to the shared memory, each thread in the warp will save an element of the data in a different one. Assuming that shared memory has 16 banks after writing each bank will contain one column. At the last step when we write from the shared memory to the global memory each warp load data from the same bank. A simple way to avoid this is by just padding the temporary array. 


.. tabs:: 

   ..  group-tab:: CUDA/HIP
        .. literalinclude:: examples/non-portable-kernel-models/cuda-matrix-transpose-v3.cu
           :language: C++
           :lines: 13-30

   ..  group-tab:: SYCL
        .. literalinclude:: examples/non-portable-kernel-models/sycl-matrix-transpose-v3.cpp
           :language: C++
           :lines: 11-28

By padding the array the data is slightly shifting it resulting in no bank conflicts. The effective bandwidth for this kernel is `697 GB/s`. 

.. admonition:: Using sharing memory as a cache - In short
   :class: dropdown

   - Shared memory can significantly improve performance in operations like matrix transpose.
   - Shared memory reduces global memory accesses and exploits the high bandwidth and low latency of shared memory.
   - An optimized implementation utilizes shared memory, loads data coalescedly, and performs transpose operations.
   - The optimized implementation uses a 2D grid of thread blocks and a shared memory tile size determined by a constant.
   - The kernel loads data from global memory into the shared memory tile and uses a synchronization barrier.
   - To avoid bank conflicts in shared memory, padding the temporary array is a simple solution.


Reductions
~~~~~~~~~~

`Reductions` refer to operations in which the elements of an array are aggregated in a single value through operations such as summing, finding the maximum or minimum, or performing logical operations. 

In the serial approach, the reduction is performed sequentially by iterating through the collection of values and accumulating the result step by step. This will be enough for small sizes, but for big problems this results in significant time spent in this part of an application. On a GPU, this approach is not feasible. Using just one thread to do this operation means the rest of the GPU is wasted. Doing reduction in parallel is a little tricky. In order for a thread to do work, it needs to have some partial result to use. If we launch, for example, a kernel performing a simple vector summation, ``sum[0]+=a[tid]``, with `N` threads we notice that this would result in undefined behaviour. GPUs have mechanisms to access the memory and lock the access for other threads while 1 thread is doing some operations to a given data via **atomics**, however this means that the memory access gets again to be serialized. There is not much gain. 
We note that when doing reductions the order of the iterations is not important (barring the typical non-associative behavior of floating-point operations). Also we can we might have to divide our problem in several subsets and do the reduction operation for each subset separately. On the GPUs, since the GPU threads are grouped in blocks, the size of the subset based on that. Inside the block, threads can cooperate with each other, they can share data via the shared memory and can be synchronized as well. All threads read the data to be reduced, but now we have significantly less partial results to deal with. In general, the size of the block ranges from 256 to 1024 threads. In case of very large problems, after this procedure if we are left too many partial results this step can be repeated.

At the block level we still have to perform a reduction in an efficient way. Doing it serially means that we are not using all GPU cores (roughly 97% of the computing capacity is wasted). Doing it naively parallel using **atomics**, but on the shared memory is also not a good option. Going back back to the fact the reduction operations are commutative and associative we can set each thread to "reduce" two elements of the local part of the array. Shared memory can be used to store the partial "reductions" as shown below in the code:

.. tabs:: 
         
   ..  group-tab:: CUDA/HIP

      .. code-block:: C++
         
         #define tpb 512 // size in this case has to be known at compile time
         // this kernel has to be launched with at least N/2 threads
         __global__ void reduction_one(double x, double *sum, int N){
           int ibl=blockIdx.y+blockIdx.x*gridDim.y;
           int ind=threadIdx.x+blockDim.x*ibl;
           
           __shared__ double shtmp[2*tpb];  
           shtmp[threadIdx.x]=0; // for sums we initiate with 0, for other operations should be different
           if(ind<N/2)
           {
              shtmp[threadIdx.x]=x[ind];
           }
           if(ind+N/2<N) 
           {
              shtmp[threadIdx.x+tpb]=x[ind+N/2];
           }
           __syncthreads();
           for(int s=tpb;s>0;s>>=1){
             if(threadIdx.x<s){
                shtmp[threadIdx.x]+=shtmp[threadIdx.x+s];}
             __syncthreads(); 
           }
           if(threadIdx.x==0)
           {
             sum[ibl]=shtmp[0]; // each block saves its partial result to an array 
             // atomicAdd(&sum[0], shene[0]); // alternatively could aggregate everything together at index 0. Only use when there not many partial sums left
           }
         }

   ..  group-tab:: SYCL
        .. literalinclude:: examples/non-portable-kernel-models/sycl-reduction.cpp
           :language: C++
           :lines: 9-51

In the kernel we have each GPU performing thread a reduction of two elements from the local portion of the array. If we have `tpb` GPU threads per block, we utilize them to store `2xtpb elements` in the local shared memory. To ensure synchronization until all data is available in the shared memory, we employ the `syncthreads()` function.

Next, we instruct each thread to "reduce" the element in the array at `threadIdx.x` with the element at `threadIdx.x+tpb`. As this operation saves the result back into the shared memory, we once again employ `syncthreads()`. By doing this, we effectively halve the number of elements to be reduced.

This procedure can be repeated, but now we only utilize `tpb/2 threads`. Each thread is responsible for "reducing" the element in the array at `threadIdx.x` with the element at `threadIdx.x+tpb/2`. After this step, we are left with `tpb/4` numbers to be reduced. We continue applying this procedure until only one number remains.

At this point, we can either "reduce" the final number with a global partial result using atomic read and write operations, or we can save it into an array for further processing.

.. figure:: img/concepts/Reduction.png
   :align: center
   
   Schematic representation on the reduction algorithm with 8 GPU threads.
   
For a detail analysis of how to optimize reduction operations in CUDA/HIP check this presentation `Optimizing Parallel Reduction in CUDA <https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>`_  

.. admonition:: Reductions - In short
   :class: dropdown

   - Reductions refer to aggregating elements of an array into a single value through operations like summing, finding maximum or minimum, or performing logical operations.
   - Performing reductions sequentially in a serial approach is inefficient for large problems, while parallel reduction on GPUs offers better performance.
   - Parallel reduction on GPUs involves dividing the problem into subsets, performing reductions within blocks of threads using shared memory, and repeatedly reducing the number of elements (two per GPU thread) until only one remains.


Overlapping Computations and Memory transfer. CUDA/HIP Streams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modern GPUs can overlap independent operations. They can do transfers between CPU and GPU and execute kernels in the same time, or they can execute kernels concurrently. CUDA/HIP streams are independent execution units, a sequence of operations that execute in issue-order on the GPU. The operations issue in different streams can be executed concurrently. 

Consider the previous case of vector addition, which involves copying data from CPU to GPU, computations and then copying back the result to GPU. In this way nothing can be overlap. 



We can improve the performance by dividing the problem in smaller independent parts. Let's consider 5 streams and consider the case where copy in one direction and computation take the same amount of time. 

.. figure:: img/concepts/StreamsTimeline.png
   :align: center


After the first and second stream copy data to the GPU, the GPU is practically occupied all time. We can see that significant performance  improvements can be obtained by eliminating the time in which the GPU is idle, waiting for data to arrive from the CPU. This very useful for problems where there is often communication to the CPU because the GPU memory can not fit all the problem or the application runs in a multi-GPU set up and communication is needed often.  

We can apply this to the vector addition problem above. 

.. tabs:: 
         
   ..  group-tab:: CUDA

      .. code-block:: C++
         
         // Distribute kernel for 'n_streams' streams, and record each stream's timing
         for (int i = 0; i < n_streams; ++i) {
           int offset = i * stream_size;
           cudaEventRecord(start_event[i], stream[i]); // stamp the moment when the kernel is submitted on stream i

           cudaMemcpyAsync( &Ad[offset],  &Ah[offset], N/n_streams*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
           cudaMemcpyAsync( &Bd[offset],  &Bh[offset], N/n_streams*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
           vector_add<<<gridsize / n_streams, blocksize, 0, stream[i]>>>(&Ad[offset], &Bd[offset], &Cd[offset], N/n_streams); //each call processes N/n_streams elements
           cudaMemcpyAsync( &Ch[offset],  &Cd[offset], N/n_streams*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);

           cudaEventRecord(stop_event[i], stream[i]);  // stamp the moment when the kernel on stream i finished
         }
      
   ..  group-tab:: HIP

      .. code-block:: C++    
         
         // Distribute kernel for 'n_streams' streams, and record each stream's timing
         for (int i = 0; i < n_streams; ++i) {
           int offset = i * (N/stream_size);
           hipEventRecord(start_event[i], stream[i]); // stamp the moment when the kernel is submitted on stream i

           hipMemcpyAsync( &Ad[offset],  &Ah[offset], N/n_streams*sizeof(float), hipMemcpyHostToDevice, stream[i]);
           hipMemcpyAsync( &Bd[offset],  &Bh[offset], N/n_streams*sizeof(float), hipMemcpyHostToDevice, stream[i]);
           vector_add<<<gridsize / n_streams, blocksize, 0, stream[i]>>>(&Ad[offset], &Bd[offset], &Cd[offset], N/n_streams); //each call processes N/n_streams elements
           hipMemcpyAsync( &Ch[offset],  &Cd[offset], N/n_streams*sizeof(float), hipMemcpyDeviceToHost, stream[i]);

           hipEventRecord(stop_event[i], stream[i]);  // stamp the moment when the kernel on stream i finished
         }
         ...

Instead of having one copy to gpu, one execution of the kernel and one copy back, we now have several of these calls independent of each other. 

Note that even when streams are not explicitly used it is possible to launch all the GPU operations asynchronous and overlap CPU operations (such I/O) and GPU operations. 
In order to learn more about how to improve performance using streams check the NVIDIA blog `How to Overlap Data Transfers in CUDA C/C++ <https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/>`_.

.. admonition:: Streams - In short
   :class: dropdown

   - CUDA/HIP streams are independent execution contexts on the GPU that allow for concurrent execution of operations issued in different streams.
   - Using streams can improve GPU performance by overlapping operations such as data transfers between CPU and GPU and kernel executions.
   - By dividing a problem into smaller independent parts and utilizing multiple streams, the GPU can avoid idle time, resulting in significant performance improvements, especially for problems with frequent CPU communication or multi-GPU setups.


Pros and cons of native programming models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are advantages and limitations to CUDA and HIP:

CUDA Pros:
   1. Performance Boost: CUDA is designed for NVIDIA GPUs and delivers excellent performance.
   2. Wide Adoption: CUDA is popular, with many resources and tools available.
   3. Mature Ecosystem: NVIDIA provides comprehensive libraries and tools for CUDA programming.

HIP Pros:
   1. Portability: HIP is portable across different GPU architectures.
   2. Open Standards: HIP is based on open standards, making it more accessible.
   3. Growing Community: The HIP community is growing, providing more resources and support.

Cons:
   0. Exclusive for GPUs
   1. Vendor Lock-in: CUDA is exclusive to NVIDIA GPUs, limiting compatibility.
   2. Learning Curve: Both CUDA and HIP require learning GPU programming concepts.
   3. Limited Hardware Support: HIP may face limitations on older or less common GPUs.



.. keypoints::

   - CUDA and HIP are two GPU programming models
   - Memory optimizations are very important
   - Asynchronous launching can be used to overlap operations and avoid idle GPU
