.. _gpu-prog-models-2:

GPU programming models 2 (detailed)
===================================

.. questions::

   - What are the key differences between different GPU programming approaches?
   - How should I choose which framework to use for my project?

.. objectives::

   - Understand basic examples in different GPU programming frameworks
   - Perform a quick cost-benefit analysis in the context of own code projects

.. instructor-note::

   - X min teaching
   - X min exercises


Directive-based frameworks
--------------------------

WRITEME begin

- What is OpenMP offloading?
- What is OpenACC?
- What are the differences between the two?

WRITEME end

The most common directive-based models for GPU parallel programming are OpenMP offloading and OpenACC. 
The parallelization is done by introducing directives in places which are targeted for parallelization. 
OpenACC is known to be more **descriptive**, which means the programmer uses directives to 
tell the compiler how/where to parallelize the code and to move the data. OpenMP offloading approach, 
on the other hand, is known to be more **prescriptive**, where the programmer uses directives to 
tell the compiler more explicitly how/where to parallelize the code, instead of letting the compiler decides.

In OpenMP/OpenACC the compiler directives are specified by using **#pragma** in C/C++ or as 
special comments identified by unique sentinels in Fortran. Compilers can ignore the 
directives if the support for OpenMP/OpenACC is not enabled.

The compiler directives are used for various purposes: for thread creation, workload 
distribution (work sharing), data-environment management, serializing sections of code or 
for synchronization of work among the threads.

Execution model 
~~~~~~~~~~~~~~~

OpenMP and OpenACC use the fork-join model of parallel execution. The program begins as a single 
thread of execution, the **master** thread. Everything is executed sequentially until the 
first parallel region construct is encountered. 

.. figure:: img/levels/threads.png
   :align: center

When a parallel region is encountered, master thread creates a group of threads, 
becomes the master of this group of threads, and is assigned the thread index 0 within 
the group. There is an implicit barrier at the end of the parallel regions. 

Memory Model
~~~~~~~~~~~~

WRITEME begin

- shared vs private variables
- global memory
- etc

WRITEME end


Directives
~~~~~~~~~~


OpenACC
^^^^^^^

In OpenACC, one of the most commonly used directives is ``kernels``,
which defines a region to be transferred into a series of kernels to be executed in sequence on a GPU. 
Work sharing is defined automatically for the separate kernels, but tuning prospects is limited.


.. challenge:: Example: ``kernels``

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: examples/acc/vec_add_kernels.c 
                        :language: cpp
                        :emphasize-lines: 17

      .. tab:: Fortran

         .. literalinclude:: examples/acc/vec_add_kernels.f90
                        :language: fortran
                        :emphasize-lines: 14,18



.. note:: 

    - data was created/destroyed on the device
    - data was transferred between the host and the device
    - the loop was parallized and execution was offloaded on the device


The other approach of OpenACC to define parallel regions is to use ``parallel`` directive.
Contrary to the ``kernels`` directive, the ``parallel`` directive is more explicit and requires 
more analysis by the programmer. Work sharing has to be defined manually using the ``loop`` directive, 
and refined tuning is possible to achieve. The above example can be re-write as the following:


.. challenge:: Example: ``parallel loop``

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: examples/acc/vec_add_loop.c 
                        :language: cpp
                        :emphasize-lines: 17

      .. tab:: Fortran

         .. literalinclude:: examples/acc/vec_add_loop.f90
                        :language: fortran
                        :emphasize-lines: 14,18



Sometimes we can obtain a little more performance by guiding the compiler to make specific choices. 
OpenACC has four levels of parallelism for offloading execution: 

  - **gang** coarse garin: the iterations are distributed among the gangs
  - **worker** fine grain: worker's threads are activated within gangs and iterations are shared among the threads 
  - **vector** each worker activtes its threads working in SIMT fashion and the work is shared among the threads
  - **seq** the iterations are executed sequentially

By default, when using ``parallel loop`` only, ``gang``, ``worker`` and ``vector`` parallelism are automatically decided and applied by the compiler. 



.. challenge:: Examples of nested loops with 

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c
             	:emphasize-lines: 3

		  #pragma acc parallel 
                  {
                  #pragma acc loop gang worker vector
                      for (i = 0; i < NX; i++) {
                          data[i] = 1.0;
                      }
                  }
		  

      .. tab:: Fortran

             .. code-block:: fortran
             	:emphasize-lines: 2,9,11,19,21,23

		  !$acc parallel 
		  !$acc loop gang worker vector
		  do i = 1, nx
                     data1(i) = 1.0
                  end do
		  !$acc end parallel

		  !$acc parallel 
		  !$acc loop gang worker
		  do j = 1, ny
		  !$acc loop vector
                     do i = 1, nx
                        data2(i,j) = 1.0
                     end do
                  end do
		  !$acc end parallel

		  !$acc parallel 
		  !$acc loop gang
		  do k = 1, nz
		  !$acc loop worker
		     do j = 1, ny
		  !$acc loop vector
                        do i = 1, nx
                           data3(i,j,k) = 1.0
                        end do
                     end do
                  end do
		  !$acc end parallel





.. note:: 

    There is no thread synchronization at ``gang`` level, which means there maybe a risk of race condition.
    The programmer could add clauses like ``num_gangs``, ``num_workers`` and ``vector_length`` within the parallel region to specify the number of 
    gangs, workers and vector length. The optimal numbers are highly architecture-dependent though.


#.. image:: img/gang_worker_vector.png

This image represents a single gang. When parallelizing our for loops, the loop iterations will be broken up evenly among a number of gangs. Each gang will contain a number of threads. These threads are organized into blocks. A worker is a row of threads. In the above graphic, there are 3 workers, which means that there are 3 rows of threads. The vector refers to how long each row is. So in the above graphic, the vector is 8, because each row is 8 threads long.



OpenMP Offloading
^^^^^^^^^^^^^^^^^

With OpenMP, the ``TARGET`` directive is used for device offloading. 

.. challenge:: Example: ``TARGET`` construct 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: examples/omp/vec_add_target.c 
                        :language: cpp
                        :emphasize-lines: 16

      .. tab:: Fortran

         .. literalinclude:: examples/omp/vec_add_target.f90
                        :language: fortran
                        :emphasize-lines: 14,18


Compared to the OpenACC's ``kernels`` directive, the ``target`` directive will not parallelise the underlying loop. 
To achieve proper parallelisation, one needs to be more prescriptive and specify what one wants. 
OpenMP offloading offers multiple levels of parallelism as well:

  - **teams** coarse grain: the iterations are distributed among the teams
  - **distribute** distributes the iterations across the master threads in the teams, but no worksharing among the threads within one team
  - **parallel do/for** fine grain: threads are activated within one team and worksharing among them
  - **SIMD** like the ``vector`` directive for OpenACC

.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c
             	:emphasize-lines: 3

		  #pragma omp target 
                  {
                  #pragma omp teams loop
                      for (i = 0; i < NX; i++) {
                          vecC[i] = vecA[i] + vecB[i];
                      }
                  }
		  


      .. tab:: Fortran

             .. code-block:: fortran
             	:emphasize-lines: 2,6

		  !$omp target 
		  !$omp teams distribute parallel do SIMD
		  do i = 1, nx
                     data1(i) = 1.0
                  end do
		  !$omp end teams distribute parallel do SIMD
		  !$omp end target


		  !$omp target 
		  !$omp teams distribute
		  do j = 1, ny
		  !$omp parallel do SIMD
                     do i = 1, nx
                        data2(i,j) = 1.0
                     end do
                  !$omp end parallel do SIMD
                  end do
		  !$omp end teams distribute
		  !$omp end target

		  !$omp target 
		  !$omp teams distribute
		  do k = 1, nz
		  !$omp parallel do
		     do j = 1, ny
		  !$omp SIMD
                        do i = 1, nx
                           data3(i,j,k) = 1.0
                        end do
                  !$omp end SIMD
                     end do
                  !$omp end parallel do
                  end do
		  !$omp end teams distribute
		  !$omp end target




.. note:: 

    Together with compiler directives, **clauses** that  can used to control  
    the parallelism of regions of code. The clauses specify additional behaviour the user wants 
    to occur and they refer to how the variables are visible to the threads (private or shared), 
    synchronization, scheduling, control, etc. The clauses are appended in the code to the directives.


Examples
~~~~~~~~

Vector addition
^^^^^^^^^^^^^^^

Example of a trivially parallelizable problem using the *loop* workshare construct:

TODO: test, simplify and harmonize all versions below

.. tabs::

   .. tab:: OpenMP C/C++
      
      .. code-block:: C++
            
         #include <stdio.h>
         #include <math.h>
         #define NX 102400

         int main(void){
             double vecA[NX],vecB[NX],vecC[NX];

             /* Initialize vectors */
             for (int i = 0; i < NX; i++) {
                 vecA[i] = 1.0;
                 vecB[i] = 1.0;
             }  

             #pragma omp parallel
             {
                 #pragma omp for
                 for (int i = 0; i < NX; i++) {
                    vecC[i] = vecA[i] * vecB[i];
                 }
             }
         }
                              
   .. tab:: OpenMP Fortran
      
      .. code-block:: Fortran
         
         program dotproduct
             implicit none  
 
             integer, parameter :: nx = 102400
             real, dimension(nx) :: vecA,vecB,vecC
             real, parameter :: r=0.2
             integer :: i

             ! Initialization of vectors
             do i = 1, nx
                vecA(i) = r**(i-1)
                vecB(i) = 1.0
             end do     

             !$omp parallel 
             !$omp do
                 do i=1,NX
                     vecC(i) = vecA(i) * vecB(i)
                 enddo  
             !$omp end do
             !$omp end parallel
         end program dotproduct

   .. tab:: OpenACC C/C++
      
      .. code-block:: C++

         #include <stdio.h>
         #include <openacc.h>
         #define NX 102400

         int main(void) {
             double vecA[NX], vecB[NX], vecC[NX];
             double sum;

             /* Initialization of the vectors */
             for (int i = 0; i < NX; i++) {
                 vecA[i] = 1.0;
                 vecB[i] = 1.0;
             }

             #pragma acc data copy(vecA,vecB,vecC)
             {
                 #pragma acc parallel
                 {
                 #pragma acc loop
                     for (int i = 0; i < NX; i++) {
                         vecC[i] = vecA[i] * vecB[i];
                     }
                 }
             }
         }         

   .. tab:: OpenACC Fortran

      .. code-block:: Fortran

         program dotproduct
             implicit none
 
             integer, parameter :: nx = 102400
             real, dimension(:), allocatable :: vecA,vecB,vecC
             real, parameter :: r=0.2
             integer :: i

             allocate (vecA(nx), vecB(nx),vecC(nx))
             ! Initialization of vectors
             do i = 1, nx
                vecA(i) = r**(i-1)
                vecB(i) = 1.0
             end do     

             !$acc data copy(vecA,vecB,vecC)
             !$acc parallel 
             !$acc loop
                 do i=1,NX
                     vecC(i) = vecA(i) * vecB(i)
                 enddo  
             !$acc end loop
             !$acc end parallel
             !$acc end data
         end program dotproduct

Reduction
^^^^^^^^^

Example of a *reduction* loop without race condition by using private variables:

.. tabs::

   .. tab:: OpenMP C/C++
      
      .. code-block:: C++
            
         #pragma omp parallel for shared(x,y,n) private(i) reduction(+:asum){
            for(i=0; i < n; i++) {
                  asum = asum + x[i] * y[i];
            }
         }
                              
   .. tab:: OpenMP Fortran
      
      .. code-block:: Fortran
         
         !$omp parallel do shared(x,y,n) private(i) reduction(+:asum)
            do i = 1, n
               asum = asum + x(i)*y(i)
            end do
         !$omp end parallel

   .. tab:: OpenACC C/C++
      
      .. code-block:: C++

         WRITEME

   .. tab:: OpenACC Fortran
      
      .. code-block:: Fortran
         
         WRITEME

Pros and cons of directive-based frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- incremental programming
- Porting of existing software requires less work
- Same code can be compiled to CPU and GPU versions easily using compiler flag
- low learning curve, do not need to know low-level hardware details
- good portability


WRITEME

Kernel-based approaches
-----------------------

Native programming models (non-portable kernels)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA
^^^^

HIP
^^^

Pros and cons of native programming models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cross-platform portability ecosystems (portable kernels)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal of the portability ecosystems is to allow the same code to run on multiple architectures, therefore reducing code duplication. They are usually based on C++, and use function objects/lambda functions to define the loop body (ie, the kernel), which can run on multiple architectures like CPU, GPU, and FPGA from different vendors. Unlike in many conventional CUDA or HIP implementations, a kernel needs to be written only once if one prefers to run it on CPU and GPU for example. Some notable cross-platform portability ecosystems are Kokkos, SYCL, and RAJA. Kokkos and RAJA are individual projects whereas SYCL is a standard that is followed by several projects implementing (and extending) it, notably `Intel DPC++ <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html>`_, `Open SYCL <https://github.com/OpenSYCL/OpenSYCL>`_ (formerly hipSYCL), `triSYCL <https://github.com/triSYCL/triSYCL>`_, and `ComputeCPP <https://developer.codeplay.com/products/computecpp/ce/home/>`_.

Kokkos
^^^^^^

Kokkos is an open-source performance portability ecosystem for parallelization on large heterogeneous hardware architectures of which development has mostly taken place on Sandia National Laboratories. The project started in 2011 as a parallel C++ programming model, but have since expanded into a more broad ecosystem including Kokkos Core (the programming model), Kokkos Kernels (math library), and Kokkos Tools (debugging, profiling and tuning tools). By preparing proposals for the C++ standard committee, the project also aims to influence the ISO/C++ language standard such that, eventually, Kokkos capabilities will become native to the language standard. A more detailed introduction is found `HERE <https://www.sandia.gov/news/publications/hpc-annual-reports/article/kokkos/>`_.

The Kokkos library provides an abstraction layer for a variety of different custom or native languages such as OpenMP, CUDA, and HIP. Therefore, it allows better portability across different hardware manufactured by different vendors, but introduces an additional dependency to the software stack. For example, when using CUDA, only CUDA installation is required, but when using Kokkos with NVIDIA GPUs, Kokkos and CUDA installation are both required. Kokkos is not a very popular choice for parallel programming, and therefore, learning and using Kokkos can be more difficult compared to more established programming models such as CUDA, for which a much larger amount of search results and stackoverflow discussions can be found.

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

When starting to write a project using Kokkos, the first step is understand Kokkos initialization and finalization. Kokkos must be initialized by calling ``Kokkos::initialize(int& argc, char* argv[])`` and finalized by calling ``Kokkos::finalize()``. More details are given in `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Initialization.html>`_.

Kokkos uses an execution space model to abstract the details of parallel hardware. The execution space instances map to the available backend options such as CUDA, OpenMP, HIP, or SYCL. If the execution space is not explicitly chosen by the programmer in the source code, the default execution space ``Kokkos::DefaultExecutionSpace`` is used .This is chosen when the Kokkos library is compiled. The Kokkos execution space model is described in more detail in `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-spaces>`_.

Similarly, Kokkos uses a memory space model for different types of memory, such as host memory or device memory. If not defined explicitly, Kokkos uses the default memory space specified during Kokkos compilation. `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#kokkos-memory-spaces>`_.

The following is an example of a Kokkos program that initializes Kokkos and prints the execution space and memory space instances: 

.. tabs:: 

   .. tab:: C++
      
      .. code-block:: C++

         #include <Kokkos_Core.hpp>
         #include <iostream>
         
         int main(int argc, char* argv[]) {
           Kokkos::initialize(argc, argv);
           std::cout << "Execution Space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
           std::cout << "Memory Space: " << typeid(Kokkos::DefaultMemorySpace).name() << std::endl;
           Kokkos::finalize();
           return 0;
         }

With Kokkos, the data can be accessed either through raw pointers or through Kokkos Views. With raw pointers, the memory allocation into the default memory space can be done using ``Kokkos::kokkos_malloc(n * sizeof(int))``. Kokkos Views are a data type that provides a way to access data more efficiently in memory corresponding to a certain Kokkos memory space, such as host memory or device memory. A 1-dimensional view of type int* can be created by ``Kokkos::View<int*> a("a", n)``, where ``a`` is a label, and ``n`` is the size in integers. Kokkos determines the optimal layout for the data at compile time for best overall performance as a function of the computer architecture. Furthermore, Kokkos handles the deallocation of such memory automatically. More details about Kokkos Views are found `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/View.html>`_.

Finally, Kokkos provides three different parallel operations: ``parallel_for``, ``parallel_reduce``, and ``parallel_scan``. The ``parallel_for`` operation is used to execute a loop in parallel. The ``parallel_reduce`` operation is used to execute a loop in parallel and reduce the results to a single value. The ``parallel_scan`` operation is used to execute a loop in parallel and scan the results. The usage of ``parallel_for`` and ``parallel_reduce`` are demonstrated in the examples later in this chapter. More detail is `HERE <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html>`_.



SYCL
^^^^


Pros and cons of cross-platform portability ecosystems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The amount of code duplication is minimized

    The same code can be compiled to multiple architectures from different vendors

    Higher level of abstraction, does not require as much knowledge of the underlying architecture

    Less matured ecosystem compared to CUDA, more uncertainty about future

    Less learning resources (stackoverflow, course material, documentation)


Examples
~~~~~~~~

Parallel for with Unified Memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

   .. tab:: OpenCL

      .. code-block:: C++

          // We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
          #define CL_HPP_MINIMUM_OPENCL_VERSION 110
          #define CL_HPP_TARGET_OPENCL_VERSION 110
          #include <CL/opencl.hpp>

          // For larger kernels, we can store source in a separate file
          static const std::string kernel_source{
              "__kernel void dot(__global const int *a, __global const int *b, __global "
              "int *c) {\n"
              "    int i = get_global_id(0);\n"
              "    c[i] = a[i] * b[i];\n"
              "}"};

          int main(int argc, char *argv[]) {

            cl::Device device = cl::Device::getDefault();
            cl::Context context(device);
            cl::CommandQueue queue(context, device);

            // Compile OpenCL program for found device.
            cl::Program program(context, kernel_source);
            program.build(device);
            cl::Kernel kernel_dot(program, "dot");

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
            cl::Buffer dev_c(context, CL_MEM_READ_WRITE, n * sizeof(int));

            // We must use cl::Kernel::setArg to pass arguments to device
            kernel_dot.setArg(0, dev_a);
            kernel_dot.setArg(1, dev_b);
            kernel_dot.setArg(2, dev_c);

            // We don't need to apply any offset to thread IDs
            const auto offset = cl::NullRange;
            queue.enqueueNDRangeKernel(kernel_dot, offset, n);

            queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, n * sizeof(int), c.data());

            // Print results
            for (unsigned i = 0; i < n; i++) {
              printf("c[%d] = %d\n", i, c[i]);
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

      .. code-block:: C

         WRITEME

   .. tab:: HIP

      .. code-block:: C

         WRITEME

Reduction
^^^^^^^^^
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

.. keypoints::

   - k1
   - k2
