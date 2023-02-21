.. _gpu-levels:

GPU programming types
=====================

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
special comments identified by unique sentinels in Fortran. Compilers can ingnore the 
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



		  




With OpenMP, the ``TARGET`` directive is used for device offloading. 

.. challenge:: Example: ``TARGET`` construct 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: examples/acc/vec_add_target.c 
                        :language: cpp
                        :emphasize-lines: 16

      .. tab:: Fortran

         .. literalinclude:: examples/acc/vec_add_target.f90
                        :language: fortran
                        :emphasize-lines: 14,18


Compared to the OpenACC's ``kernels`` directive, the ``target`` directive will not parallelise the underlying loop. 
To achieve proper parallelisation, one needs to be more prescriptive and specify what one wants:

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
		  !$omp teams loop
		  do i = 1, nx
                     vecC(i) = vecA(i) + vecB(i)
                  end do
		  !$omp end teams loop
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

WRITEME

Kernel-based approaches
-----------------------

Native programming models (non-portable kernels)

- CUDA
- HIP

Cross-platform portability libraries (portable kernels)

The goal of the portability libraries is to allow the same code to run on multiple architectures, therefore reducing code duplication. They are usually based on C++, and use function objects/lambda functions to define the loop body (ie, the kernel), which can run on multiple architectures like CPU, GPU, and FPGA from different vendors. Unlike in many conventional CUDA or HIP implementations, a kernel needs to be written only once if one prefers to run it on CPU and GPU for example. Some notable cross-platform portability libraries are Kokkos, SYCL, and Raja.

Kokkos

Kokkos is an open-source performance portability library for parallelization on large heterogeneous hardware architectures of which development has mostly taken place on Sandia National Laboratories. The project started in 2011 as a parallel C++ programming model, but have since expanded into a more broad ecosystem including Kokkos Core (the programming model), Kokkos Kernels (math library), and Kokkos Tools (debugging, profiling and tuning tools). By preparing proposals for the C++ standard committee, the project also aims to influence the ISO/C++ language standard such that, eventually, Kokkos capabilities will become native to the language standard. A more detailed introduction is found `HERE <https://www.sandia.gov/news/publications/hpc-annual-reports/article/kokkos/>`_.

The Kokkos library provides an abstraction layer for a variety of different custom or native languages such as OpenMP, CUDA, and HIP. Therefore, it allows better portability across different hardware manufactured by different vendors, but introduces an additional dependency to the software stack. For example, when using CUDA, only CUDA installation is required, but when using Kokkos with NVIDIA GPUs, Kokkos and CUDA installation are both required. Kokkos is not a very popular choice for parallel programming, and therefore, learning and using Kokkos can be more difficult compared to more established programming models such as CUDA, for which a large amount of search results and stackoverflow discussions can be found.

SYCL...

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
           
             // Initialize values on host
             for (unsigned i = 0; i < n; i++)
               a[i] = i;
           
             // Print parallel from the device
             Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
               printf("a[%d] = %d\n", i, a[i]);
             });
           }
  
           // Finalize Kokkos
           Kokkos::finalize();
           return 0;
         }

   .. tab:: SYCL

      .. code-block:: C

         WRITEME

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
          
              // Allocate on Kokkos host memory space
              Kokkos::View<int*, Kokkos::HostSpace> h_a("h_a", n);
          
              // Allocate on Kokkos default memory space (eg, GPU memory)
              Kokkos::View<int*> a("a", n);
            
              // Initialize values
              for (unsigned i = 0; i < n; i++)
                h_a[i] = i;
              
              // Copy from host to device
              Kokkos::deep_copy(a, h_a);
            
              // Print parallel from the device
              Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
                printf("a[%d] = %d\n", i, a[i]);
              });
            }
            
            // Finalize Kokkos
            Kokkos::finalize();
            return 0;
          }


   .. tab:: SYCL

      .. code-block:: C

         WRITEME

   .. tab:: CUDA

      .. code-block:: C

         WRITEME

   .. tab:: HIP

      .. code-block:: C

         WRITEME

Parallel for with streams
^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs:: 

   .. tab:: Kokkos

      .. code-block:: C++

         #include <Kokkos_Core.hpp>
         
         int main(int argc, char* argv[]) {
         
           // Initialize Kokkos
           Kokkos::initialize(argc, argv);
         
           {
             unsigned nincr = 5;
             unsigned nx = 1000;
         
             // Allocate on Kokkos default memory space (eg, GPU memory)
             Kokkos::View<int*> a("a", nx);
         
             // Create execution space instances (streams) for each increment
             auto ex = Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(),1,1,1,1,1);
           
             for(unsigned incr = 0; incr < nincr; incr++) {
               Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ex[incr], nx / nincr * incr, nx / nincr * (incr + 1)), KOKKOS_LAMBDA(const int i) {
                 a[i] = i;
               });
             }

             // Sync execution space instances (streams)
             for(unsigned incr = 0; incr < nincr; incr++)
               ex[incr].fence();
           }
           
           // Finalize Kokkos
           Kokkos::finalize();
           return 0;
         }


   .. tab:: SYCL

      .. code-block:: C

         WRITEME

   .. tab:: CUDA

      .. code-block:: C

         WRITEME

   .. tab:: HIP

      .. code-block:: C

         WRITEME

Vector addition
^^^^^^^^^^^^^^^

.. tabs:: 

   .. tab:: CUDA C 

      .. code-block:: C

         __global__ void vecAdd(int *a_d, int *b_d, int *c_d, int N)
         {
             int i = blockIdx.x * blockDim.x + threadIdx.x;
             if(i<N)
             {
               c_d[i] = a_d[i] + b_d[i];
             }
         }

   .. tab:: CUDA Fortran

      .. code-block:: Fortran

         WRITEME

   .. tab:: HIP C

      .. code-block:: C

         WRITEME

   .. tab:: HIP Fortran

      .. code-block:: Fortran
sum
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
           
             // Print parallel from the device
             Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int i, int &lsum) {
               lsum += i;
             }, sum);
           }
  
           // Finalize Kokkos
           Kokkos::finalize();
           return 0;
         }


   .. tab:: SYCL

      .. code-block:: C

         WRITEME

   .. tab:: CUDA

      .. code-block:: C

         WRITEME

   .. tab:: HIP

      .. code-block:: C

         WRITEME


Pros and cons of kernel-based frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Easy to work with
- Porting of existing software requires less work
- Same code can be compiled to CPU and GPU versions easily



- Get access to all features of the GPU hardware
- More optimization possibilities




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
