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

- CUDA
- HIP


Examples
~~~~~~~~

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

         WRITEME


Reduction
^^^^^^^^^

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
