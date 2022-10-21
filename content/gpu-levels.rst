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

OpenMP is de facto standard for threaded based parallelism. It is relatively easy to 
implement. The whole technology suite contains the library routines, the compiler 
directives and environment variables. The parallelization is done providing "hints" 
(directives) about the regions of code which are targeted for parallelization. 
The compiler then chooses how to implement these hints as best as possible. 
The compiler directives are comments in Fortran and pragmas in C/C++. 
If there is no OpenMP support in the system they become comments and the code works just 
as any other serial code.


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

In OpenMP/OpenACC the compiler directives are specified by using **#pragma** in C/C++ or as 
special comments identified by unique sentinels in Fortran. Compilers can ingnore the 
directives if the support for OpenMP/OpenACC is not enabled.


Parallel regions 
~~~~~~~~~~~~~~~~

The compiler directives are used for various purposes: for thread creation, workload 
distribution (work sharing), data-environment management, serializing sections of code or 
for synchronization of work among the threads. The parallel regions are created using the 
**parallel** construct. When this construct is encounter additional thread are forked to 
carry out the work enclose in it. 

.. figure:: img/levels/omp-parallel.png
   :scale: 70%
   :align: center
    
   Outside of a parallel region there is only one thread, while inside there are N threads 
   

Clauses
~~~~~~~

TODO: simplify paragraphs on clauses

Together with compiler directives, OpenMP provides **clauses** that  can used to control 
the parallelism of regions of code. The clauses specify additional behaviour the user wants 
to occur and they refer to how the variables are visible to the threads (private or shared), 
synchronization, scheduling, control, etc. The clauses are appended in the code to the 
directives.


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

         int main(int argc, char argv[]){
             double vecA[NX],vecB[NX],vecC[NX];

             /* Initialize vectors */
             for (int i = 0; i < NX; i++) {
                 vecA[i] = pow(r, i-1);
                 vecB(i) = 1.0;
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
         #ifdef _OPENACC
         #include <openacc.h>
         #endif

         #define NX 102400

         int main(void) {
             double vecA[NX], vecB[NX], vecC[NX];
             double sum;

             /* Initialization of the vectors */
             for (int i = 0; i < NX; i++) {
                 vecA[i] = pow(r, i-1);
                 vecB(i) = 1.0;
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

         WRITEME

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