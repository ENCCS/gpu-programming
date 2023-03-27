.. _directive-based-models:

Directive-based models
======================

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

  - **gang** coarse grain: the iterations are distributed among the gangs
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


.. keypoints::

   - k1
   - k2
