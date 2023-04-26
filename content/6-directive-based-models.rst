.. _directive-based-models:

Directive-based models
======================

.. questions::

   - q1
   - q2

.. objectives::

   - o1
   - o2

.. instructor-note::

   - 60 min teaching
   - 30 min exercises

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



.. note:: 

    There is no thread synchronization at ``gang`` level, which means there maybe a risk of race condition.
    The programmer could add clauses like ``num_gangs``, ``num_workers`` and ``vector_length`` within the parallel region to specify the number of 
    gangs, workers and vector length. The optimal numbers are highly architecture-dependent though.





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
  - **SIMD** like the ``vector`` directive in OpenACC


.. note:: 

    Since OpenMP 5.0, there is a new ``loop`` directive available, which has the similar functionality as the corresponding one in OpenACC.


.. challenge:: Syntax for ``loop`` directive

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c
             	:emphasize-lines: 1

                  #pragma omp target teams loop
                      for (i = 0; i < NX; i++) {
                          vecC[i] = vecA[i] + vecB[i];
                      }
		  


      .. tab:: Fortran

             .. code-block:: fortran
             	:emphasize-lines: 1,5

		  !$omp target teams loop
		  do i = 1, nx
                     vecC(i) = vecA(i) + vecB(i)
                  end do
		  !$omp end target teams loop



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


.. tabs::

   .. tab:: OpenMP 
      
      .. tabs::

         .. tab::  C/C++

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

		   #pragma omp target teams distribute parallel for
		   {
		       for (int i = 0; i < NX; i++) {
			  vecC[i] = vecA[i] + vecB[i];
		       }
		   }
               }
                              
         .. tab::  Fortran
      
            .. code-block:: Fortran
         
               program vecsum
                   implicit none
 
                   integer, parameter :: nx = 102400
                   real, dimension(nx) :: vecA,vecB,vecC
                   integer :: i

                   ! Initialization of vectors
                   do i = 1, nx
                      vecA(i) = 1.0
                      vecB(i) = 1.0
                   end do     

                  !$omp target teams distribute parallel do
                       do i=1,nx
                           vecC(i) = vecA(i) + vecB(i)
                       enddo  
                  !$omp end target teams distribute parallel do
               end program vecsum

   .. tab:: OpenACC 
      
      .. tabs::

         .. tab:: C/C++
      
            .. code-block:: C++

               #include <stdio.h>
               #include <openacc.h>
               #define NX 102400

	       int main(void) {
		   double vecA[NX], vecB[NX], vecC[NX];

		   /* Initialization of the vectors */
		   for (int i = 0; i < NX; i++) {
		       vecA[i] = 1.0;
		       vecB[i] = 1.0;
		   }
		   #pragma acc parallel loop
		   {
		       for (int i = 0; i < NX; i++) {
			   vecC[i] = vecA[i] + vecB[i];
		       }
		   }
	       }         

	 .. tab:: Fortran

	    .. code-block:: Fortran

	       program vecsum
		   implicit none

		   integer, parameter :: nx = 102400
		   real, dimension(:), allocatable :: vecA,vecB,vecC
		   integer :: i

		   allocate (vecA(nx), vecB(nx),vecC(nx))
		   ! Initialization of vectors
		   do i = 1, nx
		      vecA(i) = 1.0
		      vecB(i) = 1.0
		   end do     

		   !$acc parallel loop
		       do i=1,nx
			   vecC(i) = vecA(i) + vecB(i)
		       enddo  
		   !$acc end parallel loop
	       end program vecsum



Data Movement
~~~~~~~~~~~~~

Due to distinct memory spaces on host and device, transferring data becomes inevitable. 
New directives are needed to specify how variables are transferred from the host to the device data environment. 
The common transferred items consist of arrays (array sections), scalars, pointers, and structure elements. 
Various data clauses used for data movement is summarised in the following table

.. csv-table::
   :widths: auto
   :delim: ;

   ``OpenMP`` ; ``OpenACC`` ; 
   ``map(to:list)`` ; ``copyin(list)`` ; On entering the region, variables in the list are initialized on the device using the original values from the host
   ``map(from:list)`` ; ``copyout(list)`` ;  At the end of the target region, the values from variables in the list are copied into the original variables on the host. On entering the region, the initial value of the variables on the device is not initialized       
   ``map(tofrom:list)`` ; ``copy(list)`` ; the effect of both a map-to and a map-from
   ``map(alloc:list)`` ;  ``create(list)`` ; On entering the region, data is allocated and uninitialized on the device


.. +----------------------+-------------------+----------------------------------------------+
   |                      |                   |                                              |
   +======================+==================================================================+
   | OpenMP               | OpenACC           |                                              |
   +----------------------+-------------------+----------------------------------------------+
   | ``map(to:list)``     | ``copyin(list)``  |On entering the region, variables in the list |
   |                      |                   |are initialized on the device using the       |
   |                      |                   |original values from the host                 |
   +----------------------+-------------------+----------------------------------------------+
   | ``map(from:list)``   | ``copyout(list)`` | At the end of the target region, the values  |
   |                      |                   |from variables in the list are copied into    |
   |                      |                   |the original variables on the host. On        |
   |                      |                   |entering the region, the initial value of the |
   |                      |                   |variables on the device is not initialized    |
   +----------------------+-------------------+----------------------------------------------+
   | ``map(tofrom:list)`` | ``copy(list)``    |the effect of both a map-to/copyin and        |
   |                      |                   |a map-from/copyout                            |
   +----------------------+-------------------+----------------------------------------------+
   | ``map(alloc:list)``  | ``create(list)``  |On entering the region, data is allocated and |
   |                      |                   |uninitialized on the device                   |
   +----------------------+-------------------+----------------------------------------------+
 
   

.. note::

	When mapping data arrays or pointers, be careful about the array section notation:
	  - In C/C++: array[lower-bound:length]. The notation :N is equivalent to 0:N.
	  - In Fortran:array[lower-bound:upper-bound]. The notation :N is equivalent to 1:N.


Data region
^^^^^^^^^^^

The specific data clause combined with the data directive constitutes the start of a data region.
How the directives create storage, transfer data, and remove storage on the device are clasiffied as two categories: 
structured data region and unstructured data region. 
A structured data region is convenient for providing persistent data on the device which could be used for subseqent GPU directives.
However it is inconvenient in real applications using structured data region, therefore the unstructured data region  
with much more freedom in creating and deleting of data on the device at any appropriate point is adopted.


.. challenge:: Syntax for structured data region

.. tabs::

   .. tab:: OpenMP 

      .. tabs::

	 .. tab:: C/C++

		.. code-block:: c

		     #pragma omp target data [clauses]
			{structured-block}


	 .. tab:: Fortran

		.. code-block:: fortran

		     !$omp target data [clauses]
		        structured-block
		     !$omp end target data


   .. tab:: OpenACC 

      .. tabs::

	 .. tab:: C/C++

		.. code-block:: c

		     #pragma acc data [clauses]
	                {structured-block}



	 .. tab:: Fortran

		.. code-block:: fortran

		     !$acc data [clauses]
		        structured-block
		     !$acc end data



.. challenge:: Syntax for unstructured data region

.. tabs::

   .. tab:: OpenMP 

      .. tabs::

	 .. tab:: C/C++

		.. code-block:: c

		     #pragma omp target enter data [clauses]

		.. code-block:: c

		     #pragma omp target exit data
	   


	 .. tab:: Fortran

		.. code-block:: fortran

		     !$omp target enter data [clauses] 

		.. code-block:: fortran

		     !$omp target exit data


   .. tab:: OpenACC 

      .. tabs::

	 .. tab:: C/C++

		.. code-block:: c

		     #pragma acc enter data [clauses]

		.. code-block:: c

		     #pragma acc exit data



	 .. tab:: Fortran

		.. code-block:: fortran

		     !$acc enter data [clauses] 

		.. code-block:: fortran

		     !$acc exit data



.. keypoints::

  Structured Data Region
    - start and end points within a single subroutine
    - Memory exists within the data region

  Unstructured Data Region
    - multiple start and end points across different subroutines
    - Memory exists until explicitly deallocated


Optimize Data Transfers
^^^^^^^^^^^^^^^^^^^^^^^

- Explicitely transfer the data as much as possible
- Reduce the amount of data mapping between host and device, get rid of unnecessary data transfer
- Try to keep data environment residing on the device as long as possible




Pros and cons of directive-based frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Incremental programming
- Porting of existing software requires less work
- Same code can be compiled to CPU and GPU versions easily using compiler flag
- Low learning curve, do not need to know low-level hardware details
- Good portability


.. keypoints::

   - k1
   - k2
