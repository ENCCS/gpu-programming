.. _multiple-gpus:

Multiple GPU programming with MPI
=================================

.. questions::

   - q1
   - q2

.. objectives::

   - To learn about combining MPI with either OpenACC or OpenMP offloading
   - To learn about implementing GPU-awareness MPI approach 

.. instructor-note::

   - 30 min teaching
   - 20 min exercises

Introduction
------------

Exploring multiple GPUs (Graphics Processing Units) across distributed nodes offers the potential to fully leveraging the capacity of modern HPC (High-Performance Computing) systems at a large scale. Here one of the approaches to accelerate computing on distributed systems is to combine MPI (Message Passing Interface) with a GPU programming model such as OpenACC and OpenMP application programming interfaces (APIs). This combination is motivated by both the simplicity of these APIs, and the widespread use of MPI.   

In this guide we provide readers, who are familiar with MPI, with insighits on implementing a hybrid model in which the MPI communication framework is combined with either OpenACC or OpenMP APIs. A special focus will be on performing point-to-point (e.g. `MPI_Send` and `MPI_Recv`) and collective operations (e.g. `MPI_Allreduce`) from OpenACC and OpenMP APIs. Here we address two scenarios: (i) a scenario in which MPI operations are performed in the CPU-host followed by an offload to the GPU-device; and (ii) a scenario in which MPI operations are performed between a pair of GPUs without involving the CPU-host memory. The latter scenario is referred to as GPU-awareness MPI, and has the advantage of reducing the computing time caused by transferring data via the host-memory during heterogenous communications, thus rendering HPC applications efficient. 

This guide is organized as follows: we first introduce how to assign each MPI rank to a GPU device within the same node. We consider a situation in which the host and the device have a distinct memory. This is followed by a presentation on the hybrid MPI-OpenACC/OpenMP offloading with and without the GPU-awareness MPI. Exercises to help understanding these concepts are provided at the end.

Assigning MPI-ranks to GPU-devices
----------------------------------

Accelerating MPI applications to utilise multiple GPUs on distributed nodes requires as a first step assigning each MPI rank to a GPU device, such that two MPI ranks do not use the same GPU device. This is necessarily in order to prevent the application from a potential crash. This is because GPUs are designed to handle multiple threading tasks, but not multiple MPI ranks. 

One of the way to ensure that two MPI ranks do not use the same GPU, is to determine which MPI processes run on the same node, such that each process can be assigned to a GPU device within the same node. This can be done, for instance, by splitting the world communicator into sub-groups of communicators (or sub-communicators) using the routine `MPI_COMM_SPLIT_TYPE()`. 

.. challenge:: Example: ``Assign device``

   .. tabs::

      .. tab:: MPI-OpenACC

         .. literalinclude:: examples/assignDevice_acc.f90
                     :language: fortran
                     :emphasize-lines: 27-29
		     
      .. tab:: MPI-OpenMP

         .. literalinclude:: examples/assignDevice_omp.f90
                     :language: fortran
                     :emphasize-lines: 27-29		     


Here, the size of each sub-communicator corresponds to the number of GPUs per node (which is also the number of tasks per node), and each sub-communicator contains a list of processes indicated by a rank. These processes have a shared-memory region defined by the argument `MPI_COMM_TYPE_SHARED` (see the `MPI report <https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf>`_) for more details). Calling the routine `MPI_COMM_SPLIT_TYPE()` returns a sub-communicator labelled in the code above *”host_comm”*, and in which MPI-ranks are ranked from 0 to number of processes per node -1. These MPI ranks are in turn assigned to different GPU devices within the same node. This procedure is done according to which directive-based model is implemented. The retrieved MPI ranks are then stored in the variable **myDevice**. The variable is passed to an OpenACC or OpenMP routine as indicated in the code below. 

.. tabs::

   .. tab:: OpenACC

      .. code-block:: fortran

         ! Set a device number in OpenACC
         acc_set_device_num(myDevice, acc_get_device_type())

   .. tab:: OpenMP

      .. code-block:: fortran

         ! Set a device number in OpenMP 
         omp_set_default_device(myDevice)


On the other hand, one can check the total number of devices available on the host by using the following functions:

.. challenge:: Example: ``number of devices``

   .. tabs::

      .. tab:: OpenACC
      
      	.. code-block:: fortran

           ! Returns the number of devices available for offloading
           acc_get_num_devices(acc_get_device_type())

      .. tab:: OpenMP
      
      	.. code-block:: fortran

           ! Returns the number of devices available for offloading
           omp_get_num_devices()

	 
Another useful function for retrieving the device number of a specific device, which is useful, e.g., to map data to a specific device is
	
.. tabs::

   .. tab:: OpenACC
     
      .. code-block:: fortran
 	
         acc_get_device_num()

   .. tab:: OpenMP

      .. code-block:: fortran
	 
       	 omp_get_device_num()

The syntax of assigning MPI ranks to GPU devices is summarised below

.. challenge:: Example: ``Set device``

   .. tabs::

      .. tab:: MPI-OpenACC
	 
         .. literalinclude:: examples/assignDevice_acc.f90
                     :language: fortran
                     :emphasize-lines: 1,54

      .. tab:: MPI-OpenMP
	 
         .. literalinclude:: examples/assignDevice_omp.f90
                     :language: fortran
                     :emphasize-lines: 1,54


Hybrid MPI-OpenACC/OpenMP without GPU-awareness approach
--------------------------------------------------------

After covering how to assign each MPI-rank to a GPU device, we now address the concept of combining MPI with either
OpenACC or OpenMP offloading. In this approach, calling an MPI routine from an OpenACC or OpenMP API requires updating the data in the CPU host before and after an MPI call. In this scenario, the data is copied back and forth between the host and the device before and after each MPI call. In the hybrid MPI-OpenACC model, the procedure is defined by specifying the directive `update host()` for copying the data from the device to the host before an MPI call; and by the directive `update device()` specified after an MPI call for copying the data back to the device. Similarly in the hybrid MPI-OpenMP. Here, updating the data in the host can be done by specifying the OpenMP directives `update device() from()` and `update device() to()`, respectively, for copying the data from the device to the host and back to the device.

To illustrate the concept of the hybrid MPI-OpenACC/OpenMP, we show below an example of an implementation that involves the MPI functions `MPI_Send()` and `MPI_Recv()`.

.. challenge:: Example: ``Update host/device directives``

   .. tabs::

      .. tab:: MPI-OpenACC
	 
         .. literalinclude:: examples/mpiacc.f90
                     :language: fortran
                     :emphasize-lines: 67,79

      .. tab:: MPI-OpenMP

         .. literalinclude:: examples/mpiomp.f90
                     :language: fortran
                     :emphasize-lines: 68,80

Despite the simplicity of implementing the hybrid MPI-OpenACC/OpenMP offloading, it suffers from a low performance caused by an explicit transfer of data between the host and the device before and after calling an MPI routine. This constitutes a bottleneck in GPU-programming. To improve the performance affected by the host staging during the data transfer, one can implement the GPU-awareness MPI approach as described in the following section.
	  
Hybrid MPI-OpenACC/OpenMP with GPU-awareness approach 
-----------------------------------------------------

The concept of the GPU-aware MPI enables an MPI library to directly access the GPU-device memory without necessarily using the CPU-host memory as an intermediate buffer (see e.g. `here` <https://docs.open-mpi.org/en/v5.0.0rc9/networking/cuda.html>`_). This offers the benefit of transferring data from one GPU to another GPU without the involvement of the CPU-host memory.
	  
To be specific, in the GPU-awareness approach, the device pointers point to the data allocated in the GPU memory space (data should be present in the GPU device). Here, the pointers are passed as arguments to an MPI routine that is supported by the GPU memory. As MPI routines can directly access GPU memory, it offers the possibility of communicating between pairs of GPUs without transferring data back to the host. 

In the hybrid MPI-OpenACC model, the concept is defined by combining the directive `host_data` together with the clause
`use_device(list_array)`. This combination enables the access to the arrays listed in the clause `use_device(list_array)` from the host (see `here <https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf>`_). The list of arrays, which are already present in the GPU-device memory, are directly passed to an MPI routine without a need of a staging host-memory for copying the data. Note that for initially copying data to GPU, we use unstructured data blocks characterized by the directives `enter data` and `exit data`. The unstructured data has the advantage of allowing to allocate and deallocate arrays within a data region.

To illustarte the concept of the GPU-awareness MPI, we show below two examples that make use of point-to-point and collective operations from OpenACC and OpenMP APIs. In the first code example, the device pointer **f** is passed to the MPI functions `MPI_Send()` and `MP_Recv()`; and in the second one, the pointer **SumToT** is passed to the MPI function `MPI_Allreduce`. Here, the MPI operations `MPI_Send` and `MPI_Recv` as well as `MPI_Allreduce` are performed between a pair of GPUs without passing through the CPU-host memory. 

.. challenge:: Example: ``GPU-awareness: MPI_Send & MPI_Recv``

   .. tabs::

      .. tab:: GPU-aware MPI with OpenACC
	 
         .. literalinclude:: examples/mpiacc_gpuaware.f90
                     :language: fortran
                     :emphasize-lines: 67,76

      .. tab:: GPU-aware MPI with OpenMP
	 
         .. literalinclude:: examples/mpiomp_gpuaware.f90
                     :language: fortran
                     :emphasize-lines: 68,77


.. challenge:: Example: ``GPU-awareness: MPI_Allreduce``

   .. tabs::

      .. tab:: GPU-aware MPI with OpenACC
	 
         .. literalinclude:: examples/mpiacc_gpuaware.f90
                     :language: fortran
                     :emphasize-lines: 92,96

      .. tab:: GPU-aware MPI with OpenMP
	 
         .. literalinclude:: examples/mpiomp_gpuaware.f90
                     :language: fortran
                     :emphasize-lines: 95,99 

The GPU-aware MPI with OpenACC/OpenMP APIs has the capability of directly communicating between a pair of GPUs within a single node. However, performing the GPU-to-GPU communication across multiple nodes requires the the GPUDirect RDMA (Remote Direct Memory Access) technology. This technology can further improve performance by reducing latency.

Compilation process
-------------------

The compilation process of the hybrid MPI-OpenACC and MPI-OpenMP offloading is described below. This description is given for a Cray compiler of the wrapper `ftn`. On LUMI-G, the following modules may be necessary before compiling (see the `LUMI documentation <https://docs.lumi-supercomputer.eu/development/compiling/prgenv/>`_ for further details about the available programming environments): 

``
ml CrayEnv
ml PrgEnv-cray
ml cray-mpich
ml rocm
ml craype-accel-amd-gfx90a
``

.. challenge:: Example: ``Compilation process``

   .. tabs::

      .. tab:: Compiling MPI-OpenACC

         ``
         $ ftn -hacc -o mycode.mpiacc.exe mycode_mpiacc.f90
         ``

      .. tab:: Compiling MPI-OpenMP

         ``
         $ ftn -homp -o mycode.mpiomp.exe mycode_mpiomp.f90
         ``

.. note:: 

Here, the flags `hacc` and `homp` enable the OpenACC and OpenMP directives in the hybrid MPI-OpenACC and MPI-OpenMP applications, respectively.

**Enabling GPU-aware support**

To enable the GPU-aware support in MPICH library, one needs to set the following environment variable before running the application.

``
$ export MPICH_GPU_SUPPORT_ENABLED=1
``

Conclusion
----------
In conclusion, we have presented an overview of a GPU-hybrid programming by integrating GPU-directive models, specifically OpenACC and OpenMP APIs, with the MPI library. The approach adopted here allows us to utilise multiple GPU-devices not only within a single node but it extends to distributed nodes. In particular, we have addressed GPU-aware MPI approach, which has the advantage of enabling a direct interaction between an MPI library and a GPU-device memory. In other words, it permits performing MPI operations between a pair of GPUs, thus reducing the computing time caused by the data locality. 
 
Exercises
---------

We consider an MPI fortran code that solves a 2D-Laplace equation. Accelerate the code with either OpenACC or OpenMP API by following these steps:

**Exercise I: Set a GPU device**

1. Implement OpenACC/OpenMP functions that enable assigning each MPI rank to a GPU device.

**Exercise II: Accelerate loops**

2. Implement unstructured data blocks (i.e. `enter data` and `exit data` directives).

3. Include the necessary directives to accelerate the loops.

**Exercise III: Apply traditional MPI-OpenACC/OpenMP**

4. Implement the directives that enable updating the data in the host before calling an MPI functions (i.e. in OpenAC `update host()` for copying the data from GPU to CPU; and the directive `update device()` for copying the data from the CPU to GPU. In OpenMP, the directives are `update device() from()` and `update device() to()`, respectively, for copying the data from the GPU to CPU and from the CPU to the GPU).

5. Compile and run the code.

**Exercise IV: Implement GPU-aware support**

6. Implement the directives that enable to pass a device pointer to an MPI function (i.e. In OpenACC it is `host_data use_device()` and in OpenMP it is `data use_device_ptr()`).

7. Compile and run the code.

8. Evaluate the execution time in of the code in the exercises **III** and **IV**, and compare it with a pure MPI implementation.  

References
----------

`GPU-aware MPI <https://documentation.sigma2.no/code_development/guides/gpuaware_mpi.html>`_.

`MPI documentation <https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf>`_.

`OpenACC specification <https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf>`_.

`OpenMP specification <https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf>`_.

`LUMI documentation <https://docs.lumi-supercomputer.eu/development/compiling/prgenv/>`_.

`OpenACC vs OpenMP offloading <https://documentation.sigma2.no/code_development/guides/converting_acc2omp/openacc2openmp.html>`_.

`OpenACC course <https://github.com/HichamAgueny/GPU-course>`_.

