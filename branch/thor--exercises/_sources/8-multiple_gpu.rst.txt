.. _multiple-gpus:

Multiple GPU programming with MPI
=================================

.. questions::

   - What approach should be adopted to extend the synchronous OpenACC and OpenMP offloading models to utilise multiple GPUs across multiple nodes? 

.. objectives::

   - To learn about combining MPI with either OpenACC or OpenMP offloading models.
   - To learn about implementing GPU-awareness MPI approach. 

.. instructor-note::

   - 30 min teaching
   - 30 min exercises

Introduction
------------

Exploring multiple GPUs (Graphics Processing Units) across distributed nodes offers the potential to fully leveraging the capacity of modern HPC (High-Performance Computing) systems at a large scale. Here one of the approaches to accelerate computing on distributed systems is to combine MPI (Message Passing Interface) with a GPU programming model such as OpenACC and OpenMP application programming interfaces (APIs). This combination is motivated by both the simplicity of these APIs, and the widespread use of MPI.   

In this guide we provide readers, who are familiar with MPI, with insighits on implementing a hybrid model in which the MPI communication framework is combined with either OpenACC or OpenMP APIs. A special focus will be on performing point-to-point (e.g. `MPI_Send` and `MPI_Recv`) and collective operations (e.g. `MPI_Allreduce`) from OpenACC and OpenMP APIs. Here we address two scenarios: (i) a scenario in which MPI operations are performed in the CPU-host followed by an offload to the GPU-device; and (ii) a scenario in which MPI operations are performed between a pair of GPUs without involving the CPU-host memory. The latter scenario is referred to as GPU-awareness MPI, and has the advantage of reducing the computing time caused by transferring data via the host-memory during heterogenous communications, thus rendering HPC applications efficient. 

This guide is organized as follows: we first introduce how to assign each MPI rank to a GPU device within the same node. We consider a situation in which the host and the device have a distinct memory. This is followed by a presentation on the hybrid MPI-OpenACC/OpenMP offloading with and without the GPU-awareness MPI. Exercises to help understanding these concepts are provided at the end.

Assigning MPI-ranks to GPU-devices
----------------------------------

Accelerating MPI applications to utilise multiple GPUs on distributed nodes requires as a first step assigning each MPI rank to a GPU device, such that two MPI ranks do not use the same GPU device. This is necessarily in order to prevent the application from a potential crash. This is because GPUs are designed to handle multiple threading tasks, but not multiple MPI ranks. 

One of the way to ensure that two MPI ranks do not use the same GPU, is to determine which MPI processes run on the same node, such that each process can be assigned to a GPU device within the same node. This can be done, for instance, by splitting the world communicator into sub-groups of communicators (or sub-communicators) using the routine `MPI_COMM_SPLIT_TYPE()`. 

.. tabs::

    .. tab:: Splitting communicator in MPI

         .. literalinclude:: examples/mpi_acc/assignDevice_acc.f90
            :language: fortran
            :lines: 22-29


Here, the size of each sub-communicator corresponds to the number of GPUs per node (which is also the number of tasks per node), and each sub-communicator contains a list of processes indicated by a rank. These processes have a shared-memory region defined by the argument `MPI_COMM_TYPE_SHARED` (see the `MPI report <https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf>`_) for more details). Calling the routine `MPI_COMM_SPLIT_TYPE()` returns a sub-communicator labelled in the code above *”host_comm”*, and in which MPI-ranks are ranked from 0 to number of processes per node -1. These MPI ranks are in turn assigned to different GPU devices within the same node. This procedure is done according to which directive-based model is implemented. The retrieved MPI ranks are then stored in the variable **myDevice**. The variable is passed to an OpenACC or OpenMP routine as indicated in the code below. 

.. typealong:: Example: ``Assign device``

   .. tabs::

      .. tab:: OpenACC

         .. literalinclude:: examples/mpi_acc/assignDevice_acc.f90
            :language: fortran
            :lines: 34-40

      .. tab:: OpenMP

         .. literalinclude:: examples/mpi_omp/assignDevice_omp.f90
            :language: fortran
            :lines: 34-40


Another useful function for retrieving the device number of a specific device, which is useful, e.g., to map data to a specific device is
	
.. tabs::

   .. tab:: OpenACC
     
      .. code-block:: fortran
 	
         acc_get_device_num()

   .. tab:: OpenMP

      .. code-block:: fortran
	 
       	 omp_get_device_num()

The syntax of assigning MPI ranks to GPU devices is summarised below

.. typealong:: Example: ``Set device``

   .. tabs::

      .. tab:: MPI-OpenACC
	 
         .. literalinclude:: examples/mpi_acc/assignDevice_acc.f90
            :language: fortran
            :lines: 15-40

      .. tab:: MPI-OpenMP
	 
         .. literalinclude:: examples/mpi_omp/assignDevice_omp.f90
                     :language: fortran
                     :lines: 15-40


Hybrid MPI-OpenACC/OpenMP without GPU-awareness approach
--------------------------------------------------------

After covering how to assign each MPI-rank to a GPU device, we now address the concept of combining MPI with either
OpenACC or OpenMP offloading. In this approach, calling an MPI routine from an OpenACC or OpenMP API requires updating the data in the CPU host before and after an MPI call. In this scenario, the data is copied back and forth between the host and the device before and after each MPI call. In the hybrid MPI-OpenACC model, the procedure is defined by specifying the directive `update host()` for copying the data from the device to the host before an MPI call; and by the directive `update device()` specified after an MPI call for copying the data back to the device. Similarly in the hybrid MPI-OpenMP. Here, updating the data in the host can be done by specifying the OpenMP directives `update device() from()` and `update device() to()`, respectively, for copying the data from the device to the host and back to the device.

To illustrate the concept of the hybrid MPI-OpenACC/OpenMP, we show below an example of an implementation that involves the MPI functions `MPI_Send()` and `MPI_Recv()`.


.. typealong:: Example: ``Update host/device directives``

   .. tabs::

      .. tab:: MPI-OpenACC

         .. literalinclude:: examples/mpi_acc/mpiacc.f90
                     :language: fortran
                     :lines: 62-77

      .. tab:: MPI-OpenMP

         .. literalinclude:: examples/mpi_omp/mpiomp.f90
                     :language: fortran
                     :lines: 63-78


Here we present a code example that combines MPI with OpenACC/OpenMP API.

.. typealong:: Example: ``Update host/device directives``

   .. tabs::

      .. tab:: MPI-OpenACC
	 
         .. literalinclude:: examples/mpi_acc/mpiacc.f90
                     :language: fortran
                     :lines: 60-94

      .. tab:: MPI-OpenMP

         .. literalinclude:: examples/mpi_omp/mpiomp.f90
                     :language: fortran
                     :lines: 61-97

Despite the simplicity of implementing the hybrid MPI-OpenACC/OpenMP offloading, it suffers from a low performance caused by an explicit transfer of data between the host and the device before and after calling an MPI routine. This constitutes a bottleneck in GPU-programming. To improve the performance affected by the host staging during the data transfer, one can implement the GPU-awareness MPI approach as described in the following section.
	  
Hybrid MPI-OpenACC/OpenMP with GPU-awareness approach 
-----------------------------------------------------

The concept of the GPU-aware MPI enables an MPI library to directly access the GPU-device memory without necessarily using the CPU-host memory as an intermediate buffer (see e.g. `here <https://docs.open-mpi.org/en/v5.0.0rc9/networking/cuda.html>`__). This offers the benefit of transferring data from one GPU to another GPU without the involvement of the CPU-host memory.
	  
To be specific, in the GPU-awareness approach, the device pointers point to the data allocated in the GPU memory space (data should be present in the GPU device). Here, the pointers are passed as arguments to an MPI routine that is supported by the GPU memory. As MPI routines can directly access GPU memory, it offers the possibility of communicating between pairs of GPUs without transferring data back to the host. 

In the hybrid MPI-OpenACC model, the concept is defined by combining the directive `host_data` together with the clause
`use_device(list_array)`. This combination enables the access to the arrays listed in the clause `use_device(list_array)` from the host (see `here <https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf>`__). The list of arrays, which are already present in the GPU-device memory, are directly passed to an MPI routine without a need of a staging host-memory for copying the data. Note that for initially copying data to GPU, we use unstructured data blocks characterized by the directives `enter data` and `exit data`. The unstructured data has the advantage of allowing to allocate and deallocate arrays within a data region.

To illustarte the concept of the GPU-awareness MPI, we show below two examples that make use of point-to-point and collective operations from OpenACC and OpenMP APIs. In the first code example, the device pointer **f** is passed to the MPI functions `MPI_Send()` and `MP_Recv()`; and in the second one, the pointer **SumToT** is passed to the MPI function `MPI_Allreduce`. Here, the MPI operations `MPI_Send` and `MPI_Recv` as well as `MPI_Allreduce` are performed between a pair of GPUs without passing through the CPU-host memory. 

.. typealong:: Example: ``GPU-awareness: MPI_Send & MPI_Recv``

   .. tabs::

      .. tab:: GPU-aware MPI with OpenACC
	 
         .. literalinclude:: examples/mpi_acc/mpiacc_gpuaware.f90
                     :language: fortran
                     :lines: 65-74

      .. tab:: GPU-aware MPI with OpenMP
	 
         .. literalinclude:: examples/mpi_omp/mpiomp_gpuaware.f90
                     :language: fortran
                     :lines: 66-75


.. typealong:: Example: ``GPU-awareness: MPI_Allreduce``

   .. tabs::

      .. tab:: GPU-aware MPI with OpenACC
	 
         .. literalinclude:: examples/mpi_acc/mpiacc_gpuaware.f90
                     :language: fortran
                     :lines: 90-94

      .. tab:: GPU-aware MPI with OpenMP
	 
         .. literalinclude:: examples/mpi_omp/mpiomp_gpuaware.f90
                     :language: fortran
                     :lines: 93-97 


We provide below a code example that illustrates the implementation of the MPI functions `MPI_Send()`, `MPI_Recv()` and `MPI_Allreduce()` within an OpenACC/OpenMP API. This implementation is specifically designed to support GPU-aware MPI operations. 

.. typealong:: Example: ``GPU-awareness approach``

   .. tabs::

      .. tab:: GPU-aware MPI with OpenACC

         .. literalinclude:: examples/mpi_acc/mpiacc_gpuaware.f90
                     :language: fortran
                     :lines: 60-97

      .. tab:: GPU-aware MPI with OpenMP

         .. literalinclude:: examples/mpi_omp/mpiomp_gpuaware.f90
                     :language: fortran
                     :lines: 60-100

The GPU-aware MPI with OpenACC/OpenMP APIs has the capability of directly communicating between a pair of GPUs within a single node. However, performing the GPU-to-GPU communication across multiple nodes requires the the GPUDirect RDMA (Remote Direct Memory Access) technology. This technology can further improve performance by reducing latency.

Compilation process
-------------------

The compilation process of the hybrid MPI-OpenACC and MPI-OpenMP offloading is described below. This description is given for a Cray compiler of the wrapper `ftn`. On LUMI-G, the following modules may be necessary before compiling (see the `LUMI documentation <https://docs.lumi-supercomputer.eu/development/compiling/prgenv/>`_ for further details about the available programming environments): 

.. code-block::

	 ml CrayEnv
	 ml PrgEnv-cray
	 ml cray-mpich
	 ml rocm
	 ml craype-accel-amd-gfx90a


.. typealong:: Example: ``Compilation process``

   .. tabs::

      .. tab:: Compiling MPI-OpenACC

         $ ftn -hacc -o mycode.mpiacc.exe mycode_mpiacc.f90

      .. tab:: Compiling MPI-OpenMP

         $ ftn -homp -o mycode.mpiomp.exe mycode_mpiomp.f90


Here, the flags `hacc` and `homp` enable the OpenACC and OpenMP directives in the hybrid MPI-OpenACC and MPI-OpenMP applications, respectively.

**Enabling GPU-aware support**

To enable the GPU-aware support in MPICH library, one needs to set the following environment variable before running the application.

.. code-block::

     $ export MPICH_GPU_SUPPORT_ENABLED=1


Conclusion
----------
In conclusion, we have presented an overview of a GPU-hybrid programming by integrating GPU-directive models, specifically OpenACC and OpenMP APIs, with the MPI library. The approach adopted here allows us to utilise multiple GPU-devices not only within a single node but it extends to distributed nodes. In particular, we have addressed GPU-aware MPI approach, which has the advantage of enabling a direct interaction between an MPI library and a GPU-device memory. In other words, it permits performing MPI operations between a pair of GPUs, thus reducing the computing time caused by the data locality. 
 
Exercises
---------

We consider an MPI fortran code that solves a 2D-Laplace equation, and which is partially accelerated. The focus of the exercises is to complete the acceleration using either OpenACC or OpenMP API by following these steps. 

.. callout:: Access exercise material

   Code examples for the exercises below can be accessed in the `content/examples/exercise_multipleGPU` subdirectory of this repository. To access them, you need to clone the repository:

   .. code-block:: console

      $ git clone https://github.com/ENCCS/gpu-programming.git
      $ cd gpu-programming/content/examples/exercise_multipleGPU
      $ ls

.. challenge:: Exercise I: Set a GPU device

   1. Implement OpenACC/OpenMP functions that enable assigning each MPI rank to a GPU device.

   1.1 Compile and run the code on multiple GPUs.

.. challenge:: Exercise II: Apply traditional MPI-OpenACC/OpenMP

   2.1 Incoporate the OpenACC directives `*update host()*` and `*update device()*` before and after calling an MPI function, respectively. 

   .. note:: 
      The OpenACC directive `*update host()*` is used to transfer data from GPU to CPU within a data region; while the directive `*update device()*` is used to transfer the data from CPU to GPU. 

   2.2 Incorporate the OpenMP directives `*update device() from()*` and `*update device() to()*` before and after calling an MPI function, respectively.

   .. note:: 
      The OpenMP directive `*update device() from()*` is used to transfer data from GPU to CPU within a data region; while the directive `*update device() to()*` is used to transfer the data from CPU to GPU. 

   2.3 Compile and run the code on multiple GPUs.

.. challenge:: Exercise III: Implement GPU-aware support

   3.1 Incorporate the OpenACC directive `*host_data use_device()*` to pass a device pointer to an MPI function.

   3.2 Incorporate the OpenMP directive `*data use_device_ptr()*` to pass a device pointer to an MPI function.

   3.3 Compile and run the code on multiple GPUs.

.. challenge:: Exercise IV: Evaluate the performance

   1. Evaluate the execution time of the accelerated codes in the exercises **II** and **III**, and compare it with that of a pure MPI implementation.  

See also
--------

- `GPU-aware MPI <https://documentation.sigma2.no/code_development/guides/gpuaware_mpi.html>`_.
- `MPI documentation <https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf>`_.
- `OpenACC specification <https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf>`_.
- `OpenMP specification <https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf>`_.
- `LUMI documentation <https://docs.lumi-supercomputer.eu/development/compiling/prgenv/>`_.
- `OpenACC vs OpenMP offloading <https://documentation.sigma2.no/code_development/guides/converting_acc2omp/openacc2openmp.html>`_.
- `OpenACC course <https://github.com/HichamAgueny/GPU-course>`_.


