.. _multiple-gpus:

Multiple GPU programming with MPI
=================================

.. questions::

   - q1
   - q2

.. objectives::

   - Combining MPI with either OpenACC or OpenMP offloading
   - Implementing GPU-awareness MPI approach 

.. instructor-note::

   - X min teaching
   - X min exercises

Introduction
------------

Exploring multiple-GPUs across distributed nodes offers the potential to fully leveraging the capacity of modern HPC (High-Performance Computing) systems at a large scale. This has the advantage of addressing complex scientific problems that would be barely possible on conventional CPU-based HPC systems. Such a computational problem can benefit from available GPU programming models to further accelerate computing. Here combining a GPU programming model such as OpenACC and OpenMP application programming interfaces (APIs) with MPI (message passing interface) is a clear choice. 

In this section we guide readers, who are familiar with MPI, in implementing a hybrid model in which the MPI communication framework is combined with either OpenACC or OpenMP APIs. A special focus will be on performing point-to-point (e.g. `MPI_Send` and `MPI_Recv`) and collective operations (e.g. `MPI_Allreduce`) from OpenACC and OpenMP APIs. Here we address two scenarios: (i) a scenario in which MPI operations are performed in the CPU-host followed by offloading to the GPU-device; and (ii) a scenario in which MPI operations are performed between a pair of GPUs without involving the CPU-host. The latter is referred to as GPU-awareness MPI, and has the advantage of reducing the computing time caused by transferring data via the host-memory during heterogenous communications, thus rendering HPC applications efficient. 

This guide is organized as follow: we first introduce how to assign each MPI rank to a GPU device within the same node. We consider a situation in which the host and the device have a distinct memory. This is followed by a presentation on the hybrid MPI-OpenACC/OpenMP offloading with and without the GPU-awareness MPI. Exercises to help understanding these concepts are provided at the end.

Assigning MPI-ranks to GPU-devices
----------------------------------

Accelerating MPI applications to utilise multiple GPUs on distributed nodes requires as a first step assigning each MPI rank to a GPU device, such that two MPI ranks do not use the same GPU device. This is necessarily in order to prevent the application from a potential crash. This is because GPUs are designed to handle multiple threading tasks, but not multiple MPI ranks. 

One of the way to ensure that two MPI ranks do not use the same GPU, is to determine which MPI processes run on the same node, such that each process can be assigned to a GPU device within the same node. This can be done by splitting the world communicator into sub-groups of communicators (or sub-communicators) using the routine `MPI_COMM_S PLIT_TYPE()`

The size of each sub-communicator corresponds the number of GPUs per node (which is also the number of tasks per node).

.. tab:: Fortran

         .. literalinclude:: examples/assignDevice_acc.f90
                        :language: fortran
                        :emphasize-lines: 27-29
                        
Here each sub-communicator contains a list of processes indicated by a rank. These processes have a shared-memory region defined by the argument 
`MPI_COMM_TYPE_SHARED` (see ref. <xref linkend="ref-mpi"/> for more details). Calling the routine `MPI_COMM_SPLIT_TYPE()` returns a sub-communicator 
labelled *”host_comm”* in which MPI-ranks are ranked from 0 to number processes per node -1. These MPI ranks are in turn assigned to different GPU 
devices within the same node. This procedure is done according to which directive-based model is implemented. 

The retrieved MPI ranks are stored in the variable **myDevice**. The variable is passed to an OpenACC or OpenMP functions 

.. challenge:: Example: ``set device``

   .. tabs::

      .. tab:: OpenACC

         acc_set_device_num(myDevice, acc_get_device_type())

      .. tab:: OpenMP

         omp_set_default_device(myDevice)
.. note:: 


On the other hand, one can check the total number of devices available on the host by using the functions:

.. challenge:: Example: ``number of devices``

   .. tabs::

      .. tab:: OpenACC

         acc_get_num_devices(acc_get_device_type())

      .. tab:: OpenMP

         omp_get_num_devices()
.. note:: 

Another useful function for retrieving the device number of a specific device, which is useful, e.g., to map data to a specific device
	
.. tabs::

      .. tab:: OpenACC

         acc_get_device_num()

      .. tab:: OpenMP

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

.. note:: 


Hybrid MPI-OpenACC/OpenMP without GPU-awareness approach
--------------------------------------------------------

After covering how to assign each MPI-rank to a GPU device within the same node, we now address the concept of combining MPI with either
OpenACC or OpenMP offloading. In this approach calling an MPI routine from an OpenACC or OpenMP API requires updating the data in the CPU host before and after an MPI call. In this scenario, the data are copied back and forth between the host and the device before and after each MPI call. In the hybrid MPI-OpenACC model, the procedure is defined by specifying the directive `update host()` for copying the data froma device to a host before an MPI call; and by the directive `update device()` specified after an MPI call for copying the data back to a device. Similarly in the hybrid MPI-OpenMP. Here, updating the data in a host can be done by specifying the OpenMP directives `update device() from()` and `update device() to()`, respectively, for copying the data from a device to a host and back to the device.

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

.. note:: 

Despite the simplicity of implementing the hybrid MPI-OpenACC/OpenMP offloading, it suffers from a low performance caused by an explicit transfer of data between a host and a device before and after calling an MPI routine. This constitutes a bottleneck in GPU-programming. To improve the performance affected by the host staging during the data transfer, one can implement the GPU-awareness MPI approach as described in the following section.
	  
Hybrid MPI-OpenACC/OpenMP with GPU-awareness approach 
-----------------------------------------------------

The concept of the GPU-aware MPI enables an MPI library to directly access the GPU-device memory without necessarily using the CPU-host memory as an intermediate buffer. This offers the benefit of transferring data from one GPU to another GPU without involving the CPU-host.
	  
To be specific, in the GPU-awareness approach, the device pointers point to the data allocated in the GPU memory space (data should be present in the GPU device). Here, the pointers are passed as arguments to an MPI routine that is supported by the GPU memory. Note that not all the MPI routines are supported by the GPU memory (see here TOBE INCLUDED). As MPI routines can directly access GPU memory, it offers the possibility of communicating between pairs of GPUs without transferring data back to the host. 

In the hybrid MPI-OpenACC model, the concept is defined by combining the directive `host_data` together with the clause
`use_device(list_array)`. This combination enables the access to the arrays listed in the clause `use_device(list_array)` from the host (see [here](#https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.2-final.pdf)). The list of arrays, which are already present in the GPU-device memory, are directly passed to an MPI routine without a need of a staging host-memory for copying the data.


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

.. note:: 


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

.. note:: 
The GPU-aware MPI with OpenACC/OpenMP offloading has the capability of directly communicating between a pair of GPUs within a single node. However, performing the GPU-to-GPU communication across multiple nodes requires the the GPUDirect RDMA (Remote Direct Memory Access) technology. This technology can further improve performance by reducing latency.

Compilation process
-------------------

Conclusion
----------

Exercises
---------
