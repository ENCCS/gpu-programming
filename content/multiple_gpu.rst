.. _gpu-concepts:

Multiple GPU programming
========================

.. questions::

   - q1
   - q2

.. objectives::

   - o1
   - o2

.. instructor-note::

   - X min teaching
   - X min exercises

Introduction
------------

Exploring multiple-GPUs across distributed nodes offers the potential to fully leveraging the capacity of modern HPC (High-Performance Computing) 
systems at a large scale. This has the advantage of addressing complex scientific problems that would be barely possible on conventional 
CPU-based HPC systems. Such a computational problem can benefit from available GPU programming models to further accelerate computing. 
Here combining a GPU programming model such as OpenACC and OpenMP application programming interfaces (APIs) with MPI (message passing interface) 
is a clear choice. 

In this section we guide readers, who are familiar with MPI, in implementing a hybrid model in which the MPI communication framework is combined with either OpenACC or OpenMP APIs. A special focus will be on performing point-to-point (e.g. `MPI_Send` and `MPI_Recv`) and collective operations (e.g. `MPI_Allreduce`) between a pair of GPUs . This concept is referred to as GPU-aware MPI, and will be presented in the context of both the hybrid MPI-OpenACC and MPI-OpenMP models. Its implementation has the advantage of reducing the computing time caused by transferring data via the host-memory during heterogenous communications, thus rendering HPC applications efficient. For reference, a scenario in which the GPU-non-aware MPI will also be addressed.

GPU-awareness MPI approach
--------------------------

The GPU-awareness approach simply means how to make a GPU-device memory aware or not aware of the existence of an MPI library, such that a device 
pointer can be passed to the MPI library, which is GPU-accelerated. 

In the following, we first introduce how to assign each MPI rank to a specific GPU device. We consider a situation in which the host and the device 
have a distinct memory. This is followed by a presentation on GPU-non- aware MPI and GPU-aware MPI concepts. Exercises to help understanding these concepts are provided at the end.

Assigning MPI-ranks to GPU-devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accelerating MPI applications to utilise multiple GPUs on distributed nodes requires as a first step assigning each MPI rank to a GPU device, such that 
two MPI ranks do not use the same GPU device. This is necessarily in order to prevent the application from a potential crash. This is because GPUs are 
designed to handle multiple threading tasks, but not multiple MPI ranks. 

One of the way to ensure that two MPI ranks do not use the same GPU, is to determine which MPI processes run on the same node, such that each process 
can be assigned to a GPU device within the same node. This can be done by splitting the world communicator into sub-groups of communicators 
(or sub-communicators) using the routine `MPI_COMM_S PLIT_TYPE()`

The size of each sub-communicator corresponds the number of GPUs per node (which is also the number of tasks per node).

```console
call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,&amp;
                               MPI_INFO_NULL, host_comm,ierr)
call MPI_COMM_RANK(host_comm, myDevice,ierr)
```
Here each sub-communicator contains a list of processes indicated by a rank. These processes have a shared-memory region defined by the argument 
`MPI_COMM_TYPE_SHARED` (see ref. <xref linkend="ref-mpi"/> for more details). Calling the routine `MPI_COMM_SPLIT_TYPE()` returns a sub-communicator 
labelled *”host_comm”* in which MPI-ranks are ranked from 0 to number processes per node -1. These MPI ranks are in turn assigned to different GPU 
devices within the same node. This procedure is done according to which directive-based model is implemented. 

The retrieved MPI ranks are stored in the variable **myDevice**. The variable is passed to an OpenACC or OpenMP functions 

````{group-tab} OpenACC
```console
acc_set_device_num(myDevice, acc_get_device_type())
```
````
````{group-tab} OpenMP

```console
omp_set_default_device(myDevice)
```
````


On the other hand, one can check the total number of devices available on the host by using the functions:

````{group-tab} OpenACC
```console
acc_get_num_devices(acc_get_device_type())
```
````
````{group-tab} OpenMP

```console
omp_get_num_devices()
```
````	 

Another useful function for retrieving the device number of a specific device, which is useful, e.g., to map data to a specific device

````{group-tab} OpenACC
```console
acc_get_device_num()
```
````
````{group-tab} OpenMP

```console
omp_get_device_num()
```
````	

The syntax of assigning MPI ranks to GPU devices is summarised below

````{group-tab} MPI-OpenACC
```console
program assignDevice

      use mpi
      use openacc

      implicit none
       integer status(MPI_STATUS_SIZE)
       integer :: myid,ierr,nproc
       integer :: host_rank,host_comm
       integer :: myDevice,numDevice

! Initialise MPI communication.
call MPI_INIT(ierr)
! Get number of active processes (from 0 to nproc-1).
call MPI_COMM_SIZE(MPI_COMM_WORLD, nproc, ierr )
! Identify the ID rank (process).
call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr )

! Split the world communicator into subgroups of commu, each of which
! contains processes that run on the same node, and which can create a
! shared memory region (via the type MPI_COMM_TYPE_SHARED).
! The call returns a new communicator "host_comm", which is created by
! each subgroup.
call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,&amp;
                               MPI_INFO_NULL, host_comm,ierr)
call MPI_COMM_RANK(host_comm,host_rank,ierr)

myDevice = host_rank

!Sets the device number and the device type to be used
call acc_set_device_num(myDevice, acc_get_device_type())

!Returns the number of devices available on the host
      numDevice = acc_get_num_devices(acc_get_device_type())

write(*,'(A,I3,A,A,A,I3,A,I3)') "MPI-rank ", myid, " - Node ", trim(name), " - GPU_ID ", myDevice, " - GPUs-per-node ", numDevice

call MPI_FINALIZE( ierr )

       end
```
````
````{group-tab} MPI-OpenMP

```console
program assignDevice

      use mpi
      use omp_lib

      implicit none
       integer status(MPI_STATUS_SIZE)
       integer :: myid,ierr,nproc
       integer :: host_rank,host_comm
       integer :: myDevice,numDevice

! Initialise MPI communication.
call MPI_INIT(ierr)
! Get number of active processes (from 0 to nproc-1).
call MPI_COMM_SIZE(MPI_COMM_WORLD, nproc, ierr )
! Identify the ID rank (process).
call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr )

! Split the world communicator into subgroups of commu, each of which
! contains processes that run on the same node, and which can create a
! shared memory region (via the type MPI_COMM_TYPE_SHARED).
! The call returns a new communicator "host_comm", which is created by
! each subgroup.
call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,&amp;
                               MPI_INFO_NULL, host_comm,ierr)
call MPI_COMM_RANK(host_comm,host_rank,ierr)

myDevice = host_rank

!Sets the device number to use in device constructs by setting the
!initial value of the default-device-var 
call omp_set_default_device(myDevice)

! Returns the number of devices available for offloading.
     numDevice = omp_get_num_devices()

write(*,'(A,I3,A,A,A,I3,A,I3)') "MPI-rank ", myid, " - Node ", trim(name), " - GPU_ID ", myDevice, " - GPUs-per-node ", numDevice

call MPI_FINALIZE( ierr )

       end
```
````	


GPU-no-aware MPI
~~~~~~~~~~~~~~~


GPU-aware MPI
~~~~~~~~~~~~~

Exercises
~~~~~~~~~

Compilation process
~~~~~~~~~~~~~~~~~~~

Conclusion
----------
