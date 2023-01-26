.. _gpu-history:

Why GPUs?
=========

.. questions::

   - What is Moore's law?
   - What problem do GPUs solve?

.. objectives::

   - Explain the historical development of microprocessors and how GPUs enable 
     continued scaling of computational power

.. instructor-note::

   - X min teaching
   - X min exercises


Moore's law
-----------

The number of transistors in a dense integrated circuit doubles about every two years.
More transistors means smaller size of a single element, so higher core frequency can be achieved.
However, power consumption scales as frequency in third power, so the growth in the core frequency has slowed down significantly.
Higher performance of a single node has to rely on its more complicated structure and still can be achieved with SIMD, branch prediction, etc.

.. figure:: img/history/microprocessor-trend-data.png
   :align: center

   The evolution of microprocessors.
   The number of transistors per chip increase every 2 years or so.
   However it can no longer be explored by the core frequency due to power consumption limits.
   Before 2000, the increase in the single core clock frequency was the major source of the 
   increase in the performance. Mid 2000 mark a transition towards multi-core processors.

Achieving performance has been based on two main strategies over the years:

    - Increase the single processor performance: 
    - More recently, increase the number of physical cores.


Computing in parallel
---------------------

The underlying idea of parallel computing is to split a computational problem into smaller 
subtasks. Many subtasks can then be solved *simultaneously* by multiple processing units. 

.. figure:: img/history/compp.png
   :align: center
   
   Computing in parallel.

How a problem is split into smaller subtasks depends fully on the problem. 
There are various paradigms and programming approaches how to do this. 

Distributed- vs. Shared-Memory Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of computing problems are not trivially parallelizable, which means that the subtasks 
need to have access from time to time to some of the results computed by other subtasks. 
The way subtasks exchange needed information depends on the available hardware.

.. figure:: img/history/distributed_vs_shared.png
   :align: center
   
   Distributed- vs shared-memory parallel computing.

In a distributed memory environment each computing unit operates independently from the 
others. It has its own memory and it  **cannot** access the memory in other nodes. 
The communication is done via network and each computing unit runs a separate copy of the 
operating system. In a shared memory machine all computing units have access to the memory 
and can read or modify the variables within.

Processes and threads
~~~~~~~~~~~~~~~~~~~~~

The type of environment (distributed- or shared-memory) determines the programming model. 
There are two types of parallelism possible, process based and :abbr:`thread` based. 

.. figure:: img/history/processes-threads.png
   :align: center

For distributed memory machines, a process-based parallel programming model is employed. 
The processes are independent execution units which have their *own memory* address spaces. 
They are created when the parallel program is started and they are only terminated at the 
end. The communication between them is done explicitly via message passing like MPI.

On the shared memory architectures it is possible to use a thread based parallelism.  
The threads are light execution units and can be created and destroyed at a relatively 
small cost. The threads have their own state information but they *share* the *same memory* 
adress space. When needed the communication is done though the shared memory. 


Both approaches have their advantages and disadvantages.  Distributed machines are 
relatively cheap to build and they  have an "infinite " capacity. In principle one could 
add more and more computing units. In practice the more computing units are used the more 
time consuming is the communication. The shared memory systems can achive good performance 
and the programing model is quite simple. However they are limited by the memory capacity 
and by the access speed. In addition in the shared parallel model it is much easier to 
create race conditions.


Graphics processing units
-------------------------

The Graphics processing units (GPU) have been the most common accelerators during the last few years, the term GPU sometimes is used interchangeably with the term accelerator.
GPUs were initially developed for highly-parallel task of graphic processing.
Over the years, were used more and more in HPC.
GPUs are a specialized parallel hardware for floating point operations.
GPUs are co-processors for traditional CPUs: CPU still controls the work flow, delegating highly-parallel tasks to the GPU.
Based on highly parallel architectures, which allows to take advantage of the increasing number of transistors.

Using GPUs allows one to achieve very high performance per node.
As a result, the single GPU-equipped workstation can outperform small CPU-based cluster for some type of computational tasks.
The drawback is: usually major rewrites of programs is required.


Energy efficiency
-----------------





.. keypoints::

   - k1
   - k2
