.. _gpu-hardware:

Introduction to GPU hardware
============================

.. questions::

   - q1
   - q2

.. objectives::

   - o1
   - o2

.. instructor-note::

   - X min teaching
   - X min exercises


Overview of GPU hardware
------------------------

.. figure:: img/hardware/CPUAndGPU.png
    :align: center

    A comparison of the CPU and GPU architecture.
    CPU (left) has complex core structure and pack several cores on a single chip.
    GPU cores are very simple in comparison, they also share data and control between each other.
    This allows to pack more cores on a single chip, thus achieving very high compute density.

One of the most important features that allows the accelerators to reach this high performance is their scalability.
Computational cores on accelerators are usually grouped into multiprocessors.
The multiprocessors share the data and logical elements.
This allows to achieve a very high density of a compute elements on a GPU.
This also allows for better scaling: more multiprocessors means more raw performance and this is very easy to achieve with more transistors available.


Accelerators are a separate main circuit board with the processor, memory, power management, etc.
It is connected to the motherboard with CPUs via PCIe bus.
Having its own memory means that the data has to be copied to and from it.
CPU acts as a main processor, controlling the execution workflow.
It copies the data from its own memory to the GPU memory, executes the program and copies the results back.
GPUs runs tens of thousands of threads simultaneously on thousands of cores and does not do much of the data management.
With many cores trying to access the memory simultaneously and with little cache available, the accelerator can run out of memory very quickly.
This makes the data management and its access pattern is essential on the GPU.
Accelerators like to be overloaded with the number of threads, because they can switch between threads very quickly.
This allows to hide the memory operations: while some threads wait, others can compute.



How do GPUs differ from CPUs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CPUs and GPUs were designed with different goals in mind. While the CPU 
is designed to excel at executing a sequence of operations, called a thread, 
as fast as possible and can execute a few tens of these threads in parallel, 
the GPU is designed to excel at executing many thousands of them in parallel. 
GPUs were initially developed for highly-parallel task of graphic processing 
and therefore designed such that more transistors are devoted to data processing 
rather than data caching and flow control. More transistors dedicated to 
data processing is beneficial for highly parallel computations; the GPU can 
hide memory access latencies with computation, instead of relying on large data caches 
and complex flow control to avoid long memory access latencies, 
both of which are expensive in terms of transistors.




.. list-table::  
   :widths: 100 100
   :header-rows: 1

   * - CPU
     - GPU
   * - General purpose
     - Highly specialized for parallelism
   * - Good for serial processing
     - Good for parallel processing
   * - Great for task parallelism
     - Great for data parallelism
   * - Low latency per thread
     - High-throughput
   * - Large area dedicated cache and control
     - Hundreds of floating-point execution units


Summary
^^^^^^^

- GPUs are highly parallel devices that can execute certain parts of the program in many parallel threads.
- CPU controls the works flow and makes all the allocations and data transfers.
- In order to use the GPU efficiently, one has to split their the problem  in many parts that can run simultaneously.


.. keypoints::

   - k1
   - k2
