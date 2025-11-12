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