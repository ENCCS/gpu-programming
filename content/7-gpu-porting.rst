.. _gpu-porting:

Preparing code for GPU porting
==============================

.. questions::

   - q1
   - q2

.. objectives::

   - o1
   - o2

.. instructor-note::

   - X min teaching
   - X min exercises

Taking advantage of the parallel processing capability of the GPUs requires modifying the original code. However some work is required before writing actual code running on the GPUs:

* identify (or decide)  the parts of the code targeted by the porting. These are computational intensive parts of the code such as loops or matrix operations
* if a cpu library is used one should identify the equivalent one on the GPUs. For example BLAS library has cu/hipBLAS, or mkl equivalents. 
* when porting a loop directly,  works needs to be done to **refactor** it in a way that is suitable for the GPUs.(example missing here). This involves splitting the loop in several steps or changing some operations to reflect the independence of the operations between different iterations or give a better memory access. Each "step" of the original loop is then mapped to a kernel which is executed many gpu threads, each gpu thread correspoding to an iteration. 


.. keypoints::

   - k1
   - k2
