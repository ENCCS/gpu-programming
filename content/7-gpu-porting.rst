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

Taking advantage of the parallel processing capability of the GPUs requires modifying the original. However some work is required before writing actual code running on the GPUs:

* identify (or decide)  the parts of the code targeted by the porting. These are computational intensive parts of the code such as loops or matrix operations
* if a cpu library is used one should idnetify the equivalent one on the GPUs. For example BLAS library has cu/hipBLAS, or mkl equivlanents. 
* when porting a loop directly works needs to be done to refactor it in a way that is suitable for the GPUs.(example missing here). Loops are mapped to kernels which are executed by many gpu threads. Due to data dependency (the gpu threads have very limited communications among themselves) and performance considerations, it is possible that a CPU loop to be splitted in many kernels.


.. keypoints::

   - k1
   - k2
