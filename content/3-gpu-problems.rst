.. _gpu-problems:

What problems fit to GPU?
=========================

.. questions::

   - What are the strenghts and weaknesses of GPUs?
   - What makes a particular problem suitable for GPU-porting?
   - Why are GPUs so ubiquitous in machine learning applications?

.. objectives::

   - o1
   - o2

.. instructor-note::

   - 20 min teaching
   - 10 min exercises



.. keypoints::

   - k1
   - k2


Answer from `Stack Exchange <https://scicomp.stackexchange.com/questions/943/what-kinds-of-problems-lend-themselves-well-to-gpu-computing>`__:

   *From a metaphorical point of view, the GPU can be seen as a person lying on a bed 
   of nails. The person lying on top is the data and in the base of each nail there 
   is a processor, so the nail is actually an arrow pointing from processor to memory. 
   All nails are in a regular pattern, like a grid. If the body is well spread, 
   it feels good (performance is good), if the body only touches some spots of the 
   nail bed, then the pain is bad (bad performance).*


GPU computing is well-suited to problems that involve large amounts of data parallelism. 
Specifically, you can expect good performance on GPUs for:

- Large-scale matrix and vector operations, which are common in machine learning, scientific computing, and image processing.
- Fourier transforms, also common in machine learning, scientific computing, and image processing.
- Monte Carlo simulations, used across finance, physics, and other fields to simulate complex systems.
- Molecular dynamics simulations, which are used in chemistry, biochemistry and physics.
- Computational fluid dynamics, used in engineering, physics, and other fields.
- Convolutional neural networks and computer vision algorithms.
- Big data analytics, such as clustering, classification, and regression.
- Graphics rendering, which GPUs were originally designed for.

What are GPUs not good for
--------------------------


Examples of GPU accelerated software
------------------------------------

FIXME: show a few simple examples of CPU vs GPU versions of algorithms and roughly what speedup 
one can get 

Computational Chemistry
-----------------------

A great deal of computational resources are spent in Quantum Chemical calculations which involve
the solution of the Hartree-Fock eigenvalue problem, which requires the diagonalization of the
Fock matrix whose elements are given by:
   
.. math::
    F_{\alpha \beta} = H^{\textrm{core}}_{\alpha \beta} + \sum_{\gamma \delta}D_{\gamma \delta} \left [ (\alpha \beta|\gamma \delta) - \frac{1}{2} (\alpha \delta|\gamma \beta) \right ],

The first term is related to the one electron contributions and the second term is related to the 
electron repulsion integrals (ERIs), in parenthesis, weighted by the by the density matrix 
:math:`D_{\gamma \delta}`. One of the most expensive parts in the solution of the HF equations is the 
processing (digestion) of the ERIs, one algorithm to do this task is as follows:

.. figure:: img/concepts/algorithms.svg
    :width: 200
    :align: center

    Algorithm for processing ERIs [JCTC, 17, 7486, (2021)]

This algorithm is suitable for GPUs as it involves many arithmetic operations. In addition to this,
there are symmetries and properties of the integrals that could be used to rearrange the loops in
an efficient manner that fit GPU architectures. 
