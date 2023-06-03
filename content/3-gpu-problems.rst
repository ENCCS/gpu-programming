.. _gpu-problems:

What problems fit to GPU?
=========================

.. questions::

   - What are the strenghts and weaknesses of GPUs?
   - What makes a particular problem suitable for GPU-porting?
   - Why are GPUs so ubiquitous in machine learning applications?

.. objectives::

   - Get a feeling for the type of use cases that GPUs excel at.

.. instructor-note::

   - 10 min teaching
   - 10 min exercises


What are GPUs good for
----------------------


Answer from `Stack Exchange <https://scicomp.stackexchange.com/questions/943/what-kinds-of-problems-lend-themselves-well-to-gpu-computing>`__:

   *From a metaphorical point of view, the GPU can be seen as a person lying on a bed 
   of nails. The person lying on top is the data and in the base of each nail there 
   is a processor, so the nail is actually an arrow pointing from processor to memory. 
   All nails are in a regular pattern, like a grid. If the body is well spread, 
   it feels good (performance is good), if the body only touches some spots of the 
   nail bed, then the pain is bad (bad performance).*


GPU computing is well-suited to problems that involve large amounts of data parallelism. 
Specifically, you can expect good performance on GPUs for:

- **Large-scale matrix and vector operations**: Common in machine learning, scientific computing, and image processing.
- **Fourier transforms**: Also common in machine learning, scientific computing, and image processing.
- **Monte Carlo simulations**: Used across finance, physics, and other fields to simulate complex systems.
- **Molecular dynamics simulations**: Used in chemistry, biochemistry and physics.
- **Computational fluid dynamics**: Used in engineering, physics, and other fields.
- **Convolutional neural networks** and **computer vision algorithms**.
- **Big data analytics**: Clustering, classification, regression, etc.
- **Graphics rendering**: Original use-case for GPUs.

What are GPUs not good for
--------------------------

Not all programming problems can efficiently leverage the parallelism offered by GPUs. 
Some types of problems that do not fit well on a GPU include:

- **Sequential tasks**: Problems that require a series of dependent steps, 
  where each step relies on the outcome of the previous step, are not well-suited 
  for parallel processing. Examples include recursive algorithms, certain dynamic 
  programming problems, and some graph traversal algorithms.

- **Fine-grained branching**: GPUs perform best when the code being executed across 
  different threads follows a similar control flow. When there is extensive 
  branching (i.e., many ``if`` statements) within a kernel or algorithm, performance 
  may suffer due to the divergence in execution paths among the GPU threads.

- **Low arithmetic intensity**: GPUs excel at performing a large number of mathematical 
  operations quickly. If a problem has low arithmetic intensity (i.e., a low ratio of 
  arithmetic operations to memory accesses), the GPU may not be able to efficiently utilize 
  its computational power, leading to underperformance.

- **Small data sets**: If the problem involves a small data set that does not require significant 
  parallelism, using a GPU may not result in noticeable performance gains. In such cases, 
  the overhead of transferring data between the CPU and GPU, and the time spent initializing the GPU, 
  may outweigh any potential benefits.

- **Limited parallelism**: Some algorithms have inherent limitations on the degree of parallelism that can be 
  achieved. In these cases, using a GPU may not lead to significant performance improvements.

- **Memory-bound problems**: GPUs generally have less memory available compared to CPUs, and their memory bandwidth 
  can be a limiting factor. If a problem requires a large amount of memory or involves memory-intensive operations, 
  it may not be well-suited for a GPU.

Examples of GPU acceleration
----------------------------

FIXME: show a few simple examples of CPU vs GPU versions of algorithms and roughly what speedup 
one can get 

VASP is a popular software package used for electronic structure calculations. The figure below show the speedup observed in a recent benchmark study on the Perlmutter and Cori supercomputers.

.. figure:: img/problems/vasp_gpu.png
   :align: center

   VASP GPU speedup for benchmark Si128 acfdtr. The horizontal axis shows the number of nodes, and the vertical axis shows the GPU speedup of VASP (Time(CPU)/Time(GPU)). (Recent unpublished benchmarks of VASP on NVIDIA A100 GPUs unpublished).


To give a flavor of what type of performance gains we can achieve by porting a calculations to a GPU 
(if we're lucky!), let's look at a few simple cases:



Computational Chemistry
^^^^^^^^^^^^^^^^^^^^^^^

A great deal of computational resources are spent in Quantum Chemical calculations which involve
the solution of the Hartree-Fock eigenvalue problem, which requires the diagonalization of the
Fock matrix whose elements are given by:
   
.. math::
    F_{\alpha \beta} = H^{\textrm{core}}_{\alpha \beta} + \sum_{\gamma \delta}D_{\gamma \delta} \left [ (\alpha \beta|\gamma \delta) - \frac{1}{2} (\alpha \delta|\gamma \beta) \right ],

The first term is related to the one electron contributions and the second term is related to the 
electron repulsion integrals (ERIs), in parenthesis, weighted by the by the density matrix 
:math:`D_{\gamma \delta}`. One of the most expensive parts in the solution of the Hartree-Fock equations is the 
processing (digestion) of the ERIs, one algorithm to do this task is as follows:

.. figure:: img/concepts/algorithms.svg
    :width: 200
    :align: center

    Algorithm for processing ERIs [`JCTC, 17, 7486, (2021) <https://doi.org/10.1021/acs.jctc.1c00720>`__]

This algorithm is suitable for GPUs as it involves many arithmetic operations. In addition to this,
there are symmetries and properties of the integrals that could be used to rearrange the loops in
an efficient manner that fit GPU architectures. 


Humanities
^^^^^^^^^^

**Language models and NLP (natural language processing)**

With the recent popularity of ChatGPT, the use of language models has come into the mainstream, 
however such models have been used in the humanities many years already. One of the biggest goals of humanities 
researchers is working with textual data which has increased exponentially over recent years due to the rise in 
social media. Analyzing such textual data to gain insights into questions of sociology, linguistics and various 
other fields have become increasingly reliant on using language models. Along with language models, 
the need for GPU access has become essential.


**Archeology**

The field of archeology also makes use of GPUs in their 3D modelling 
and rendering work. The biggest problem with archeological sites is that once they are excavated, 
they are destroyed, so any researchers who aren't present at the site, would lose valuable insights into how 
it looked when it was found. However, with recent developments in technology and accessibility to high-performance 
computing, they are able to generate extremely detailed renderings of the excavation sites which act as a way to 
preserve the site for future researchers to gain critical insights and contribute to the research. 

Exercises
---------

.. challenge:: Discussion

   - What type of problems have you used GPUs for?
   - How large was the performance boost?


.. challenge:: Good and bad use cases for GPU porting

   Which of the following computational tasks is likely to gain the least performance benefit from being ported to a GPU?

   1. Training a large, deep neural network.
   2. Performing a Monte Carlo simulation with a large number of independent trials.
   3. Executing an algorithm with heavy use of recursion and frequent branching.
   4. Processing a large image with a convolutional filter.

   .. solution::

      The right answer is option 3. GPUs do not handle recursion and branching as effectively as more 
      data-heavy algorithms.



.. keypoints::

   - GPUs excel in processing tasks with high data parallelism, such as large-scale matrix operations, Fourier transforms, and big data analytics. 
   - GPUs struggle with sequential tasks, problems with extensive control flow divergence, low arithmetic intensity tasks, small data sets, and memory-bound problems.
