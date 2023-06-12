.. _example-heat:

GPU programming example: stencil computation
============================================

.. questions::

   - How do I compile and run code developed using different programming models and frameworks?
   - What can I expect from the GPU-ported programs in terms of performance gains / trends and how do I estimate this?

.. objectives::

   - To show a self-contained example of parallel computation executed on CPU and GPU using different programming models
   - To show differences and consequences of implementing the same algorithm in natural "style" of different models/ frameworks
   - To discuss how to assess theoretical and practical performance scaling of GPU codes

.. instructor-note::

   - 45 min teaching
   - 45 min exercises

Problem: heat flow in two-dimensional area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heat flows in objects according to local temperature differences, as if seeking local equilibrium. The following example defines a rectangular area with two always-warm sides (temperature 70 and 85), two cold sides (temperature 20 and 5) and a cold disk at the center. Because of heat diffusion, temperature of neighboring patches of the area is bound to equalize, changing the overall distribution:

.. figure:: img/stencil/heat_montage.png
   :align: center
   
   Over time, the temperature distribution progresses from the initial state toward an end state where upper triangle is warm and lower is cold. The average temperature tends to (70 + 85 + 20 + 5) / 4 = 45.

Technique: stencil computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heat transfer in the system above is governed by the partial differential equation(s) describing local variation of the temperature field in time and space. That is, the rate of change of the temperature field :math:`u(x, y, t)` over two spatial dimensions :math:`x` and :math:`y` and time :math:`t` (with rate coefficient :math:`\alpha`) can be modelled via the equation

.. math::
   \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial x^2}\right)
   
The standard way to numerically solve differential equations is to *discretize* them, i. e. to consider only a set/ grid of specific area points at specific moments in time. That way, partial derivatives :math:`{\partial u}` are converted into differences between adjacent grid points :math:`u^{m}(i,j)`, with :math:`m, i, j` denoting time and spatial grid points, respectively. Temperature change in time at a certain point can now be computed from the values of neighboring points at earlier time; the same expression, called *stencil*, is applied to every point on the grid.

.. figure:: img/stencil/stencil.svg
   :align: center

   This simplified model uses an 8x8 grid of data in light blue in state
   :math:`m`, each location of which has to be updated based on the
   indicated 5-point stencil in yellow to move to the next time point
   :math:`m+1`.

.. challenge:: Discussion: stencil applications

   Stencil computation is a common occurrence in solving numerical problems. Have you already encountered it? Can you think of a problem that could be formulated this way in your field / area of expertise?
   
   .. solution::
      
      One obvious choice is *convolution* operation, used in image processing to apply various filter kernels; in some contexts, "convolution" and "stencil" are used almost interchangeably.

Technical considerations
------------------------

**1. How fast and/ or accurate can the solution be?**

Spatial resolution of the temperature field is controlled by the number/ density of the grid points. As the full grid update is required to proceed from one time point to the next, stencil computation is the main target of parallelization (on CPU or GPU).

Moreover, in many cases the chosen time step cannot be arbitrarily large, otherwise the numerical differentiation will fail, and dense/ accurate grids imply small time steps (see inset below), which makes efficient spatial update even more important.

.. solution:: Optional: stencil expression and time-step limit
   
   Differential equation shown above can be discretized using different schemes. For this example, temperature values at each grid point :math:`u^{m}(i,j)` are updated from one time point (:math:`m`) to the next (:math:`m+1`), using the following expressions:
      
   .. math::
       u^{m+1}(i,j) = u^m(i,j) + \Delta t \alpha \nabla^2 u^m(i,j) ,
   
   where
   
   .. math::
      \nabla^2 u  &= \frac{u(i-1,j)-2u(i,j)+u(i+1,j)}{(\Delta x)^2} \\
          &+ \frac{u(i,j-1)-2u(i,j)+u(i,j+1)}{(\Delta y)^2} ,
   
   and :math:`\Delta x`, :math:`\Delta y`, :math:`\Delta t` are step sizes in space and time, respectively.
   
   Time-update schemes also have a limit on the maximum allowed time step :math:`\Delta t`. For the current scheme, it is equal to
   
   .. math::
      \Delta t_{max} = \frac{(\Delta x)^2 (\Delta y)^2}{2 \alpha ((\Delta x)^2 + (\Delta y)^2)}

**2. What to do with area boundaries?**

Naturally, stencil expression can't be applied directly to the outermost grid points that have no outer neighbors. This can be solved by either changing the expression for those points or by adding an additional layer of grid that is used in computing update, but not updated itself -- points of fixed temperature for the sides are being used in this example.

**3. How could the algorithm be optimized further?**

In `an earlier episode <https://enccs.github.io/gpu-programming/9-non-portable-kernel-models/#memory-optimizations>`_, importance of efficient memory access was already stressed. In the following examples, each grid point (and its neighbors) is treated mostly independently; however, this also means that for 5-point stencil each value of the grid point may be read up to 5 times from memory (even if it's the fast GPU memory). By rearranging the order of mathematical operations, it may be possible to reuse these values in a more efficient way.

Another point to note is that even if the solution is propagated in small time steps, not every step might actually be needed for output. Once some *local* region of the field is updated, mathematically nothing prevents it from being updated for the second time step -- even if the rest of the field is still being recalculated -- as long as :math:`t = m-1` values for the region boundary are there when needed. (Of course, this is more complicated to implement and would only give benefits in certain cases.)

.. challenge:: Poll: which programming model/ framework are you most interested in today?

   - OpenMP offloading (C++)
   - SYCL
   - *Python* (``numba``/CUDA)
   - Julia

The following table will aid you in navigating the rest of this section:

.. admonition:: Episode guide

   - `Sequential and OpenMP-threaded code <https://enccs.github.io/gpu-programming/13-examples/#sequential-and-thread-parallel-program-in-c>`__ in C++, including compilation/ running instructions
   - `Naive GPU parallelization <https://enccs.github.io/gpu-programming/13-examples/#gpu-parallelization-first-steps>`__, including SYCL compilation instructions
   - `GPU code with device data management <https://enccs.github.io/gpu-programming/13-examples/#gpu-parallelization-data-movement>`__ (OpenMP, SYCL)
   - `Python implementation <https://enccs.github.io/gpu-programming/13-examples/#python-jit-and-gpu-acceleration>`__, including running instructions on `Google Colab <https://colab.research.google.com/>`__
   - `Julia implementation <https://enccs.github.io/gpu-programming/13-examples/#julia-gpu-acceleration>`__, including running instructions

Sequential and thread-parallel program in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. callout:: Trying out code examples

   Source files of the examples presented for the rest of this episode are available in the `content/examples/stencil/ <https://github.com/ENCCS/gpu-programming/tree/main/content/examples/stencil/>`_ directory.
   To download them to your home directory on the cluster, you can use Git:
   
   .. code-block:: console

      $ git clone https://github.com/ENCCS/gpu-programming.git
      $ cd gpu-programming/content/examples/stencil/
      $ ls

   .. warning::

      Don't forget to ``git pull`` for the latest updates if you already have the content from the first day of the workshop!

If we assume the grid point values to be truly independent *for a single time step*, stencil application procedure may be straighforwardly written as a loop over the grid points, as shown below in tab "Stencil update". (General structure of the program and the default parameter values for the problem model are also provided for reference.) CPU-thread parallelism can then be enabled by a single OpenMP ``#pragma``:

.. tabs::

   .. tab:: Stencil update

         .. literalinclude:: examples/stencil/base/core.cpp 
                        :language: cpp
                        :emphasize-lines: 25

   .. tab:: Main function

         .. literalinclude:: examples/stencil/base/main.cpp 
                        :language: cpp
                        :emphasize-lines: 37
 
   .. tab:: Default params

         .. literalinclude:: examples/stencil/base/heat.h 
                        :language: cpp
                        :lines: 7-34

.. callout::

      If you will be using the Git-stored versions of the executables, you should also make them... well, executable:
      
      .. code-block:: console
      
         $ cd lumi
         $ chmod 770 stencil*

.. solution:: Optional: compiling the executables and running OpenMP-CPU tests

   Executable files for the OpenMP-enabled variants are provided together with the source code. However, if you'd like to compile them yourself, follow the instructions below:
   
   .. code-block:: console

      module load LUMI/22.08
      module load partition/G
      module load rocm/5.3.3
      
      cd base/
      make all
   
   Afterwards login into an interactive node and test the executables:
   
   .. code-block:: console

      $ srun --account=project_465000485 --partition=standard-g --nodes=1 --cpus-per-task=1 --ntasks-per-node=1 --gpus-per-node=1 --time=1:00:00 --pty bash
      
      $ ./stencil
      $ ./stencil_off
      $ ./stencil_data
      
      $ exit
      
   If everything works well, the output should look similar to this:
   
   .. code-block:: console

      $ ./stencil
      Average temperature, start: 59.763305
      Average temperature at end: 59.281239
      Control temperature at end: 59.281239
      Iterations took 1.395 seconds.
      $ ./stencil_off
      Average temperature, start: 59.763305
      Average temperature at end: 59.281239
      Control temperature at end: 59.281239
      Iterations took 4.269 seconds.
      $ ./stencil_data   
      Average temperature, start: 59.763305
      Average temperature at end: 59.281239
      Control temperature at end: 59.281239
      Iterations took 1.197 seconds.
      $ 

   Changing number of default OpenMP threads is somewhat tricky to do interactively, so OpenMP-CPU "scaling" tests are done via provided batch script (make sure (f. e.f, using ``squeue --me``) that there is no currently running interactive allocation):
   
   .. code-block:: console

      $ sbatch test-omp.slurm
      (to see the job status, enter command below)
      $ squeue --me
      (job should finish in a couple of minutes; let's also minimize extraneous output)
      $ more job.o<job ID> | grep Iterations
    
   The expected output is:
   
   .. code-block:: console
   
      Iterations took 1.390 seconds.
      Iterations took 13.900 seconds.
      Iterations took 0.194 seconds.
      Iterations took 1.728 seconds.
      Iterations took 0.069 seconds.
      Iterations took 0.547 seconds.
      (... 18 lines in total ...)

CPU parallelization: timings
----------------------------

For later comparison, some benchmarks of the thread-parallel executable are provided below:

.. list-table:: Run times of OpenMP-enabled executable, s
   :widths: 25 25 25
   :header-rows: 1
   
   * - Job size
     - 1 CPU core
     - 32 CPU cores
   * - S:2000 T:500
     - 1.390
     - 0.061
   * - S:2000 T:5000
     - 13.900
     - 0.550
   * - S:20000 T:50
     - 15.200
     - 12.340

A closer look reveals that the computation time scales very nicely with increasing time steps:

.. figure:: img/stencil/heat-omp-T.png
   :align: center
   
However, for larger grid sizes the parallelization becomes inefficient -- as the individual chunks of the grid get too large to fit into CPU cache, threads become bound by the speed of RAM reads/writes:

.. figure:: img/stencil/heat-omp-S.png
   :align: center

.. challenge:: Discussion: heat flow computation scaling

   1. How is heat flow computation expected to scale with respect to the number of time steps?
   
      a. Linearly
      b. Quadratically
      c. Exponentially
   
   2. How is stencil application (grid update) expected to scale with respect to the size of the grid side?
   
      a. Linearly
      b. Quadratically
      c. Exponentially
   
   3. (Optional) Do you expect GPU-accelerated computations to suffer from the memory effects observed above? Why/ why not?
   
   .. solution::
   
      1. The answer is a: since each time-step update is sequential and involves a similar number of operations, then the update time will be more or less constant.
      2. The answer is b: since stencil application is independent for every grid point, the update time will be proportional to the number of points i.e. side * side.
      3. GPU computations are indeed sensitive to memory access patterns and tend to resort to (GPU) memory quickly. However, the effect above arises because multiple active CPU threads start competing for access to RAM. In contrast, "over-subscribing" the GPU with large amount of threads executing the same kernel (stencil update on a grid point) tends to hide memory access latencies; increasing grid size might actually help to achieve this.


GPU parallelization: first steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's apply several techniques presented in previous episodes to make stencil update GPU-parallel.

OpenMP (or OpenACC) offloading requires to define a region to be executed in parallel as well as data that shall be copied over/ used in GPU memory. Similarly, SYCL programming model offers convenient ways to define execution kernels, context to run them in (called queue) and simplified CPU-GPU transfer of needed data.

Changes of stencil update code for OpenMP and SYCL are shown in the tabs below:

.. tabs::

   .. tab:: OpenMP (naive)

         .. literalinclude:: examples/stencil/base/core-off.cpp 
                        :language: cpp
                        :emphasize-lines: 25-26
         
   .. tab:: SYCL (naive)

         .. literalinclude:: examples/stencil/sycl/core-naive.cpp 
                        :language: cpp
                        :emphasize-lines: 31,35

.. callout:: Loading modules on LUMI
   
   As SYCL is placed on top of ROCm/HIP (or CUDA) software stack, even running SYCL executables may require respective modules to be loaded. On current nodes, it can be done as follows:
   
   .. code-block:: console
   
      salloc -A project_465000485 -N 1 -t 1:00:0 -p standard-g --gpus-per-node=1
      
      module load LUMI/22.08
      module load partition/G
      module load rocm/5.3.3
      module use /project/project_465000485/Easy_Build_Installations/modules/LUMI/22.08/partition/G/
      module load hipSYCL

.. solution:: Optional: compiling the SYCL executables

   As previously, you are welcome to generate your own executables. This time we will be using the interactive allocation:
   
   .. code-block:: console

      $ cd ../sycl/
      (give the following lines some time, probably a couple of min)
      $ syclcc -O2 -o stencil_naive core-naive.cpp io.cpp main-naive.cpp pngwriter.c setup.cpp utilities.cpp
      $ syclcc -O2 -o stencil core.cpp io.cpp main.cpp pngwriter.c setup.cpp utilities.cpp
      
      $ srun stencil_naive
      $ srun stencil

.. challenge:: Exercise: naive GPU ports

   In the interactive allocation, run (using ``srun``) provided or compiled executables ``base/stencil``, ``base/stencil_off`` and ``sycl/stencil_naive``. Try changing problem size parameters:
   
   - ``srun stencil_naive 2000 2000 5000``
   
   To look for:
   
   - How computation times change? 
   - Do the results align to your expectations?
   
   .. solution::
   
      If you ran the program (or looked up output of earlier sections), you might already know that the GPU-"ported" versions actually run slower than the single-CPU-core version! In fact, the scaling behavior of all three variants is similar and expected, which is a good sign; only the "computation unit cost" is different. You can compare benchmark summaries in the tabs below:

      .. tabs::

         .. tab:: Sequential

            .. figure:: img/stencil/heat-seq.png
               :align: center

         .. tab:: OpenMP (naive)

            .. figure:: img/stencil/heat-off.png
               :align: center

         .. tab:: SYCL (naive)

            .. figure:: img/stencil/heat-sycl0.png
               :align: center

GPU parallelization: data movement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Why the porting approach above seems to be grossly inefficient?

On each step, we:

- re-allocate GPU memory, 
- copy the data from CPU to GPU, 
- perform the computation, 
- then copy the data back.

But overhead can be reduced by taking care to minimize data transfers between *host* and *device* memory:

- allocate GPU memory once at the start of the program,
- only copy the data from GPU to CPU when we need it,
- swap the GPU buffers between timesteps, like we do with CPU buffers. (OpenMP does this automatically.)

Changes of stencil update code as well as the main program are shown in tabs below. 

.. tabs::

   .. tab:: OpenMP

         .. literalinclude:: examples/stencil/base/core-data.cpp
                        :language: cpp
                        :emphasize-lines: 25,40-75
   
   .. tab:: SYCL

         .. literalinclude:: examples/stencil/sycl/core.cpp
                        :language: cpp
                        :emphasize-lines: 13-14,27-28,41-55

   .. tab:: Python

         .. literalinclude:: examples/stencil/python/core_cuda.py
                        :language: py
                        :lines: 6-34
                        :emphasize-lines: 14-16,18

   .. tab:: main() (SYCL)

         .. literalinclude:: examples/stencil/sycl/main.cpp 
                        :language: cpp
                        :emphasize-lines: 38-39,44-45,51,56,59,75

.. challenge:: Exercise: updated GPU ports

   In the interactive allocation, run (using ``srun``) provided or compiled executables ``base/stencil_data`` and ``sycl/stencil``. Try changing problem size parameters:
   
   - ``srun stencil 2000 2000 5000``
   
   To look for:
      
   - How computation times change this time around?
   - What largest grid and/or longest propagation time can you get in 10 s on your machine?
   
   .. solution::
   
      .. tabs::
      
         .. tab:: OpenMP data mapping
         
            Using GPU offloading with mapped device data, it is possible to achieve performance gains compared to thread-parallel version for larger grid sizes, due to the fact that the latter version becomes essentially RAM-bound, but the former does not.
            
            .. figure:: img/stencil/heat-map.png
               :align: center
               
         .. tab:: SYCL device buffers
         
            Because of the more explicit programming approach, SYCL GPU port is still 10 times faster than OpenMP offloaded version, comparable with thread-parallel CPU version running on all cores of a single node. Moreover, the performance scales perfectly with respect to both grid size and number of time steps (grid updates) computed.
            
            .. figure:: img/stencil/heat-sycl2.png
               :align: center            

Python: JIT and GPU acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned `previously <https://enccs.github.io/gpu-programming/6-language-support/#numba>`_, Numba package allows developers to just-in-time (JIT) compile Python code to run fast on CPUs, but can also be used for JIT compiling for GPUs (although AMD GPU support is at the moment deprecated for Numba versions > 0.53). JIT seems to work well on loop-based, computationally heavy functions, so trying it out is a nice choice for initial source version:

.. tabs::

   .. tab:: Stencil update

         .. literalinclude:: examples/stencil/python/core.py
                        :language: py
                        :lines: 6-29
                        :emphasize-lines: 17
   
   .. tab:: Data generation

         .. literalinclude:: examples/stencil/python/heat.py
                        :language: py
                        :lines: 57-78
                        :emphasize-lines: 1

The alternative approach would be to rewrite stencil update code in NumPy style, exploiting loop vectorization.

.. callout:: Trying out Python examples

   You can run provided code examples on Google Colab using instructions provided in the `Setup <https://enccs.github.io/gpu-programming/0-setup/#running-on-google-colab>`_, your local machine, or LUMI node (non-GPU variants). On LUMI, you can set up Python distribution as following:
   
   .. code-block:: console

      $ module load cray-python/3.9.13.1
      (install needed dependencies locally)
      $ pip3 install --user numba matplotlib
      $ cd ../python/
      (make sure you have active allocation)
      $ srun python3 main.py

Short summary of a typical Colab run is provided below:

.. list-table:: Run times of Numba JIT-enabled Python program, s
   :widths: 25 25 25 25 25
   :header-rows: 1
   
   * - Job size
     - JIT (LUMI)
     - JIT (Colab)
     - Job size
     - no JIT (Colab)
   * - S:2000 T:500
     - 1.648
     - 8.495
     - S:200 T:50
     - 5.318
   * - S:2000 T:200
     - 0.787
     - 3.524
     - S:200 T:20
     - 1.859
   * - S:1000 T:500
     - 0.547
     - 2.230
     - S:100 T:50
     - 1.156

Numba's ``@vectorize`` and ``@guvectorize`` decorators offer an interface to create CPU- (or GPU-) accelerated *Python* functions without explicit implementation details. However, such functions become increasingly complicated to write (and optimize by the compiler) with increasing complexity of the computations within.

However, for NVIDIA GPUs, Numba also offers direct CUDA-based kernel programming, which can be the best choice for those already familiar with CUDA. Example for stencil update written in Numba CUDA is shown in the `data movement section <https://enccs.github.io/gpu-programming/13-examples/#gpu-parallelization-data-movement>`_, tab "Python". In this case, data transfer functions ``devdata = cuda.to_device(data)`` and ``devdata.copy_to_host(data)`` (see ``main_cuda.py``) are already provided by Numba package.

.. challenge:: Exercise: CUDA acceleration in Python

   Using Google Colab (or your own machine), run provided Numba-CUDA Python program. Try changing problem size parameters:
   
   - ``args.rows, args.cols, args.nsteps = 2000, 2000, 5000`` for notebooks,
   - [``srun``] ``python3 main.py 2000 2000 5000`` for command line.
   
   To look for:
      
   - How computation times change?
   - Do you get better performance than from JIT-compiled CPU version? How far can you push the problem size?
   - Are you able to monitor the GPU usage?
   
   .. solution::
   
      Some numbers from Colab:
      
      .. list-table:: Run times of Numba CUDA Python program, s
         :widths: 25 25 25 25
         :header-rows: 1

         * - Job size
           - JIT (LUMI)
           - JIT (Colab)
           - CUDA (Colab)
         * - S:2000 T:500
           - 1.648
           - 8.495
           - 1.079
         * - S:2000 T:2000
           - 6.133
           - 36.61
           - 3.931
         * - S:5000 T:500
           - 9.478
           - 57.19
           - 6.448


Julia GPU acceleration
~~~~~~~~~~~~~~~~~~~~~~

A Julia version of the stencil example above can be found below (a simplified version of the HeatEquation module at https://github.com/ENCCS/HeatEquation.jl). 
The source files are also available in the `content/examples/stencil/julia <https://github.com/ENCCS/gpu-programming/tree/main/content/examples/stencil/julia>`_ directory of this repository.

To run the example on LUMI CPU partition, type:

.. code-block:: console

   $ # interactive CPU node
   $ srun --account=project_465000485 --partition=standard --nodes=1 --cpus-per-task=32 --ntasks-per-node=1 --time=01:00:00 --pty bash
   $ # load Julia env
   $ module purge
   $ module use /appl/local/csc/modulefiles
   $ module load julia/1.9.0
   $ # in directory with Project.toml and source files, instantiate an environment to install packages
   $ julia --project -e "using Pkg ; Pkg.instantiate()"
   $ # finally run
   $ julia --project main.jl

To run on the GPU partition, use instead the ``srun`` command 

.. code-block:: console

   $ srun --account=project_465000485 --partition=standard-g --nodes=1 --cpus-per-task=1 --ntasks-per-node=1 --gpus-per-node=1 --time=1:00:00 --pty bash


.. callout:: Optional dependency

   Note that the ``Plots.jl`` dependency is commented out in ``main.jl`` and ``Project.toml``. This saves ~2 minute precompilation time when you first instantiate the Julia environment. To generate plots, just uncomment the commented ``Plots.jl`` dependency in ``Project.toml``, instantiate again, and import and use ``Plots`` in ``main.jl``.

.. tabs::

   .. tab:: main.jl

      .. literalinclude:: examples/stencil/julia/main.jl
         :language: julia

   .. tab:: core.jl

      .. literalinclude:: examples/stencil/julia/core.jl
         :language: julia

   .. tab:: heat.jl

      .. literalinclude:: examples/stencil/julia/heat.jl
         :language: julia

   .. tab:: Project.toml

      .. literalinclude:: examples/stencil/julia/Project.toml
         :language: julia


.. challenge:: Exercise: Julia port to GPUs

   Carefully inspect all Julia source files and consider the following questions:

   1. Which functions should be ported to run on GPU?
   2. Look at the :meth:`initialize!` function and how it uses the ``arraytype`` argument. This could be done more compactly and elegantly, but this solution solves scalar indexing errors. What are scalar indexing errors?
   3. Try to start sketching GPU-ported versions of the key functions.
   4. When you have a version running on a GPU (your own or the solution provided below), try benchmarking it by adding ``@btime`` in front of :meth:`simulate!` in ``main.jl``. Benchmark also the CPU version, and compare.

   .. solution:: Hints

      - create a new function :meth:`evolve_gpu!` which contains the GPU kernelized version of :meth:`evolve!`
      - in the loop over timesteps in :meth:`simulate!`, you will need a conditional like ``if typeof(curr.data) <: ROCArray`` to call your GPU-ported function
      - you cannot pass the struct ``Field`` to the kernel. You will instead need to directly pass the array ``Field.data``. This also necessitates passing in other variables like ``curr.dx^2``, etc.


   .. solution:: More hints

      - since the data is two-dimensional, you'll need ``i = (blockIdx().x - 1) * blockDim().x + threadIdx().x`` and ``j = (blockIdx().y - 1) * blockDim().y + threadIdx().y``
      - to not overindex the 2D array, you can use a conditional like ``if i > 1 && j > 1 && i < nx+2 && j < ny+2``
      - when calling the kernel, you can set the number of threads and blocks like ``xthreads = ythreads = 16`` and ``xblocks, yblocks = cld(curr.nx, xthreads), cld(curr.ny, ythreads)``, and then call it with, e.g., ``@roc threads=(xthreads, ythreads) blocks = (xblocks, yblocks) evolve_rocm!(curr.data, prev.data, curr.dx^2, curr.dy^2, nx, ny, a, dt)``.




   .. solution:: 

      1. The :meth:`evolve!` and :meth:`simulate!` functions need to be ported. The ``main.jl`` file also needs to be updated to work with GPU arrays.
      2. "Scalar indexing" is where you iterate over a GPU array, which would be excruciatingly slow and is indeed only allowed in interactive REPL sessions. Without the if-statements in the :meth:`initialize!` function, the :meth:`generate_field!` method would be doing disallowed scalar indexing if you were running on a GPU.
      3. The GPU-ported version is found below. Try it out on both CPU and GPU and observe the speedup. Play around with array size to see if the speedup is affected. You can also play around with the ``xthreads`` and ``ythreads`` variables to see if it changes anything.

      .. tabs::

         .. tab:: main_gpu.jl

            .. literalinclude:: examples/stencil/julia/main_gpu.jl
               :language: julia

         .. tab:: core_gpu.jl

            .. literalinclude:: examples/stencil/julia/core_gpu.jl
               :language: julia


See also
~~~~~~~~

This section leans heavily on source code and material created for several other computing workshops 
by `ENCCS <https://enccs.se/>`_ and `CSC <https://csc.fi/>`_ and adapted for the purposes of this lesson.
If you want to know more about specific programming models / framework, definitely check these out!

- `OpenMP for GPU offloading <https://enccs.github.io/openmp-gpu/>`_
- `Heterogeneous programming with SYCL <https://enccs.github.io/sycl-workshop/>`_
- `Educational implementation of heat flow example (incl. MPI-aware CUDA) <https://github.com/cschpc/heat-equation/>`_

