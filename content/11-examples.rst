.. _example-heat:

Problem example: stencil computation
====================================

.. questions::

   - q1
   - q2

.. objectives::

   - To show a self-contained example of parallel computation executed on CPU (via OpenMP) and GPU (different models)
   - o2

.. instructor-note::

   - X min teaching
   - X min exercises

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

Stencil computation is a common occurrence in solving numerical equations, image processing (for 2D convolution) and other areas.

.. solution:: Stencil expression and time-step limit
   
   WRITEME


Technical considerations
------------------------

**1. How fast and/ or accurate can the solution be?**

Spatial resolution of the temperature field is controlled by the number/ density of the grid points. As the full grid update is required to proceed from one time point to the next, stencil computation is the main target of parallelization (on CPU or GPU).

Moreover, in many cases the chosen time step cannot be arbitrarily large, otherwise the numerical differentiation will fail, and dense/ accurate grids imply small time steps (see inset above), which makes efficient spatial update even more important.

**2. What to do with area boundaries?**

Naturally, stencil expression can't be applied directly to the outermost grid points that have no outer neighbors. This can be solved by either changing the expression for those points or by adding an additional layer of grid that is used in computing update, but not updated itself -- points of fixed temperature for the sides are being used in this example.

**3.**



CPU parallelization (with OpenMP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WRITEME



GPU parallelization examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~




.. keypoints::

   - k1
   - k2
