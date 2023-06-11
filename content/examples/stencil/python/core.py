# (c) 2023 ENCCS, CSC and the contributors
from heat import *

# core.py

from numba import jit


# Update the temperature values using five-point stencil
# Arguments:
#   curr: current temperature field object
#   prev: temperature field from previous time step
#   a: diffusivity
#   dt: time step
def evolve(current, previous, a, dt):
    dx2, dy2 = previous.dx**2, previous.dy**2
    curr, prev = current.data, previous.data
    # Run (possibly accelerated) update
    _evolve(curr, prev, a, dt, dx2, dy2)


@jit(nopython=True)
def _evolve(curr, prev, a, dt, dx2, dy2):
    nx, ny = prev.shape # These are the FULL dims, rows+2 / cols+2
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            curr[i, j] = prev[i, j] + a * dt * ( \
              (prev[i+1, j] - 2*prev[i, j] + prev[i-1, j]) / dx2 + \
              (prev[i, j+1] - 2*prev[i, j] + prev[i, j-1]) / dy2 )

