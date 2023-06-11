# (c) 2023 ENCCS, CSC and the contributors

# core.py

import math
from numba import cuda

# Update the temperature values using five-point stencil
# Arguments:
#   curr: current temperature field object
#   prev: temperature field from previous time step
#   a: diffusivity
#   dt: time step
def evolve(current, previous, a, dt):
    dx2, dy2 = previous.dx**2, previous.dy**2
    curr, prev = current.dev, previous.dev
    # Set thread and block sizes
    nx, ny = prev.shape # These are the FULL dims, rows+2 / cols+2
    tx, ty = (16, 16)   # Arbitrary choice
    bx, by = math.ceil(nx / tx), math.ceil(ny / ty)
    # Run numba (CUDA) kernel
    _evolve_kernel[(bx, by), (tx, ty)](curr, prev, a, dt, dx2, dy2)


@cuda.jit()
def _evolve_kernel(curr, prev, a, dt, dx2, dy2):
    nx, ny = prev.shape # These are the FULL dims, rows+2 / cols+2
    i, j = cuda.grid(2)
    if ((i >= 1) and (i < nx-1) 
        and (j >= 1) and (j < ny-1)):
        curr[i, j] = prev[i, j] + a * dt * ( \
            (prev[i+1, j] - 2*prev[i, j] + prev[i-1, j]) / dx2 + \
            (prev[i, j+1] - 2*prev[i, j] + prev[i, j-1]) / dy2 )

