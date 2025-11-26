# (c) 2023 ENCCS, CSC and the contributors
from heat import *

# core.py

import jax
import jax.numpy as jnp
from functools import partial


# Update the temperature values using five-point stencil
# Arguments:
#   curr: current temperature field object
#   prev: temperature field from previous time step
#   a: diffusivity
#   dt: time step
def evolve(current, previous, a, dt):
    dx2, dy2 = previous.dx**2, previous.dy**2
    curr, prev = current. data, previous.data
    # Run JAX-accelerated update
    result = _evolve(curr, prev, a, dt, dx2, dy2)
    current.data = np.array(result)


@partial(jax.jit, static_argnums=())
def _evolve(curr, prev, a, dt, dx2, dy2):
    curr = jnp.asarray(curr, dtype=float)
    prev = jnp.asarray(prev, dtype=float)
    nx, ny = prev.shape # These are the FULL dims, rows+2 / cols+2
    
    # Create interior slice
    i_indices = jnp.arange(1, nx-1)
    j_indices = jnp.arange(1, ny-1)
    
    # Vectorized stencil operation
    def update_interior():
        laplacian_x = (prev[2:nx, 1:ny-1] - 2*prev[1:nx-1, 1:ny-1] + prev[0:nx-2, 1:ny-1]) / dx2
        laplacian_y = (prev[1:nx-1, 2:ny] - 2*prev[1:nx-1, 1:ny-1] + prev[1:nx-1, 0:ny-2]) / dy2
        interior_update = prev[1:nx-1, 1:ny-1] + a * dt * (laplacian_x + laplacian_y)
        return interior_update
    
    interior = update_interior()
    
    # Reconstruct full array with boundaries
    curr_new = curr.at[1:nx-1, 1:ny-1].set(interior)
    return curr_new