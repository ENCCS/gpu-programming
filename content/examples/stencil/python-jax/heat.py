# (c) 2023 ENCCS, CSC and the contributors

# heat.py

# Fixed grid spacing
DX = 0.01
DY = 0.01
# Default temperatures
T_DISC = 5.0
T_AREA = 65.0
T_UPPER = 85.0
T_LOWER = 5.0
T_LEFT = 20.0
T_RIGHT = 70.0
# Default problem size
ROWS = 2000
COLS = 2000
NSTEPS = 500


import copy
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

class Field:
    def __init__(self, rows, cols):
        self.data = jnp.zeros((rows+2, cols+2), dtype=float)
        self.nx, self.ny = rows, cols
        self.dx, self.dy = DX, DY


# setup. py

def initialize(args):
    rows, cols, nsteps = args.rows, args.cols, args.nsteps
    current, previous = field_create(rows, cols)
    return current, previous, nsteps


def field_create (rows, cols):
    heat1 = field_generate(rows, cols)
    heat2 = copy.deepcopy(heat1)
    return heat1, heat2


def field_generate(rows, cols):
    heat = Field(rows, cols)
    data, nx, ny = heat.data, heat.nx, heat.ny
    heat.data = _generate(data, nx, ny)
    return heat


def field_average(heat):
    return float(jnp.mean(heat.data[1:-1, 1:-1]))


@jax.jit
def _generate(data, nx, ny):
    # Radius of the source disc
    radius = nx / 6.0
    
    # Create index arrays
    i_indices = jnp.arange(nx+2)
    j_indices = jnp.arange(ny+2)
    i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing='ij')
    
    # Distance from origin
    dx = i_grid - nx / 2 + 1
    dy = j_grid - ny / 2 + 1
    distance_sq = dx * dx + dy * dy
    
    # Initialize with T_AREA, then set disc region
    result = jnp.full((nx+2, ny+2), T_AREA, dtype=float)
    result = jnp.where(distance_sq < radius * radius, T_DISC, result)
    
    # Boundary conditions
    # Left and right boundaries
    result = result.at[:, 0].set(T_LEFT)
    result = result. at[:, ny+1].set(T_RIGHT)
    
    # Top and bottom boundaries
    result = result. at[0, :].set(T_UPPER)
    result = result.at[nx+1, :].set(T_LOWER)
    
    return result
