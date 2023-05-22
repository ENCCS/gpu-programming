from heat_params import *

# setup.py

import copy
from numba import jit


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
    _generate(data, nx, ny)
    return heat


def field_average(heat):
    return np.mean(heat.data[1:-1, 1:-1])


# core.py

def evolve(current, previous, a, dt):
    dx2, dy2 = previous.dx**2, previous.dy**2
    curr, prev = current.data, previous.data
    _evolve(curr, prev, a, dt, dx2, dy2)


@jit(nopython=True)
def _evolve(curr, prev, a, dt, dx2, dy2):
    ### Loops
    nx, ny = prev.shape # These are the FULL dims, rows+2 / cols+2
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            curr[i, j] = prev[i, j] + a * dt * ( \
              (prev[i+1, j] - 2*prev[i, j] + prev[i-1, j]) / dx2 + \
              (prev[i, j+1] - 2*prev[i, j] + prev[i, j-1]) / dy2 )
    
    ### numpy slices
    # curr[1:-1, 1:-1] = prev[1:-1, 1:-1] + a * dt * ( \
    #             (prev[2:, 1:-1] - 2*prev[1:-1, 1:-1] + prev[:-2, 1:-1]) / dx2 + \
    #             (prev[1:-1, 2:] - 2*prev[1:-1, 1:-1] + prev[1:-1, :-2]) / dy2 )

    # Comparison (2000x2000x50):
    #   Loops, no jit   --  376 s
    #   Loops, jit      --  0.4 s
    #   Slices, no jit  --  6.3 s
    #   Slices, jit     --  2.6 s


@jit(nopython=True)
def _generate(data, nx, ny):
    # Radius of the source disc
    radius = nx / 6.0
    for i in range(nx+2):
        for j in range(ny+2):
            # Distance of point i, j from the origin
            dx = i - nx / 2 + 1
            dy = j - ny / 2 + 1
            if (dx * dx + dy * dy < radius * radius):
                data[i,j] = T_DISC
            else:
                data[i,j] = T_AREA

    # Boundary conditions
    for i in range(nx+2):
        data[i,0] = T_LEFT
        data[i,ny + 1] = T_RIGHT

    for j in range(ny+2):
        data[0,j] = T_UPPER
        data[(nx + 1),j] = T_LOWER

