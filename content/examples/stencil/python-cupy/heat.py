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


class Field:
    def __init__(self, rows, cols):
        self.data = np.zeros((rows+2, cols+2), dtype=float)
        self.dev = None  # Device array
        self.nx, self.ny = rows, cols
        self.dx, self.dy = DX, DY


# setup.py

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
        data[i, ny+1] = T_RIGHT

    for j in range(ny+2):
        data[0,j] = T_UPPER
        data[nx+1, j] = T_LOWER

