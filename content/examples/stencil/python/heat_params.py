# CONSTANTS

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


import numpy as np

class Field:
    def __init__(self, rows, cols):
        self.data = np.zeros((rows+2, cols+2), dtype=float)
        self.nx, self.ny = rows, cols
        self.dx, self.dy = DX, DY

