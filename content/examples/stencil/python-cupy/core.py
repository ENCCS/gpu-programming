# (c) 2023 ENCCS, CSC and the contributors
from heat import *

# core_cupy.py

import cupy as cp
import math

# CuPy RawKernel for the stencil update
_evolve_kernel_code = """
extern "C" __global__
void evolve_kernel(
    double* curr, const double* prev, float a, float dt, float dx2, float dy2, int nx, int ny
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        int idx = i * ny + j;
        int idx_ip = (i + 1) * ny + j;
        int idx_im = (i - 1) * ny + j;
        int idx_jp = i * ny + (j + 1);
        int idx_jm = i * ny + (j - 1);
        
        curr[idx] = prev[idx] + a * dt * (
            (prev[idx_ip] - 2.0f * prev[idx] + prev[idx_im]) / dx2 +
            (prev[idx_jp] - 2.0f * prev[idx] + prev[idx_jm]) / dy2
        );
    }
}
"""

_evolve_kernel = cp.RawKernel(_evolve_kernel_code, 'evolve_kernel')

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
    nx, ny = prev.shape  # These are the FULL dims, rows+2 / cols+2
    tx, ty = (16, 16)    # Arbitrary choice
    bx, by = math.ceil(nx / tx), math.ceil(ny / ty)
    
    # Run CuPy compiled kernel
    _evolve_kernel((bx, by), (tx, ty), (curr, prev, cp.float32(a), cp.float32(dt), cp.float32(dx2), cp.float32(dy2), cp.int32(nx), cp.int32(ny)))
