import math
import numba

@numba.vectorize([numba.float64(numba.float64, numba.float64)], target='cuda')
def f_numba_gpu(x,y):
    return math.pow(x,3.0) + 4*math.sin(y)
