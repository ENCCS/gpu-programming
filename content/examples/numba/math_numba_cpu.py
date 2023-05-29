import math
import numba

@numba.vectorize([numba.float64(numba.float64, numba.float64)], target='cpu')
def f_numba_cpu(x,y):
    return math.pow(x,3.0) + 4*math.sin(y)
