---
title: High level language support
subtitle: '<img src="https://media.enccs.se/2022/12/ENCCS-Logo-Horizontal-Colour-low-res-323.png" width="50%" align="center"/>'
author: Ashwin Mohanan
institute: ENCCS/RISE
date: 2025-11-20
theme: league
transition: fade
highlightTheme: monokai  # Does not work with pandoc
---

# Outline

Introduce different libraries, show similarities and differences in capabilities

- GPU programming with Python
	
	Numba, Cupy, Jax, PyCUDA/PyOpenCL, CUDA Python / HIP Python
	
- GPU programming with Julia
	
	CUDA.jl / AMDGPU.jl / oneAPI.jl / Metal.jl and KernelAbstractions


---

## What is high-level programming?

> Primer on Python and Numpy

Either use simple data structures in Python's standard library and for-loops

```python
import random
import math
# Element-wise sum of two "arrays" represented as lists
a = [random.random() for _ in range(1000)]
b = [random.random() for _ in range(1000)]

c = []
for a_i, b_i in zip(a, b):
    c.append(a_i + math.sin(b_i))
```
---

## What is high-level programming?

> Primer on Python and Numpy

... or you use **efficient data structures** and **vectorized operations** 

with expressive syntax!

```python
import numpy as np
a = np.random.random(size=1000)
b = np.random.random(size=1000)
c = a + np.sin(b)
```



---

## Numba

<img src="https://numba.pydata.org/_static/numba-blue-horizontal-rgb.svg" width="50%"/>


- JIT compiler which supports a subset of Python and Numpy
- **CPU** execution 

	via LLVM using `llvmlite` and a C compiler 
- **GPU** execution 
	
	via CUDA using `nvcc` and `nvrtc` / `cudatoolkit`

---

## Numba

- Experimental AMD GPU support through `numba-hip` extension

```python
from numba import hip
# Use `@hip.jit` decorator, or
# Call `hip.pose_as_cuda()` and use `@cuda.jit`
```


---

## Numba: ufuncs (element-wise vectorized)

```python
import math
import numpy as np
import numba

@numba.vectorize(
	[numba.float64(numba.float64, numba.float64)], 
	target="cuda"
)
def a_plus_sin_b(a, b):
    return a + math.sin(b)

N = 10_000_000
vec = np.random.rand(N)
a_plus_sin_b(vec, vec)
```

---

## Numba: low-level

Control execution using `numba.cuda.threadIdx`, `blockDim`, `blockIdx`, `gridDim` etc.

```python
import math
import numpy as np
import numba

@numba.cuda.jit
def a_plus_sin_b(a, b, c):
    """GPU vectorized addition. Computes C = A + B"""
    # like threadIdx.x + (blockIdx.x * blockDim.x)
    thread_id = numba.cuda.grid(ndim=1)
    size = len(c)

    if thread_id < size:
        c[thread_id] = a[thread_id] + math.sin(b[thread_id])

N = 10_000_000
a = numba.cuda.to_device(np.random.random(N))
b = numba.cuda.to_device(np.random.random(N))
c = numba.cuda.device_array_like(a)

grid_size = len(a)
a_plus_sin_b.forall(grid_size)(a, b, c)
c.copy_to_host()
```

---

## Jax

<img src="https://docs.jax.dev/en/latest/_static/jax_logo_250px.png" width="20%">


- Support a subset of Python and Numpy
- Drop-in replacement for Numpy 
 
  `import jax.numpy as jnp`

 - Interoperable, uses just-in-time XLA compiler to target CPU, GPU (CUDA and ROCm) and TPU.

---

## Jax as drop-in replacement for Numpy

```python
import numpy as np

data = np.random.random((10, 10_000))
data[5, 42] = np.nan
data[7, 1111] = np.nan

# compute 90th percentile ignoring NaNs, 
# and along the rows of an array
np.nanpercentile(data, 90, axis=0)

import jax.numpy as jnp
jnp.nanpercentile(data, 90, axis=0)
```

---

## Jax as JIT compiler

```python
import numpy as np
import jax.numpy as jnp
from jax import jit

def a_plus_sin_b_numpy(x, y):
    return x + np.sin(y)

@jit
def a_plus_sin_b_jax(x, y):
    return x + jnp.sin(y)

N = 10_000_000
vec = np.random.rand(N)
a_plus_sin_b_numpy(vec, vec)
a_plus_sin_b_jax(vec, vec)  # slightly faster on CPU

import jax
print(jax.devices())  # check for CUDA
vec_d = jax.device_put(mx)

a_plus_sin_b_jax(vec_d, vec_d)  # much faster on GPU
```

---

## CuPy

<img src="https://cupy.dev/images/cupy.png" width="20%">

- Supports a subset of Python, Numpy and Scipy
- Precompiled wheels for CUDA and ROCm
- Drop-in support for Numpy and Scipy
	```python
	import cupy as cp
	import cupyx.scipy.fft as cufft
	```
- JIT compilation
- [Array API](https://data-apis.org/array-api/2023.12/index.html) support => Cupy arrays can use Numpy functions

---

## CuPy as drop-in replacement for Numpy


```python
import cupy as cp
a = cp.random.random(size=1000)
b = cp.random.random(size=1000)
c = a + cp.sin(b)
```

---

## CuPy as compiler for custom kernels

```python
a_plus_sin_b = cp.ElementwiseKernel(
   'float64 a, float64 b',
   'float64 c',
   'c = a + sin(b)',
   'a_plus_sin_b'
)
a_plus_sin_b
```

---

## Libraries for lower level implementation

- CUDA Python
- HIP Python 

HIP Python + `hip-python-as-cuda` makes it interoperable with CUDA Python code

**Third-party implementations**

- PyCUDA
- PyOpenCL

---

## Summary of Python libraries

<div class="r-fit-text">


| Feature             | Numba | Jax | CuPy | Cuda / HIP Python | PyCUDA / PyOpenCL |
| ------------------- | ----- | --- | ---- | ----------------- | ----------------- |
| Low level           | ✅     | 🤷  | ✅    | ✅                 | ✅                 |
| High level          | ✅     | ✅   | ✅    | ❌                 | ❌                 |
| Numpy compat        | ✅     | ✅   | ✅    | 🤷                | 🤷                |
| CUDA/ROCm interoperability         | 🤷    | ✅   | ✅    | ✅                 | 🤷                |
| Pre-compiled kernel | ❌     | ✅   | ✅    | ❌                 | ❌                 |
| Custom kernels      | ✅     | ✅   | ✅    | ✅                 | ✅                 |

</div>

---

<div class="r-fit-text">

## Julia


<img src="https://julialang.org/assets/infra/logo.svg" width="40%"><img src="https://cuda.juliagpu.org/dev/assets/logo.png" width="40%" align="right">


- Base arrays
- JuliaGPU project
- CUDA.jl, AMDGPU.jl, oneAPI.jl, Metal.jl
- KernelAbstractions

</div>

---

## About Julia

> Primer on Julia and the Base libary

- First, released in 2012 and inspired by MATLAB, Lua, Lisp, Python ....
- JIT compiled with its core implemented using C and LLVM


---

## Array programming in Julia

> Primer on Julia and the Base libary

```julia
a = rand(1000);
b = rand(1000);

c = a .+ sin.(b)
```

- Batteries included with `AbstractArray` and Base.Array
- Sub-types of arrays: `Vector` (1D), `Matrix` (2D),  `Array` (N-D)

---

## CUDA.jl,  AMDGPU.jl, oneAPI.jl, Metal.jl

```julia
using CUDA  # or AMDGPU or oneAPI or Metal

a = rand(1000);
a_d = CuArray(a);  # or ROCArray, oneArray, MtlArray

b_d = CUDA.rand(1000)

# Same builtin function from Base can be used
c = a_d .+ sin.(b_d)
```

---

<div class="r-fit-text">
<h2>KernelAbstractions for generic functions</h2>

```julia
using KernelAbstractions

@kernel function a_plus_sin_b!(C, @Const(A), @Const(B))
	 i = @index(Global)  # get threadIdx
	 @inbounds C[i] = A[i] + sin(B[i])
end
```

```julia
using CUDA
a_d = CUDA.rand(1000)
b_d = CUDA.rand(1000)

c_d = similar(a_d)

backend = get_backend(a_d)
a_plus_sin_b!(backend, size(c_d))(
	c_d, a_d, b_d, ndrange=size(c_d)
)
```

</div>
