# Matrix Transpose

Assuming a matrix `a` of size `(NxM)` how to improve matrix operations on GPU? In particular the transposing operation `b(i,j)=a(j,i)`. We will compare the execution times and the effective bandwidth between a simple `copy` kernel, a  `naive` transpose implementation, and two more optimized versions using `shared memory` (with and without bank conflicts). The time is mesured using the `events`. The effective bandwidth is computed as the ratio between the total memory read and written by the kernel (`2 x Total size of the Matrix in Gbytes`) and the execution time in seconds. 

## Copy kernel
The base line for our experiment is the simple copy kernel. 
```
__global__ void copy_kernel(float *in, float *out, int width, int height) {
  int x_index = blockIdx.x * tile_dim + threadIdx.x;
  int y_index = blockIdx.y * tile_dim + threadIdx.y;

  int index = y_index * width + x_index;

  out[index] = in[index];
}
```
This kernel is only reading the data from the input matrix to the output matrix. No optimizations are needed except for minor tuning in the  number of threads per block. All reads from and writes to the GPU memory are coalesced and it is maximum bandwidth that one could achive on a given machine in a kernel.

## Naive transpose
This is the first transpose version where each the reads are done in a coalesced way, but not the writing.

```

__global__ void transpose__naive_kernel(float *in, float *out, int width, int height) {
  int x_index = blockIdx.x * tile_dim + threadIdx.x;
  int y_index = blockIdx.y * tile_dim + threadIdx.y;

  int in_index = y_index * width + x_index;
  int out_index = x_index * height + y_index;

  out[out_index] = in[in_index];
}
```
The index `in_index` increases with `threadIdx.x`, two adjacent threads, `threadIdx.x` and `threadIdx.x+1`, access elements near each other in the gloabl memory. This ensures coalesced reads. On the other hand the writing is strided. Two adjacent threads write to location in memory far away from each other by `height`.

## Transpose with shared memory
Shared Memory (SM) can be used in order to avoid the uncoalesced writing mentioned above.
```
__global__ void transpose_SM_kernel(float *in, float *out, int width,
                                     int height) {
  __shared__ float tile[tile_dim][tile_dim];

  int x_tile_index = blockIdx.x * tile_dim;
  int y_tile_index = blockIdx.y * tile_dim;

  int in_index =
      (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
  int out_index =
      (x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

  tile[threadIdx.y][threadIdx.x] = in[in_index];

  __syncthreads();

  out[out_index] = tile[threadIdx.x][threadIdx.y];
}
``` 
The shared memory is local to each CU with about 100 time slower latency than the global memory. While there is an extra synchronization needed to ensure that the data has been saved locally, the gain in switching from uncoalesced to coalesced accesses outweights the loss. The reading and writing of SM can be done in any order as long as there are no bank conflicts. While the first SM access `tile[threadIdx.y][threadIdx.x] = in[in_index];` is free on bank conflicts the secone one `out[out_index] = tile[threadIdx.x][threadIdx.y];`. When bank conflicts occur the access to the data is serialized. Even so the gain of using SM is quite big.  

## Transpose with shared memory and no bank conflicts
The bank conflicts in this case can be solved in a very simple way. We pad the shared matrix. Instead of `__shared__ float tile[tile_dim][tile_dim];` we use `__shared__ float tile[tile_dim][tile_dim+1];`. Effectively this shifts the data in the banks. Hopefully this does not create other banks conflicts!!!!
```
__global__ void transpose_SM_nobc_kernel(float *in, float *out, int width,
                                     int height) {
  __shared__ float tile[tile_dim][tile_dim+1];

  int x_tile_index = blockIdx.x * tile_dim;
  int y_tile_index = blockIdx.y * tile_dim;

  int in_index =
      (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
  int out_index =
      (x_tile_index + threadIdx.y) * height + (y_tile_index + threadIdx.x);

  tile[threadIdx.y][threadIdx.x] = in[in_index];

  __syncthreads();

  out[out_index] = tile[threadIdx.x][threadIdx.y];
}
``` 

For the optimizations exercise get aquinted with the code, compile them and execute them. For each case try to tune the threads per block (by changing `tile_dim`) and find the configuration which improve the performance  the most and also the ones which do not. As a reference the `V100` has 84 Streaming Multiprocessors (nvidia equivalent of CU) and a peak bandwidth of `900 GB/s`.


In this exercise it is pretty intuitive what is needed to be done to improve the performance.  Measuring the time by events is suficient, but in general  in order to obtain more information about how various parts of the application behave a **profiler** is recommended. `HIP` does not provide us with profilers, they are provided by the back end on top of which they are running. On Nvidia platforms we can use the tools [Nsight Systems](https://docs.csc.fi/computing/nsys/) and [Nsight Compute](https://docs.csc.fi/computing/ncu/).

