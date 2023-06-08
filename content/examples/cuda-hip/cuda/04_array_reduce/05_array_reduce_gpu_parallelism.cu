//
// nvcc -O3 -DUSE_DP 05_array_reduce_gpu_parallelism.cu
// nvcc -O3          05_array_reduce_gpu_parallelism.cu
//
#include "error_checker.cuh"
#include <stdio.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 20;
const int NX = 100000000;
const int size_array = sizeof(real) * NX;
const int BLOCK_SIZE = 128;
const int GRID_SIZE = 10240;

void timing(const real *h_x);

int main(int argc, const char * argv[])
{

    printf("\n--Beginning of the main function.\n");
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nUsing Device %d: %s\n", dev, deviceProp.name);

    real *h_x = (real *) malloc(size_array);
    for (int n = 0; n < NX; ++n)
        h_x[n] = 1.23;
    real *d_x;
    CHECK(cudaMalloc(&d_x, size_array));
    CHECK(cudaMemcpy(d_x, h_x, size_array, cudaMemcpyHostToDevice));

    timing(d_x);

    free(h_x);
    CHECK(cudaFree(d_x));

    printf("\n--Ending of the main function.\n\n");
    return 0;
}

void __global__ reduce_cp(const real *d_x, real *d_y, const int NX)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (int n = bid * blockDim.x + tid; n < NX; n += stride)
        y += d_x[n];
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
        y += g.shfl_down(y, i);

    if (tid == 0)
        d_y[bid] = y;
}

real reduce(const real *d_x)
{
    const int ymem = sizeof(real) * GRID_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));

    reduce_cp<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_x, d_y, NX);
    reduce_cp<<<1, 1024, sizeof(real) * 1024>>>(d_y, d_y, GRID_SIZE);

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void timing(const real *d_x)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x); 

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("\tTime = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("\tSum = %f.\n", sum);
}


