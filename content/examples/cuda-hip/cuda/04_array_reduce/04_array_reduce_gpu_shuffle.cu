//
// nvcc -O3 -DUSE_DP 04_array_reduce_gpu_shuffle.cu
// nvcc -O3          04_array_reduce_gpu_shuffle.cu
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
const unsigned FULL_MASK = 0xffffffff;

void timing(const real *d_x, const int method);

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

    printf("\n\tUsing syncwarp:\n");
    timing(d_x, 0);
    printf("\n\tUsing shuffle:\n");
    timing(d_x, 1);
    printf("\n\tUsing cooperative group:\n");
    timing(d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));

    printf("\n--Ending of the main function.\n\n");
    return 0;
}


void __global__ reduce_syncwarp(const real *d_x, real *d_y, const int NX)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < NX) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncwarp();
    }

    if (tid == 0)
        atomicAdd(d_y, s_y[0]);
}

void __global__ reduce_shfl(const real *d_x, real *d_y, const int NX)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < NX) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    real y = s_y[tid];

    for (int offset = 16; offset > 0; offset >>= 1)
        y += __shfl_down_sync(FULL_MASK, y, offset);

    if (tid == 0)
        atomicAdd(d_y, y);
}

void __global__ reduce_cp(const real *d_x, real *d_y, const int NX)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < NX) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    real y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
        y += g.shfl_down(y, i);

    if (tid == 0)
        atomicAdd(d_y, y);
}

real reduce(const real *d_x, const int method)
{
    const int grid_size = (NX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    CHECK(cudaMalloc(&d_y, sizeof(real)));
    CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));

    switch (method)
    {
        case 0:
            reduce_syncwarp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, NX); break;
        case 1:
            reduce_shfl<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, NX); break;
        case 2:
            reduce_cp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, NX); break;
        default:
            printf("\tWrong method.\n"); exit(1);
    }

    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];
}

void timing(const real *d_x, const int method)
{
    real sum = 0;
    
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method); 

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


