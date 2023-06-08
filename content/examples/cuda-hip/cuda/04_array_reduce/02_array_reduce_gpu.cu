//
// nvcc -O3 -DUSE_DP 02_array_reduce_gpu.cu
// nvcc -O3          02_array_reduce_gpu.cu
//
#include "error_checker.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 20;
const int NX = 100000000;
const int size_array = sizeof(real) * NX;
const int BLOCK_SIZE = 128;

void timing(real *h_x, real *d_x, const int method);

int main(int argc, const char * argv[])
{

    printf("\n--Beginning of the main function.\n");
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nUsing Device %d: %s\n", dev, deviceProp.name);

    real *h_x = (real *)malloc(size_array);
    for (int n = 0; n < NX; ++n)
        h_x[n] = 1.23;
    real *d_x;
    CHECK(cudaMalloc(&d_x, size_array));

    printf("\n\tUsing global memory only:\n");
    timing(h_x, d_x, 0);
    printf("\n\tUsing static shared memory:\n");
    timing(h_x, d_x, 1);
    printf("\n\tUsing dynamic shared memory:\n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));

    printf("\n--Ending of the main function.\n\n");
    return 0;
}


void __global__ reduction_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    real *x = d_x + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
            x[tid] += x[tid + offset];
        __syncthreads();
    }
    if (tid == 0)
        d_y[blockIdx.x] = x[0];
}


void __global__ reduction_static_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[128];
    s_y[tid] = (n < NX) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if (tid == 0)
        d_y[bid] = s_y[0];
}

void __global__ reduction_dynamic_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < NX) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
            s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if (tid == 0)
        d_y[bid] = s_y[0];
}


real reduction(real *d_x, const int method)
{
    int grid_size = (NX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *) malloc(ymem);

    switch (method)
    {
        case 0:
            reduction_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y); break;
        case 1:
            reduction_static_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y); break;
        case 2:
            reduction_dynamic_shared<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y); break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));

    real result = 0.0;
    for (int n = 0; n < grid_size; ++n)
        result += h_y[n];

    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}


void timing(real *h_x, real *d_x, const int method)
{
    real sum = 0;
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        CHECK(cudaMemcpy(d_x, h_x, size_array, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduction(d_x, method);

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

