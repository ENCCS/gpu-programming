//
// nvcc -O3 -DUSE_DP 01_array_reduce_cpu.cu 
// nvcc -O3          01_array_reduce_cpu.cu
// 
#include "error_checker.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real; // for double precision
#else
    typedef float real;  // for single precision
#endif

const int NUM_REPEATS = 20;

void timing(const real *x, const int NX);
real reduce(const real *x, const int NX);

int main(int argc, const char * argv[])
{

    printf("\n--Beginning of the main function.\n");
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nUsing Device %d: %s\n", dev, deviceProp.name);

    const int NX = 100000000;
    const int size_array = sizeof(real) * NX;
    real *x = (real *) malloc(size_array);
    for (int n = 0; n < NX; ++n)
        x[n] = 1.23;

    timing(x, NX);

    free(x);

    printf("\n--Ending of the main function.\n\n");
    return 0;
}


void timing(const real *x, const int NX)
{
    real sum = 0;
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(x, NX);

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


real reduce(const real *x, const int NX)
{
    real sum = 0.0;
    for (int n = 0; n < NX; ++n)
        sum += x[n];
    return sum;
}


