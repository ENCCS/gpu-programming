#include "hip/hip_runtime.h"
//
// nvcc 05_hello.cu
//
#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    printf("Hello World from GPU (block-%d and thread-(%d, %d))!\n", bid, tidx, tidy);
}


int main(int argc, const char * argv[])
{
	printf("\n----------------------\n");
	printf("Hello World from CPU! Before calling 'hello_from_gpu' kernel function.\n");

    const dim3 block_size(4, 8);
    hipLaunchKernelGGL(hello_from_gpu, 1, block_size, 0, 0);

	printf("Hello World from CPU!  After calling 'hello_from_gpu' kernel function.\n");
	printf("\n----------------------\n");

    hipDeviceSynchronize(); // hipDeviceReset();
    return 0;
}
