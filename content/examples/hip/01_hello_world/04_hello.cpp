#include "hip/hip_runtime.h"
//
// nvcc 04_hello.cu
//
#include <stdio.h>

void __global__ hello_from_gpu()
{
    int bx=blockIdx.x; int tx=threadIdx.x;
    printf("Hello World from GPU. (BLOCK=%d and THREAD=%d)!\n", bx, tx);
}


int main(int argc, const char * argv[])
{
	printf("\n----------------------\n");
	printf("Hello World from CPU! Before calling 'hello_from_gpu' kernel function.\n");

    hipLaunchKernelGGL(hello_from_gpu, 2, 4, 0, 0);

	printf("Hello World from CPU!  After calling 'hello_from_gpu' kernel function.\n");
	printf("\n----------------------\n");

    hipDeviceSynchronize(); // hipDeviceReset();

    return 0;
}
