//
// nvcc 04_hello.cu
//
#include <stdio.h>

void __global__ hello_from_gpu()
{
    printf("Hello World from GPU. (BLOCK=%d and THREAD=%d)!\n", blockIdx.x, threadIdx.x);
}


int main(int argc, char * argv[])
{
	printf("\n----------------------\n");
	printf("Hello World from CPU! Before calling 'hello_from_gpu' kernel function.\n");

    hello_from_gpu<<<2, 4>>>();

	printf("Hello World from CPU!  After calling 'hello_from_gpu' kernel function.\n");
	printf("\n----------------------\n");

    cudaDeviceSynchronize();
    return 0;
}
