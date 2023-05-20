//
// nvcc 03_hello.cu
// run with the command './a.exe' and './a.exe > out.txt'
//	   to get results from CPU and GPU with different output orders
//
#include <stdio.h>

void __global__ hello_from_gpu()
{
    printf("\n\n---Hello World from the GPU!\n\n");
}


int main(int argc, char * argv[])
{
	printf("\n----------------------\n");
	printf("Hello World from CPU! Before calling 'hello_from_gpu' kernel function.\n");
	
    hello_from_gpu<<<1, 1>>>();

	printf("Hello World from CPU!  After calling 'hello_from_gpu' kernel function.\n");
	printf("\n----------------------\n");

	cudaDeviceSynchronize(); //cudaDeviceReset();
    return 0;
}
