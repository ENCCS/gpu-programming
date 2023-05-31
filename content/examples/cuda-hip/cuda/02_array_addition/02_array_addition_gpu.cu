//
// nvcc 02_array_addition_gpu.cu
//
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ array_addition(const double *vecA, const double *vecB, double *vecC);
void array_check(const double *vecC, const int NX);


int main(int argc, const char * argv[])
{

	printf("\n--Beginning of the main function.\n");

    const int NX = 25600004; // try 25600000, 25600004 and 25600008
	int size_array = sizeof(double) * NX;

    double *h_vecA = (double *)malloc(size_array); // 'h' for host (CPU)
    double *h_vecB = (double *)malloc(size_array);
    double *h_vecC = (double *)malloc(size_array);

    for (int i = 0; i < NX; i++)
    {
        h_vecA[i] = a;
        h_vecB[i] = b;
    }

    double *d_vecA, *d_vecB, *d_vecC; // 'd' for device (GPU)
    cudaMalloc((void **)&d_vecA, size_array);
    cudaMalloc((void **)&d_vecB, size_array);
    cudaMalloc((void **)&d_vecC, size_array);
    cudaMemcpy(d_vecA, h_vecA, size_array, cudaMemcpyHostToDevice); // copy data from host(CPU) to device(GPU)
    cudaMemcpy(d_vecB, h_vecB, size_array, cudaMemcpyHostToDevice);

    const int block_size = 128;
    int grid_size = (NX + block_size - 1) / block_size;

	printf("\n\tArray_size = %d, grid_size = %d and block_size = %d.\n", NX, grid_size, block_size);
    array_addition<<<grid_size, block_size>>>(d_vecA, d_vecB, d_vecC);

    cudaMemcpy(h_vecC, d_vecC, size_array, cudaMemcpyDeviceToHost); // copy data from device(GPU) to host(CPU)
    array_check(h_vecC, NX);

    free(h_vecA);
    free(h_vecB);
    free(h_vecC);
    cudaFree(d_vecA);
    cudaFree(d_vecB);
    cudaFree(d_vecC);

	printf("\n--Ending of the main function.\n\n");

    return 0;
}


void __global__ array_addition(const double *vecA, const double *vecB, double *vecC)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    vecC[i] = vecA[i] + vecB[i];
}


void array_check(const double *vecC, const int NX)
{
    bool has_error = false;
    for (int i = 0; i < NX; i++)
    {
        if (fabs(vecC[i] - c) > EPSILON)
		{
            has_error = true;
			break;
		}
    }
    printf("\n\tChecking array addition results >>> %s\n", has_error? "|| ERROR ||":"|| NO ERROR ||");
}

